"""Event-driven execution backtester with explicit latency semantics.

Timeline (L = one-way latency, symmetric for market data and orders):

  exchange time:   event e occurs at T
  trader sees it:  wall time T + L          (market-data latency)
  trader decides:  strategy runs on book/events with time <= T
  order reaches:   exchange at T + 2L       (order latency)
  fills happen:    at exchange time, via trade prints or marketable sweep

The exchange book is ground truth; the *visible* state at a decision equals
the exchange book after applying the event that just arrived, which by
construction contains no information newer than T.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config
from core.orderbook import OrderBook
from core.types import Fill, Side
from execution.queue_model import QueueSimulator, RestingOrder
from execution.strategies import (Cancel, DecisionContext, ParentOrder,
                                  Strategy, Submit)


# ---------------------------------------------------------------------------
# Parent-order schedule (shared across strategies for paired comparison)
# ---------------------------------------------------------------------------
def generate_parent_schedule(n_parents: int, n_events: int, split: int,
                             deadline_events: int, seed: int,
                             buffer: int = 40):
    rng = np.random.default_rng(seed)
    lo = split + config.FEATURE_WINDOW + buffer
    hi = n_events - deadline_events - max(config.HORIZONS) - buffer
    if hi <= lo:
        raise ValueError("not enough events after train split for backtest")
    starts = np.linspace(lo, hi, n_parents) + rng.uniform(
        -buffer / 2, buffer / 2, n_parents)
    starts = np.clip(starts, lo, hi).astype(int)
    sched = []
    for k, s in enumerate(starts):
        side = Side.BUY if k % 2 == 0 else Side.SELL
        sched.append((side, int(s), int(s) + deadline_events))
    return sched


# ---------------------------------------------------------------------------
# Scheduled order operations
# ---------------------------------------------------------------------------
@dataclass
class _Op:
    arrival_ts: int
    parent_id: int
    action: object
    order_id: int | None = None


class Backtester:
    def __init__(self, events, predictor, strategy: Strategy,
                 latency_ms: int, tick: float = config.TICK_SIZE,
                 n_book_levels: int = config.N_BOOK_LEVELS):
        self.events = events
        self.predictor = predictor
        self.strategy = strategy
        self.L = latency_ms
        self.tick = tick
        self.book = OrderBook(max_levels=n_book_levels)
        self.queues = QueueSimulator()
        self.orders: dict[int, RestingOrder] = {}
        self._orders_by_parent: dict[int, set] = {}
        self.fills: list[Fill] = []
        self.parents: dict[int, ParentOrder] = {}
        self._ops: list[_Op] = []
        self._next_id = 1

    # -- setup -------------------------------------------------------------
    def set_schedule(self, schedule):
        for k, (side, s0, s1) in enumerate(schedule):
            mid = self.predictor.store.mid_at(s0)
            self.parents[k] = ParentOrder(
                parent_id=k, side=side, total=config.PARENT_QTY,
                start_seq=s0, deadline_seq=s1, arrival_mid=mid)

    # -- scheduling --------------------------------------------------------
    def _new_order_id(self) -> int:
        oid = self._next_id
        self._next_id += 1
        return oid

    def _schedule(self, decision_ts: int, parent_id: int, action) -> None:
        arrival = decision_ts + 2 * self.L
        oid = None
        if isinstance(action, Submit):
            oid = self._new_order_id()
            self.parents[parent_id].pending_submits += 1
        elif isinstance(action, Cancel):
            oid = action.order_id
        self._ops.append(_Op(arrival, parent_id, action, oid))

    # -- op execution at exchange -----------------------------------------
    def _apply_due_ops(self, ts: int, equal: bool) -> None:
        remaining = []
        for op in sorted(self._ops, key=lambda o: o.arrival_ts):
            due = (op.arrival_ts < ts) or (equal and op.arrival_ts == ts)
            if due:
                self._execute_op(op)
            else:
                remaining.append(op)
        self._ops = remaining

    def _execute_op(self, op: _Op) -> None:
        a = op.action
        parent = self.parents[op.parent_id]
        if isinstance(a, Cancel):
            order = self.orders.get(op.order_id)
            if order is not None and order.alive:
                self.queues.cancel(order)
            return
        if isinstance(a, Submit):
            parent.pending_submits -= 1
            self._execute_submit(op, a, parent)

    def _fee(self, notional: float, maker: bool) -> float:
        bps = config.MAKER_FEE_BPS if maker else config.TAKER_FEE_BPS
        return notional * bps / 1e4

    def _emit_fill(self, order, parent, price, qty, ts, seq, maker,
                   deadline=False, source: str = "") -> Fill:
        notional = price * qty
        fee = self._fee(notional, maker)
        f = Fill(order_id=order.order_id, parent_id=parent.parent_id,
                 side=parent.side, price=price, qty=qty, ts=ts,
                 maker=maker, fee=fee, submit_ts=order.submit_ts,
                 is_deadline_liquidation=deadline, seq=seq,
                 source=source or ("resting_fill" if maker else "taker"),
                 mid_at_fill=self.book.mid or 0.0)
        self.fills.append(f)
        order.filled_qty += qty
        order.remaining -= qty
        order.fills.append((ts, qty))
        parent.filled += qty
        return f

    def _execute_submit(self, op: _Op, a: Submit, parent: ParentOrder) -> None:
        # ultimate guard against decision races: never execute more than the
        # parent still needs.
        allowed = parent.total - parent.filled
        if allowed <= 1e-9:
            return
        qty = min(a.qty, allowed)

        order = RestingOrder(
            order_id=op.order_id, parent_id=parent.parent_id, side=a.side,
            price=a.limit_price or 0.0, qty=qty, remaining=qty,
            submit_ts=op.arrival_ts - 2 * self.L,
            submit_seq=self._cur_seq, timeout_events=a.timeout_events)
        self.orders[order.order_id] = order
        self._orders_by_parent.setdefault(parent.parent_id, set()).add(
            order.order_id)
        deadline = a.tag in ("deadline_take",)

        # market order, or marketable limit: sweep the opposite side
        vwap, sweeps, rem = self.book.walk(a.side.value, qty)
        marketable_levels = []
        if a.limit_price is None:
            marketable_levels = sweeps
        else:
            for p, q in sweeps:
                ok = (a.side is Side.BUY and p <= a.limit_price) or \
                     (a.side is Side.SELL and p >= a.limit_price)
                if ok:
                    marketable_levels.append((p, q))
                else:
                    rem += q
        swept = 0.0
        for p, q in marketable_levels:
            q = min(q, parent.total - parent.filled)
            if q <= 0:
                continue
            self._emit_fill(order, parent, p, q, op.arrival_ts,
                            self._cur_seq, maker=False, deadline=deadline,
                            source=a.tag)
            swept += q
        # NOTE: frozen-book replay -- our fills do not mutate the recorded
        # book; subsequent recorded depth updates drive aggregate state.

        rest_qty = qty - swept
        # clamp remainder once more against any fills that just landed
        rest_qty = min(rest_qty, parent.total - parent.filled)
        if rest_qty > 1e-9:
            order.remaining = rest_qty
            order.qty = qty
            if a.limit_price is None:
                # no liquidity left: rest at the deepest cleaned price as a
                # marketable limit (rare for the sizes used here)
                order.price = marketable_levels[-1][0] if marketable_levels \
                    else self.book.mid or 0.0
            price = order.price
            mq = self.book.qty_at(
                "bid" if a.side is Side.BUY else "ask", price)
            self.queues.join(order, mq)
            order.rested = True
        elif rest_qty <= 1e-9 and order.remaining <= 1e-9:
            order.alive = False

    # -- main loop ---------------------------------------------------------
    def run(self) -> list[Fill]:
        # index parents by start; only active parents are touched per event
        starts: dict[int, list[int]] = {}
        for pid, p in self.parents.items():
            starts.setdefault(p.start_seq, []).append(pid)
        active: set[int] = set()

        for ev in self.events:
            i, T = ev.seq, ev.ts
            self._cur_seq = i
            # ops due strictly before this event
            self._apply_due_ops(T, equal=False)

            if ev.depth is not None:
                d = ev.depth
                self.book.apply_levels(d.bids, d.asks)
                for p, q in d.bids:
                    self.queues.on_level(Side.BUY, p, q)
                for p, q in d.asks:
                    self.queues.on_level(Side.SELL, p, q)
            if ev.trade is not None:
                for oid, fq in self.queues.on_trade(ev.trade):
                    order = self.orders[oid]
                    if order.alive:
                        parent = self.parents[order.parent_id]
                        fq = min(fq, max(0.0, parent.total - parent.filled))
                        if fq > 1e-9:
                            self._emit_fill(order, parent, order.price, fq,
                                            T, i, maker=True)
                        if order.remaining <= 1e-9 \
                                or parent.filled >= parent.total - 1e-9:
                            # child done, or parent target hit: stop resting
                            self.queues.cancel(order)
                            order.alive = False

            # decisions: the event arrives at the trader at wall T + L
            active.update(starts.get(i, ()))
            finished: list[int] = []
            for pid in list(active):
                parent = self.parents[pid]
                if parent.filled >= parent.total - 1e-9:
                    # target reached: make sure nothing remains resting
                    for o in self._live(parent):
                        self.queues.cancel(o)
                        o.alive = False
                    finished.append(pid)
                    continue
                # enter liquidation at deadline: cancel everything resting
                if i >= parent.deadline_seq and not parent.liquidating:
                    parent.liquidating = True
                    for o in self._live(parent):
                        self._schedule(T, pid, Cancel(o.order_id))
                if parent.liquidating:
                    # wait for cancels / in-flight submits, then take residual
                    if not self._live(parent) and parent.pending_submits == 0 \
                            and not parent.take_scheduled:
                        rem = parent.total - parent.filled
                        if rem > 1e-9:
                            self._schedule(
                                T, pid, Submit(parent.side, rem,
                                               tag="deadline_take"))
                        parent.take_scheduled = True
                        finished.append(pid)
                    continue
                ctx = self._context(i, T, parent)
                for act in self.strategy.on_decision(ctx):
                    self._schedule(T, pid, act)
            active.difference_update(finished)

            # ops due exactly at T (conservative: after the market event)
            self._apply_due_ops(T, equal=True)

        # safety: flush any late ops (e.g. deadline takes beyond last event)
        for op in sorted(self._ops, key=lambda o: o.arrival_ts):
            self._execute_op(op)
        self._ops.clear()
        return self.fills

    def _live(self, parent: ParentOrder) -> list:
        ids = self._orders_by_parent.get(parent.parent_id, set())
        return [self.orders[oid] for oid in ids if self.orders[oid].alive]

    def _context(self, i: int, T: int, parent: ParentOrder) -> DecisionContext:
        return DecisionContext(
            seq=i, ts=T, book=self.book, parent=parent,
            live=self._live(parent), predictor=self.predictor, tick=self.tick)
