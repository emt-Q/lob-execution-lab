"""The four execution strategies under comparison.

  1. MarketTake      -- immediately cross the spread (taker).
  2. JoinBest        -- post at best bid (buy) / best ask (sell); maintain
                        price priority; market-take on timeout/deadline.
  3. ImproveOneTick  -- post one tick better than the best on our side; when
                        the spread is one tick this is marketable and degrades
                        to taking, when the spread is wider it jumps the queue
                        as a maker with zero volume ahead.
  4. AdaptiveSignal  -- predicted mid move (ticks) + inventory schedule:
                        strong adverse signal or behind schedule -> take;
                        mild signal -> improve; favorable signal -> join.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.orderbook import OrderBook
from core.types import Side
from model.predictor import MidPredictor


# ---------------------------------------------------------------------------
# Actions / context
# ---------------------------------------------------------------------------
@dataclass
class Submit:
    side: Side
    qty: float
    limit_price: float | None = None    # None => market order
    timeout_events: int | None = None
    tag: str = ""


@dataclass
class Cancel:
    order_id: int


@dataclass
class ParentOrder:
    parent_id: int
    side: Side
    total: float
    start_seq: int
    deadline_seq: int
    arrival_mid: float
    filled: float = 0.0
    pending_take: float = 0.0           # qty to market-take after a cancel lands
    liquidating: bool = False           # deadline entered: wind down, no new ideas
    take_scheduled: bool = False        # deadline market take already scheduled
    pending_submits: int = 0            # submit ops in flight to the exchange


@dataclass
class DecisionContext:
    seq: int
    ts: int
    book: OrderBook
    parent: ParentOrder
    live: list                # list[RestingOrder]
    predictor: MidPredictor | None
    tick: float

    @property
    def remaining(self) -> float:
        return max(0.0, self.parent.total - self.parent.filled)

    def best_on_side(self, side: Side) -> float | None:
        return book_best(self.book, side)

    def first_live(self):
        return self.live[0] if self.live else None


def book_best(book: OrderBook, side: Side) -> float | None:
    return book.best_bid if side is Side.BUY else book.best_ask


class Strategy:
    name = "base"

    def on_decision(self, ctx: DecisionContext) -> list:
        raise NotImplementedError

    # -- shared helpers ----------------------------------------------------
    @staticmethod
    def _timeout_actions(ctx: DecisionContext, order) -> list:
        unfilled = order.remaining
        ctx.parent.pending_take = unfilled
        return [Cancel(order.order_id)]

    def _post_timeout_take(self, ctx: DecisionContext) -> list:
        if ctx.parent.pending_take > 0 and not ctx.live:
            q = ctx.parent.pending_take
            ctx.parent.pending_take = 0.0
            return [Submit(ctx.parent.side, q, tag="timeout_take")]
        return []


# ---------------------------------------------------------------------------
# 1. Immediate take
# ---------------------------------------------------------------------------
class MarketTakeStrategy(Strategy):
    name = "market_take"

    def on_decision(self, ctx: DecisionContext) -> list:
        if ctx.parent.filled <= 0 and not ctx.live and ctx.remaining > 0:
            return [Submit(ctx.parent.side, ctx.parent.total, tag="market")]
        return []


# ---------------------------------------------------------------------------
# 2. Join best bid/ask
# ---------------------------------------------------------------------------
class JoinBestStrategy(Strategy):
    name = "join_best"

    def on_decision(self, ctx: DecisionContext) -> list:
        actions = self._post_timeout_take(ctx)
        if actions:
            return actions
        order = ctx.first_live()
        best = ctx.best_on_side(ctx.parent.side)
        if best is None:
            return []
        if order is not None:
            # best on our side improved ABOVE us -> we lost priority, replace
            moved_away = (ctx.parent.side is Side.BUY and best > order.price) or \
                         (ctx.parent.side is Side.SELL and best < order.price)
            if moved_away:
                return [Cancel(order.order_id)]
            if ctx.seq - order.submit_seq >= (order.timeout_events or 10**9) \
                    and ctx.parent.pending_take <= 0:
                return self._timeout_actions(ctx, order)
            return []
        if ctx.remaining <= 0:
            return []
        q = min(ctx_children(ctx), ctx.remaining)
        return [Submit(ctx.parent.side, q, limit_price=best,
                       timeout_events=ctx_timout(ctx), tag="join")]


# ---------------------------------------------------------------------------
# 3. Improve one tick
# ---------------------------------------------------------------------------
class ImproveOneTickStrategy(Strategy):
    name = "improve_one_tick"

    def target_price(self, ctx: DecisionContext) -> float | None:
        side = ctx.parent.side
        best = ctx.best_on_side(side)
        if best is None:
            return None
        return best + ctx.tick if side is Side.BUY else best - ctx.tick

    def on_decision(self, ctx: DecisionContext) -> list:
        actions = self._post_timeout_take(ctx)
        if actions:
            return actions
        if ctx.best_on_side(ctx.parent.side) is None:
            return []
        target = self.target_price(ctx)
        if target is None:
            return []
        order = ctx.first_live()
        if order is not None:
            need_replace = (ctx.parent.side is Side.BUY and target > order.price) or \
                           (ctx.parent.side is Side.SELL and target < order.price)
            if need_replace:
                return [Cancel(order.order_id)]
            if ctx.seq - order.submit_seq >= (order.timeout_events or 10**9) \
                    and ctx.parent.pending_take <= 0:
                return self._timeout_actions(ctx, order)
            return []
        if ctx.remaining <= 0:
            return []
        q = min(ctx_children(ctx), ctx.remaining)
        return [Submit(ctx.parent.side, q, limit_price=target,
                       timeout_events=ctx_timout(ctx), tag="improve")]


# ---------------------------------------------------------------------------
# 4. Adaptive: prediction signal + inventory dynamics
# ---------------------------------------------------------------------------
@dataclass
class AdaptiveSignalStrategy(Strategy):
    name: str = "adaptive_signal"
    take_threshold: float = 0.8       # predicted ticks (side-adjusted) to take
    improve_threshold: float = 0.15   # predicted ticks to improve vs join
    patient_threshold: float = -0.3
    urgency_take: float = 0.30        # schedule slippage forcing a take
    urgency_double: float = 0.20

    def _signal(self, ctx: DecisionContext) -> float:
        if ctx.predictor is None:
            return 0.0
        import config
        e = ctx.predictor.expected_move(ctx.seq, config.PRIMARY_HORIZON)
        return e * ctx.parent.side.sign

    def _behind(self, ctx: DecisionContext) -> float:
        p = ctx.parent
        total_span = max(1, p.deadline_seq - p.start_seq)
        elapsed = (ctx.seq - p.start_seq) / total_span
        return elapsed - p.filled / p.total

    def on_decision(self, ctx: DecisionContext) -> list:
        actions = self._post_timeout_take(ctx)
        if actions:
            return actions
        sig = self._signal(ctx)
        behind = self._behind(ctx)

        order = ctx.first_live()
        if order is not None:
            # price running away from a resting quote -> cancel and take
            if sig > self.take_threshold + 0.4:
                ctx.parent.pending_take = order.remaining
                return [Cancel(order.order_id)]
            # priority maintenance, same as the plain limit strategies
            best = ctx.best_on_side(ctx.parent.side)
            if best is not None:
                moved = (ctx.parent.side is Side.BUY and best > order.price) or \
                        (ctx.parent.side is Side.SELL and best < order.price)
                if moved:
                    return [Cancel(order.order_id)]
            if ctx.seq - order.submit_seq >= (order.timeout_events or 10**9) \
                    and ctx.parent.pending_take <= 0:
                return self._timeout_actions(ctx, order)
            return []

        if ctx.remaining <= 0:
            return []
        best = ctx.best_on_side(ctx.parent.side)
        if best is None:
            return []

        child = ctx_children(ctx)
        if behind > self.urgency_double:
            child *= 2
        child = min(child, ctx.remaining)

        if sig > self.take_threshold or behind > self.urgency_take:
            return [Submit(ctx.parent.side, child, tag="signal_take")]
        if sig < self.patient_threshold:
            best = ctx.best_on_side(ctx.parent.side)
            return [Submit(ctx.parent.side, child, limit_price=best,
                           timeout_events=ctx_timout(ctx), tag="signal_join")]
        # mild / neutral signal: improve one tick
        side = ctx.parent.side
        best = ctx.best_on_side(side)
        target = best + ctx.tick if side is Side.BUY else best - ctx.tick
        return [Submit(ctx.parent.side, child, limit_price=target,
                       timeout_events=ctx_timout(ctx), tag="signal_improve")]


# context-bound helpers (kept as functions to avoid threading cfg through ctors)
def ctx_children(ctx: DecisionContext) -> float:
    import config
    return config.CHILD_QTY


def ctx_timout(ctx: DecisionContext) -> int:
    import config
    return config.LIMIT_TIMEOUT_EVENTS


STRATEGIES = {
    "market_take": MarketTakeStrategy,
    "join_best": JoinBestStrategy,
    "improve_one_tick": ImproveOneTickStrategy,
    "adaptive_signal": AdaptiveSignalStrategy,
}
