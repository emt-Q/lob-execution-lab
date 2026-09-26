"""Execution strategies: map a parent order state to child decisions.

A strategy is a pure function of a :class:`DecisionContext` and returns a
list of :class:`Action` objects (submit / cancel). The backtester owns state
transition and latency; strategies hold no state of their own.

The four strategies are:
  * MarketTake        -- immediately cross the spread (taker);
  * JoinBest          -- rest at the current best bid/ask;
  * ImproveOneTick    -- rest one tick better (queue-jump when spread is wide);
  * AdaptiveSignal    -- signal + inventory aware dynamic posting.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from core.types import Side


class ActionType(str, Enum):
    SUBMIT = "submit"
    CANCEL = "cancel"


@dataclass
class Submit:
    side: Side
    price: float
    qty: float
    post: bool                 # True = resting limit; False = marketable / take


@dataclass
class Cancel:
    order_id: int


@dataclass
class ParentOrder:
    parent_id: int
    side: Side
    total: float
    start_ts: int
    deadline_ts: int
    horizon_events: int
    filled: float = 0.0
    child_size: float = 0.0
    active_orders: dict = field(default_factory=dict)
    done: bool = False

    @property
    def remaining(self) -> float:
        return max(0.0, self.total - self.filled)


@dataclass
class DecisionContext:
    parent: ParentOrder
    seq: int
    ts: int
    best_bid: float
    best_ask: float
    tick: float
    child_qty: float
    # progress bookkeeping
    events_since_start: int
    n_children_filled: int
    last_fill_event: int
    predictor: object = None


class Strategy:
    name = "base"

    def decide(self, ctx: DecisionContext) -> list:
        raise NotImplementedError


class MarketTake(Strategy):
    name = "market_take"

    def decide(self, ctx: DecisionContext) -> list:
        qty = min(ctx.child_qty, ctx.parent.remaining)
        if qty <= 0:
            return []
        px = ctx.best_ask if ctx.parent.side is Side.BUY else ctx.best_bid
        return [Submit(ctx.parent.side, px, qty, post=False)]


class JoinBest(Strategy):
    name = "join_best"

    def decide(self, ctx: DecisionContext) -> list:
        if ctx.parent.active_orders:
            return []
        qty = min(ctx.child_qty, ctx.parent.remaining)
        if qty <= 0:
            return []
        px = ctx.best_bid if ctx.parent.side is Side.BUY else ctx.best_ask
        return [Submit(ctx.parent.side, px, qty, post=True)]


class ImproveOneTick(Strategy):
    name = "improve_one_tick"

    def decide(self, ctx: DecisionContext) -> list:
        if ctx.parent.active_orders:
            return []
        qty = min(ctx.child_qty, ctx.parent.remaining)
        if qty <= 0:
            return []
        if ctx.parent.side is Side.BUY:
            px = ctx.best_bid + ctx.tick
            marketable = px >= ctx.best_ask
        else:
            px = ctx.best_ask - ctx.tick
            marketable = px <= ctx.best_bid
        return [Submit(ctx.parent.side, px, qty, post=not marketable)]


class AdaptiveSignal(Strategy):
    """Signal + inventory schedule.

    Decision at each child opportunity:
      * strong adverse predicted move, or badly behind the linear schedule =>
        take now;
      * mild / no signal => improve one tick (queue-jump when possible);
      * predicted move in our favor => rest at best (earn the spread while the
        market comes to us);
      * existing resting quotes are pulled if the signal flips against them.
    """

    name = "adaptive_signal"

    H = 5  # prediction horizon used for execution

    def decide(self, ctx: DecisionContext) -> list:
        actions: list = []
        pred = ctx.predictor
        side = ctx.parent.side
        # signed predicted move in our favor, in ticks
        e_move = (pred.expected_move(ctx.seq, self.H) if pred else 0.0)
        e_favor = side.sign * e_move

        # cancel resting quotes if the signal now points against them
        if e_favor < -0.4:
            for oid in list(ctx.parent.active_orders):
                actions.append(Cancel(oid))

        if ctx.parent.active_orders and not actions:
            return []

        # linear schedule: expected fraction filled by now
        frac_time = ctx.events_since_start / max(ctx.parent.horizon_events, 1)
        frac_done = ctx.parent.filled / ctx.parent.total
        behind = frac_done < frac_time - 0.15

        qty = min(ctx.child_qty, ctx.parent.remaining)
        if qty <= 0:
            return actions

        strong_against = e_favor < -0.8
        if strong_against or behind:
            px = ctx.best_ask if side is Side.BUY else ctx.best_bid
            actions.append(Submit(side, px, qty, post=False))
        elif e_favor > 0.5:
            px = ctx.best_bid if side is Side.BUY else ctx.best_ask
            actions.append(Submit(side, px, qty, post=True))
        else:
            if side is Side.BUY:
                px = ctx.best_bid + ctx.tick
                marketable = px >= ctx.best_ask
            else:
                px = ctx.best_ask - ctx.tick
                marketable = px <= ctx.best_bid
            actions.append(Submit(side, px, qty, post=not marketable))
        return actions


STRATEGIES = {
    MarketTake.name: MarketTake,
    JoinBest.name: JoinBest,
    ImproveOneTick.name: ImproveOneTick,
    AdaptiveSignal.name: AdaptiveSignal,
}
