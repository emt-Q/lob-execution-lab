"""FIFO queue-position model for our resting limit orders.

A price level is represented as an ordered list of *blocks*::

    [market qty][our order][market qty][our order] ...

which is exactly price-time priority (FIFO). The model supports:
  * partial fills        -- a trade shrinks blocks front-to-back;
  * cancellations        -- unexplained aggregate decreases (depth qty change
                            minus trade volume) are attributed pro-rata across
                            *market* blocks, the standard queue-model assumption
                            (we cannot observe who cancelled; pro-rata is the
                            unbiased choice, vs. "all cancels ahead" which is
                            optimistic and "all behind" which is pessimistic);
  * queue position       -- quantity ahead of each of our blocks;
  * late joins           -- new aggregate quantity appends at the back.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.types import Side, Trade


@dataclass
class Block:
    qty: float
    order_id: int | None = None       # None => other market participant
    alive: bool = True


@dataclass
class RestingOrder:
    order_id: int
    parent_id: int
    side: Side
    price: float
    qty: float                        # original submitted qty
    remaining: float
    submit_ts: int
    submit_seq: int                   # visible-event seq at submission
    timeout_events: int | None = None
    filled_qty: float = 0.0
    alive: bool = True
    rested: bool = False        # True once it actually joined a queue (maker)
    fills: list = field(default_factory=list)  # (ts, qty)


class PriceQueue:
    def __init__(self, side: Side, price: float, market_qty: float):
        self.side = side
        self.price = price
        self.blocks: list[Block] = []
        if market_qty > 0:
            self.blocks.append(Block(market_qty))
        self.trade_volume_since_reconcile = 0.0

    def total(self) -> float:
        return sum(b.qty for b in self.blocks if b.alive)

    def market_total(self) -> float:
        return sum(b.qty for b in self.blocks if b.alive and b.order_id is None)

    def append_market(self, qty: float) -> None:
        if qty > 0:
            self.blocks.append(Block(qty))

    def append_ours(self, qty: float, order_id: int) -> None:
        self.blocks.append(Block(qty, order_id))

    def qty_ahead_of(self, order_id: int) -> float:
        ahead = 0.0
        for b in self.blocks:
            if not b.alive:
                continue
            if b.order_id == order_id:
                break
            ahead += b.qty
        return ahead

    def remove_ours(self, order_id: int) -> None:
        for b in self.blocks:
            if b.order_id == order_id:
                b.alive = False
                b.qty = 0.0

    def consume_trade(self, qty: float) -> list[tuple[int, float]]:
        """Aggressive trade of `qty` hits this passive price level."""
        fills: list[tuple[int, float]] = []
        remaining = qty
        for b in self.blocks:
            if not b.alive or remaining <= 0:
                break
            take = min(b.qty, remaining)
            b.qty -= take
            remaining -= take
            if b.order_id is not None and take > 0:
                fills.append((b.order_id, take))
        self.blocks = [b for b in self.blocks if b.alive and b.qty > 1e-12]
        self.trade_volume_since_reconcile += qty - remaining
        return fills

    def reconcile(self, aggregate_qty: float) -> None:
        """Aggregate depth (excluding us) is reported as `aggregate_qty`."""
        traded = self.trade_volume_since_reconcile
        self.trade_volume_since_reconcile = 0.0
        market_blocks = [b for b in self.blocks if b.alive and b.order_id is None]
        current_market = sum(b.qty for b in market_blocks)
        cancel = current_market - aggregate_qty
        if cancel > 1e-10:
            # cancellations: pro-rata across market blocks
            if current_market > 0:
                for b in market_blocks:
                    b.qty *= max(0.0, (current_market - cancel) / current_market)
        elif cancel < -1e-10:
            # unexplained adds (depth increase not modeled event-by-event):
            # append at back, preserving FIFO
            self.blocks.append(Block(-cancel))
        self.blocks = [b for b in self.blocks if b.alive and b.qty > 1e-12]


class QueueSimulator:
    """All price levels with at least one of our resting orders."""

    def __init__(self):
        self.queues: dict[tuple[str, float], PriceQueue] = {}

    def _key(self, side: Side, price: float):
        return (side.value, price)

    def join(self, order: RestingOrder, market_qty: float) -> None:
        key = self._key(order.side, order.price)
        q = self.queues.get(key)
        if q is None:
            q = PriceQueue(order.side, order.price, market_qty)
            self.queues[key] = q
        q.append_ours(order.remaining, order.order_id)

    def cancel(self, order: RestingOrder) -> None:
        key = self._key(order.side, order.price)
        q = self.queues.get(key)
        if q is not None:
            q.remove_ours(order.order_id)
        order.alive = False

    def on_trade(self, tr: Trade) -> list[tuple[int, float]]:
        passive = tr.aggressor_side.opposite
        key = self._key(passive, tr.price)
        q = self.queues.get(key)
        if q is None:
            return []
        return q.consume_trade(tr.qty)

    def on_level(self, side: Side, price: float, aggregate_qty: float) -> None:
        key = self._key(side, price)
        q = self.queues.get(key)
        if q is not None:
            q.reconcile(aggregate_qty)

    def qty_ahead(self, order: RestingOrder) -> float:
        key = self._key(order.side, order.price)
        q = self.queues.get(key)
        return q.qty_ahead_of(order.order_id) if q else 0.0
