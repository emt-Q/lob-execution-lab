"""L2 limit order book reconstruction (Binance spot semantics).

Binance publishes:
  * a REST snapshot  -> {lastUpdateId, bids: [[p, q], ...], asks: [...]}
  * diff depth stream -> {U: firstUpdateId, u: finalUpdateId, b, a}
A diff sets the aggregate quantity at a price level (qty 0 = delete level).

The standard local reconstruction algorithm:
  1. buffer diffs, fetch snapshot with lastUpdateId S;
  2. discard buffered diffs with u <= S;
  3. the first applied diff must satisfy U <= S+1 <= u;
  4. afterwards each diff must satisfy U == previous_u + 1 (no gap).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .types import DepthEvent


class OrderBook:
    """Aggregate L2 book. Prices map to total resting quantity."""

    def __init__(self, max_levels: int | None = None):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.max_levels = max_levels
        self.last_update_id: int = -1

    # -- mutation ----------------------------------------------------------
    def apply_snapshot(self, last_update_id: int, bids, asks) -> None:
        self.bids = {float(p): float(q) for p, q in bids if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in asks if float(q) > 0}
        self.last_update_id = last_update_id
        self._trim()

    def apply_levels(self, bids, asks) -> None:
        for p, q in bids:
            p, q = float(p), float(q)
            if q == 0.0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q
        for p, q in asks:
            p, q = float(p), float(q)
            if q == 0.0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q
        self._trim()

    def _trim(self) -> None:
        if self.max_levels is None:
            return
        if len(self.bids) > self.max_levels:
            keep = sorted(self.bids, reverse=True)[: self.max_levels]
            self.bids = {p: self.bids[p] for p in keep}
        if len(self.asks) > self.max_levels:
            keep = sorted(self.asks)[: self.max_levels]
            self.asks = {p: self.asks[p] for p in keep}

    # -- views -------------------------------------------------------------
    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return (bb + ba) / 2.0

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba - bb

    @property
    def best_bid_qty(self) -> float:
        bb = self.best_bid
        return self.bids[bb] if bb is not None else 0.0

    @property
    def best_ask_qty(self) -> float:
        ba = self.best_ask
        return self.asks[ba] if ba is not None else 0.0

    def microprice(self) -> float | None:
        """Quantity-weighted reference price (weighted by opposite qty)."""
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        qb, qa = self.bids[bb], self.asks[ba]
        if qb + qa == 0:
            return None
        return (bb * qa + ba * qb) / (qb + qa)

    def levels(self, side: str, n: int) -> list[tuple[float, float]]:
        book = self.bids if side == "bid" else self.asks
        prices = sorted(book, reverse=(side == "bid"))[:n]
        return [(p, book[p]) for p in prices]

    def qty_at(self, side: str, price: float) -> float:
        book = self.bids if side == "bid" else self.asks
        return book.get(price, 0.0)

    def walk(self, side: str, qty: float) -> tuple[float, list[tuple[float, float]], float]:
        """Simulate an aggressive order sweeping the book.

        Returns (vwap_of_consumed, [(price, qty), ...], remaining_qty).
        A BUY walks asks upward; a SELL walks bids downward.
        """
        book = self.asks if side == "buy" else self.bids
        prices = sorted(book) if side == "buy" else sorted(book, reverse=True)
        fills, remaining, notional = [], qty, 0.0
        for p in prices:
            if remaining <= 0:
                break
            take = min(book[p], remaining)
            fills.append((p, take))
            notional += p * take
            remaining -= take
        vwap = notional / (qty - remaining) if qty - remaining > 0 else 0.0
        return vwap, fills, remaining

    def snapshot_payload(self) -> dict:
        return {
            "lastUpdateId": self.last_update_id,
            "bids": [[p, q] for p, q in self.levels("bid", len(self.bids))],
            "asks": [[p, q] for p, q in self.levels("ask", len(self.asks))],
        }


@dataclass
class ReconstructionReport:
    snapshot_id: int
    applied_diffs: int
    gaps: list


class BookReconstructor:
    """Stateful validator/rebuilder following the Binance recipe."""

    def __init__(self, max_levels: int | None = None):
        self.book = OrderBook(max_levels=max_levels)
        self._buffer: deque[DepthEvent] = deque()
        self._snapshot_id: int | None = None
        self._prev_u: int | None = None
        self.gaps: list = []
        self.applied = 0

    def ingest_snapshot(self, last_update_id: int, bids, asks) -> None:
        self.book.apply_snapshot(last_update_id, bids, asks)
        self._snapshot_id = last_update_id
        self._prev_u = None
        # drop buffered diffs that predate the snapshot
        while self._buffer and self._buffer[0].final_update_id <= last_update_id:
            self._buffer.popleft()
        while self._buffer:
            self._apply_diff(self._buffer.popleft())

    def ingest_diff(self, ev: DepthEvent) -> None:
        if self._snapshot_id is None:
            self._buffer.append(ev)
            return
        self._apply_diff(ev)

    def _apply_diff(self, ev: DepthEvent) -> None:
        s = self._snapshot_id
        if self._prev_u is None:
            if not (ev.first_update_id <= s + 1 <= ev.final_update_id):
                self.gaps.append(
                    f"first diff misaligned: U={ev.first_update_id} "
                    f"snapshot={s} u={ev.final_update_id}"
                )
        elif ev.first_update_id != self._prev_u + 1:
            self.gaps.append(
                f"sequence gap: expected U={self._prev_u + 1} got {ev.first_update_id}"
            )
        self.book.apply_levels(ev.bids, ev.asks)
        self.book.last_update_id = ev.final_update_id
        self._prev_u = ev.final_update_id
        self.applied += 1

    def report(self) -> ReconstructionReport:
        return ReconstructionReport(self._snapshot_id or -1, self.applied, list(self.gaps))
