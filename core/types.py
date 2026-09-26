"""Shared event / order dataclasses."""
from __future__ import annotations

import enum
from dataclasses import dataclass, field


class Side(str, enum.Enum):
    BUY = "buy"
    SELL = "sell"

    @property
    def sign(self) -> int:
        return 1 if self is Side.BUY else -1

    @property
    def opposite(self) -> "Side":
        return Side.SELL if self is Side.BUY else Side.BUY


class EventType(str, enum.Enum):
    DEPTH = "depth"
    TRADE = "trade"


@dataclass
class DepthEvent:
    """Binance diff depth update (one event, possibly many price levels)."""
    ts: int                     # event / exchange timestamp, ms
    first_update_id: int        # U
    final_update_id: int        # u
    bids: list                  # list[[price_str, qty_str]]
    asks: list
    etype: EventType = EventType.DEPTH


@dataclass
class Trade:
    ts: int                     # trade time T, ms
    trade_id: int
    price: float
    qty: float
    is_buyer_maker: bool        # Binance 'm': True => seller-side aggressor? no:
    # m=True means the BUYER was the maker, so the aggressor was a SELL.
    etype: EventType = EventType.TRADE

    @property
    def aggressor_side(self) -> Side:
        return Side.SELL if self.is_buyer_maker else Side.BUY


@dataclass
class Fill:
    order_id: int
    parent_id: int
    side: Side
    price: float
    qty: float
    ts: int
    maker: bool
    fee: float
    submit_ts: int
    is_deadline_liquidation: bool = False
    seq: int = 0
    source: str = ""               # resting_fill | market | timeout_take | ...
    # snapshot for post-fill analysis
    mid_at_fill: float = 0.0
    markouts: dict = field(default_factory=dict)


@dataclass
class MarketEvent:
    """A unit of 'book time': either a depth update or a trade print."""
    seq: int
    ts: int
    depth: DepthEvent | None = None
    trade: Trade | None = None

    @property
    def type(self) -> EventType:
        return self.depth.etype if self.depth is not None else self.trade.etype
