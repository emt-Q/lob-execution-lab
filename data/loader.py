"""Parse recorded JSONL (live or synthetic) into typed, time-ordered events."""
from __future__ import annotations

import json

from core.types import DepthEvent, MarketEvent, Trade


def load_jsonl(path: str):
    """Returns (snapshot: dict|None, depths: list[DepthEvent], trades: list[Trade])."""
    snapshot = None
    depths: list[DepthEvent] = []
    trades: list[Trade] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            kind, d = rec["type"], rec["data"]
            if kind == "snapshot":
                snapshot = d
            elif kind == "depth":
                depths.append(DepthEvent(
                    ts=d["E"], first_update_id=d["U"], final_update_id=d["u"],
                    bids=[[float(x[0]), float(x[1])] for x in d["b"]],
                    asks=[[float(x[0]), float(x[1])] for x in d["a"]],
                ))
            elif kind == "trade":
                trades.append(Trade(
                    ts=d["T"], trade_id=d["t"],
                    price=float(d["p"]), qty=float(d["q"]),
                    is_buyer_maker=d["m"],
                ))
    return snapshot, depths, trades


def build_market_events(depths: list[DepthEvent],
                        trades: list[Trade]) -> list[MarketEvent]:
    """Merge on exchange time. At equal timestamps trades print first so the
    queue model consumes trades before reconciling the aggregate depth."""
    items: list[tuple[int, int, object]] = []
    for d in depths:
        items.append((d.ts, 1, d))
    for t in trades:
        items.append((t.ts, 0, t))
    items.sort(key=lambda x: (x[0], x[1]))
    events: list[MarketEvent] = []
    for seq, (ts, _, obj) in enumerate(items):
        if isinstance(obj, DepthEvent):
            events.append(MarketEvent(seq=seq, ts=ts, depth=obj))
        else:
            events.append(MarketEvent(seq=seq, ts=ts, trade=obj))
    return events
