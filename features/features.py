"""Feature engineering for short-horizon mid-price prediction.

All features are known strictly at the event they describe (no lookahead).
Labels are future mid moves; the frame keeps them for training but the
execution engine only ever reads features at the current event.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

import config
from core.orderbook import OrderBook
from core.types import DepthEvent, Side, Trade

FEATURE_COLS = ["qi", "micro_dev", "ofi", "trade_flow", "trade_ci", "rv",
                "spread_ticks"]


@dataclass
class FeatureStore:
    df: pd.DataFrame

    def features_at(self, seq: int) -> pd.Series:
        seq = min(max(seq, 0), len(self.df) - 1)
        return self.df.loc[seq, FEATURE_COLS].astype(float)


def build_features(raw_path: str | None = None,
                   snapshot: dict | None = None,
                   depths: list | None = None,
                   trades: list | None = None) -> FeatureStore:
    if raw_path is not None:
        from data.loader import load_jsonl
        snapshot, depths, trades = load_jsonl(raw_path)
    depths = depths or []
    trades = trades or []

    ob = OrderBook(max_levels=config.N_BOOK_LEVELS)
    if snapshot:
        ob.apply_snapshot(snapshot["lastUpdateId"], snapshot["bids"],
                          snapshot["asks"])

    # index trades/depths by timestamp; walk in exchange-time order with trades
    # at the same timestamp applied first.
    items: list[tuple[int, int, object]] = []
    for d in depths:
        items.append((d.ts, 1, d))
    for t in trades:
        items.append((t.ts, 0, t))
    items.sort(key=lambda x: (x[0], x[1]))

    W = config.FEATURE_WINDOW
    rows: list[dict] = []
    prev_bb = prev_ba = None
    ofi_terms: list[float] = []
    trade_signed: list[float] = []
    trade_signs: list[int] = []
    mid_hist: list[float] = []

    def _depth_terms(ev: DepthEvent):
        nonlocal prev_bb, prev_ba
        terms = []
        # bid side contribution
        for p, q in ev.bids:
            p, q = float(p), float(q)
            q_prev = ob.qty_at("bid", p)
            if prev_bb is not None and p >= prev_bb:
                terms.append(q - q_prev)
        if prev_bb is not None and ob.best_bid is not None and ob.best_bid < prev_bb:
            terms.append(-ob.qty_at("bid", prev_bb))
        for p, q in ev.asks:
            p, q = float(p), float(q)
            q_prev = ob.qty_at("ask", p)
            if prev_ba is not None and p <= prev_ba:
                terms.append(-(q - q_prev))
        if prev_ba is not None and ob.best_ask is not None and ob.best_ask > prev_ba:
            terms.append(-ob.qty_at("ask", prev_ba))
        prev_bb, prev_ba = ob.best_bid, ob.best_ask
        return terms

    for seq, (ts, _, obj) in enumerate(items):
        if isinstance(obj, DepthEvent):
            terms = _depth_terms(obj)
            ob.apply_levels(obj.bids, obj.asks)
            ofi_terms.append(float(np.sum(terms)) if terms else 0.0)
        else:
            tr: Trade = obj
            sgn = tr.aggressor_side.sign
            trade_signed.append(sgn * tr.qty)
            trade_signs.append(sgn)
            ofi_terms.append(0.0)

        mid = ob.mid
        if mid is not None:
            mid_hist.append(mid)

        bb, ba = ob.best_bid, ob.best_ask
        if bb is None or ba is None:
            continue
        qb, qa = ob.bids[bb], ob.asks[ba]
        qi = (qb - qa) / (qb + qa) if qb + qa > 0 else 0.0
        micro = ob.microprice()
        micro_dev = (micro - mid) / config.TICK_SIZE if micro else 0.0
        spread_t = (ba - bb) / config.TICK_SIZE

        depth_ref = max((qb + qa) / 2.0, 1e-9)
        ofi = float(np.sum(ofi_terms[-W:])) / depth_ref
        tflow = float(np.sum(trade_signed[-W:]))
        if len(trade_signs) >= 2:
            seg = trade_signs[-(W + 1):]
            same = sum(1 for i in range(1, len(seg)) if seg[i] == seg[i - 1])
            tci = (2 * same / (len(seg) - 1)) - 1
        else:
            tci = 0.0
        if len(mid_hist) >= W + 1:
            rets = np.diff(mid_hist[-(W + 1):]) / config.TICK_SIZE
            rv = float(np.std(rets))
        else:
            rv = 0.0

        rows.append({
            "seq": seq, "ts": ts, "mid": mid, "best_bid": bb, "best_ask": ba,
            "qi": qi, "micro_dev": micro_dev, "ofi": ofi,
            "trade_flow": tflow, "trade_ci": tci, "rv": rv,
            "spread_ticks": spread_t,
        })

    df = pd.DataFrame(rows)
    # labels: future signed mid move in ticks + direction class
    tick = config.TICK_SIZE
    db = config.LABEL_DEADBAND_TICKS
    for h in config.HORIZONS:
        fwd = df["mid"].shift(-h)
        move = (fwd - df["mid"]) / tick
        df[f"y{h}"] = move
        df[f"c{h}"] = np.where(move > db, 1, np.where(move < -db, -1, 0))
    return FeatureStore(df)


def save_feature_summary(store: FeatureStore, out_path: str) -> None:
    store.df.to_csv(out_path, index=False)
