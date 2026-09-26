"""Feature construction for short-horizon mid-price movement prediction.

Features (per market event, book-time index):
  qi           queue imbalance at top of book  (Qb-Qa)/(Qb+qa)
  micro_dev    microprice deviation from mid, in ticks
  ofi          Cont-Kukanov-Stoikov (2013) order-flow imbalance, window sum,
               normalized by mean top-of-book depth
  trade_flow   signed traded volume / total volume over window
               (+ = buyer-initiated aggressors)
  trade_ci     signed trade count imbalance over window
  rv           realized volatility: rolling std of mid log changes (ticks)
  spread_ticks bid-ask spread in ticks

Labels for horizons h in {1,5,20}:
  y_h          (mid_{i+h} - mid_i) / tick           (continuous)
  c_h in {-1,0,1} direction, deadband LABEL_DEADBAND_TICKS
"""
from __future__ import annotations

from collections import deque

import numpy as np
import pandas as pd

import config
from core.orderbook import OrderBook
from data.loader import build_market_events, load_jsonl

FEATURE_COLS = ["qi", "micro_dev", "ofi", "trade_flow", "trade_ci", "rv",
                "spread_ticks"]


class FeatureStore:
    def __init__(self, df: pd.DataFrame, tick: float, window: int):
        self.df = df.reset_index(drop=True)
        self.tick = tick
        self.window = window

    def features_at(self, seq: int) -> pd.Series:
        seq = min(max(seq, 0), len(self.df) - 1)
        return self.df.loc[seq, FEATURE_COLS].fillna(0.0)

    def mid_at(self, seq: int) -> float:
        seq = min(max(seq, 0), len(self.df) - 1)
        return float(self.df.loc[seq, "mid"])


def build_features(path: str | None = None,
                   snapshot=None, depths=None, trades=None,
                   window: int = config.FEATURE_WINDOW,
                   tick: float = config.TICK_SIZE,
                   max_levels: int = config.N_BOOK_LEVELS) -> FeatureStore:
    if path is not None:
        snapshot, depths, trades = load_jsonl(path)
    events = build_market_events(depths, trades)

    book = OrderBook(max_levels=max_levels)
    if snapshot is not None:
        book.apply_snapshot(snapshot["lastUpdateId"],
                            snapshot["bids"], snapshot["asks"])

    n = len(events)
    ts = np.zeros(n, dtype=np.int64)
    mids = np.full(n, np.nan)
    pb = np.full(n, np.nan)
    pa = np.full(n, np.nan)
    qb = np.full(n, np.nan)
    qa = np.full(n, np.nan)
    ofi_step = np.zeros(n)
    signed_vol = np.zeros(n)
    signed_cnt = np.zeros(n)
    total_vol = np.zeros(n)
    total_cnt = np.zeros(n)

    prev = None  # (pb, pa, qb, qa)

    for ev in events:
        i = ev.seq
        ts[i] = ev.ts
        if ev.depth is not None:
            book.apply_levels(ev.depth.bids, ev.depth.asks)
        if ev.trade is not None:
            tr = ev.trade
            sgn = tr.aggressor_side.sign
            signed_vol[i] += sgn * tr.qty
            signed_cnt[i] += sgn
            total_vol[i] += tr.qty
            total_cnt[i] += 1

        b_bid, b_ask = book.best_bid, book.best_ask
        if b_bid is None or b_ask is None:
            if prev is not None:
                pb[i], pa[i], qb[i], qa[i] = prev
            continue
        cur = (b_bid, b_ask, book.bids[b_bid], book.asks[b_ask])
        pb[i], pa[i], qb[i], qa[i] = cur
        mids[i] = (b_bid + b_ask) / 2.0

        if prev is not None and ev.depth is not None:
            ofi_step[i] = _ofi_contribution(prev, cur)
        prev = cur

    # forward-fill book state across events that didn't touch the top
    frame = pd.DataFrame({
        "ts": ts, "mid": mids, "pb": pb, "pa": pa, "qb": qb, "qa": qa,
        "ofi_step": ofi_step, "signed_vol": signed_vol, "signed_cnt": signed_cnt,
        "total_vol": total_vol, "total_cnt": total_cnt,
    })
    for c in ("mid", "pb", "pa", "qb", "qa"):
        frame[c] = frame[c].ffill()

    g = frame
    g["qi"] = (g.qb - g.qa) / (g.qb + g.qa)
    g["microprice"] = (g.pb * g.qa + g.pa * g.qb) / (g.qb + g.qa)
    g["micro_dev"] = (g.microprice - g.mid) / tick
    g["spread_ticks"] = (g.pa - g.pb) / tick

    roll_depth = (g.qb + g.qa).rolling(window, min_periods=2).mean()
    g["ofi"] = g.ofi_step.rolling(window, min_periods=2).sum() / roll_depth

    tv = g.total_vol.rolling(window, min_periods=1).sum()
    sv = g.signed_vol.rolling(window, min_periods=1).sum()
    g["trade_flow"] = sv / tv.replace(0, np.nan)
    tc = g.total_cnt.rolling(window, min_periods=1).sum()
    sc = g.signed_cnt.rolling(window, min_periods=1).sum()
    g["trade_ci"] = sc / tc.replace(0, np.nan)

    log_ret = np.log(g.mid).diff()
    g["rv"] = log_ret.rolling(window, min_periods=5).std() / np.log(1 + tick / g.mid)

    g = g.fillna(0.0)

    # labels
    for h in config.HORIZONS:
        fut = g.mid.shift(-h)
        delta = (fut - g.mid) / tick
        g[f"y{h}"] = delta
        g[f"c{h}"] = np.where(delta > config.LABEL_DEADBAND_TICKS, 1,
                              np.where(delta < -config.LABEL_DEADBAND_TICKS, -1, 0))

    keep = ["ts", "mid", "pb", "pa", "qb", "qa"] + FEATURE_COLS + \
           [f"y{h}" for h in config.HORIZONS] + [f"c{h}" for h in config.HORIZONS]
    return FeatureStore(g[keep], tick, window)


def _ofi_contribution(prev, cur) -> float:
    pb0, pa0, qb0, qa0 = prev
    pb1, pa1, qb1, qa1 = cur
    if pb1 > pb0:
        e_b = qb1
    elif pb1 < pb0:
        e_b = -qb0
    else:
        e_b = qb1 - qb0

    if pa1 > pa0:
        e_a = -qa0
    elif pa1 < pa0:
        e_a = qa1
    else:
        e_a = qa0 - qa1
    return e_b + e_a


def save_feature_summary(store: FeatureStore, out_csv: str) -> None:
    store.df.to_csv(out_csv, index=False)
