"""Execution-quality metrics.

Sign conventions
----------------
side sign s = +1 buy, -1 sell.
  markout_h  = s * (mid_{seq+h} - fill_px) / tick     positive = FAVORABLE
  adverse selection AS_h = -markout                  positive = adverse
  implementation shortfall (bps) =
      s * (fill_vwap - arrival_mid) / arrival_mid * 1e4        positive = cost
  net cost (USD) = s * (fill_notional - arrival_mid*qty) + fees
  net PnL vs arrival = -net cost
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from core.types import Fill, Side
from execution.queue_model import RestingOrder
from execution.strategies import ParentOrder
from features.features import FeatureStore


# ---------------------------------------------------------------------------
# Markouts
# ---------------------------------------------------------------------------
def attach_markouts(fills: list[Fill], store: FeatureStore,
                    horizons=config.HORIZONS) -> None:
    df = store.df
    n = len(df)
    mids = df["mid"].to_numpy()
    for f in fills:
        for h in horizons:
            j = min(f.seq + h, n - 1)
            m_fut = mids[j]
            if m_fut is None or np.isnan(m_fut):
                continue
            mk_ticks = f.side.sign * (m_fut - f.price) / store.tick
            mk_bps = f.side.sign * (m_fut - f.price) / f.price * 1e4
            f.markouts[f"markout{h}_ticks"] = mk_ticks
            f.markouts[f"markout{h}_bps"] = mk_bps


def fill_ledger(fills: list[Fill]) -> pd.DataFrame:
    rows = []
    for f in fills:
        rows.append({
            "order_id": f.order_id, "parent_id": f.parent_id,
            "side": f.side.value, "price": f.price, "qty": f.qty,
            "ts": f.ts, "seq": f.seq, "maker": f.maker, "fee": f.fee,
            "source": f.source,
            "submit_ts": f.submit_ts,
            "ttf_ms": f.ts - f.submit_ts,
            "deadline": f.is_deadline_liquidation,
            **{f"markout{h}_ticks": f.markouts.get(f"markout{h}_ticks", np.nan)
               for h in config.HORIZONS},
            **{f"markout{h}_bps": f.markouts.get(f"markout{h}_bps", np.nan)
               for h in config.HORIZONS},
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Parent-level ledger
# ---------------------------------------------------------------------------
def parent_ledger(parents: dict[int, ParentOrder], fills: list[Fill],
                  store: FeatureStore) -> pd.DataFrame:
    by: dict[int, list[Fill]] = {}
    for f in fills:
        by.setdefault(f.parent_id, []).append(f)
    rows = []
    mids = store.df["mid"].to_numpy()
    for pid, p in parents.items():
        fs = by.get(pid, [])
        qty = sum(f.qty for f in fs)
        notional = sum(f.price * f.qty for f in fs)
        fees = sum(f.fee for f in fs)
        vwap = notional / qty if qty > 0 else np.nan
        is_bps = p.side.sign * (vwap - p.arrival_mid) / p.arrival_mid * 1e4 \
            if qty > 0 else np.nan
        net_usd = p.side.sign * (notional - p.arrival_mid * qty) + fees \
            if qty > 0 else np.nan

        # inventory schedule deviation on the event grid
        grid = np.arange(p.start_seq, p.deadline_seq + 1)
        cum = np.zeros_like(grid, dtype=float)
        for f in fs:
            idx = f.seq - p.start_seq
            if 0 <= idx < len(grid):
                cum[idx] += f.qty
        cum = np.cumsum(cum)
        sched = p.total * (grid - p.start_seq) / max(1, p.deadline_seq - p.start_seq)
        inv_dev = np.mean(np.abs(cum - sched)) / p.total
        inv_max = np.max(np.abs(cum - sched)) / p.total

        rows.append({
            "parent_id": pid, "side": p.side.value, "qty": qty,
            "vwap": vwap, "arrival_mid": p.arrival_mid,
            "is_bps": is_bps, "fees_bps": fees / notional * 1e4 if notional else 0,
            "net_cost_usd": net_usd,
            "maker_qty": sum(f.qty for f in fs if f.maker),
            "taker_qty": sum(f.qty for f in fs if not f.maker),
            "deadline_qty": sum(f.qty for f in fs if f.is_deadline_liquidation),
            "timeout_qty": sum(f.qty for f in fs if f.source == "timeout_take"),
            "inventory_dev": inv_dev, "inventory_max": inv_max,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Strategy x latency summary
# ---------------------------------------------------------------------------
def summarize(strategy: str, latency_ms: int,
              parents: dict[int, ParentOrder], fills: list[Fill],
              orders: dict[int, RestingOrder], store: FeatureStore) -> dict:
    fl = fill_ledger(fills)
    pl = parent_ledger(parents, fills, store)

    total_qty = pl.qty.sum() if len(pl) else 0.0
    # resting (maker) children: fill probability
    rested = [o for o in orders.values() if o.rested]
    rested_filled = [o for o in rested if o.filled_qty > 1e-9]
    fill_prob = len(rested_filled) / len(rested) if rested else np.nan

    # time-to-fill for resting children that got maker fills
    ttf = []
    for o in rested_filled:
        first = min(t for t, _ in o.fills)
        ttf.append(first - o.submit_ts)
    ttf = np.array(ttf)

    out = {
        "strategy": strategy,
        "latency_ms": latency,
        "fill_probability": fill_prob,
        "ttf_ms_median": float(np.median(ttf)) if len(ttf) else np.nan,
        "ttf_ms_mean": float(np.mean(ttf)) if len(ttf) else np.nan,
        "maker_qty_fraction": pl.maker_qty.sum() / total_qty if total_qty else np.nan,
        "deadline_qty_fraction": pl.deadline_qty.sum() / total_qty if total_qty else np.nan,
        "timeout_qty_fraction": pl.timeout_qty.sum() / total_qty if total_qty else np.nan,
        "implementation_shortfall_bps": pl.is_bps.mean() if len(pl) else np.nan,
        "fees_bps": pl.fees_bps.mean() if len(pl) else np.nan,
        "net_pnl_bps": -pl.is_bps.mean() if len(pl) else np.nan,
        "net_pnl_usd_per_parent": -pl.net_cost_usd.mean() if len(pl) else np.nan,
        "inventory_dev": pl.inventory_dev.mean() if len(pl) else np.nan,
        "inventory_max": pl.inventory_max.mean() if len(pl) else np.nan,
    }
    for h in config.HORIZONS:
        col = f"markout{h}_ticks"
        vals = fl[col].dropna() if len(fl) else pd.Series(dtype=float)
        out[f"markout{h}_ticks"] = vals.mean() if len(vals) else np.nan
        out[f"adverse_selection{h}_ticks"] = -vals.mean() if len(vals) else np.nan
    return out
