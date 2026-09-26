"""Experiment orchestration and figure generation.

Produces under results/ and reports/figures/:
  summary.csv                 strategy x latency metrics
  accuracy_sweep.csv          adaptive performance vs signal strength
  condition_analysis.csv      posting-vs-taking advantage by market condition
  alpha_decay.csv             apparent vs fill-conditioned signal alpha
  fill_ledger_*.parquet/csv   per-fill and per-parent ledgers (L=0)
  six diagnostic figures
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config
from analysis.metrics import (attach_markouts, fill_ledger, parent_ledger,
                              summarize)
from execution.backtester import Backtester, generate_parent_schedule
from execution.strategies import STRATEGIES
from features.features import FEATURE_COLS
from model.predictor import MidPredictor

SIGNAL_GRID = [0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 1.0]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _set_mix(predictor: MidPredictor, mix: float) -> None:
    predictor.signal_mix = mix
    for m in predictor.models.values():
        m.signal_mix = mix


def _run_one(events, predictor, strategy_name, latency, schedule, store):
    strat = STRATEGIES[strategy_name]()
    bt = Backtester(events, predictor, strat, latency)
    bt.set_schedule(schedule)
    bt.run()
    attach_markouts(bt.fills, store)
    return bt


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------
def run_main_sweep(events, store: FeatureStore_like, schedule, predictor):
    rows = []
    ledgers = {}
    for latency in config.LATENCIES_MS:
        for name in STRATEGIES:
            bt = _run_one(events, predictor, name, latency, schedule, store)
            rows.append(summarize(name, latency, bt.parents, bt.fills,
                                  bt.orders, store))
            ledgers[(name, latency)] = bt
    return pd.DataFrame(rows), ledgers


# ---------------------------------------------------------------------------
# prediction accuracy sweep (adaptive strategy)
# ---------------------------------------------------------------------------
def run_accuracy_sweep(events, store, schedule, predictor):
    rows = []
    for mix in SIGNAL_GRID:
        _set_mix(predictor, mix)
        ev = predictor.evaluation()
        acc = float(ev.loc[ev.horizon == config.PRIMARY_HORIZON,
                           "directional_accuracy"].iloc[0])
        acc3 = float(ev.loc[ev.horizon == config.PRIMARY_HORIZON,
                            "balanced_accuracy"].iloc[0])
        for latency in config.LATENCIES_MS:
            bt = _run_one(events, predictor, "adaptive_signal", latency,
                          schedule, store)
            s = summarize("adaptive_signal", latency, bt.parents, bt.fills,
                          bt.orders, store)
            s["signal_mix"] = mix
            s["model_directional_accuracy"] = acc
            s["model_balanced_accuracy"] = acc3
            rows.append(s)
    _set_mix(predictor, 0.0)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# condition analysis: when does posting beat taking?
# ---------------------------------------------------------------------------
def run_condition_analysis(ledgers, store: FeatureStore_like):
    rows = []
    for latency in config.LATENCIES_MS:
        take_pl = parent_ledger(ledgers[("market_take", latency)].parents,
                                ledgers[("market_take", latency)].fills, store)
        for sname in ("join_best", "improve_one_tick", "adaptive_signal"):
            bt = ledgers[(sname, latency)]
            pl = parent_ledger(bt.parents, bt.fills, store)
            m = pl.merge(take_pl[["parent_id", "is_bps"]], on="parent_id",
                         suffixes=("", "_take"))
            # positive advantage => posting better than taking
            m["advantage_bps"] = m.is_bps_take - m.is_bps
            df = store.df
            starts = [bt.parents[i].start_seq for i in m.parent_id]
            m["spread_b"] = df["spread_ticks"].to_numpy()[starts]
            m["qi_b"] = df["qi"].to_numpy()[starts]
            m["rv_b"] = df["rv"].to_numpy()[starts]
            m["ofi_b"] = df["ofi"].to_numpy()[starts]
            m["strategy_cmp"] = sname
            m["latency_ms"] = latency
            rows.append(m[["parent_id", "strategy_cmp", "latency_ms",
                           "advantage_bps", "spread_b", "qi_b", "rv_b",
                           "ofi_b"]])
    out = pd.concat(rows, ignore_index=True)

    # bucketed summary at L = 0
    cond = out[out.latency_ms == 0].copy()
    cond["spread_bin"] = pd.cut(cond.spread_b, [-1, 1.01, 2.01, 1e9],
                                labels=["1 tick", "2 ticks", "3+ ticks"])
    for c in ("qi_b", "rv_b", "ofi_b"):
        cond[c + "_bin"] = pd.qcut(cond[c].rank(method="first"), 3,
                                   labels=["low", "mid", "high"])
    buckets = []
    for dim in ("spread_bin", "qi_b_bin", "rv_b_bin", "ofi_b_bin"):
        for sname, g in cond.groupby("strategy_cmp", observed=True):
            for b, gg in g.groupby(dim, observed=True):
                buckets.append({"dimension": dim, "bucket": str(b),
                                "strategy_cmp": sname,
                                "advantage_bps": gg.advantage_bps.mean(),
                                "n": len(gg)})
    return out, pd.DataFrame(buckets)


# ---------------------------------------------------------------------------
# alpha decay: apparent vs fill-conditioned signal
# ---------------------------------------------------------------------------
def run_alpha_decay(ledgers, store, predictor):
    """Apparent edge vs fill-conditioned edge, on one consistent axis.

    Apparent: at every test event, imagine we could trade the side the signal
    favors *at the mid*. The expected favorable edge is |E[move]| and the
    realized favorable edge is sign(E[move]) * realized move. This is the
    headline alpha a backtest naively shows.

    Fill-conditioned: for our actual resting orders, the expected favorable
    edge is E[move] * (our side sign); the realized favorable edge is the
    side-adjusted markout. Resting orders only fill when aggressors hit --
    precisely the adverse states -- so this edge is what we truly earn.
    """
    df = store.df
    h = config.PRIMARY_HORIZON
    test_idx = np.arange(predictor.split + config.FEATURE_WINDOW,
                         len(df) - h - 1)
    em = np.full(len(df), 0.0)
    for i in test_idx:
        em[i] = predictor.expected_move(i, h)
    y = df[f"y{h}"].to_numpy()

    app_expected = np.abs(em[test_idx])
    app_realized = np.sign(em[test_idx]) * y[test_idx]

    # expected/realized favorable edge for each filled resting child
    fill_exp, fill_real = [], []
    for sname in ("join_best", "improve_one_tick"):
        bt = ledgers[(sname, 0)]
        for o in bt.orders.values():
            if not (o.rested and o.filled_qty > 1e-9):
                continue
            side_sign = bt.parents[o.parent_id].side.sign
            fill_exp.append(predictor.expected_move(o.submit_seq, h)
                            * side_sign)
            fs = [f for f in bt.fills if f.order_id == o.order_id]
            fill_real.append(np.mean(
                [f.markouts.get(f"markout{h}_ticks", np.nan) for f in fs]))
    fill_exp = np.array(fill_exp)
    fill_real = np.array(fill_real)

    # shared "expected favorable edge" bins, pooled across both populations
    pooled = np.concatenate([app_expected, fill_exp])
    edges = np.unique(np.quantile(pooled, np.linspace(0, 1, 6)))
    edges[0] -= 1e-9
    labels = ["Q1\n(weak/against)", "Q2", "Q3", "Q4", "Q5\n(strong favor)"]

    rows = []
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        m_app = (app_expected >= lo) & (app_expected < hi)
        m_fill = (fill_exp >= lo) & (fill_exp < hi)
        rows.append({
            "signal_bucket": labels[k],
            "apparent_expected_ticks": app_expected[m_app].mean()
            if m_app.any() else np.nan,
            "apparent_realized_ticks": app_realized[m_app].mean()
            if m_app.any() else np.nan,
            "fill_conditioned_ticks": fill_real[m_fill].mean()
            if m_fill.any() else np.nan,
            "n_events": int(m_app.sum()),
            "n_fills": int(m_fill.sum()),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def _save(fig, name):
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, dpi=130)
    plt.close(fig)


def make_figures(summary, acc_sweep, cond_buckets, alpha_decay):
    colors = {"market_take": "#d1495b", "join_best": "#2e86ab",
              "improve_one_tick": "#6a994e", "adaptive_signal": "#e08e45"}

    # 1. strategy x latency, 4 panels
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [
        ("implementation_shortfall_bps", "Implementation shortfall (bps)"),
        ("markout5_ticks", "5-tick markout (ticks, + favorable)"),
        ("fill_probability", "Resting-order fill probability"),
        ("ttf_ms_median", "Median time-to-fill (ms)"),
    ]
    for ax, (col, title) in zip(axes.ravel(), panels):
        for name, g in summary.groupby("strategy"):
            g = g.sort_values("latency_ms")
            ax.plot(g.latency_ms, g[col], marker="o", label=name,
                    color=colors.get(name))
        ax.set_title(title); ax.set_xlabel("latency (ms)"); ax.grid(alpha=.3)
    axes.ravel()[0].legend(fontsize=8)
    _save(fig, "fig1_strategy_latency.png")

    # 2. markout term structure (L=0)
    fig, ax = plt.subplots(figsize=(8, 5))
    width, x = 0.2, np.arange(3)
    for k, (name, g) in enumerate(summary[summary.latency_ms == 0].groupby("strategy")):
        vals = [g[f"markout{h}_ticks"].iloc[0] for h in (1, 5, 20)]
        ax.bar(x + (k - 1.5) * width, vals, width, label=name,
               color=colors.get(name))
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x, ["1 tick", "5 tick", "20 tick"])
    ax.set_ylabel("markout (ticks, + favorable)")
    ax.set_title("Post-fill markout term structure (L=0)")
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    _save(fig, "fig2_markout_term_structure.png")

    # 3. accuracy breakeven
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for L, g in acc_sweep.groupby("latency_ms"):
        g = g.sort_values("model_directional_accuracy")
        ax.plot(g.model_directional_accuracy, -g.implementation_shortfall_bps,
                marker="o", label=f"adaptive {L}ms")
    for L, g in summary[summary.strategy == "market_take"].groupby("latency_ms"):
        ax.axhline(-g.implementation_shortfall_bps.iloc[0], ls="--", lw=1,
                   color="#d1495b")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("model directional accuracy at 5-tick horizon")
    ax.set_ylabel("net PnL vs arrival (bps)")
    ax.set_title("Prediction accuracy needed to offset queue + latency")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, "fig3_accuracy_breakeven.png")

    # 4. conditions
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.2), sharey=True)
    for ax, dim in zip(axes, ("spread_bin", "qi_b_bin", "rv_b_bin", "ofi_b_bin")):
        sub = cond_buckets[cond_buckets.dimension == dim]
        buckets = sub.bucket.unique()
        x = np.arange(len(buckets))
        for k, sname in enumerate(sub.strategy_cmp.unique()):
            gg = sub[sub.strategy_cmp == sname].set_index("bucket").reindex(buckets)
            ax.bar(x + (k - 1) * 0.27, gg.advantage_bps, 0.27, label=sname,
                   color=colors.get(sname))
        ax.axhline(0, color="k", lw=.8)
        ax.set_xticks(x, buckets, fontsize=8)
        ax.set_title(dim.replace("_bin", "").replace("_b", ""))
        ax.grid(alpha=.3, axis="y")
    axes[0].set_ylabel("posting advantage vs take (bps)")
    axes[0].legend(fontsize=7)
    _save(fig, "fig4_posting_conditions.png")

    # 5. alpha decay
    fig, ax = plt.subplots(figsize=(8.5, 5))
    x = np.arange(len(alpha_decay))
    ax.bar(x - 0.2, alpha_decay.apparent_realized_ticks, 0.4,
           label="apparent edge if traded at mid", color="#2e86ab")
    ax.bar(x + 0.2, alpha_decay.fill_conditioned_ticks, 0.4,
           label="fill-conditioned edge (our resting orders)",
           color="#d1495b")
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(x, alpha_decay.signal_bucket, fontsize=8)
    ax.set_ylabel(f"realized favorable {config.PRIMARY_HORIZON}-tick edge "
                  "(ticks)")
    ax.set_title("Why order-flow alpha dies after fill selectivity")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    _save(fig, "fig5_alpha_decay.png")

    # 6. net PnL after fees across latency
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for name, g in summary.groupby("strategy"):
        g = g.sort_values("latency_ms")
        ax.plot(g.latency_ms, g.net_pnl_bps, marker="o", label=name,
                color=colors.get(name))
    ax.axhline(0, color="k", lw=.8)
    ax.set_xlabel("latency (ms)")
    ax.set_ylabel("net PnL vs arrival (bps, incl. fees)")
    ax.set_title("Latency sensitivity: when adverse selection erodes edge")
    ax.legend(fontsize=8); ax.grid(alpha=.3)
    _save(fig, "fig6_latency_sensitivity.png")


# Type alias just for readability (FeatureStore avoids circular import cost)
FeatureStore_like = object
