"""End-to-end pipeline on deterministic synthetic L2 data.

Steps: generate data -> build features -> train walk-forward models ->
run strategy x latency sweep -> accuracy / condition / alpha experiments ->
metrics CSVs + figures.

Live Binance data: `python data/binance_client.py --minutes 30`, then
`python run_all.py --raw data/raw/btcusdt_live.jsonl`.
"""
from __future__ import annotations

import argparse

import config
from analysis.experiments import (make_figures, run_accuracy_sweep,
                                  run_alpha_decay, run_condition_analysis,
                                  run_main_sweep)
from analysis.metrics import fill_ledger, parent_ledger
from data.loader import build_market_events, load_jsonl
from data.synthetic import generate
from execution.backtester import generate_parent_schedule
from features.features import build_features
from model.predictor import MidPredictor


def main(raw_path: str | None = None) -> None:
    raw = raw_path or str(config.RAW_DIR / "btcusdt_synth.jsonl")
    try:
        load_jsonl(raw)
    except FileNotFoundError:
        generate(raw)

    snapshot, depths, trades = load_jsonl(raw)
    events = build_market_events(depths, trades)
    print(f"Loaded {len(events)} market events "
          f"({sum(e.depth is not None for e in events)} depth, "
          f"{sum(e.trade is not None for e in events)} trades)")

    store = build_features(snapshot=snapshot, depths=depths, trades=trades)
    predictor = MidPredictor(store)
    print("\nModel evaluation (test half):")
    print(predictor.evaluation().to_string(index=False, float_format="%.3f"))

    schedule = generate_parent_schedule(
        config.N_PARENT_ORDERS, len(events), predictor.split,
        config.PARENT_HORIZON_EVENTS, config.RANDOM_SEED)

    print("\nRunning strategy x latency sweep ...")
    summary, ledgers = run_main_sweep(events, store, schedule, predictor)
    print("Running prediction-accuracy sweep ...")
    acc_sweep = run_accuracy_sweep(events, store, schedule, predictor)
    print("Running condition + alpha-decay analyses ...")
    cond, cond_buckets = run_condition_analysis(ledgers, store)
    alpha = run_alpha_decay(ledgers, store, predictor)

    # persist
    summary.to_csv(config.RESULTS_DIR / "summary.csv", index=False)
    acc_sweep.to_csv(config.RESULTS_DIR / "accuracy_sweep.csv", index=False)
    cond.to_csv(config.RESULTS_DIR / "condition_analysis.csv", index=False)
    cond_buckets.to_csv(config.RESULTS_DIR / "condition_buckets.csv", index=False)
    alpha.to_csv(config.RESULTS_DIR / "alpha_decay.csv", index=False)
    for (name, L), bt in ledgers.items():
        if L == 0:
            fill_ledger(bt.fills).to_csv(
                config.RESULTS_DIR / f"fill_ledger_{name}.csv", index=False)
            parent_ledger(bt.parents, bt.fills, store).to_csv(
                config.RESULTS_DIR / f"parent_ledger_{name}.csv", index=False)

    make_figures(summary, acc_sweep, cond_buckets, alpha)

    print("\n=== Strategy x latency summary ===")
    cols = ["strategy", "latency_ms", "fill_probability", "ttf_ms_median",
            "markout1_ticks", "markout5_ticks", "markout20_ticks",
            "implementation_shortfall_bps", "net_pnl_bps",
            "inventory_dev"]
    print(summary[cols].to_string(index=False, float_format="%.3f"))
    print("\nArtifacts written to results/ and reports/figures/.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=None, help="path to recorded JSONL")
    args = ap.parse_args()
    main(args.raw)
