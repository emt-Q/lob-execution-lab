"""Step 4: run a single strategy/latency backtest and print metrics."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from analysis.metrics import attach_markouts, fill_ledger, summarize
from data.loader import build_market_events, load_jsonl
from execution.backtester import Backtester, generate_parent_schedule
from execution.strategies import STRATEGIES
from features.features import build_features
from model.predictor import MidPredictor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="join_best", choices=list(STRATEGIES))
    ap.add_argument("--latency", type=int, default=0)
    args = ap.parse_args()

    raw = config.RAW_DIR / f"{config.SYMBOL.lower()}_live.jsonl"
    if not raw.exists():
        raw = config.RAW_DIR / f"{config.SYMBOL.lower()}_synth.jsonl"
    snapshot, depths, trades = load_jsonl(str(raw))
    events = build_market_events(depths, trades)
    store = build_features(snapshot=snapshot, depths=depths, trades=trades)
    predictor = MidPredictor(store)
    schedule = generate_parent_schedule(
        config.N_PARENT_ORDERS, len(events), predictor.split,
        config.PARENT_HORIZON_EVENTS, config.RANDOM_SEED)

    bt = Backtester(events, predictor, STRATEGIES[args.strategy](), args.latency)
    bt.set_schedule(schedule)
    bt.run()
    attach_markouts(bt.fills, store)
    s = summarize(args.strategy, args.latency, bt.parents, bt.fills,
                  bt.orders, store)
    for k, v in s.items():
        print(f"{k:32s} {v}")
    fill_ledger(bt.fills).to_csv(
        config.RESULTS_DIR / f"fill_ledger_{args.strategy}_{args.latency}ms.csv",
        index=False)


if __name__ == "__main__":
    main()
