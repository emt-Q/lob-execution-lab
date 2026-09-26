"""End-to-end smoke test on a small synthetic stream."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from analysis.metrics import attach_markouts
from data.synthetic import SyntheticLOB
from execution.backtester import Backtester, generate_parent_schedule
from execution.strategies import STRATEGIES
from features.features import build_features
from model.predictor import MidPredictor


def _small_pipeline(tmp_path):
    import json
    gen = SyntheticLOB(n_events=6000, seed=11)
    p = tmp_path / "small.jsonl"
    with open(p, "w") as f:
        for line in gen.lines():
            f.write(line + "\n")
    from data.loader import load_jsonl
    snap, depths, trades = load_jsonl(str(p))
    store = build_features(snapshot=snap, depths=depths, trades=trades)
    predictor = MidPredictor(store)
    schedule = generate_parent_schedule(
        12, len(store.df), predictor.split, 300, seed=3, buffer=20)
    return store, predictor, schedule


def test_all_strategies_complete(tmp_path):
    import json
    from data.loader import build_market_events, load_jsonl
    gen = SyntheticLOB(n_events=6000, seed=11)
    p = tmp_path / "small.jsonl"
    with open(p, "w") as f:
        for line in gen.lines():
            f.write(line + "\n")
    snap, depths, trades = load_jsonl(str(p))
    events = build_market_events(depths, trades)
    store = build_features(snapshot=snap, depths=depths, trades=trades)
    predictor = MidPredictor(store)
    schedule = generate_parent_schedule(
        12, len(events), predictor.split, 300, seed=3, buffer=20)

    for name, cls in STRATEGIES.items():
        for L in (0, 50):
            bt = Backtester(events, predictor, cls(), L)
            bt.set_schedule(schedule)
            bt.run()
            attach_markouts(bt.fills, store)
            # every parent fully executed (resting + deadline liquidation)
            for parent in bt.parents.values():
                assert abs(parent.filled - parent.total) < 1e-6
