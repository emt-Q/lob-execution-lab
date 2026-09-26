"""Step 3: train walk-forward models and print out-of-sample evaluation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from features.features import build_features
from model.predictor import MidPredictor


def main() -> None:
    raw = config.RAW_DIR / f"{config.SYMBOL.lower()}_live.jsonl"
    if not raw.exists():
        raw = config.RAW_DIR / f"{config.SYMBOL.lower()}_synth.jsonl"
    store = build_features(str(raw))
    predictor = MidPredictor(store)
    ev = predictor.evaluation()
    print(ev.to_string(index=False))
    ev.to_csv(config.RESULTS_DIR / "model_evaluation.csv", index=False)


if __name__ == "__main__":
    main()
