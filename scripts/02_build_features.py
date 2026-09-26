"""Step 2: build the feature frame and persist a CSV snapshot."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from features.features import build_features, save_feature_summary


def main() -> None:
    raw = config.RAW_DIR / f"{config.SYMBOL.lower()}_live.jsonl"
    if not raw.exists():
        raw = config.RAW_DIR / f"{config.SYMBOL.lower()}_synth.jsonl"
    store = build_features(str(raw))
    out = config.PROCESSED_DIR / "features.csv"
    save_feature_summary(store, str(out))
    print(f"Feature frame {store.df.shape} -> {out}")


if __name__ == "__main__":
    main()
