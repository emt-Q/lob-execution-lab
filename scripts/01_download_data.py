"""Step 1: acquire data.

Tries to record live Binance L2 (depth + trades); if the WebSocket is
unreachable it falls back to the deterministic synthetic generator, which
emits the identical schema.
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config
from data.binance_client import record
from data.synthetic import generate

LIVE_MINUTES = 30


def main() -> None:
    live_path = config.RAW_DIR / f"{config.SYMBOL.lower()}_live.jsonl"
    try:
        asyncio.run(record(config.SYMBOL, LIVE_MINUTES * 60, str(live_path)))
        print("Live data saved:", live_path)
    except Exception as exc:  # network / geo-block / timeout
        print(f"Live recording unavailable ({exc!r}); using synthetic data.")
        generate(str(config.RAW_DIR / f"{config.SYMBOL.lower()}_synth.jsonl"))


if __name__ == "__main__":
    main()
