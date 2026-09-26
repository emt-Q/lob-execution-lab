"""Record Binance public L2 data (diff depth + trades) to a JSONL file.

File format (one JSON object per line), identical for live and synthetic data:
  {"type": "snapshot", "ts": <fetch wall ms>, "data": {lastUpdateId, bids, asks}}
  {"type": "depth",    "data": {e,E,s,U,u,b,a}}
  {"type": "trade",    "data": {e,E,s,t,p,q,T,m}}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import requests
import websockets

import config


def fetch_snapshot(rest_base: str, symbol: str, limit: int = 1000) -> dict:
    url = f"{rest_base}/api/v3/depth"
    r = requests.get(url, params={"symbol": symbol, "limit": limit}, timeout=15)
    r.raise_for_status()
    return r.json()


async def record(
    symbol: str,
    duration_s: float,
    out_path: str,
    endpoint: str | None = None,
) -> str:
    ep = config.ENDPOINTS[endpoint or config.DEFAULT_ENDPOINT]
    symbol_l = symbol.lower()
    snap = fetch_snapshot(ep["rest"], symbol)
    stream = f"{symbol_l}@depth@100ms/{symbol_l}@trade"
    uri = f"{ep['ws']}/stream?streams={stream}"

    n_depth = n_trade = 0
    with open(out_path, "w") as f:
        f.write(json.dumps({"type": "snapshot", "ts": int(time.time() * 1000),
                            "data": snap}) + "\n")
        async with websockets.connect(uri, ping_interval=20, open_timeout=15) as ws:
            deadline = time.monotonic() + duration_s
            while time.monotonic() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=max(deadline - time.monotonic(), 0.1))
                except asyncio.TimeoutError:
                    continue
                env = json.loads(raw)
                d = env.get("data", env)
                kind = d.get("e")
                if kind == "depthUpdate":
                    f.write(json.dumps({"type": "depth", "data": d}) + "\n")
                    n_depth += 1
                elif kind == "trade":
                    f.write(json.dumps({"type": "trade", "data": d}) + "\n")
                    n_trade += 1
    print(f"Recorded {n_depth} depth updates, {n_trade} trades -> {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--endpoint", default=config.DEFAULT_ENDPOINT,
                    choices=list(config.ENDPOINTS))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = args.out or str(config.RAW_DIR / f"{args.symbol.lower()}_live.jsonl")
    asyncio.run(record(args.symbol, args.minutes * 60, out, args.endpoint))


if __name__ == "__main__":
    main()
