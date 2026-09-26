"""Global configuration for the LOB execution lab.

All monetary quantities are in quote currency (USDT), all sizes in base
currency (e.g. BTC). Timestamps are integer exchange milliseconds unless a
function explicitly says otherwise.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = ROOT / "results"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS / "figures"
for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Market / instrument
# ---------------------------------------------------------------------------
SYMBOL = os.getenv("SYMBOL", "BTCUSDT")
TICK_SIZE = 0.01          # BTCUSDT tick = 1 cent
LOT_SIZE = 0.00001        # BTCUSDT base lot

# Binance spot standard fees (VIP 0). Maker/taker are equal at 10 bps on the
# base tier; the values can be overridden for BNB discount or other venues.
MAKER_FEE_BPS = float(os.getenv("MAKER_FEE_BPS", 1.0))
TAKER_FEE_BPS = float(os.getenv("TAKER_FEE_BPS", 1.0))

# ---------------------------------------------------------------------------
# Endpoints. api.binance.com geo-blocks some regions (HTTP 451); the public
# data mirror data-api.binance.vision serves identical REST/WS market data.
# ---------------------------------------------------------------------------
ENDPOINTS = {
    "vision": {
        "rest": "https://data-api.binance.vision",
        "ws": "wss://data-stream.binance.vision",
    },
    "binance.us": {
        "rest": "https://api.binance.us",
        "ws": "wss://stream.binance.us:9443",
    },
    "binance.com": {
        "rest": "https://api.binance.com",
        "ws": "wss://stream.binance.com:9443",
    },
}
DEFAULT_ENDPOINT = os.getenv("BINANCE_ENDPOINT", "vision")

# ---------------------------------------------------------------------------
# Feature / model horizons (in quote events, i.e. "ticks" of book time)
# ---------------------------------------------------------------------------
HORIZONS = (1, 5, 20)
FEATURE_WINDOW = 20          # events used for rolling OFI / flow / vol
PRIMARY_HORIZON = 5
LABEL_DEADBAND_TICKS = 0.25  # |delta mid| below this = "flat" class

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
LATENCIES_MS = (0, 10, 50, 100)
N_PARENT_ORDERS = 200
PARENT_QTY = 0.5             # base currency per parent order
PARENT_HORIZON_EVENTS = 200  # deadline, in market events
CHILD_QTY = 0.1
LIMIT_TIMEOUT_EVENTS = 60    # join/improve: market-take remainder after this
N_BOOK_LEVELS = 50           # depth levels retained in reconstructed book

RANDOM_SEED = 42

# Synthetic data (offline fallback / deterministic test bed)
SYNTH_EVENTS = 90_000
SYNTH_SEED = 7


@dataclass
class BacktestConfig:
    symbol: str = SYMBOL
    tick_size: float = TICK_SIZE
    lot_size: float = LOT_SIZE
    maker_fee_bps: float = MAKER_FEE_BPS
    taker_fee_bps: float = TAKER_FEE_BPS
    horizons: tuple = HORIZONS
    latencies_ms: tuple = LATENCIES_MS
    n_parent_orders: int = N_PARENT_ORDERS
    parent_qty: float = PARENT_QTY
    parent_horizon_events: int = PARENT_HORIZON_EVENTS
    child_qty: float = CHILD_QTY
    limit_timeout_events: int = LIMIT_TIMEOUT_EVENTS
    n_book_levels: int = N_BOOK_LEVELS
    seed: int = RANDOM_SEED
    extra: dict = field(default_factory=dict)
