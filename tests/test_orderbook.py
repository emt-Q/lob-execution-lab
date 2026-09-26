"""Tests for L2 book reconstruction."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.orderbook import BookReconstructor, OrderBook


def test_snapshot_and_views():
    ob = OrderBook()
    ob.apply_snapshot(100,
                      [["100.00", "2.0"], ["99.99", "3.0"]],
                      [["100.01", "1.5"], ["100.02", "4.0"]])
    assert ob.best_bid == 100.00
    assert ob.best_ask == 100.01
    assert ob.mid == 100.005
    assert ob.spread == pytest.approx(0.01, abs=1e-9)
    assert ob.best_bid_qty == 2.0
    assert ob.best_ask_qty == 1.5


def test_diff_sets_and_deletes():
    ob = OrderBook()
    ob.apply_snapshot(1, [["100.00", "2"]],
                      [["100.01", "1"], ["100.02", "3"]])
    ob.apply_levels([["100.00", "2.5"], ["99.99", "1.0"]], [["100.01", "0"]])
    assert ob.best_bid_qty == 2.5
    assert ob.best_ask == 100.02  # 100.01 deleted
    assert ob.qty_at("bid", 99.99) == 1.0


def test_walk():
    ob = OrderBook()
    ob.apply_snapshot(1, [["99.99", "5"]],
                      [["100.01", "2"], ["100.02", "3"]])
    vwap, fills, rem = ob.walk("buy", 4)
    assert rem == 0
    assert fills == [(100.01, 2), (100.02, 2)]
    assert abs(vwap - 100.015) < 1e-9


def test_microprice():
    ob = OrderBook()
    ob.apply_snapshot(1, [["100.00", "3"]], [["100.02", "1"]])
    # (100*1 + 100.02*3)/4
    assert abs(ob.microprice() - 100.015) < 1e-9


def test_reconstructor_alignment():
    rec = BookReconstructor()
    rec.ingest_diff(_diff(2, 3, bids=[["100.00", "2"]], asks=[["100.01", "1"]]))
    rec.ingest_snapshot(2, [["100.00", "2"]], [["100.01", "1"]])
    # buffered diff u=3 covers snapshot+1: U=2 <= 3 <= 3 -> applied
    assert rec.applied == 1
    assert rec.gaps == []


def test_reconstructor_gap_detection():
    rec = BookReconstructor()
    rec.ingest_snapshot(10, [["100.00", "1"]], [["100.01", "1"]])
    rec.ingest_diff(_diff(13, 14))  # expected U=12
    assert len(rec.gaps) == 1


def _diff(U, u, bids=None, asks=None):
    from core.types import DepthEvent
    return DepthEvent(ts=1, first_update_id=U, final_update_id=u,
                      bids=bids or [], asks=asks or [])
