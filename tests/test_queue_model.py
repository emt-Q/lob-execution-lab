"""Tests for the FIFO queue model."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.types import Side, Trade
from execution.queue_model import QueueSimulator, RestingOrder


def _order(oid, side=Side.BUY, price=100.0, qty=1.0):
    return RestingOrder(order_id=oid, parent_id=0, side=side, price=price,
                        qty=qty, remaining=qty, submit_ts=0, submit_seq=0)


def _bookkeep(o, fills):
    # mirrors Backtester._emit_fill accounting for direct queue tests
    for oid, q in fills:
        o.remaining -= q
        o.filled_qty += q


def _trade(price, qty, aggressor: Side):
    return Trade(ts=1, trade_id=1, price=price, qty=qty,
                 is_buyer_maker=(aggressor is Side.SELL))


def test_no_fill_while_ahead():
    qs = QueueSimulator()
    o = _order(1)
    qs.join(o, market_qty=5.0)
    assert qs.qty_ahead(o) == 5.0
    fills = qs.on_trade(_trade(100.0, 3.0, Side.SELL))
    assert fills == []
    assert qs.qty_ahead(o) == 2.0
    assert o.remaining == 1.0


def test_partial_then_full_fill():
    qs = QueueSimulator()
    o = _order(1, qty=2.0)
    qs.join(o, market_qty=1.0)
    f1 = qs.on_trade(_trade(100.0, 2.0, Side.SELL))
    assert f1 == [(1, 1.0)]           # 1 ahead + 1 of ours
    _bookkeep(o, f1)
    assert o.remaining == 1.0
    f2 = qs.on_trade(_trade(100.0, 1.0, Side.SELL))
    assert f2 == [(1, 1.0)]
    _bookkeep(o, f2)
    assert o.remaining == 0.0


def test_fifo_two_own_orders():
    qs = QueueSimulator()
    o1 = _order(1, qty=1.0)
    o2 = _order(2, qty=1.0)
    qs.join(o1, market_qty=1.0)
    qs.join(o2, market_qty=1.0)       # market total unchanged since join
    # o2 ahead must include o1
    assert qs.qty_ahead(o2) == 2.0
    fills = qs.on_trade(_trade(100.0, 2.0, Side.SELL))
    assert fills == [(1, 1.0)]        # 1 market + o1, o2 not reached
    fills = qs.on_trade(_trade(100.0, 1.0, Side.SELL))
    assert fills == [(2, 1.0)]


def test_reconcile_pro_rata_cancel():
    qs = QueueSimulator()
    o = _order(1)
    qs.join(o, market_qty=4.0)
    # trade consumes 1 of market qty, then depth reports aggregate market = 1
    qs.on_trade(_trade(100.0, 1.0, Side.SELL))
    qs.on_level(Side.BUY, 100.0, aggregate_qty=1.0)
    # 3 remained after trade; cancel 2 pro-rata -> 1 ahead
    assert abs(qs.qty_ahead(o) - 1.0) < 1e-9


def test_new_adds_append_behind():
    qs = QueueSimulator()
    o = _order(1)
    qs.join(o, market_qty=2.0)
    qs.on_level(Side.BUY, 100.0, aggregate_qty=5.0)
    # extra qty appends behind us: ahead unchanged
    assert qs.qty_ahead(o) == 2.0


def test_cancel_removes_block():
    qs = QueueSimulator()
    o = _order(1)
    qs.join(o, market_qty=0.0)
    qs.cancel(o)
    assert o.alive is False
    assert qs.qty_ahead(o) == 0.0
