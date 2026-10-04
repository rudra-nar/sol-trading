"""Tests for the fill simulator.

Phase 3 acceptance tests:
- Market order fill, limit fill, stop-loss hit, take-profit hit
- Round-trip cost = 2 * qty * price * fee_rate (within tolerance)
- Funding payment = rate * position_size * mark_price
"""

from __future__ import annotations

from sol_ew.core.types import Bar
from sol_ew.sim.simulator import Side, Simulator


def _bar(
    idx: int = 0,
    o: float = 100.0,
    h: float = 105.0,
    l: float = 95.0,
    c: float = 102.0,
    base_ts: int = 0,
    interval_ms: int = 300_000,  # 5m bars
) -> Bar:
    """Create a synthetic bar."""
    open_ts = base_ts + idx * interval_ms
    close_ts = open_ts + interval_ms - 1
    return Bar(
        tf="5m", open_ts=open_ts, close_ts=close_ts,
        o=o, h=h, l=l, c=c,
        volume=1000.0, taker_buy_volume=500.0, n_trades=100,
    )


# ============================================================================
# Market Order Tests
# ============================================================================


class TestMarketOrder:
    """Test market order fill logic."""

    def test_long_market_order(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)

        fills = sim.on_bar(_bar(0, o=100.0))
        assert len(fills) == 1
        assert fills[0].side == Side.LONG
        assert fills[0].price == 100.0
        assert fills[0].is_maker is False
        assert sim.position is not None
        assert sim.position.side == Side.LONG

    def test_short_market_order(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_market_order("sell1", Side.SHORT, qty=1.0)

        fills = sim.on_bar(_bar(0, o=100.0))
        assert len(fills) == 1
        assert fills[0].side == Side.SHORT
        assert sim.position is not None
        assert sim.position.side == Side.SHORT

    def test_market_order_with_slippage(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=10.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)

        fills = sim.on_bar(_bar(0, o=100.0))
        # Long slips up: 100 + 100 * 10/10000 = 100.1
        assert fills[0].price == 100.0 + 100.0 * 10.0 / 10_000
        assert fills[0].price > 100.0

    def test_short_slippage_goes_down(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=10.0)
        sim.submit_market_order("sell1", Side.SHORT, qty=1.0)

        fills = sim.on_bar(_bar(0, o=100.0))
        assert fills[0].price < 100.0


# ============================================================================
# Limit Order Tests
# ============================================================================


class TestLimitOrder:
    """Test limit order fill logic."""

    def test_long_limit_filled(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_limit_order("lim1", Side.LONG, qty=1.0, price=96.0)

        # Bar low touches 95, so limit at 96 should fill
        fills = sim.on_bar(_bar(0, o=100.0, h=105.0, l=95.0, c=102.0))
        assert len(fills) == 1
        assert fills[0].price == 96.0
        assert fills[0].is_maker is True

    def test_long_limit_not_filled(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_limit_order("lim1", Side.LONG, qty=1.0, price=90.0)

        # Bar low is 95, limit at 90 should NOT fill
        fills = sim.on_bar(_bar(0, o=100.0, h=105.0, l=95.0, c=102.0))
        assert len(fills) == 0
        assert len(sim.pending_orders) == 1

    def test_short_limit_filled(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_limit_order("lim1", Side.SHORT, qty=1.0, price=104.0)

        # Bar high reaches 105, so limit at 104 should fill
        fills = sim.on_bar(_bar(0, o=100.0, h=105.0, l=95.0, c=102.0))
        assert len(fills) == 1
        assert fills[0].price == 104.0

    def test_maker_fee_on_limit(self) -> None:
        sim = Simulator(fee_maker=0.0002, fee_taker=0.0005, slippage_bps=0.0)
        sim.submit_limit_order("lim1", Side.LONG, qty=10.0, price=96.0)

        fills = sim.on_bar(_bar(0, l=95.0))
        expected_fee = 10.0 * 96.0 * 0.0002
        assert abs(fills[0].fee - expected_fee) < 0.0001


# ============================================================================
# Stop-Loss Tests
# ============================================================================


class TestStopLoss:
    """Test stop-loss execution."""

    def test_long_stop_hit(self) -> None:
        sim = Simulator(slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)
        sim.on_bar(_bar(0, o=100.0, h=105.0, l=99.0, c=101.0))
        sim.set_stop_loss(97.0)

        # Bar that hits stop
        fills = sim.on_bar(_bar(1, o=99.0, h=100.0, l=96.0, c=97.0))
        sl_fills = [f for f in fills if f.reason == "stop_loss"]
        assert len(sl_fills) == 1
        assert sl_fills[0].price <= 97.0
        assert sim.position is None

    def test_short_stop_hit(self) -> None:
        sim = Simulator(slippage_bps=0.0)
        sim.submit_market_order("sell1", Side.SHORT, qty=1.0)
        sim.on_bar(_bar(0, o=100.0))
        sim.set_stop_loss(103.0)

        # Bar that hits stop
        fills = sim.on_bar(_bar(1, o=101.0, h=104.0, l=100.0, c=103.0))
        sl_fills = [f for f in fills if f.reason == "stop_loss"]
        assert len(sl_fills) == 1
        assert sl_fills[0].price >= 103.0
        assert sim.position is None

    def test_stop_gap_through(self) -> None:
        """Stop should fill at bar open if it gaps through the stop price."""
        sim = Simulator(slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)
        sim.on_bar(_bar(0, o=100.0, h=105.0, l=99.0, c=101.0))
        sim.set_stop_loss(97.0)

        # Gap down: bar opens at 95, well below stop at 97
        fills = sim.on_bar(_bar(1, o=95.0, h=96.0, l=94.0, c=95.5))
        sl_fills = [f for f in fills if f.reason == "stop_loss"]
        assert len(sl_fills) == 1
        # Should fill at bar open (gap-through), not stop price
        assert sl_fills[0].price == 95.0

    def test_stop_not_hit(self) -> None:
        sim = Simulator(slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)
        sim.on_bar(_bar(0, o=100.0))
        sim.set_stop_loss(90.0)

        fills = sim.on_bar(_bar(1, o=100.0, h=105.0, l=95.0, c=102.0))
        sl_fills = [f for f in fills if f.reason == "stop_loss"]
        assert len(sl_fills) == 0
        assert sim.position is not None


# ============================================================================
# Take-Profit Tests
# ============================================================================


class TestTakeProfit:
    """Test take-profit execution."""

    def test_long_tp_hit(self) -> None:
        sim = Simulator(slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)
        sim.on_bar(_bar(0, o=100.0))
        sim.set_take_profit(110.0)

        fills = sim.on_bar(_bar(1, o=105.0, h=112.0, l=104.0, c=111.0))
        tp_fills = [f for f in fills if f.reason == "take_profit"]
        assert len(tp_fills) == 1
        assert tp_fills[0].price == 110.0
        assert sim.position is None

    def test_short_tp_hit(self) -> None:
        sim = Simulator(slippage_bps=0.0)
        sim.submit_market_order("sell1", Side.SHORT, qty=1.0)
        sim.on_bar(_bar(0, o=100.0))
        sim.set_take_profit(90.0)

        fills = sim.on_bar(_bar(1, o=95.0, h=96.0, l=89.0, c=91.0))
        tp_fills = [f for f in fills if f.reason == "take_profit"]
        assert len(tp_fills) == 1
        assert tp_fills[0].price == 90.0
        assert sim.position is None


# ============================================================================
# Round-Trip Cost Test (Acceptance Criterion)
# ============================================================================


class TestRoundTripCost:
    """Round-trip cost = 2 * qty * price * fee_rate (within tolerance)."""

    def test_round_trip_taker_fees(self) -> None:
        fee_rate = 0.0005
        sim = Simulator(
            initial_capital=100_000.0,
            fee_maker=0.0002, fee_taker=fee_rate,
            slippage_bps=0.0,
        )

        qty = 10.0
        price = 100.0

        # Open long
        sim.submit_market_order("open", Side.LONG, qty=qty)
        sim.on_bar(_bar(0, o=price, h=price + 5, l=price - 5, c=price))

        # Close long (market order on opposite side)
        sim.submit_market_order("close", Side.SHORT, qty=qty, reduce_only=True)
        sim.on_bar(_bar(1, o=price, h=price + 5, l=price - 5, c=price))

        # Round-trip fee should be approximately 2 * qty * price * fee_rate
        expected_fee = 2 * qty * price * fee_rate
        assert abs(sim.total_fees - expected_fee) < 0.01, (
            f"Expected round-trip fee ~{expected_fee:.4f}, got {sim.total_fees:.4f}"
        )

    def test_round_trip_maker_fees(self) -> None:
        fee_rate = 0.0002
        sim = Simulator(
            initial_capital=100_000.0,
            fee_maker=fee_rate, fee_taker=0.0005,
            slippage_bps=0.0,
        )

        qty = 5.0
        price = 150.0

        # Open with limit
        sim.submit_limit_order("open", Side.LONG, qty=qty, price=price)
        sim.on_bar(_bar(0, o=price + 1, h=price + 5, l=price - 5, c=price))

        # Close with limit
        sim.submit_limit_order("close", Side.SHORT, qty=qty, price=price, reduce_only=True)
        sim.on_bar(_bar(1, o=price - 1, h=price + 5, l=price - 5, c=price))

        expected_fee = 2 * qty * price * fee_rate
        assert abs(sim.total_fees - expected_fee) < 0.01


# ============================================================================
# Funding Tests (Acceptance Criterion)
# ============================================================================


class TestFunding:
    """Funding payment = rate * position_size * mark_price."""

    def test_funding_applied_after_8h(self) -> None:
        sim = Simulator(slippage_bps=0.0, funding_interval_ms=8 * 3_600_000)

        # Open position
        sim.submit_market_order("buy1", Side.LONG, qty=10.0)
        sim.on_bar(_bar(0, o=100.0, c=100.0, base_ts=0))

        initial_cash = sim.cash

        # Feed bars across an 8h funding boundary
        # 8h = 28_800_000 ms, 5m bars = 300_000 ms each, need 96 bars
        for i in range(1, 100):
            sim.on_bar(_bar(i, o=100.0, h=105.0, l=95.0, c=100.0, base_ts=0))

        # Funding should have been applied
        assert sim.total_funding > 0
        # Cash should have decreased (long pays funding when rate positive)
        assert sim.cash < initial_cash

    def test_no_funding_without_position(self) -> None:
        sim = Simulator(slippage_bps=0.0)
        for i in range(100):
            sim.on_bar(_bar(i, o=100.0, c=100.0))
        assert sim.total_funding == 0.0


# ============================================================================
# Equity & Drawdown Tests
# ============================================================================


class TestEquity:
    """Test equity curve and drawdown tracking."""

    def test_initial_equity(self) -> None:
        sim = Simulator(initial_capital=10_000.0)
        sim.on_bar(_bar(0))
        assert sim.equity == 10_000.0

    def test_equity_tracks_unrealized(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)
        sim.on_bar(_bar(0, o=100.0, c=100.0))  # Entry

        # Price goes up
        sim.on_bar(_bar(1, o=101.0, c=110.0))
        # Equity should be > initial (unrealized profit)
        assert sim.equity > 10_000.0

    def test_drawdown_tracking(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)
        sim.on_bar(_bar(0, o=100.0, c=100.0))

        # Price spike up then down
        sim.on_bar(_bar(1, o=100.0, h=150.0, l=100.0, c=150.0))
        sim.on_bar(_bar(2, o=150.0, h=150.0, l=100.0, c=100.0))

        assert sim.max_drawdown > 0.0

    def test_equity_curve_grows(self) -> None:
        sim = Simulator(initial_capital=10_000.0)
        for i in range(5):
            sim.on_bar(_bar(i))
        assert len(sim.equity_curve) == 5


# ============================================================================
# State Tests
# ============================================================================


class TestSimState:
    """Test snapshot/restore."""

    def test_snapshot_restore(self) -> None:
        sim = Simulator(initial_capital=10_000.0, slippage_bps=0.0)
        sim.submit_market_order("buy1", Side.LONG, qty=1.0)
        sim.on_bar(_bar(0, o=100.0))
        sim.set_stop_loss(95.0)

        snap = sim.snapshot()

        sim2 = Simulator(initial_capital=10_000.0)
        sim2.restore(snap)
        assert sim2.position is not None
        assert sim2.position.stop_loss == 95.0
        assert abs(sim2.cash - sim.cash) < 0.01

    def test_cancel_order(self) -> None:
        sim = Simulator()
        sim.submit_limit_order("lim1", Side.LONG, qty=1.0, price=90.0)
        assert len(sim.pending_orders) == 1
        assert sim.cancel_order("lim1") is True
        assert len(sim.pending_orders) == 0
        assert sim.cancel_order("nonexistent") is False
