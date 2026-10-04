"""Tests for bar builder and resampler.

Includes property tests as required by Phase 2 acceptance:
- No bar is emitted before its close_ts.
- The bar aggregate equals the aggregate over its constituents.
- HTF alignment test (invariant 5).
"""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from sol_ew.bars.builder import BarBuilder, bar_from_kline_event
from sol_ew.bars.resample import BarResampler, MultiTimeframeResampler, _align_ts
from sol_ew.core.types import Bar


def _make_1m_bar(
    idx: int,
    base_ts: int = 0,
    o: float = 100.0,
    h: float = 105.0,
    l: float = 98.0,
    c: float = 103.0,
    degraded: bool = False,
) -> Bar:
    """Create a synthetic 1m bar."""
    open_ts = base_ts + idx * 60_000
    close_ts = open_ts + 59_999
    return Bar(
        tf="1m", open_ts=open_ts, close_ts=close_ts,
        o=o, h=h, l=l, c=c,
        volume=1000.0, taker_buy_volume=500.0, n_trades=100,
        degraded=degraded,
    )


# ============================================================================
# Bar Builder Tests
# ============================================================================


class TestBarBuilder:
    """Test BarBuilder from kline events."""

    def test_closed_kline_emits_bar(self) -> None:
        builder = BarBuilder(tf="1m")
        bar = builder.on_kline(
            open_ts=0, close_ts=59999,
            o=100.0, h=105.0, l=98.0, c=103.0,
            volume=1000.0, taker_buy_volume=500.0, n_trades=100,
            is_closed=True,
        )
        assert bar is not None
        assert bar.tf == "1m"
        assert bar.o == 100.0
        assert bar.close_ts == 59999

    def test_unclosed_kline_returns_none(self) -> None:
        builder = BarBuilder(tf="1m")
        bar = builder.on_kline(
            open_ts=0, close_ts=59999,
            o=100.0, h=105.0, l=98.0, c=103.0,
            volume=1000.0, taker_buy_volume=500.0, n_trades=100,
            is_closed=False,
        )
        assert bar is None

    def test_gap_detection(self) -> None:
        builder = BarBuilder(tf="1m")
        # First bar
        bar1 = builder.on_kline(
            open_ts=0, close_ts=59999,
            o=100.0, h=105.0, l=98.0, c=103.0,
            volume=1000.0, taker_buy_volume=500.0, n_trades=100,
            is_closed=True,
        )
        assert bar1 is not None
        assert bar1.degraded is False

        # Second bar with gap (skip 2 minutes)
        bar2 = builder.on_kline(
            open_ts=180000, close_ts=239999,
            o=103.0, h=106.0, l=101.0, c=104.0,
            volume=1200.0, taker_buy_volume=600.0, n_trades=120,
            is_closed=True,
        )
        assert bar2 is not None
        assert bar2.degraded is True

    def test_consecutive_bars_not_degraded(self) -> None:
        builder = BarBuilder(tf="1m")
        for i in range(5):
            bar = builder.on_kline(
                open_ts=i * 60000, close_ts=(i + 1) * 60000 - 1,
                o=100.0 + i, h=105.0 + i, l=98.0 + i, c=103.0 + i,
                volume=1000.0, taker_buy_volume=500.0, n_trades=100,
                is_closed=True,
            )
            assert bar is not None
            assert bar.degraded is False

    def test_bars_emitted_counter(self) -> None:
        builder = BarBuilder(tf="1m")
        assert builder.bars_emitted == 0
        for i in range(3):
            builder.on_kline(
                open_ts=i * 60000, close_ts=(i + 1) * 60000 - 1,
                o=100.0, h=105.0, l=98.0, c=103.0,
                volume=1000.0, taker_buy_volume=500.0, n_trades=100,
                is_closed=True,
            )
        assert builder.bars_emitted == 3

    def test_snapshot_restore(self) -> None:
        builder = BarBuilder(tf="1m")
        builder.on_kline(
            open_ts=0, close_ts=59999,
            o=100.0, h=105.0, l=98.0, c=103.0,
            volume=1000.0, taker_buy_volume=500.0, n_trades=100,
            is_closed=True,
        )
        snap = builder.snapshot()

        builder2 = BarBuilder(tf="1m")
        builder2.restore(snap)
        assert builder2.bars_emitted == 1


class TestBarFromKlineEvent:
    """Test bar_from_kline_event helper."""

    def test_closed_event(self) -> None:
        event_data = {
            "k": {
                "t": 0, "T": 59999,
                "o": "100.0", "h": "105.0", "l": "98.0", "c": "103.0",
                "v": "1000.0", "V": "500.0", "n": 100,
                "x": True,
            }
        }
        bar = bar_from_kline_event(event_data)
        assert bar is not None
        assert bar.o == 100.0
        assert bar.close_ts == 59999

    def test_unclosed_event(self) -> None:
        event_data = {
            "k": {
                "t": 0, "T": 59999,
                "o": "100.0", "h": "105.0", "l": "98.0", "c": "103.0",
                "v": "1000.0", "x": False,
            }
        }
        bar = bar_from_kline_event(event_data)
        assert bar is None

    def test_no_kline_key(self) -> None:
        assert bar_from_kline_event({}) is None


# ============================================================================
# Bar Resampler Tests
# ============================================================================


class TestBarResampler:
    """Test single-pair resampling."""

    def test_5m_from_1m(self) -> None:
        resampler = BarResampler(source_tf="1m", target_tf="5m")
        results: list[Bar] = []

        # Feed 10 1m bars (should produce 2 5m bars)
        for i in range(10):
            bar = _make_1m_bar(i, o=100.0 + i, h=110.0 + i, l=95.0 + i, c=102.0 + i)
            result = resampler.on_bar(bar)
            if result is not None:
                results.append(result)

        # The last period is still forming, flush it
        final = resampler.flush()
        if final:
            results.append(final)

        assert len(results) == 2
        for r in results:
            assert r.tf == "5m"

    def test_5m_ohlcv_aggregation(self) -> None:
        """Verify OHLCV aggregation is correct."""
        resampler = BarResampler(source_tf="1m", target_tf="5m")

        bars_in = []
        for i in range(5):
            bar = _make_1m_bar(i, o=100.0 + i, h=110.0 + i, l=90.0 - i, c=105.0 + i)
            bars_in.append(bar)
            resampler.on_bar(bar)

        # Feed one more bar from next period to trigger emission
        trigger = _make_1m_bar(5)
        result = resampler.on_bar(trigger)

        assert result is not None
        # Open should be first bar's open
        assert result.o == bars_in[0].o
        # High should be max of all highs
        assert result.h == max(b.h for b in bars_in)
        # Low should be min of all lows
        assert result.l == min(b.l for b in bars_in)
        # Close should be last bar's close
        assert result.c == bars_in[-1].c
        # Volume should be sum
        assert abs(result.volume - sum(b.volume for b in bars_in)) < 0.01
        # N trades should be sum
        assert result.n_trades == sum(b.n_trades for b in bars_in)

    def test_degraded_propagation(self) -> None:
        """If any constituent bar is degraded, the HTF bar is degraded."""
        resampler = BarResampler(source_tf="1m", target_tf="5m")

        for i in range(5):
            bar = _make_1m_bar(i, degraded=(i == 3))
            resampler.on_bar(bar)

        # Trigger emission
        result = resampler.on_bar(_make_1m_bar(5))
        assert result is not None
        assert result.degraded is True

    def test_no_degraded_when_clean(self) -> None:
        resampler = BarResampler(source_tf="1m", target_tf="5m")
        for i in range(5):
            resampler.on_bar(_make_1m_bar(i))
        result = resampler.on_bar(_make_1m_bar(5))
        assert result is not None
        assert result.degraded is False

    def test_1h_from_1m(self) -> None:
        resampler = BarResampler(source_tf="1m", target_tf="1h")
        results: list[Bar] = []

        for i in range(120):
            result = resampler.on_bar(_make_1m_bar(i))
            if result is not None:
                results.append(result)

        final = resampler.flush()
        if final:
            results.append(final)

        assert len(results) == 2
        assert all(r.tf == "1h" for r in results)

    def test_bar_emitted_at_close(self) -> None:
        """HTF bar close_ts must be the end of its period."""
        resampler = BarResampler(source_tf="1m", target_tf="5m")

        for i in range(5):
            resampler.on_bar(_make_1m_bar(i))
        result = resampler.on_bar(_make_1m_bar(5))

        assert result is not None
        # 5m bar starting at 0 should close at 299999 (5*60000 - 1)
        assert result.close_ts == 5 * 60_000 - 1

    def test_snapshot_restore(self) -> None:
        resampler = BarResampler(source_tf="1m", target_tf="5m")
        for i in range(3):
            resampler.on_bar(_make_1m_bar(i))

        snap = resampler.snapshot()
        assert resampler.has_pending

        resampler2 = BarResampler(source_tf="1m", target_tf="5m")
        resampler2.restore(snap)
        assert resampler2.has_pending


# ============================================================================
# Multi-Timeframe Resampler Tests
# ============================================================================


class TestMultiTimeframeResampler:
    """Test cascading multi-timeframe resampler."""

    def test_produces_all_timeframes(self) -> None:
        mtf = MultiTimeframeResampler(source_tf="1m", target_tfs=["5m", "15m"])
        all_completed: dict[str, list[Bar]] = {"5m": [], "15m": []}

        for i in range(30):
            completed = mtf.on_bar(_make_1m_bar(i))
            for tf, bar in completed.items():
                all_completed[tf].append(bar)

        final = mtf.flush()
        for tf, bar in final.items():
            all_completed[tf].append(bar)

        assert len(all_completed["5m"]) == 6  # 30/5
        assert len(all_completed["15m"]) == 2  # 30/15

    def test_htf_not_emitted_before_close(self) -> None:
        """Invariant 5: a higher-TF bar must not be available before its close."""
        mtf = MultiTimeframeResampler(source_tf="1m", target_tfs=["5m"])

        for i in range(4):
            completed = mtf.on_bar(_make_1m_bar(i))
            # 5m bar should NOT be emitted during its first 4 constituents
            assert "5m" not in completed

    def test_snapshot_restore(self) -> None:
        mtf = MultiTimeframeResampler(source_tf="1m", target_tfs=["5m"])
        for i in range(3):
            mtf.on_bar(_make_1m_bar(i))
        snap = mtf.snapshot()

        mtf2 = MultiTimeframeResampler(source_tf="1m", target_tfs=["5m"])
        mtf2.restore(snap)
        # Should be able to continue accumulating


class TestAlignTs:
    """Test timestamp alignment."""

    def test_5m_alignment(self) -> None:
        assert _align_ts(0, 300_000) == 0
        assert _align_ts(60_000, 300_000) == 0
        assert _align_ts(299_999, 300_000) == 0
        assert _align_ts(300_000, 300_000) == 300_000

    def test_1h_alignment(self) -> None:
        assert _align_ts(0, 3_600_000) == 0
        assert _align_ts(1_800_000, 3_600_000) == 0
        assert _align_ts(3_599_999, 3_600_000) == 0
        assert _align_ts(3_600_000, 3_600_000) == 3_600_000


# ============================================================================
# Property Tests (Phase 2 Acceptance)
# ============================================================================


@st.composite
def random_1m_bars(draw: st.DrawFn) -> list[Bar]:
    """Generate a random sequence of consecutive 1m bars."""
    n = draw(st.integers(min_value=5, max_value=60))
    bars = []
    for i in range(n):
        o = draw(st.floats(min_value=50.0, max_value=200.0, allow_nan=False, allow_infinity=False))
        h = o + draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
        l = o - draw(st.floats(min_value=0.01, max_value=10.0, allow_nan=False, allow_infinity=False))
        c = draw(st.floats(min_value=l, max_value=h, allow_nan=False, allow_infinity=False))
        vol = draw(st.floats(min_value=0.0, max_value=100000.0, allow_nan=False, allow_infinity=False))
        bars.append(Bar(
            tf="1m",
            open_ts=i * 60_000,
            close_ts=(i + 1) * 60_000 - 1,
            o=o, h=h, l=l, c=c,
            volume=vol,
            taker_buy_volume=vol * 0.5,
            n_trades=draw(st.integers(min_value=0, max_value=10000)),
        ))
    return bars


@given(bars=random_1m_bars())
@settings(max_examples=50)
def test_property_no_early_emission(bars: list[Bar]) -> None:
    """Property: No bar is emitted before its close_ts."""
    resampler = BarResampler(source_tf="1m", target_tf="5m")
    for bar in bars:
        result = resampler.on_bar(bar)
        if result is not None:
            # The emitted bar's close_ts must be <= the current bar's open_ts
            # (it was emitted because the current bar started a new period)
            assert result.close_ts < bar.open_ts, (
                f"HTF bar close_ts={result.close_ts} >= current bar open_ts={bar.open_ts}"
            )


@given(bars=random_1m_bars())
@settings(max_examples=50)
def test_property_aggregate_equality(bars: list[Bar]) -> None:
    """Property: bar aggregate equals aggregate over constituents."""
    resampler = BarResampler(source_tf="1m", target_tf="5m")
    period_bars: list[Bar] = []
    current_period = -1

    for bar in bars:
        period = bar.open_ts // 300_000

        if current_period >= 0 and period != current_period:
            # Check the emitted bar against accumulated constituents
            result = resampler.on_bar(bar)
            if result is not None and period_bars:
                assert result.o == period_bars[0].o
                assert result.h == max(b.h for b in period_bars)
                assert result.l == min(b.l for b in period_bars)
                assert result.c == period_bars[-1].c
                assert abs(result.volume - sum(b.volume for b in period_bars)) < 0.01
            period_bars = [bar]
        else:
            resampler.on_bar(bar)
            period_bars.append(bar)

        current_period = period
