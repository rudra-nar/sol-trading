"""Tests for ZigZag pivot detector.

Acceptance tests:
- No-repaint: once a pivot is emitted, it never changes or disappears.
- Pivots alternate HIGH/LOW.
- Warm-up: no pivots before ATR is available.
- Bull/bear symmetry: mirrored price series produces mirrored pivots.
"""

from __future__ import annotations

from sol_ew.agents.pivot_wave.zigzag import ZigZag, compute_atr
from sol_ew.core.types import Bar


def _bar(
    idx: int,
    o: float = 100.0,
    h: float = 105.0,
    l: float = 95.0,
    c: float = 102.0,
    tf: str = "1h",
) -> Bar:
    """Create a synthetic bar."""
    open_ts = idx * 3_600_000
    close_ts = open_ts + 3_599_999
    return Bar(
        tf=tf, open_ts=open_ts, close_ts=close_ts,
        o=o, h=h, l=l, c=c,
        volume=1000.0, taker_buy_volume=500.0, n_trades=100,
    )


def _trending_bars(n: int, start: float = 100.0, step: float = 2.0, tf: str = "1h") -> list[Bar]:
    """Create bars with a clear uptrend."""
    bars = []
    for i in range(n):
        mid = start + i * step
        bars.append(_bar(i, o=mid, h=mid + 3, l=mid - 3, c=mid + 1, tf=tf))
    return bars


def _zigzag_bars(tf: str = "1h") -> list[Bar]:
    """Create bars with a clear zigzag pattern: up, down, up, down.

    Pattern: rise to 120, drop to 90, rise to 130, drop to 85
    Designed to produce clear pivots with ATR period=5, mult=1.5.
    """
    bars = []
    # Phase 1: gradual rise to 120 (bars 0-14, warmup)
    for i in range(15):
        mid = 100 + i * 1.5
        bars.append(_bar(len(bars), o=mid, h=mid + 2, l=mid - 2, c=mid + 1, tf=tf))

    # Phase 2: spike to 130 then sharp drop (bars 15-24)
    for i in range(5):
        mid = 125 + i * 2
        bars.append(_bar(len(bars), o=mid, h=mid + 3, l=mid - 1, c=mid + 2, tf=tf))
    for i in range(5):
        mid = 130 - i * 8
        bars.append(_bar(len(bars), o=mid, h=mid + 1, l=mid - 3, c=mid - 2, tf=tf))

    # Phase 3: recover up (bars 25-34)
    for i in range(10):
        mid = 95 + i * 5
        bars.append(_bar(len(bars), o=mid, h=mid + 3, l=mid - 2, c=mid + 2, tf=tf))

    # Phase 4: drop again (bars 35-44)
    for i in range(10):
        mid = 145 - i * 6
        bars.append(_bar(len(bars), o=mid, h=mid + 1, l=mid - 3, c=mid - 2, tf=tf))

    return bars


class TestComputeAtr:
    """Test ATR computation."""

    def test_simple_atr(self) -> None:
        bars = [
            _bar(0, o=100, h=110, l=90, c=105),
            _bar(1, o=105, h=115, l=95, c=100),
            _bar(2, o=100, h=108, l=92, c=106),
        ]
        atr = compute_atr(bars, period=2)
        assert atr > 0

    def test_insufficient_bars(self) -> None:
        assert compute_atr([], period=14) == 0.0
        assert compute_atr([_bar(0)], period=14) == 0.0


class TestZigZagWarmup:
    """Test that no pivots are emitted before ATR warm-up."""

    def test_no_pivots_during_warmup(self) -> None:
        zz = ZigZag(tf="1h", atr_period=14, mult=3.0)
        for i in range(14):
            result = zz.on_bar(_bar(i, h=100 + i, l=90 + i))
            assert result is None, f"Pivot emitted during warmup at bar {i}"

    def test_warmed_up_after_sufficient_bars(self) -> None:
        zz = ZigZag(tf="1h", atr_period=5, mult=1.5)
        for i in range(10):
            zz.on_bar(_bar(i))
        assert zz.warmed_up


class TestZigZagNoRepaint:
    """Test the no-repaint invariant."""

    def test_pivots_never_change(self) -> None:
        """Once a pivot is emitted, feeding more bars never changes it."""
        zz = ZigZag(tf="1h", atr_period=5, mult=1.5)
        bars = _zigzag_bars()

        emitted_pivots = []
        for bar in bars:
            pivot = zz.on_bar(bar)
            if pivot is not None:
                emitted_pivots.append(pivot)

        # Feed more bars -- existing pivots must not change
        for i in range(10):
            mid = 80 + i * 2
            zz.on_bar(_bar(len(bars) + i, o=mid, h=mid + 2, l=mid - 2, c=mid + 1))

        # All previously emitted pivots should still be in the list unchanged
        current_pivots = zz.pivots
        for orig in emitted_pivots:
            matching = [p for p in current_pivots if p.ts == orig.ts and p.kind == orig.kind]
            assert len(matching) == 1, f"Pivot at ts={orig.ts} changed or disappeared"
            assert matching[0].price == orig.price, "Pivot price changed (repaint!)"
            assert matching[0].confirm_ts == orig.confirm_ts, "Pivot confirm_ts changed"

    def test_pivots_are_append_only(self) -> None:
        """The pivot list only grows, never shrinks."""
        zz = ZigZag(tf="1h", atr_period=5, mult=1.5)
        bars = _zigzag_bars()

        prev_count = 0
        for bar in bars:
            zz.on_bar(bar)
            current_count = len(zz.pivots)
            assert current_count >= prev_count, "Pivot list shrunk!"
            prev_count = current_count


class TestZigZagAlternation:
    """Test that pivots alternate HIGH/LOW."""

    def test_alternating_kinds(self) -> None:
        zz = ZigZag(tf="1h", atr_period=5, mult=1.5)
        bars = _zigzag_bars()

        for bar in bars:
            zz.on_bar(bar)

        pivots = zz.pivots
        if len(pivots) >= 2:
            for i in range(1, len(pivots)):
                assert pivots[i].kind != pivots[i - 1].kind, (
                    f"Consecutive pivots {i-1} and {i} have same kind={pivots[i].kind}"
                )


class TestZigZagConfirmation:
    """Test that confirm_ts is always at or after the confirming bar."""

    def test_confirm_ts_after_pivot_ts(self) -> None:
        zz = ZigZag(tf="1h", atr_period=5, mult=1.5)
        bars = _zigzag_bars()

        for bar in bars:
            pivot = zz.on_bar(bar)
            if pivot is not None:
                # confirm_ts must be >= ts (confirmation comes after the extreme)
                assert pivot.confirm_ts >= pivot.ts, (
                    f"Pivot confirmed before it occurred: ts={pivot.ts}, confirm_ts={pivot.confirm_ts}"
                )
                # confirm_idx must be >= idx
                assert pivot.confirm_idx >= pivot.idx


    def test_bullish_bearish_mirror(self) -> None:
        """Mirrored price series should produce pivots with mirrored kinds.

        Note: The zigzag always starts looking for HIGH, so the initial
        direction bias means the count may differ by 1. We verify that
        the alternation pattern holds and that common pivots are mirrored.
        """
        zz_bull = ZigZag(tf="1h", atr_period=5, mult=1.5)
        zz_bear = ZigZag(tf="1h", atr_period=5, mult=1.5)

        # Use symmetric OHLC (o=c=mid) so mirror is exact
        bull_bars = []
        for i in range(30):
            if i < 10:
                mid = 100 + i * 3
            elif i < 20:
                mid = 130 - (i - 10) * 5
            else:
                mid = 80 + (i - 20) * 4
            bull_bars.append(_bar(i, o=mid, h=mid + 2, l=mid - 2, c=mid))

        bear_bars = []
        for i, bb in enumerate(bull_bars):
            m = 200 - bb.o
            bear_bars.append(_bar(i, o=m, h=m + 2, l=m - 2, c=m))

        for bar in bull_bars:
            zz_bull.on_bar(bar)
        for bar in bear_bars:
            zz_bear.on_bar(bar)

        bull_pivots = zz_bull.pivots
        bear_pivots = zz_bear.pivots

        # Both should produce pivots
        assert len(bull_pivots) >= 1
        assert len(bear_pivots) >= 1

        # Pivots at the same bar index should have opposite kinds
        bull_by_idx = {p.idx: p for p in bull_pivots}
        bear_by_idx = {p.idx: p for p in bear_pivots}
        common_idx = set(bull_by_idx.keys()) & set(bear_by_idx.keys())
        for idx in common_idx:
            assert bull_by_idx[idx].kind == -bear_by_idx[idx].kind, (
                f"At idx={idx}: bull kind={bull_by_idx[idx].kind}, "
                f"bear kind={bear_by_idx[idx].kind} (should be opposite)"
            )


class TestZigZagSnapshot:
    """Test snapshot/restore."""

    def test_snapshot_restore(self) -> None:
        zz = ZigZag(tf="1h", atr_period=5, mult=1.5)
        for i in range(20):
            zz.on_bar(_bar(i, h=100 + i * 2, l=90 + i * 2))

        snap = zz.snapshot()
        assert snap["warmed_up"] is True
        assert snap["bar_idx"] == 20

        zz2 = ZigZag(tf="1h", atr_period=5, mult=1.5)
        zz2.restore(snap)
        assert zz2.warmed_up
        assert zz2.bar_count == 20
