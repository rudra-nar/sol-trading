"""Tests for the Clock module."""

import time

from sol_ew.core.clock import Clock


class TestReplayClock:
    """Test replay-mode clock behavior."""

    def test_initial_time_is_zero(self) -> None:
        clock = Clock(mode="replay")
        assert clock.now == 0

    def test_advance(self) -> None:
        clock = Clock(mode="replay")
        clock.advance(5000)
        assert clock.now == 5000

    def test_advance_monotonic(self) -> None:
        clock = Clock(mode="replay")
        clock.advance(1000)
        clock.advance(2000)
        assert clock.now == 2000

    def test_advance_same_time(self) -> None:
        clock = Clock(mode="replay")
        clock.advance(1000)
        clock.advance(1000)  # same time is fine
        assert clock.now == 1000

    def test_advance_backwards_raises(self) -> None:
        clock = Clock(mode="replay")
        clock.advance(2000)
        try:
            clock.advance(1000)
            raise AssertionError("Should raise ValueError")
        except ValueError:
            pass

    def test_snapshot_restore(self) -> None:
        clock = Clock(mode="replay")
        clock.advance(5000)
        snap = clock.snapshot()

        clock2 = Clock(mode="replay")
        clock2.restore(snap)
        assert clock2.now == 5000
        assert clock2.mode == "replay"


class TestLiveClock:
    """Test live-mode clock behavior."""

    def test_live_returns_system_time(self) -> None:
        clock = Clock(mode="live")
        now = clock.now
        sys_ms = int(time.time() * 1000)
        assert abs(now - sys_ms) < 1000  # within 1 second

    def test_live_advance_raises(self) -> None:
        clock = Clock(mode="live")
        try:
            clock.advance(1000)
            raise AssertionError("Should raise RuntimeError")
        except RuntimeError:
            pass


class TestInvalidMode:
    """Test invalid clock modes."""

    def test_invalid_mode_raises(self) -> None:
        try:
            Clock(mode="invalid")
            raise AssertionError("Should raise ValueError")
        except ValueError:
            pass
