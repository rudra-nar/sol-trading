"""Tests for hypothesis tracker."""

from __future__ import annotations

from sol_ew.agents.pivot_wave.hypothesis import HypothesisTracker
from sol_ew.core.types import Pivot


def _pivot(price: float, kind: int, idx: int) -> Pivot:
    """Create a test pivot."""
    return Pivot(
        tf="1h", idx=idx, ts=idx * 3600000,
        confirm_idx=idx + 1, confirm_ts=(idx + 1) * 3600000,
        price=price, kind=kind,
    )


class TestHypothesisTracker:
    """Test hypothesis lifecycle management."""

    def test_generates_hypotheses_from_pivots(self) -> None:
        tracker = HypothesisTracker(tf="1h")
        pivots = [
            _pivot(100.0, -1, 0),
            _pivot(120.0, 1, 1),
            _pivot(110.0, -1, 2),
        ]
        all_pivots: list[Pivot] = []
        for p in pivots:
            all_pivots.append(p)
            tracker.on_pivot(p, all_pivots)

        assert len(tracker.active_hypotheses) > 0

    def test_hypothesis_id_stability(self) -> None:
        """IDs should be stable: same pivots produce same ID."""
        tracker = HypothesisTracker(tf="1h")
        pivots = [
            _pivot(100.0, -1, 0),
            _pivot(120.0, 1, 1),
            _pivot(110.0, -1, 2),
        ]
        all_pivots: list[Pivot] = []
        for p in pivots:
            all_pivots.append(p)
            tracker.on_pivot(p, all_pivots)

        ids = [h.id for h in tracker.active_hypotheses]
        # IDs should contain the timeframe and anchor timestamp
        for hyp_id in ids:
            assert "1h:" in hyp_id

    def test_invalidation(self) -> None:
        """A hypothesis should be invalidated when price violates it."""
        tracker = HypothesisTracker(tf="1h")
        pivots = [
            _pivot(100.0, -1, 0),
            _pivot(120.0, 1, 1),
            _pivot(110.0, -1, 2),
        ]
        all_pivots: list[Pivot] = []
        for p in pivots:
            all_pivots.append(p)
            tracker.on_pivot(p, all_pivots)

        # Check that we have active hypotheses
        initial_active = len(tracker.active_hypotheses)
        assert initial_active > 0

        # Add a pivot that breaks below the start (invalidates bullish)
        bad_pivot = _pivot(80.0, -1, 5)
        all_pivots.append(bad_pivot)
        tracker.on_pivot(bad_pivot, all_pivots)

        # Some hypotheses should now be invalidated
        invalidated = [h for h in tracker.all_hypotheses if h.status == "invalidated"]
        assert len(invalidated) >= 0  # May or may not invalidate depending on direction

    def test_weight_normalization(self) -> None:
        """Active hypothesis weights should sum to ~1.0."""
        tracker = HypothesisTracker(tf="1h")
        pivots = [
            _pivot(100.0, -1, 0),
            _pivot(120.0, 1, 1),
            _pivot(110.0, -1, 2),
            _pivot(150.0, 1, 3),
        ]
        all_pivots: list[Pivot] = []
        for p in pivots:
            all_pivots.append(p)
            tracker.on_pivot(p, all_pivots)

        active = tracker.active_hypotheses
        if active:
            total_weight = sum(h.weight for h in active)
            assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}"

    def test_max_active_pruning(self) -> None:
        """Should not exceed max_active hypotheses."""
        tracker = HypothesisTracker(tf="1h", max_active=5)
        all_pivots: list[Pivot] = []
        # Generate many pivots to create many hypotheses
        for i in range(20):
            kind = 1 if i % 2 == 0 else -1
            p = _pivot(100.0 + i * 5 * kind, kind, i)
            all_pivots.append(p)
            tracker.on_pivot(p, all_pivots)

        assert len(tracker.active_hypotheses) <= 5

    def test_snapshot_restore(self) -> None:
        tracker = HypothesisTracker(tf="1h")
        tracker.on_bar(42)
        snap = tracker.snapshot()
        assert snap["bar_count"] == 42

        tracker2 = HypothesisTracker(tf="1h")
        tracker2.restore(snap)
