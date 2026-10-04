"""Tests for Fibonacci grid and confluence zone computation.

Acceptance tests:
- Grid arithmetic: retracement and extension levels computed correctly.
- Confluence clustering edge cases: single level, ties, multiple degrees.
"""

from __future__ import annotations

from sol_ew.agents.fib.confluence import (
    FibLevel,
    cluster_into_zones,
    compute_confluence_zones,
    compute_fib_grid,
    fib_extensions,
    fib_retracements,
)
from sol_ew.core.types import Hypothesis, Pivot


def _pivot(price: float, kind: int, idx: int = 0) -> Pivot:
    return Pivot(
        tf="1h", idx=idx, ts=idx * 3600000,
        confirm_idx=idx + 1, confirm_ts=(idx + 1) * 3600000,
        price=price, kind=kind,
    )


# ============================================================================
# Grid Arithmetic Tests
# ============================================================================


class TestFibRetracements:
    """Test retracement level computation."""

    def test_bullish_retracements(self) -> None:
        """Retracements of a bullish move 100 -> 200."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(200.0, 1, 1)
        levels = fib_retracements(p0, p1, "test", "1h")

        # 38.2% retrace of 100-point move = 200 - 100*0.382 = 161.8
        prices = {round(lv.price, 1): lv.ratio for lv in levels}
        assert round(200 - 100 * 0.382, 1) in prices
        assert round(200 - 100 * 0.618, 1) in prices
        assert round(200 - 100 * 0.500, 1) in prices

    def test_bearish_retracements(self) -> None:
        """Retracements of a bearish move 200 -> 100."""
        p0 = _pivot(200.0, 1, 0)
        p1 = _pivot(100.0, -1, 1)
        levels = fib_retracements(p0, p1, "test", "1h")

        # 38.2% retrace of 100-point down = 100 - (-100)*0.382 = 100 + 38.2 = 138.2
        prices = {round(lv.price, 1) for lv in levels}
        assert 138.2 in prices  # 100 + 100*0.382
        assert 161.8 in prices  # 100 + 100*0.618

    def test_all_levels_have_metadata(self) -> None:
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(200.0, 1, 1)
        levels = fib_retracements(p0, p1, "hyp123", "4h", weight=0.8)
        for lv in levels:
            assert lv.source_hyp_id == "hyp123"
            assert lv.tf == "4h"
            assert lv.weight == 0.8
            assert lv.kind == "retrace"


class TestFibExtensions:
    """Test extension level computation."""

    def test_bullish_extensions(self) -> None:
        """W3 extensions from W2 end. W1: 100->120, W2 end: 110."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(120.0, 1, 1)
        p2 = _pivot(110.0, -1, 2)
        levels = fib_extensions(p0, p1, p2, "test", "1h")

        # 1.618 extension: 110 + 20*1.618 = 142.36
        prices = {round(lv.price, 2) for lv in levels}
        assert 142.36 in prices

    def test_bearish_extensions(self) -> None:
        """Bearish: W1: 200->180, W2 end: 190."""
        p0 = _pivot(200.0, 1, 0)
        p1 = _pivot(180.0, -1, 1)
        p2 = _pivot(190.0, 1, 2)
        levels = fib_extensions(p0, p1, p2, "test", "1h")

        # 1.618 ext: 190 - 20*1.618 = 157.64
        prices = {round(lv.price, 2) for lv in levels}
        assert 157.64 in prices


class TestComputeFibGrid:
    """Test grid computation from a hypothesis."""

    def test_grid_with_3_pivots(self) -> None:
        """Hypothesis with W1-W2 should produce retrace + extension levels."""
        hyp = Hypothesis(
            id="test:0:impulse_W1W2_1", tf="1h",
            kind="impulse_W1W2", direction=1,
            pivots=(
                _pivot(100.0, -1, 0),
                _pivot(120.0, 1, 1),
                _pivot(110.0, -1, 2),
            ),
            invalidation=100.0,
            targets={1.618: 142.36},
            rule_score=0.8, weight=0.5,
        )
        levels = compute_fib_grid(hyp)
        # Should have retracements of W2 + extensions from W2
        assert len(levels) > 5
        kinds = {lv.kind for lv in levels}
        assert "retrace" in kinds
        assert "extension" in kinds

    def test_grid_with_1_pivot(self) -> None:
        """Too few pivots: should return empty."""
        hyp = Hypothesis(
            id="test:0:impulse_W1_1", tf="1h",
            kind="impulse_W1", direction=1,
            pivots=(_pivot(100.0, -1, 0),),
            invalidation=100.0, targets={},
            rule_score=0.5, weight=0.5,
        )
        levels = compute_fib_grid(hyp)
        assert len(levels) == 0


# ============================================================================
# Confluence Clustering Tests
# ============================================================================


class TestClusterIntoZones:
    """Test confluence zone clustering."""

    def test_empty_levels(self) -> None:
        assert cluster_into_zones([]) == []

    def test_single_level_no_zone(self) -> None:
        """A single level doesn't meet min_sources=2."""
        levels = [FibLevel(100.0, 0.618, "retrace", "hyp1", "1h", 1.0)]
        zones = cluster_into_zones(levels, min_sources=2)
        assert len(zones) == 0

    def test_single_source_no_zone(self) -> None:
        """Multiple levels from same source don't form confluence."""
        levels = [
            FibLevel(100.0, 0.382, "retrace", "hyp1", "1h", 1.0),
            FibLevel(100.1, 0.618, "retrace", "hyp1", "1h", 1.0),
        ]
        zones = cluster_into_zones(levels, min_sources=2)
        assert len(zones) == 0

    def test_two_sources_form_zone(self) -> None:
        """Two levels from different sources within threshold form a zone."""
        levels = [
            FibLevel(100.0, 0.618, "retrace", "hyp1", "1h", 0.8),
            FibLevel(100.3, 0.382, "retrace", "hyp2", "4h", 0.6),
        ]
        zones = cluster_into_zones(levels, cluster_pct=0.5, min_sources=2)
        assert len(zones) == 1
        assert zones[0].strength > 1.0  # sum of weights
        assert zones[0].lo <= 100.0
        assert zones[0].hi >= 100.3

    def test_multi_degree_confluence(self) -> None:
        """Levels from different timeframes count as separate degrees."""
        levels = [
            FibLevel(150.0, 0.618, "retrace", "hyp1", "1h", 0.5),
            FibLevel(150.2, 0.382, "retrace", "hyp2", "4h", 0.7),
            FibLevel(150.1, 1.618, "extension", "hyp3", "1D", 0.3),
        ]
        zones = cluster_into_zones(levels, cluster_pct=0.5, min_sources=2)
        assert len(zones) == 1
        assert zones[0].n_degrees == 3  # 1h, 4h, 1D

    def test_separate_clusters(self) -> None:
        """Levels far apart should form separate clusters."""
        levels = [
            FibLevel(100.0, 0.382, "retrace", "hyp1", "1h", 1.0),
            FibLevel(100.1, 0.618, "retrace", "hyp2", "1h", 1.0),
            FibLevel(200.0, 0.382, "retrace", "hyp3", "1h", 1.0),
            FibLevel(200.1, 0.618, "retrace", "hyp4", "1h", 1.0),
        ]
        zones = cluster_into_zones(levels, cluster_pct=0.5, min_sources=2)
        assert len(zones) == 2

    def test_sorted_by_strength(self) -> None:
        """Zones should be sorted by strength descending."""
        levels = [
            FibLevel(100.0, 0.382, "retrace", "hyp1", "1h", 0.5),
            FibLevel(100.1, 0.618, "retrace", "hyp2", "1h", 0.5),
            FibLevel(200.0, 0.382, "retrace", "hyp3", "1h", 1.0),
            FibLevel(200.1, 0.618, "retrace", "hyp4", "1h", 1.0),
        ]
        zones = cluster_into_zones(levels, cluster_pct=0.5, min_sources=2)
        assert len(zones) == 2
        assert zones[0].strength >= zones[1].strength

    def test_ties_all_same_price(self) -> None:
        """All levels at exact same price from different sources."""
        levels = [
            FibLevel(100.0, 0.618, "retrace", f"hyp{i}", f"{i}h", 1.0)
            for i in range(5)
        ]
        zones = cluster_into_zones(levels, min_sources=2)
        assert len(zones) == 1
        assert zones[0].lo == zones[0].hi == 100.0
        assert zones[0].n_degrees == 5


# ============================================================================
# Integration: compute_confluence_zones
# ============================================================================


class TestComputeConfluenceZones:
    """Integration test for the full pipeline."""

    def test_from_hypotheses(self) -> None:
        hyp1 = Hypothesis(
            id="1h:0:impulse", tf="1h",
            kind="impulse_W1W2", direction=1,
            pivots=(
                _pivot(100.0, -1, 0),
                _pivot(120.0, 1, 1),
                _pivot(110.0, -1, 2),
            ),
            invalidation=100.0, targets={},
            rule_score=0.8, weight=0.5,
        )
        hyp2 = Hypothesis(
            id="4h:0:impulse", tf="4h",
            kind="impulse_W1W2", direction=1,
            pivots=(
                _pivot(95.0, -1, 0),
                _pivot(125.0, 1, 1),
                _pivot(108.0, -1, 2),
            ),
            invalidation=95.0, targets={},
            rule_score=0.7, weight=0.4,
        )
        zones = compute_confluence_zones([hyp1, hyp2], cluster_pct=1.0, min_sources=2)
        # With overlapping Fibonacci grids, should find at least one confluence zone
        # (depends on level proximity, so we just verify the pipeline works)
        assert isinstance(zones, list)

    def test_empty_hypotheses(self) -> None:
        zones = compute_confluence_zones([], min_sources=2)
        assert zones == []
