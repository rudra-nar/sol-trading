"""Tests for Elliott Wave validation rules.

Tests each hard rule with passing and failing hand-built sequences,
both directions (bullish and bearish).
"""

from __future__ import annotations

from sol_ew.agents.pivot_wave.wave_rules import (
    rule_abc_b_within,
    rule_abc_c_beyond_a,
    rule_w2_retrace,
    rule_w3_not_shortest,
    rule_w4_no_overlap,
    validate_corrective,
    validate_impulse,
)
from sol_ew.core.types import Pivot


def _pivot(price: float, kind: int, idx: int = 0) -> Pivot:
    """Create a simple test pivot."""
    return Pivot(
        tf="1h", idx=idx, ts=idx * 3600000,
        confirm_idx=idx + 1, confirm_ts=(idx + 1) * 3600000,
        price=price, kind=kind,
    )


# ============================================================================
# Rule 1: Wave 2 retrace (both directions)
# ============================================================================


class TestW2Retrace:
    """Rule 1: Wave 2 never retraces more than 100% of Wave 1."""

    def test_bullish_valid(self) -> None:
        """Bullish W1 (100->120), W2 retraces to 110: valid."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(120.0, 1, 1)
        p2 = _pivot(110.0, -1, 2)
        result = rule_w2_retrace(p0, p1, p2)
        assert result.valid is True

    def test_bullish_invalid(self) -> None:
        """Bullish W1 (100->120), W2 retraces below 100: invalid."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(120.0, 1, 1)
        p2 = _pivot(95.0, -1, 2)
        result = rule_w2_retrace(p0, p1, p2)
        assert result.valid is False

    def test_bearish_valid(self) -> None:
        """Bearish W1 (100->80), W2 retraces to 90: valid."""
        p0 = _pivot(100.0, 1, 0)
        p1 = _pivot(80.0, -1, 1)
        p2 = _pivot(90.0, 1, 2)
        result = rule_w2_retrace(p0, p1, p2)
        assert result.valid is True

    def test_bearish_invalid(self) -> None:
        """Bearish W1 (100->80), W2 retraces above 100: invalid."""
        p0 = _pivot(100.0, 1, 0)
        p1 = _pivot(80.0, -1, 1)
        p2 = _pivot(105.0, 1, 2)
        result = rule_w2_retrace(p0, p1, p2)
        assert result.valid is False

    def test_ideal_fib_scores_high(self) -> None:
        """W2 at 61.8% retrace should score well."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(200.0, 1, 1)
        # 61.8% retrace: 200 - 100*0.618 = 138.2
        p2 = _pivot(138.2, -1, 2)
        result = rule_w2_retrace(p0, p1, p2)
        assert result.valid is True
        assert result.score > 0.7


# ============================================================================
# Rule 2: Wave 3 not shortest (both directions)
# ============================================================================


class TestW3NotShortest:
    """Rule 2: Wave 3 is never the shortest of Waves 1, 3, 5."""

    def test_w3_longest_valid(self) -> None:
        """W3 is the longest: always valid."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(120.0, 1, 1)   # W1 = 20
        p2 = _pivot(110.0, -1, 2)
        p3 = _pivot(170.0, 1, 3)   # W3 = 60
        p4 = _pivot(155.0, -1, 4)
        p5 = _pivot(180.0, 1, 5)   # W5 = 25
        result = rule_w3_not_shortest(p0, p1, p2, p3, p4, p5)
        assert result.valid is True

    def test_w3_shortest_invalid(self) -> None:
        """W3 is the shortest: invalid."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(140.0, 1, 1)   # W1 = 40
        p2 = _pivot(130.0, -1, 2)
        p3 = _pivot(145.0, 1, 3)   # W3 = 15 (shortest!)
        p4 = _pivot(135.0, -1, 4)
        p5 = _pivot(180.0, 1, 5)   # W5 = 45
        result = rule_w3_not_shortest(p0, p1, p2, p3, p4, p5)
        assert result.valid is False

    def test_partial_validation(self) -> None:
        """With only 4 pivots, partial validation should work."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(120.0, 1, 1)
        p2 = _pivot(110.0, -1, 2)
        p3 = _pivot(160.0, 1, 3)
        result = rule_w3_not_shortest(p0, p1, p2, p3, p3)
        assert result.valid is True


# ============================================================================
# Rule 3: Wave 4 no overlap (both directions)
# ============================================================================


class TestW4NoOverlap:
    """Rule 3: Wave 4 never enters the price territory of Wave 1."""

    def test_bullish_valid(self) -> None:
        """Bullish: W4 stays above W1 end."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(120.0, 1, 1)  # W1 end
        p3 = _pivot(160.0, 1, 3)
        p4 = _pivot(140.0, -1, 4)  # W4 end above p1 (120)
        result = rule_w4_no_overlap(p0, p1, p3, p4)
        assert result.valid is True

    def test_bullish_invalid(self) -> None:
        """Bullish: W4 overlaps W1 territory."""
        p0 = _pivot(100.0, -1, 0)
        p1 = _pivot(120.0, 1, 1)
        p3 = _pivot(160.0, 1, 3)
        p4 = _pivot(115.0, -1, 4)  # W4 end BELOW p1 (120) = overlap
        result = rule_w4_no_overlap(p0, p1, p3, p4)
        assert result.valid is False

    def test_bearish_valid(self) -> None:
        """Bearish: W4 stays below W1 end."""
        p0 = _pivot(200.0, 1, 0)
        p1 = _pivot(180.0, -1, 1)
        p3 = _pivot(140.0, -1, 3)
        p4 = _pivot(160.0, 1, 4)  # W4 above p1 for bearish? No -- for bearish p4 < p1
        # For bearish: p1 < p0, direction is down. W4 high should be below p1.
        # p4 (160) < p1 (180)? YES, valid.
        result = rule_w4_no_overlap(p0, p1, p3, p4)
        assert result.valid is True

    def test_bearish_invalid(self) -> None:
        """Bearish: W4 overlaps W1."""
        p0 = _pivot(200.0, 1, 0)
        p1 = _pivot(180.0, -1, 1)
        p3 = _pivot(140.0, -1, 3)
        p4 = _pivot(185.0, 1, 4)  # Above p1 (180) = overlap
        result = rule_w4_no_overlap(p0, p1, p3, p4)
        assert result.valid is False


# ============================================================================
# Corrective rules (both directions)
# ============================================================================


class TestABCRules:
    """Test corrective ABC rules."""

    def test_b_within_bearish_valid(self) -> None:
        """Bearish A: B stays below origin."""
        origin = _pivot(120.0, 1, 0)
        pa = _pivot(100.0, -1, 1)  # A goes down
        pb = _pivot(110.0, 1, 2)   # B bounces but stays below origin
        result = rule_abc_b_within(pa, pb, origin)
        assert result.valid is True

    def test_b_beyond_origin_invalid(self) -> None:
        """B goes beyond origin: invalid."""
        origin = _pivot(120.0, 1, 0)
        pa = _pivot(100.0, -1, 1)
        pb = _pivot(125.0, 1, 2)   # B exceeds origin
        result = rule_abc_b_within(pa, pb, origin)
        assert result.valid is False

    def test_c_beyond_a_valid(self) -> None:
        """C extends beyond A."""
        pa = _pivot(100.0, -1, 1)  # A ended at low
        pc = _pivot(90.0, -1, 3)   # C goes lower
        result = rule_abc_c_beyond_a(pa, pc)
        assert result.valid is True

    def test_c_not_beyond_a_invalid(self) -> None:
        """C doesn't extend beyond A: invalid (truncated C)."""
        pa = _pivot(100.0, -1, 1)
        pc = _pivot(105.0, -1, 3)  # C higher than A end
        result = rule_abc_c_beyond_a(pa, pc)
        assert result.valid is False


# ============================================================================
# Composite validation
# ============================================================================


class TestCompositeValidation:
    """Test full impulse/corrective validation."""

    def test_valid_bullish_impulse(self) -> None:
        pivots = [
            _pivot(100.0, -1, 0),   # start
            _pivot(120.0, 1, 1),     # W1 end
            _pivot(110.0, -1, 2),    # W2 end (50% retrace)
            _pivot(160.0, 1, 3),     # W3 end (longest)
            _pivot(140.0, -1, 4),    # W4 end (above p1=120)
            _pivot(180.0, 1, 5),     # W5 end
        ]
        valid, score, _results = validate_impulse(pivots)
        assert valid is True
        assert score > 0.5

    def test_invalid_impulse_w4_overlap(self) -> None:
        pivots = [
            _pivot(100.0, -1, 0),
            _pivot(120.0, 1, 1),
            _pivot(110.0, -1, 2),
            _pivot(160.0, 1, 3),
            _pivot(115.0, -1, 4),    # W4 overlaps W1!
        ]
        valid, _score, _results = validate_impulse(pivots)
        assert valid is False

    def test_valid_corrective(self) -> None:
        pivots = [
            _pivot(120.0, 1, 0),     # origin
            _pivot(100.0, -1, 1),    # A end
            _pivot(110.0, 1, 2),     # B end (within origin)
            _pivot(90.0, -1, 3),     # C end (beyond A)
        ]
        valid, _score, _results = validate_corrective(pivots)
        assert valid is True

    def test_too_few_pivots(self) -> None:
        """With < 3 pivots, validation should pass vacuously."""
        valid, _score, _results = validate_impulse([_pivot(100.0, -1, 0)])
        assert valid is True
