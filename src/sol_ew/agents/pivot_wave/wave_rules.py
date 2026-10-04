"""Elliott Wave validation rules.

Implements the hard rules that every valid wave count must satisfy:

Impulse (5-wave):
  Rule 1: Wave 2 never retraces more than 100% of Wave 1.
  Rule 2: Wave 3 is never the shortest of Waves 1, 3, 5.
  Rule 3: Wave 4 never enters the price territory of Wave 1
           (no overlap; except in leading/ending diagonals).

Corrective (ABC):
  Rule 4: Wave B does not move beyond the origin of Wave A.
  Rule 5: Wave C must move beyond the end of Wave A.

Guidelines (soft, scored but not mandatory):
  G1: Wave 2 typically retraces 50-78.6% of Wave 1.
  G2: Wave 3 typically extends to 161.8% of Wave 1.
  G3: Wave 4 typically retraces 38.2% of Wave 3.
  G4: Alternation: if Wave 2 is sharp, Wave 4 should be flat (and vice versa).

Each rule returns (valid: bool, score: float) where score is 0-1.
"""

from __future__ import annotations

from dataclasses import dataclass

from sol_ew.core.types import Pivot


@dataclass(frozen=True)
class RuleResult:
    """Result of evaluating a wave rule."""
    name: str
    valid: bool
    score: float  # 0.0 = worst, 1.0 = ideal
    message: str = ""


# ============================================================================
# Pivot helpers
# ============================================================================


def _wave_length(start: Pivot, end: Pivot) -> float:
    """Signed price movement of a wave."""
    return end.price - start.price


def _wave_abs(start: Pivot, end: Pivot) -> float:
    """Absolute price movement of a wave."""
    return abs(end.price - start.price)


def _retrace_ratio(retracement: float, original: float) -> float:
    """Compute retracement ratio (0 = no retrace, 1 = 100% retrace)."""
    if abs(original) < 1e-10:
        return 0.0
    return abs(retracement) / abs(original)


# ============================================================================
# Hard Rules (impulse)
# ============================================================================


def rule_w2_retrace(p0: Pivot, p1: Pivot, p2: Pivot) -> RuleResult:
    """Rule 1: Wave 2 never retraces more than 100% of Wave 1.

    p0 = start of W1, p1 = end of W1 (start of W2), p2 = end of W2.
    """
    w1 = _wave_length(p0, p1)
    w2 = _wave_length(p1, p2)

    # W2 must retrace W1 (opposite direction)
    # For bullish impulse: W1 up, W2 down. W2 must not go below p0.
    if w1 > 0:
        # Bullish: p2 must be above p0
        valid = p2.price > p0.price
    else:
        # Bearish: p2 must be below p0
        valid = p2.price < p0.price

    ratio = _retrace_ratio(w2, w1)
    # Score: ideal retrace is 0.50-0.786
    if 0.50 <= ratio <= 0.786:
        score = 1.0
    elif 0.382 <= ratio <= 0.886:
        score = 0.7
    elif ratio < 1.0:
        score = max(0.3, 1.0 - ratio)
    else:
        score = 0.0

    return RuleResult(
        name="W2_retrace",
        valid=valid,
        score=score,
        message=f"W2 retraces {ratio:.1%} of W1 (max 100%)",
    )


def rule_w3_not_shortest(
    p0: Pivot, p1: Pivot, p2: Pivot, p3: Pivot, p4: Pivot, p5: Pivot | None = None,
) -> RuleResult:
    """Rule 2: Wave 3 is never the shortest of Waves 1, 3, 5.

    Needs at least W1 and W3 to partially validate. Full validation
    requires p5 (end of W5).
    """
    w1_len = _wave_abs(p0, p1)
    w3_len = _wave_abs(p2, p3)

    if p5 is None:
        # Partial check: W3 must be at least as long as W1
        # (if W3 is shorter, it COULD end up shortest, which is a warning)
        valid = w3_len >= w1_len * 0.5  # Lenient until W5 known
        score = min(1.0, w3_len / w1_len) if w1_len > 0 else 0.5
        return RuleResult(
            name="W3_not_shortest",
            valid=valid,
            score=score,
            message=f"W3/W1 ratio: {w3_len/w1_len:.2f} (W5 not yet known)",
        )

    w5_len = _wave_abs(p4, p5)
    shortest = min(w1_len, w3_len, w5_len)
    valid = w3_len > shortest or (w3_len == shortest and w3_len >= w1_len)

    # Score based on how much W3 exceeds the shorter of W1, W5
    other_min = min(w1_len, w5_len)
    score = min(1.0, w3_len / other_min) if other_min > 0 else 0.5

    return RuleResult(
        name="W3_not_shortest",
        valid=valid,
        score=score,
        message=f"W1={w1_len:.2f}, W3={w3_len:.2f}, W5={w5_len:.2f}",
    )


def rule_w4_no_overlap(p0: Pivot, p1: Pivot, p3: Pivot, p4: Pivot) -> RuleResult:
    """Rule 3: Wave 4 never enters the price territory of Wave 1.

    For bullish: p4 (end of W4) must stay above p1 (end of W1 = start of W2).
    Actually: W4 low must stay above W1 high (p1.price for bullish).
    More precisely: p4.price must not overlap with p0-to-p1 territory.
    """
    w1_direction = p1.price - p0.price

    if w1_direction > 0:
        # Bullish impulse: W4 low (p4.price since kind=-1) must be > W1 peak area
        # The overlap boundary is p1.price (end of W1)
        valid = p4.price > p1.price
        margin = (p4.price - p1.price) / (p3.price - p1.price) if (p3.price - p1.price) != 0 else 0
    else:
        # Bearish impulse: W4 high (p4.price since kind=+1) must be < W1 trough
        valid = p4.price < p1.price
        margin = (p1.price - p4.price) / (p1.price - p3.price) if (p1.price - p3.price) != 0 else 0

    score = min(1.0, max(0.0, margin)) if valid else 0.0

    return RuleResult(
        name="W4_no_overlap",
        valid=valid,
        score=score,
        message=f"W4 end={p4.price:.2f}, W1 end={p1.price:.2f}, valid={valid}",
    )


# ============================================================================
# Hard Rules (corrective ABC)
# ============================================================================


def rule_abc_b_within(pa: Pivot, pb: Pivot, origin: Pivot) -> RuleResult:
    """Rule 4: Wave B does not move beyond the origin of Wave A.

    For bearish correction (A down): B must not go above origin.
    For bullish correction (A up): B must not go below origin.
    """
    a_direction = pa.price - origin.price

    if a_direction < 0:
        # Bearish A: B must stay below origin
        valid = pb.price < origin.price
    else:
        # Bullish A: B must stay above origin
        valid = pb.price > origin.price

    score = 1.0 if valid else 0.0
    return RuleResult(
        name="ABC_B_within",
        valid=valid,
        score=score,
        message=f"B={pb.price:.2f}, origin={origin.price:.2f}",
    )


def rule_abc_c_beyond_a(pa: Pivot, pc: Pivot) -> RuleResult:
    """Rule 5: Wave C must move beyond the end of Wave A.

    C extends further than A in the same direction.
    """
    a_direction = pa.kind  # The end-of-A pivot kind indicates direction

    if a_direction == -1:
        # A ended at a low, C should go lower
        valid = pc.price < pa.price
    else:
        # A ended at a high, C should go higher
        valid = pc.price > pa.price

    score = 1.0 if valid else 0.0
    return RuleResult(
        name="ABC_C_beyond_A",
        valid=valid,
        score=score,
        message=f"A end={pa.price:.2f}, C end={pc.price:.2f}",
    )


# ============================================================================
# Guidelines (soft scoring)
# ============================================================================


def guideline_w2_fib(p0: Pivot, p1: Pivot, p2: Pivot) -> RuleResult:
    """G1: Wave 2 typically retraces 50-78.6% of Wave 1."""
    w1 = _wave_abs(p0, p1)
    w2 = _wave_abs(p1, p2)
    if w1 < 1e-10:
        return RuleResult(name="G1_W2_fib", valid=True, score=0.5)

    ratio = w2 / w1
    # Ideal: 0.500, 0.618, 0.786
    ideal_ratios = [0.500, 0.618, 0.786]
    min_dist = min(abs(ratio - r) for r in ideal_ratios)
    score = max(0.0, 1.0 - min_dist * 5)  # 0.2 deviation = 0 score

    return RuleResult(
        name="G1_W2_fib",
        valid=True,
        score=score,
        message=f"W2/W1={ratio:.3f} (ideal: 0.500-0.786)",
    )


def guideline_w3_extension(p0: Pivot, p1: Pivot, p2: Pivot, p3: Pivot) -> RuleResult:
    """G2: Wave 3 typically extends to 161.8% of Wave 1."""
    w1 = _wave_abs(p0, p1)
    w3 = _wave_abs(p2, p3)
    if w1 < 1e-10:
        return RuleResult(name="G2_W3_ext", valid=True, score=0.5)

    ratio = w3 / w1
    # Ideal: 1.618, 2.618, 3.618
    ideal_ratios = [1.618, 2.618, 3.618]
    min_dist = min(abs(ratio - r) for r in ideal_ratios)
    score = max(0.0, 1.0 - min_dist * 2)

    return RuleResult(
        name="G2_W3_ext",
        valid=True,
        score=score,
        message=f"W3/W1={ratio:.3f} (ideal: 1.618, 2.618)",
    )


def guideline_w4_fib(p2: Pivot, p3: Pivot, p4: Pivot) -> RuleResult:
    """G3: Wave 4 typically retraces 38.2% of Wave 3."""
    w3 = _wave_abs(p2, p3)
    w4 = _wave_abs(p3, p4)
    if w3 < 1e-10:
        return RuleResult(name="G3_W4_fib", valid=True, score=0.5)

    ratio = w4 / w3
    # Ideal: 0.236, 0.382, 0.500
    ideal_ratios = [0.236, 0.382, 0.500]
    min_dist = min(abs(ratio - r) for r in ideal_ratios)
    score = max(0.0, 1.0 - min_dist * 5)

    return RuleResult(
        name="G3_W4_fib",
        valid=True,
        score=score,
        message=f"W4/W3={ratio:.3f} (ideal: 0.382)",
    )


# ============================================================================
# Composite validation
# ============================================================================


def validate_impulse(pivots: list[Pivot]) -> tuple[bool, float, list[RuleResult]]:
    """Validate an impulse wave pattern from a sequence of pivots.

    Expects pivots: [p0, p1, p2, p3, ...] where p0 is the start of W1,
    p1 is end of W1/start of W2, etc.

    Args:
        pivots: List of pivots forming the wave count.

    Returns:
        Tuple of (all_valid, composite_score, list of rule results).
    """
    results: list[RuleResult] = []

    if len(pivots) < 3:
        return True, 0.5, []  # Not enough pivots to validate

    p0, p1, p2 = pivots[0], pivots[1], pivots[2]

    # Rule 1: W2 retrace
    r1 = rule_w2_retrace(p0, p1, p2)
    results.append(r1)

    # Guidelines
    g1 = guideline_w2_fib(p0, p1, p2)
    results.append(g1)

    if len(pivots) >= 4:
        p3 = pivots[3]
        # Rule 2: W3 not shortest (partial)
        r2 = rule_w3_not_shortest(p0, p1, p2, p3, pivots[4] if len(pivots) > 4 else p3)
        results.append(r2)
        # Guideline: W3 extension
        g2 = guideline_w3_extension(p0, p1, p2, p3)
        results.append(g2)

    if len(pivots) >= 5:
        p3, p4 = pivots[3], pivots[4]
        # Rule 3: W4 no overlap
        r3 = rule_w4_no_overlap(p0, p1, p3, p4)
        results.append(r3)
        # Guideline: W4 fib
        g3 = guideline_w4_fib(p2, p3, p4)
        results.append(g3)

    if len(pivots) >= 6:
        p5 = pivots[5]
        # Full Rule 2 with W5
        r2_full = rule_w3_not_shortest(p0, p1, p2, pivots[3], pivots[4], p5)
        results.append(r2_full)

    all_valid = all(r.valid for r in results if not r.name.startswith("G"))
    hard_scores = [r.score for r in results if not r.name.startswith("G")]
    soft_scores = [r.score for r in results if r.name.startswith("G")]

    # Composite: 70% hard rules + 30% guidelines
    hard_avg = sum(hard_scores) / len(hard_scores) if hard_scores else 1.0
    soft_avg = sum(soft_scores) / len(soft_scores) if soft_scores else 0.5
    composite = 0.7 * hard_avg + 0.3 * soft_avg

    return all_valid, composite, results


def validate_corrective(pivots: list[Pivot]) -> tuple[bool, float, list[RuleResult]]:
    """Validate a corrective ABC pattern.

    Expects pivots: [origin, end_A, end_B, end_C].

    Args:
        pivots: List of 3-4 pivots forming the ABC.

    Returns:
        Tuple of (all_valid, composite_score, list of rule results).
    """
    results: list[RuleResult] = []

    if len(pivots) < 3:
        return True, 0.5, []

    origin = pivots[0]
    pa = pivots[1]

    if len(pivots) >= 3:
        pb = pivots[2]
        r4 = rule_abc_b_within(pa, pb, origin)
        results.append(r4)

    if len(pivots) >= 4:
        pc = pivots[3]
        r5 = rule_abc_c_beyond_a(pa, pc)
        results.append(r5)

    all_valid = all(r.valid for r in results)
    scores = [r.score for r in results]
    composite = sum(scores) / len(scores) if scores else 0.5

    return all_valid, composite, results
