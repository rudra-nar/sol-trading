"""Fibonacci grid and confluence zone computation.

Computes Fibonacci retracement and extension levels from wave pivots,
then clusters overlapping levels across multiple degrees into confluence
zones ranked by strength.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sol_ew.core.types import Hypothesis, Pivot, Zone

logger = logging.getLogger(__name__)

# Standard Fibonacci ratios
RETRACE_RATIOS = (0.236, 0.382, 0.500, 0.618, 0.786)
EXTENSION_RATIOS = (1.000, 1.272, 1.618, 2.000, 2.618, 3.618)


@dataclass(frozen=True)
class FibLevel:
    """A single Fibonacci level with source metadata."""

    price: float
    ratio: float
    kind: str  # "retrace" or "extension"
    source_hyp_id: str
    tf: str
    weight: float = 1.0  # hypothesis weight


def fib_retracements(p_start: Pivot, p_end: Pivot, hyp_id: str, tf: str, weight: float = 1.0) -> list[FibLevel]:
    """Compute Fibonacci retracement levels between two pivots.

    Args:
        p_start: Start of the wave move (e.g., Wave 1 start).
        p_end: End of the wave move (e.g., Wave 1 end).
        hyp_id: Source hypothesis ID.
        tf: Timeframe.
        weight: Hypothesis weight.

    Returns:
        List of FibLevel objects for each retracement ratio.
    """
    diff = p_end.price - p_start.price
    levels = []
    for ratio in RETRACE_RATIOS:
        price = p_end.price - diff * ratio
        levels.append(FibLevel(
            price=price, ratio=ratio, kind="retrace",
            source_hyp_id=hyp_id, tf=tf, weight=weight,
        ))
    return levels


def fib_extensions(p_start: Pivot, p_end: Pivot, p_base: Pivot, hyp_id: str, tf: str, weight: float = 1.0) -> list[FibLevel]:
    """Compute Fibonacci extension levels projected from a base pivot.

    Args:
        p_start: Start of the reference wave (e.g., Wave 1 start).
        p_end: End of the reference wave (e.g., Wave 1 end).
        p_base: Base pivot to project from (e.g., Wave 2 end).
        hyp_id: Source hypothesis ID.
        tf: Timeframe.
        weight: Hypothesis weight.

    Returns:
        List of FibLevel objects for each extension ratio.
    """
    diff = abs(p_end.price - p_start.price)
    direction = 1 if p_end.price > p_start.price else -1
    levels = []
    for ratio in EXTENSION_RATIOS:
        price = p_base.price + direction * diff * ratio
        levels.append(FibLevel(
            price=price, ratio=ratio, kind="extension",
            source_hyp_id=hyp_id, tf=tf, weight=weight,
        ))
    return levels


def compute_fib_grid(hypothesis: Hypothesis) -> list[FibLevel]:
    """Compute all Fibonacci levels for a single hypothesis.

    Generates retracements and extensions depending on how many
    pivots the hypothesis has.

    Args:
        hypothesis: An active wave hypothesis.

    Returns:
        List of all FibLevel objects for this hypothesis.
    """
    pivots = hypothesis.pivots
    levels: list[FibLevel] = []
    hyp_id = hypothesis.id
    tf = hypothesis.tf
    w = hypothesis.weight

    if len(pivots) < 2:
        return levels

    # Retracement of the latest completed wave
    levels.extend(fib_retracements(pivots[-2], pivots[-1], hyp_id, tf, w))

    # If we have W1 (p0, p1) and W2 (p1, p2): project W3 from p2
    if len(pivots) >= 3:
        levels.extend(fib_extensions(pivots[0], pivots[1], pivots[2], hyp_id, tf, w))

    # If we have up to W3 end: project W5 targets
    if len(pivots) >= 5:
        levels.extend(fib_extensions(pivots[2], pivots[3], pivots[4], hyp_id, tf, w))

    return levels


def cluster_into_zones(
    all_levels: list[FibLevel],
    cluster_pct: float = 0.5,
    min_sources: int = 2,
) -> list[Zone]:
    """Cluster nearby Fibonacci levels into confluence zones.

    Levels within `cluster_pct`% of each other are grouped. A zone is
    only emitted if it has contributions from at least `min_sources`
    different hypotheses.

    Args:
        all_levels: All FibLevel objects from all hypotheses/degrees.
        cluster_pct: Percentage threshold for clustering (0.5 = 0.5%).
        min_sources: Minimum number of distinct sources to form a zone.

    Returns:
        List of Zone objects sorted by strength (descending).
    """
    if not all_levels:
        return []

    # Sort by price
    sorted_levels = sorted(all_levels, key=lambda lv: lv.price)

    # Greedy clustering
    clusters: list[list[FibLevel]] = []
    current_cluster: list[FibLevel] = [sorted_levels[0]]

    for lv in sorted_levels[1:]:
        cluster_center = sum(l.price for l in current_cluster) / len(current_cluster)
        threshold = cluster_center * cluster_pct / 100.0
        if abs(lv.price - cluster_center) <= threshold:
            current_cluster.append(lv)
        else:
            clusters.append(current_cluster)
            current_cluster = [lv]
    clusters.append(current_cluster)

    # Convert clusters to zones
    zones: list[Zone] = []
    for cluster in clusters:
        # Unique sources
        unique_sources = set()
        unique_tfs = set()
        for lv in cluster:
            unique_sources.add(lv.source_hyp_id)
            unique_tfs.add(lv.tf)

        if len(unique_sources) < min_sources:
            continue

        prices = [lv.price for lv in cluster]
        strength = sum(lv.weight for lv in cluster)
        source_labels = tuple(
            f"{lv.source_hyp_id}:{lv.kind}:{lv.ratio}" for lv in cluster
        )

        zones.append(Zone(
            lo=min(prices),
            hi=max(prices),
            strength=strength,
            n_degrees=len(unique_tfs),
            sources=source_labels,
        ))

    # Sort by strength descending
    zones.sort(key=lambda z: -z.strength)
    return zones


def compute_confluence_zones(
    hypotheses: list[Hypothesis],
    cluster_pct: float = 0.5,
    min_sources: int = 2,
) -> list[Zone]:
    """Compute confluence zones from multiple hypotheses across degrees.

    This is the main entry point: takes all active hypotheses, computes
    their Fibonacci grids, and clusters into zones.

    Args:
        hypotheses: Active hypotheses from all timeframes/degrees.
        cluster_pct: Clustering threshold percentage.
        min_sources: Minimum distinct sources per zone.

    Returns:
        List of Zone objects sorted by strength.
    """
    all_levels: list[FibLevel] = []
    for hyp in hypotheses:
        all_levels.extend(compute_fib_grid(hyp))

    return cluster_into_zones(all_levels, cluster_pct, min_sources)
