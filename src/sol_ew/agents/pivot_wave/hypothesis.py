"""Elliott Wave hypothesis tracker.

Manages the lifecycle of wave hypotheses at each degree:
  active -> invalidated (price violates a hard rule)
  active -> superseded  (a better-fitting hypothesis emerges)
  active -> completed   (full 5-wave or ABC pattern confirmed)
  active -> expired     (too old / too many bars without progress)

Hypothesis IDs are stable: f"{tf}:{anchor_pivot_ts}:{kind}".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sol_ew.agents.pivot_wave.wave_rules import (
    validate_corrective,
    validate_impulse,
)
from sol_ew.core.types import Hypothesis, Pivot

logger = logging.getLogger(__name__)

# Maximum age in bars before a hypothesis expires
MAX_AGE_BARS = 500


def _make_hyp_id(tf: str, anchor_ts: int, kind: str) -> str:
    """Generate a stable hypothesis ID."""
    return f"{tf}:{anchor_ts}:{kind}"


def _direction_from_pivots(pivots: tuple[Pivot, ...]) -> int:
    """Infer direction from pivot sequence: +1 bullish, -1 bearish."""
    if len(pivots) < 2:
        return 0
    return 1 if pivots[1].price > pivots[0].price else -1


@dataclass
class HypothesisTracker:
    """Tracks and manages wave hypotheses for a single timeframe.

    On each new pivot, the tracker:
    1. Checks existing hypotheses for invalidation.
    2. Extends existing hypotheses with the new pivot.
    3. Generates new hypotheses from recent pivot patterns.
    4. Scores and normalizes weights.
    """

    tf: str
    max_active: int = 20
    max_age_bars: int = MAX_AGE_BARS

    def __post_init__(self) -> None:
        self._hypotheses: dict[str, Hypothesis] = {}
        self._bar_count: int = 0

    def on_pivot(self, pivot: Pivot, all_pivots: list[Pivot]) -> list[Hypothesis]:
        """Process a new confirmed pivot.

        Args:
            pivot: The newly confirmed pivot.
            all_pivots: Full pivot history for this timeframe.

        Returns:
            List of newly created or updated hypotheses.
        """
        updated: list[Hypothesis] = []

        # 1. Check invalidation of existing hypotheses
        self._check_invalidations(pivot)

        # 2. Try to extend existing hypotheses
        for hyp_id, hyp in list(self._hypotheses.items()):
            if hyp.status != "active":
                continue
            extended = self._try_extend(hyp, pivot)
            if extended is not None:
                self._hypotheses[hyp_id] = extended
                updated.append(extended)

        # 3. Generate new hypotheses from recent pivots
        new_hyps = self._generate_new(pivot, all_pivots)
        for hyp in new_hyps:
            if hyp.id not in self._hypotheses:
                self._hypotheses[hyp.id] = hyp
                updated.append(hyp)

        # 4. Expire old hypotheses
        self._expire_old()

        # 5. Normalize weights
        self._normalize_weights()

        # 6. Prune if too many
        self._prune_excess()

        return updated

    def on_bar(self, bar_idx: int) -> None:
        """Track bar progression for aging hypotheses."""
        self._bar_count = bar_idx

    def _check_invalidations(self, pivot: Pivot) -> None:
        """Check if new pivot invalidates any existing hypotheses."""
        for hyp_id, hyp in list(self._hypotheses.items()):
            if hyp.status != "active":
                continue

            # Check if price has violated the invalidation level
            if hyp.direction == 1 and pivot.kind == -1:
                # Bullish hyp: invalidated if a new low goes below invalidation
                if pivot.price < hyp.invalidation:
                    hyp.status = "invalidated"
                    logger.debug("Hypothesis %s invalidated at %.2f", hyp_id, pivot.price)
            elif hyp.direction == -1 and pivot.kind == 1 and pivot.price > hyp.invalidation:
                hyp.status = "invalidated"
                logger.debug("Hypothesis %s invalidated at %.2f", hyp_id, pivot.price)

    def _try_extend(self, hyp: Hypothesis, pivot: Pivot) -> Hypothesis | None:
        """Try to extend a hypothesis with a new pivot.

        Returns a new Hypothesis with the pivot appended if it fits,
        or None if the pivot doesn't extend this hypothesis.
        """
        # Check if the new pivot is compatible
        n_pivots = len(hyp.pivots)
        if n_pivots == 0:
            return None

        last = hyp.pivots[-1]
        # Pivots must alternate kind
        if pivot.kind == last.kind:
            return None

        # Build extended pivot sequence
        new_pivots = (*hyp.pivots, pivot)
        pivot_list = list(new_pivots)

        # Validate the extended sequence
        if hyp.kind.startswith("impulse"):
            valid, score, _ = validate_impulse(pivot_list)
        elif hyp.kind.startswith("abc"):
            valid, score, _ = validate_corrective(pivot_list)
        else:
            return None

        if not valid:
            return None

        # Determine new kind label based on pivot count
        kind = self._classify_pattern(hyp.kind, len(new_pivots))

        # Update invalidation level
        invalidation = self._compute_invalidation(new_pivots, hyp.direction)

        # Update targets
        targets = self._compute_targets(new_pivots, hyp.direction)

        # Check if pattern is complete
        status = "active"
        if (hyp.kind.startswith("impulse") and len(new_pivots) >= 6) or (hyp.kind.startswith("abc") and len(new_pivots) >= 4):
            status = "completed"

        return Hypothesis(
            id=hyp.id,
            tf=hyp.tf,
            kind=kind,
            direction=hyp.direction,
            pivots=new_pivots,
            invalidation=invalidation,
            targets=targets,
            rule_score=score,
            born_ts=hyp.born_ts,
            status=status,
        )

    def _generate_new(self, pivot: Pivot, all_pivots: list[Pivot]) -> list[Hypothesis]:
        """Generate new hypotheses from the most recent pivots."""
        new: list[Hypothesis] = []

        if len(all_pivots) < 2:
            return new

        # Look for W1-W2 patterns (start of impulse)
        # Need at least 3 pivots: p0, p1, p2
        if len(all_pivots) >= 3:
            p0, p1, p2 = all_pivots[-3], all_pivots[-2], all_pivots[-1]
            # Check if this forms a valid W1-W2
            valid, score, _ = validate_impulse([p0, p1, p2])
            if valid and score > 0.3:
                direction = _direction_from_pivots((p0, p1))
                hyp_id = _make_hyp_id(self.tf, p0.ts, f"impulse_W1W2_{direction}")
                if hyp_id not in self._hypotheses:
                    invalidation = self._compute_invalidation((p0, p1, p2), direction)
                    targets = self._compute_targets((p0, p1, p2), direction)
                    hyp = Hypothesis(
                        id=hyp_id,
                        tf=self.tf,
                        kind="impulse_W1W2",
                        direction=direction,
                        pivots=(p0, p1, p2),
                        invalidation=invalidation,
                        targets=targets,
                        rule_score=score,
                        born_ts=pivot.confirm_ts,
                        status="active",
                    )
                    new.append(hyp)

        # Look for ABC start
        if len(all_pivots) >= 2:
            origin, pa = all_pivots[-2], all_pivots[-1]
            direction = -1 if pa.price < origin.price else 1
            hyp_id = _make_hyp_id(self.tf, origin.ts, f"abc_A_{direction}")
            if hyp_id not in self._hypotheses:
                hyp = Hypothesis(
                    id=hyp_id,
                    tf=self.tf,
                    kind="abc_A",
                    direction=direction,
                    pivots=(origin, pa),
                    invalidation=origin.price,
                    targets={},
                    rule_score=0.5,
                    born_ts=pivot.confirm_ts,
                    status="active",
                )
                new.append(hyp)

        return new

    @staticmethod
    def _classify_pattern(base_kind: str, n_pivots: int) -> str:
        """Update the kind label based on pivot count."""
        if base_kind.startswith("impulse"):
            stages = {2: "impulse_W1", 3: "impulse_W1W2", 4: "impulse_W3",
                      5: "impulse_W4", 6: "impulse_W5"}
            return stages.get(n_pivots, f"impulse_W{n_pivots}")
        if base_kind.startswith("abc"):
            stages = {2: "abc_A", 3: "abc_B", 4: "abc_C"}
            return stages.get(n_pivots, f"abc_{n_pivots}")
        return base_kind

    @staticmethod
    def _compute_invalidation(pivots: tuple[Pivot, ...], direction: int) -> float:
        """Compute the invalidation price for a hypothesis."""
        if not pivots:
            return 0.0
        if direction == 1:
            # Bullish: invalidated below the start
            return pivots[0].price
        # Bearish: invalidated above the start
        return pivots[0].price

    @staticmethod
    def _compute_targets(pivots: tuple[Pivot, ...], direction: int) -> dict[float, float]:
        """Compute Fibonacci target prices."""
        targets: dict[float, float] = {}
        if len(pivots) < 2:
            return targets

        # Wave 1 length for projection
        w1_len = abs(pivots[1].price - pivots[0].price)
        if w1_len < 1e-10:
            return targets

        base = pivots[-1].price
        for ratio in [1.0, 1.272, 1.618, 2.0, 2.618]:
            if direction == 1:
                targets[ratio] = base + w1_len * ratio
            else:
                targets[ratio] = base - w1_len * ratio

        return targets

    def _expire_old(self) -> None:
        """Expire hypotheses that are too old."""
        for hyp in self._hypotheses.values():
            if hyp.status != "active":
                continue
            age = self._bar_count - (hyp.born_ts // 1000)  # Rough age estimate
            if age > self.max_age_bars * 1000:
                hyp.status = "expired"

    def _normalize_weights(self) -> None:
        """Normalize weights among active hypotheses."""
        active = [h for h in self._hypotheses.values() if h.status == "active"]
        if not active:
            return
        total_score = sum(h.rule_score for h in active)
        if total_score <= 0:
            return
        for h in active:
            h.weight = h.rule_score / total_score

    def _prune_excess(self) -> None:
        """Remove lowest-scoring hypotheses if over the limit."""
        active = [h for h in self._hypotheses.values() if h.status == "active"]
        if len(active) <= self.max_active:
            return
        # Sort by score, remove lowest
        active.sort(key=lambda h: h.rule_score, reverse=True)
        for h in active[self.max_active:]:
            h.status = "superseded"

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def active_hypotheses(self) -> list[Hypothesis]:
        """All currently active hypotheses."""
        return [h for h in self._hypotheses.values() if h.status == "active"]

    @property
    def all_hypotheses(self) -> list[Hypothesis]:
        """All hypotheses (including inactive)."""
        return list(self._hypotheses.values())

    def get(self, hyp_id: str) -> Hypothesis | None:
        """Get a hypothesis by ID."""
        return self._hypotheses.get(hyp_id)

    def snapshot(self) -> dict[str, object]:
        """Capture state for replayability."""
        return {
            "tf": self.tf,
            "bar_count": self._bar_count,
            "n_hypotheses": len(self._hypotheses),
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore from snapshot."""
        self._bar_count = int(state["bar_count"])  # type: ignore[arg-type]
