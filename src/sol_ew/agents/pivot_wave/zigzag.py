"""ATR-based ZigZag pivot detector.

Detects swing highs and lows using an ATR-multiplier threshold. Once a pivot
is emitted it NEVER changes or disappears (no repaint). All downstream logic
must use confirm_idx/confirm_ts, not the pivot bar itself.

Invariants:
- A pivot is confirmed only after the threshold reversal is observed.
- confirm_ts is the close_ts of the bar that confirmed the pivot.
- Pivots alternate: HIGH, LOW, HIGH, LOW, ...
- The sequence is append-only.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sol_ew.core.types import Bar, Pivot

logger = logging.getLogger(__name__)


def compute_atr(bars: list[Bar], period: int) -> float:
    """Compute Average True Range over the last `period` bars.

    True Range = max(H-L, |H-prev_close|, |L-prev_close|)

    Args:
        bars: List of closed bars (most recent last).
        period: ATR lookback period.

    Returns:
        ATR value. Returns 0 if insufficient bars.
    """
    if len(bars) < 2:
        return 0.0

    trs: list[float] = []
    for i in range(1, len(bars)):
        prev_c = bars[i - 1].c
        h = bars[i].h
        l = bars[i].l
        tr = max(h - l, abs(h - prev_c), abs(l - prev_c))
        trs.append(tr)

    if not trs:
        return 0.0

    # Use the last `period` true ranges
    recent = trs[-period:] if len(trs) >= period else trs
    return sum(recent) / len(recent)


@dataclass
class ZigZag:
    """ATR-based ZigZag pivot detector with no-repaint guarantee.

    The algorithm works as follows:
    1. Maintain a candidate extreme (high or low).
    2. As bars arrive, update the candidate if price extends further.
    3. When price reverses by ATR * mult from the candidate, confirm
       the pivot and switch direction.
    4. The pivot's confirm_idx/confirm_ts is set to the CONFIRMING bar,
       not the extreme bar.

    Attributes:
        tf: Timeframe this zigzag operates on.
        atr_period: ATR lookback period.
        mult: ATR multiplier for reversal threshold.
    """

    tf: str
    atr_period: int = 14
    mult: float = 3.0

    def __post_init__(self) -> None:
        self._bars: list[Bar] = []
        self._bar_idx: int = 0
        self._pivots: list[Pivot] = []

        # Candidate state
        self._looking_for: int = 0  # 0=undecided, +1=looking for high, -1=looking for low
        self._candidate_price: float = 0.0
        self._candidate_idx: int = 0
        self._candidate_ts: int = 0
        self._warmed_up: bool = False

    def on_bar(self, bar: Bar) -> Pivot | None:
        """Process a closed bar and potentially emit a confirmed pivot.

        Args:
            bar: A closed bar at this zigzag's timeframe.

        Returns:
            A confirmed Pivot if one was just confirmed, None otherwise.
        """
        self._bars.append(bar)
        current_idx = self._bar_idx
        self._bar_idx += 1

        # Need minimum bars for ATR
        if len(self._bars) < self.atr_period + 1:
            return None

        atr = compute_atr(self._bars, self.atr_period)
        if atr <= 0:
            return None

        threshold = atr * self.mult

        # Initialize on first valid bar
        if self._looking_for == 0:
            self._looking_for = 1  # Start by looking for a high
            self._candidate_price = bar.h
            self._candidate_idx = current_idx
            self._candidate_ts = bar.open_ts
            self._warmed_up = True
            return None

        result: Pivot | None = None

        if self._looking_for == 1:  # Looking for swing HIGH
            if bar.h > self._candidate_price:
                # Extend the candidate high
                self._candidate_price = bar.h
                self._candidate_idx = current_idx
                self._candidate_ts = bar.open_ts
            elif self._candidate_price - bar.l >= threshold:
                # Confirmed HIGH pivot: price dropped enough from candidate
                result = Pivot(
                    tf=self.tf,
                    idx=self._candidate_idx,
                    ts=self._candidate_ts,
                    confirm_idx=current_idx,
                    confirm_ts=bar.close_ts,
                    price=self._candidate_price,
                    kind=1,  # swing high
                )
                self._pivots.append(result)
                # Switch to looking for LOW, starting with current bar's low
                self._looking_for = -1
                self._candidate_price = bar.l
                self._candidate_idx = current_idx
                self._candidate_ts = bar.open_ts

        elif self._looking_for == -1:  # Looking for swing LOW
            if bar.l < self._candidate_price:
                # Extend the candidate low
                self._candidate_price = bar.l
                self._candidate_idx = current_idx
                self._candidate_ts = bar.open_ts
            elif bar.h - self._candidate_price >= threshold:
                # Confirmed LOW pivot: price rose enough from candidate
                result = Pivot(
                    tf=self.tf,
                    idx=self._candidate_idx,
                    ts=self._candidate_ts,
                    confirm_idx=current_idx,
                    confirm_ts=bar.close_ts,
                    price=self._candidate_price,
                    kind=-1,  # swing low
                )
                self._pivots.append(result)
                # Switch to looking for HIGH
                self._looking_for = 1
                self._candidate_price = bar.h
                self._candidate_idx = current_idx
                self._candidate_ts = bar.open_ts

        return result

    @property
    def pivots(self) -> list[Pivot]:
        """All confirmed pivots (append-only)."""
        return list(self._pivots)

    @property
    def last_pivot(self) -> Pivot | None:
        """Most recently confirmed pivot."""
        return self._pivots[-1] if self._pivots else None

    @property
    def warmed_up(self) -> bool:
        """Whether enough bars have been processed for ATR calculation."""
        return self._warmed_up

    @property
    def bar_count(self) -> int:
        """Number of bars processed."""
        return self._bar_idx

    def snapshot(self) -> dict[str, object]:
        """Capture state for replayability."""
        return {
            "tf": self.tf,
            "atr_period": self.atr_period,
            "mult": self.mult,
            "bar_idx": self._bar_idx,
            "looking_for": self._looking_for,
            "candidate_price": self._candidate_price,
            "candidate_idx": self._candidate_idx,
            "candidate_ts": self._candidate_ts,
            "warmed_up": self._warmed_up,
            "n_pivots": len(self._pivots),
            # Store bars for ATR continuity
            "recent_bars": [
                {"o": b.o, "h": b.h, "l": b.l, "c": b.c,
                 "open_ts": b.open_ts, "close_ts": b.close_ts,
                 "volume": b.volume, "taker_buy_volume": b.taker_buy_volume,
                 "n_trades": b.n_trades, "degraded": b.degraded}
                for b in self._bars[-self.atr_period * 2:]
            ],
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore from snapshot."""
        self._bar_idx = int(state["bar_idx"])  # type: ignore[arg-type]
        self._looking_for = int(state["looking_for"])  # type: ignore[arg-type]
        self._candidate_price = float(state["candidate_price"])  # type: ignore[arg-type]
        self._candidate_idx = int(state["candidate_idx"])  # type: ignore[arg-type]
        self._candidate_ts = int(state["candidate_ts"])  # type: ignore[arg-type]
        self._warmed_up = bool(state["warmed_up"])
        # Restore recent bars for ATR
        self._bars = [
            Bar(tf=self.tf, **b)  # type: ignore[arg-type]
            for b in state.get("recent_bars", [])  # type: ignore[union-attr]
        ]
