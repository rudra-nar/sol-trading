"""Bar resampler -- aggregates lower-timeframe bars into higher-timeframe bars.

Resamples 1m -> 5m -> 15m -> 1h -> 4h -> 1D. A higher-timeframe bar is
emitted ONLY after its close_ts (invariant 5: a 1h bar is available to the
5m logic only after the 1h close_ts). The forming HTF bar is never exposed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sol_ew.core.types import Bar
from sol_ew.data.validate import interval_to_ms

logger = logging.getLogger(__name__)

# Timeframe hierarchy for resampling
TF_HIERARCHY = ["1m", "5m", "15m", "1h", "4h", "1D"]

# How many of the lower TF bars make one higher TF bar
TF_MULTIPLES = {
    ("1m", "5m"): 5,
    ("1m", "15m"): 15,
    ("1m", "1h"): 60,
    ("1m", "4h"): 240,
    ("1m", "1D"): 1440,
    ("5m", "15m"): 3,
    ("5m", "1h"): 12,
    ("5m", "4h"): 48,
    ("5m", "1D"): 288,
}


def _align_ts(ts: int, interval_ms: int) -> int:
    """Align a timestamp to the start of its interval period.

    For daily bars, aligns to UTC midnight.

    Args:
        ts: Timestamp in UTC milliseconds.
        interval_ms: Interval duration in milliseconds.

    Returns:
        Aligned timestamp.
    """
    return (ts // interval_ms) * interval_ms


@dataclass
class BarResampler:
    """Aggregates lower-timeframe bars into a higher-timeframe bar.

    Accumulates constituent bars and emits the HTF bar only when the
    period closes. The forming bar is never exposed (invariant 5).

    Attributes:
        source_tf: Input bar timeframe (e.g. "1m").
        target_tf: Output bar timeframe (e.g. "5m").
    """

    source_tf: str
    target_tf: str

    def __post_init__(self) -> None:
        self._target_interval_ms = interval_to_ms(self.target_tf)
        self._current_period_start: int = 0
        self._o: float = 0.0
        self._h: float = 0.0
        self._l: float = 0.0
        self._c: float = 0.0
        self._volume: float = 0.0
        self._taker_buy_volume: float = 0.0
        self._n_trades: int = 0
        self._bar_count: int = 0
        self._degraded: bool = False
        self._bars_emitted: int = 0

    def on_bar(self, bar: Bar) -> Bar | None:
        """Process an incoming lower-timeframe bar.

        Accumulates OHLCV data and emits a higher-timeframe bar when
        the period closes.

        Args:
            bar: A closed lower-timeframe bar.

        Returns:
            A completed higher-timeframe Bar, or None if the period
            is still forming.
        """
        period_start = _align_ts(bar.open_ts, self._target_interval_ms)
        result: Bar | None = None

        # If we've moved to a new period, emit the completed previous one
        if self._bar_count > 0 and period_start != self._current_period_start:
            result = self._emit()

        # Start new period or accumulate
        if self._bar_count == 0 or period_start != self._current_period_start:
            self._current_period_start = period_start
            self._o = bar.o
            self._h = bar.h
            self._l = bar.l
            self._c = bar.c
            self._volume = bar.volume
            self._taker_buy_volume = bar.taker_buy_volume
            self._n_trades = bar.n_trades
            self._degraded = bar.degraded
            self._bar_count = 1
        else:
            # Accumulate within current period
            self._h = max(self._h, bar.h)
            self._l = min(self._l, bar.l)
            self._c = bar.c
            self._volume += bar.volume
            self._taker_buy_volume += bar.taker_buy_volume
            self._n_trades += bar.n_trades
            self._degraded = self._degraded or bar.degraded
            self._bar_count += 1

        return result

    def flush(self) -> Bar | None:
        """Flush any remaining accumulated data as a final bar.

        Call at end of replay to emit the last incomplete period.

        Returns:
            The final bar if data was accumulated, None otherwise.
        """
        if self._bar_count > 0:
            return self._emit()
        return None

    def _emit(self) -> Bar:
        """Emit the accumulated bar and reset state."""
        close_ts = self._current_period_start + self._target_interval_ms - 1
        bar = Bar(
            tf=self.target_tf,
            open_ts=self._current_period_start,
            close_ts=close_ts,
            o=self._o,
            h=self._h,
            l=self._l,
            c=self._c,
            volume=self._volume,
            taker_buy_volume=self._taker_buy_volume,
            n_trades=self._n_trades,
            degraded=self._degraded,
        )
        self._bar_count = 0
        self._bars_emitted += 1
        return bar

    @property
    def bars_emitted(self) -> int:
        """Total number of bars emitted."""
        return self._bars_emitted

    @property
    def has_pending(self) -> bool:
        """Whether there's an incomplete bar being accumulated."""
        return self._bar_count > 0

    def snapshot(self) -> dict[str, object]:
        """Capture state for replayability."""
        return {
            "source_tf": self.source_tf,
            "target_tf": self.target_tf,
            "current_period_start": self._current_period_start,
            "o": self._o, "h": self._h, "l": self._l, "c": self._c,
            "volume": self._volume,
            "taker_buy_volume": self._taker_buy_volume,
            "n_trades": self._n_trades,
            "bar_count": self._bar_count,
            "degraded": self._degraded,
            "bars_emitted": self._bars_emitted,
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore from snapshot."""
        self._current_period_start = int(state["current_period_start"])  # type: ignore[arg-type]
        self._o = float(state["o"])  # type: ignore[arg-type]
        self._h = float(state["h"])  # type: ignore[arg-type]
        self._l = float(state["l"])  # type: ignore[arg-type]
        self._c = float(state["c"])  # type: ignore[arg-type]
        self._volume = float(state["volume"])  # type: ignore[arg-type]
        self._taker_buy_volume = float(state["taker_buy_volume"])  # type: ignore[arg-type]
        self._n_trades = int(state["n_trades"])  # type: ignore[arg-type]
        self._bar_count = int(state["bar_count"])  # type: ignore[arg-type]
        self._degraded = bool(state["degraded"])
        self._bars_emitted = int(state["bars_emitted"])  # type: ignore[arg-type]


class MultiTimeframeResampler:
    """Manages multiple resamplers to produce all required timeframes.

    Feeds 1m bars through a cascade of resamplers to produce 5m, 15m,
    1h, 4h, and 1D bars. All resamplers emit bars only on close.
    """

    def __init__(self, source_tf: str = "1m", target_tfs: list[str] | None = None) -> None:
        """Initialize with source timeframe and target timeframes.

        Args:
            source_tf: Input bar timeframe.
            target_tfs: List of target timeframes to produce.
                Defaults to ["5m", "15m", "1h", "4h", "1D"].
        """
        if target_tfs is None:
            target_tfs = ["5m", "15m", "1h", "4h", "1D"]
        self._source_tf = source_tf
        self._resamplers: dict[str, BarResampler] = {}
        for tf in target_tfs:
            self._resamplers[tf] = BarResampler(source_tf=source_tf, target_tf=tf)

    def on_bar(self, bar: Bar) -> dict[str, Bar]:
        """Process an incoming source bar through all resamplers.

        Args:
            bar: A closed source-timeframe bar.

        Returns:
            Dict mapping timeframe to completed Bar for any timeframes
            that completed a period on this bar. Empty dict if none completed.
        """
        completed: dict[str, Bar] = {}
        for tf, resampler in self._resamplers.items():
            result = resampler.on_bar(bar)
            if result is not None:
                completed[tf] = result
        return completed

    def flush(self) -> dict[str, Bar]:
        """Flush all resamplers at end of replay.

        Returns:
            Dict of final bars for each timeframe with pending data.
        """
        completed: dict[str, Bar] = {}
        for tf, resampler in self._resamplers.items():
            result = resampler.flush()
            if result is not None:
                completed[tf] = result
        return completed

    def snapshot(self) -> dict[str, object]:
        """Capture state for replayability."""
        return {
            tf: resampler.snapshot()
            for tf, resampler in self._resamplers.items()
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore from snapshot."""
        for tf, resampler in self._resamplers.items():
            if tf in state:
                resampler.restore(state[tf])  # type: ignore[arg-type]
