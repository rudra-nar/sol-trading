"""Bar builder -- constructs Bar objects from kline events or raw trades.

Bars are emitted ONLY on close (invariant 1: no agent may read a bar before
its close_ts). Bars built over a data gap are flagged degraded=True.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sol_ew.core.types import Bar
from sol_ew.data.validate import interval_to_ms

logger = logging.getLogger(__name__)


@dataclass
class BarBuilder:
    """Builds Bar objects from incoming kline events.

    Processes closed kline events and emits fully formed Bar objects.
    Tracks gap state to flag degraded bars.

    Invariants enforced:
    - A bar is only emitted after its close_ts (never a forming bar).
    - Bars spanning a data gap are flagged degraded=True.
    """

    tf: str
    _interval_ms: int = 0
    _last_close_ts: int = 0
    _gap_active: bool = False
    _bars_emitted: int = 0

    def __post_init__(self) -> None:
        self._interval_ms = interval_to_ms(self.tf)

    def on_kline(
        self,
        open_ts: int,
        close_ts: int,
        o: float,
        h: float,
        l: float,
        c: float,
        volume: float,
        taker_buy_volume: float,
        n_trades: int,
        is_closed: bool,
    ) -> Bar | None:
        """Process a kline event and return a Bar if the kline is closed.

        Args:
            open_ts: Bar open time (UTC ms).
            close_ts: Bar close time (UTC ms).
            o, h, l, c: OHLC prices.
            volume: Total volume.
            taker_buy_volume: Taker buy volume.
            n_trades: Number of trades in the bar.
            is_closed: Whether this kline is final/closed.

        Returns:
            A Bar if the kline is closed, None otherwise.
        """
        if not is_closed:
            return None

        # Detect gaps: if the gap between this bar and the last is > 1.5x interval
        degraded = False
        if self._last_close_ts > 0:
            expected_open = self._last_close_ts + 1  # next ms after last close
            gap = open_ts - expected_open
            if gap > self._interval_ms * 0.5:
                logger.warning(
                    "Gap detected in %s bars: %d ms gap at ts=%d",
                    self.tf, gap, open_ts,
                )
                degraded = True
                self._gap_active = True
            else:
                self._gap_active = False

        self._last_close_ts = close_ts
        self._bars_emitted += 1

        return Bar(
            tf=self.tf,
            open_ts=open_ts,
            close_ts=close_ts,
            o=o, h=h, l=l, c=c,
            volume=volume,
            taker_buy_volume=taker_buy_volume,
            n_trades=n_trades,
            degraded=degraded,
        )

    @property
    def bars_emitted(self) -> int:
        """Total number of bars emitted."""
        return self._bars_emitted

    def snapshot(self) -> dict[str, object]:
        """Capture state for replayability."""
        return {
            "tf": self.tf,
            "last_close_ts": self._last_close_ts,
            "gap_active": self._gap_active,
            "bars_emitted": self._bars_emitted,
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore state from snapshot."""
        self._last_close_ts = int(state["last_close_ts"])  # type: ignore[arg-type]
        self._gap_active = bool(state["gap_active"])
        self._bars_emitted = int(state["bars_emitted"])  # type: ignore[arg-type]


def bar_from_kline_event(event_data: dict[str, object], tf: str = "1m") -> Bar | None:
    """Create a Bar directly from a Binance kline websocket event.

    Args:
        event_data: The raw event data dict containing 'k' kline object.
        tf: Timeframe label for the bar.

    Returns:
        A Bar if the kline is closed, None otherwise.
    """
    k = event_data.get("k")
    if k is None:
        return None

    is_closed = k.get("x", False)  # type: ignore[union-attr]
    if not is_closed:
        return None

    return Bar(
        tf=tf,
        open_ts=int(k["t"]),  # type: ignore[index,arg-type]
        close_ts=int(k["T"]),  # type: ignore[index,arg-type]
        o=float(k["o"]),  # type: ignore[index,arg-type]
        h=float(k["h"]),  # type: ignore[index,arg-type]
        l=float(k["l"]),  # type: ignore[index,arg-type]
        c=float(k["c"]),  # type: ignore[index,arg-type]
        volume=float(k["v"]),  # type: ignore[index,arg-type]
        taker_buy_volume=float(k.get("V", 0)),  # type: ignore[union-attr]
        n_trades=int(k.get("n", 0)),  # type: ignore[union-attr]
    )
