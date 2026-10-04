"""Clock — simulation-aware time provider.

Provides a consistent time source for both live and replay modes.
All timestamps are UTC integer milliseconds.
"""

from __future__ import annotations


class Clock:
    """Provides the current time for agents, supporting both live and replay modes.

    In replay mode, time advances as bars are replayed.
    In live mode, uses system clock.
    """

    def __init__(self, mode: str = "replay") -> None:
        """Initialize the clock.

        Args:
            mode: "replay" or "live"
        """
        if mode not in ("replay", "live"):
            raise ValueError(f"Invalid clock mode: {mode!r}. Must be 'replay' or 'live'.")
        self._mode = mode
        self._current_ts: int = 0

    @property
    def mode(self) -> str:
        """Current clock mode."""
        return self._mode

    @property
    def now(self) -> int:
        """Current time in UTC milliseconds.

        In replay mode, returns the last time set via advance().
        In live mode, returns system time.
        """
        if self._mode == "live":
            import time

            return int(time.time() * 1000)
        return self._current_ts

    def advance(self, ts: int) -> None:
        """Advance replay clock to the given timestamp.

        Only valid in replay mode. Time must be monotonically non-decreasing.

        Args:
            ts: New current time in UTC milliseconds.

        Raises:
            RuntimeError: If called in live mode.
            ValueError: If ts would move time backwards.
        """
        if self._mode == "live":
            raise RuntimeError("Cannot advance clock in live mode.")
        if ts < self._current_ts:
            raise ValueError(
                f"Clock cannot go backwards: current={self._current_ts}, requested={ts}"
            )
        self._current_ts = ts

    def snapshot(self) -> dict[str, object]:
        """Capture current state for replayability."""
        return {"mode": self._mode, "current_ts": self._current_ts}

    def restore(self, state: dict[str, object]) -> None:
        """Restore state from a snapshot."""
        self._mode = str(state["mode"])
        self._current_ts = int(state["current_ts"])  # type: ignore[arg-type]
