"""Event replayer -- reads stored events and emits them through the same handler
interface the live feed uses.

Events are emitted in recv_ts order (live-equivalent). This ensures that
live trading and backtest call the same agent entry points with the same
event types (invariant 9).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import pyarrow.parquet as pq

from sol_ew.data.recorder import RawEvent

logger = logging.getLogger(__name__)


class EventHandler(Protocol):
    """Protocol for handling replayed events.

    Same interface used by the live feed, ensuring identical code paths.
    """

    def on_event(self, event: RawEvent) -> None:
        """Handle a single event."""
        ...


class EventReplayer:
    """Replays stored events in recv_ts order through a handler.

    Reads from the Parquet partitions written by the recorder and emits
    events chronologically. Supports filtering by stream type and time range.
    """

    def __init__(
        self,
        data_dir: Path,
        handler: EventHandler | Callable[[RawEvent], None],
        streams: list[str] | None = None,
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> None:
        """Initialize the replayer.

        Args:
            data_dir: Root data directory containing recorded/ subdirectory.
            handler: Event handler (protocol or callable).
            streams: If provided, only replay these stream types. None = all.
            start_ts: If provided, skip events before this timestamp (UTC ms).
            end_ts: If provided, skip events after this timestamp (UTC ms).
        """
        self._data_dir = data_dir
        self._handler = handler
        self._streams = streams
        self._start_ts = start_ts
        self._end_ts = end_ts

    def replay(self) -> int:
        """Replay all events in recv_ts order.

        Returns:
            Total number of events replayed.
        """
        recorded_dir = self._data_dir / "recorded"
        if not recorded_dir.exists():
            logger.warning("No recorded data found at %s", recorded_dir)
            return 0

        # Collect all events from all streams
        all_events: list[RawEvent] = []

        stream_dirs = [d for d in recorded_dir.iterdir() if d.is_dir()]
        for stream_dir in stream_dirs:
            stream_name = stream_dir.name
            if self._streams and stream_name not in self._streams:
                continue

            events = self._load_stream_events(stream_dir, stream_name)
            all_events.extend(events)

        # Sort by recv_ts for live-equivalent ordering
        all_events.sort(key=lambda e: e.recv_ts)

        # Emit events
        count = 0
        for event in all_events:
            if self._start_ts is not None and event.recv_ts < self._start_ts:
                continue
            if self._end_ts is not None and event.recv_ts > self._end_ts:
                continue

            if callable(self._handler) and not hasattr(self._handler, "on_event"):
                self._handler(event)
            else:
                self._handler.on_event(event)  # type: ignore[union-attr]
            count += 1

        logger.info("Replayed %d events", count)
        return count

    def _load_stream_events(self, stream_dir: Path, stream_name: str) -> list[RawEvent]:
        """Load events from a stream's Parquet partitions.

        Args:
            stream_dir: Directory containing Hive-partitioned Parquet files.
            stream_name: Name of the stream.

        Returns:
            List of RawEvent objects.
        """
        parquet_files = sorted(stream_dir.rglob("*.parquet"))
        events: list[RawEvent] = []

        for pq_file in parquet_files:
            try:
                table = pq.read_table(pq_file)
                df = table.to_pandas()
            except Exception as e:
                logger.warning("Error reading %s: %s", pq_file, e)
                continue

            for _, row in df.iterrows():
                data = row.get("data", "{}")
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except json.JSONDecodeError:
                        data = {}

                event = RawEvent(
                    stream=stream_name,
                    exchange_ts=int(row["exchange_ts"]),
                    recv_ts=int(row["recv_ts"]),
                    data=data,
                )
                events.append(event)

        logger.debug("Loaded %d events from %s", len(events), stream_name)
        return events


class KlineReplayer:
    """Specialized replayer for historical kline Parquet files.

    Reads from the partitioned kline data (downloaded by download.py)
    and emits synthetic RawEvent objects in chronological order,
    compatible with the same handler interface as live data.
    """

    def __init__(
        self,
        data_dir: Path,
        symbol: str,
        interval: str,
        handler: EventHandler | Callable[[RawEvent], None],
        start_ts: int | None = None,
        end_ts: int | None = None,
    ) -> None:
        self._data_dir = data_dir
        self._symbol = symbol
        self._interval = interval
        self._handler = handler
        self._start_ts = start_ts
        self._end_ts = end_ts

    def replay(self) -> int:
        """Replay kline data as events.

        Returns:
            Total number of kline events emitted.
        """
        kline_dir = (
            self._data_dir / "raw" / "klines"
            / f"symbol={self._symbol}" / f"interval={self._interval}"
        )
        if not kline_dir.exists():
            logger.warning("No kline data found at %s", kline_dir)
            return 0

        parquet_files = sorted(kline_dir.rglob("part.parquet"))
        count = 0

        for pq_file in parquet_files:
            try:
                table = pq.read_table(pq_file)
                df = table.to_pandas()
            except Exception as e:
                logger.warning("Error reading %s: %s", pq_file, e)
                continue

            df = df.sort_values("open_time")

            for _, row in df.iterrows():
                close_ts = int(row["close_time"])

                if self._start_ts is not None and close_ts < self._start_ts:
                    continue
                if self._end_ts is not None and close_ts > self._end_ts:
                    continue

                # Emit as a kline event at close_ts (bar is "known" only at close)
                kline_data = {
                    "s": self._symbol,
                    "t": int(row["open_time"]),
                    "T": close_ts,
                    "o": str(row["open"]),
                    "h": str(row["high"]),
                    "l": str(row["low"]),
                    "c": str(row["close"]),
                    "v": str(row["volume"]),
                    "n": int(row["count"]) if "count" in row.index else 0,
                    "V": str(row["taker_buy_volume"]) if "taker_buy_volume" in row.index else "0",
                    "x": True,  # is final/closed bar
                }

                event = RawEvent(
                    stream=f"kline_{self._interval}",
                    exchange_ts=close_ts,
                    recv_ts=close_ts,  # In replay, recv_ts = exchange_ts
                    data={"k": kline_data, "e": "kline"},
                )

                if callable(self._handler) and not hasattr(self._handler, "on_event"):
                    self._handler(event)
                else:
                    self._handler.on_event(event)  # type: ignore[union-attr]
                count += 1

        logger.info("Replayed %d kline events for %s %s", count, self._symbol, self._interval)
        return count
