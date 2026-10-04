"""Live data recorder for exchange public websocket feeds.

Records kline_1m, aggTrade, depth@100ms, markPrice, funding, and liquidation
events. Writes append-only Parquet partitions by hour with both exchange_ts
and recv_ts. Reconnects with exponential backoff. Detects gaps on restart.

PAPER TRADING ONLY -- reads public/read-only data only.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# Binance Futures websocket endpoint
WS_BASE = "wss://fstream.binance.com"

# Maximum events to buffer before flushing to disk
BUFFER_FLUSH_SIZE = 5000
# Flush interval in seconds (even if buffer not full)
BUFFER_FLUSH_INTERVAL = 60.0
# REST depth snapshot interval in seconds
DEPTH_SNAPSHOT_INTERVAL = 300.0
# Clock offset sample interval in seconds
CLOCK_SAMPLE_INTERVAL = 60.0
# Maximum reconnection backoff in seconds
MAX_BACKOFF = 60.0


@dataclass(frozen=True)
class RawEvent:
    """A raw event from the exchange feed.

    Both exchange_ts and recv_ts are recorded for latency analysis
    and clock offset tracking.
    """

    stream: str       # "kline_1m", "aggTrade", "depth", "markPrice", etc.
    exchange_ts: int   # UTC ms from exchange
    recv_ts: int       # UTC ms when received locally
    data: dict[str, object]  # raw event payload


@dataclass
class RecorderState:
    """Persistent state for gap detection across restarts."""

    last_event_ts: dict[str, int] = field(default_factory=dict)  # stream -> last exchange_ts
    start_ts: int = 0
    total_events: int = 0
    gaps_detected: int = 0

    def snapshot(self) -> dict[str, object]:
        """Capture state for persistence."""
        return {
            "last_event_ts": dict(self.last_event_ts),
            "start_ts": self.start_ts,
            "total_events": self.total_events,
            "gaps_detected": self.gaps_detected,
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore from persisted state."""
        self.last_event_ts = dict(state.get("last_event_ts", {}))  # type: ignore[arg-type]
        self.start_ts = int(state.get("start_ts", 0))  # type: ignore[arg-type]
        self.total_events = int(state.get("total_events", 0))  # type: ignore[arg-type]
        self.gaps_detected = int(state.get("gaps_detected", 0))  # type: ignore[arg-type]


class EventBuffer:
    """Buffers events and flushes to Parquet partitions by hour.

    Partition path: {data_dir}/recorded/{stream}/year=YYYY/month=MM/day=DD/hour=HH/part.parquet
    """

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._buffers: dict[str, list[dict[str, object]]] = {}
        self._last_flush_time = time.monotonic()

    def add(self, event: RawEvent) -> None:
        """Add an event to the buffer."""
        stream = event.stream
        if stream not in self._buffers:
            self._buffers[stream] = []
        self._buffers[stream].append({
            "exchange_ts": event.exchange_ts,
            "recv_ts": event.recv_ts,
            "data": json.dumps(event.data, default=str),
        })

    def should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        total = sum(len(b) for b in self._buffers.values())
        time_elapsed = time.monotonic() - self._last_flush_time
        return total >= BUFFER_FLUSH_SIZE or time_elapsed >= BUFFER_FLUSH_INTERVAL

    def flush(self) -> int:
        """Flush all buffers to Parquet files.

        Returns:
            Total number of events flushed.
        """
        total_flushed = 0
        for stream, events in self._buffers.items():
            if not events:
                continue
            total_flushed += len(events)
            df = pd.DataFrame(events)
            # Group by hour for partitioning
            df["_hour_ts"] = (df["recv_ts"] // 3_600_000) * 3_600_000
            for hour_ts, group in df.groupby("_hour_ts"):
                ts = pd.Timestamp(int(hour_ts), unit="ms", tz="UTC")  # type: ignore[arg-type]
                path = (
                    self._data_dir / "recorded" / stream
                    / f"year={ts.year}" / f"month={ts.month:02d}"
                    / f"day={ts.day:02d}" / f"hour={ts.hour:02d}"
                    / "part.parquet"
                )
                group_out = group.drop(columns=["_hour_ts"])
                self._append_parquet(path, group_out)

        self._buffers.clear()
        self._last_flush_time = time.monotonic()
        return total_flushed

    @staticmethod
    def _append_parquet(path: Path, df: pd.DataFrame) -> None:
        """Append DataFrame to a Parquet file (creates if not exists)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if path.exists():
            existing = pq.read_table(path)
            # Unify schemas by casting to the new table's schema
            try:
                table = pa.concat_tables([existing, table], promote_options="default")
            except (pa.ArrowInvalid, pa.ArrowTypeError):
                # If schemas are incompatible, cast existing to match new
                existing = existing.select([c for c in table.column_names if c in existing.column_names])
                table = pa.concat_tables([existing, table], promote_options="permissive")
        pq.write_table(table, path, compression="snappy")


def _now_ms() -> int:
    """Current UTC time in milliseconds."""
    return int(time.time() * 1000)


def _build_stream_url(symbol: str, streams: list[str]) -> str:
    """Build the combined websocket stream URL.

    Args:
        symbol: Trading pair in lowercase (e.g. "solusdt").
        streams: List of stream suffixes.

    Returns:
        Full websocket URL.
    """
    stream_names = [f"{symbol}@{s}" for s in streams]
    return f"{WS_BASE}/stream?streams={'/'.join(stream_names)}"


def _extract_exchange_ts(stream_type: str, data: dict[str, object]) -> int:
    """Extract the exchange timestamp from an event.

    Args:
        stream_type: The type of stream (e.g., "kline", "aggTrade").
        data: The event data dict.

    Returns:
        Exchange timestamp in UTC milliseconds.
    """
    # Most Binance events have "E" as event time
    if "E" in data:
        return int(data["E"])  # type: ignore[arg-type]
    # Trade events
    if "T" in data:
        return int(data["T"])  # type: ignore[arg-type]
    return _now_ms()


def _classify_stream(raw_stream: str) -> str:
    """Classify a raw Binance stream name into a category.

    Args:
        raw_stream: Raw stream name like "solusdt@kline_1m".

    Returns:
        Category string like "kline_1m", "aggTrade", etc.
    """
    # Format: "symbol@streamType" or "symbol@streamType@speed"
    parts = raw_stream.split("@", 1)
    if len(parts) < 2:
        return "unknown"
    return parts[1]


async def run_recorder(
    symbol: str,
    data_dir: Path,
    state_file: Path | None = None,
    stop_event: asyncio.Event | None = None,
) -> None:
    """Run the live data recorder.

    Connects to Binance public websocket, records events to Parquet.
    Reconnects with exponential backoff on disconnection. Detects
    gaps on restart by comparing with persisted state.

    Args:
        symbol: Trading pair (e.g. "SOLUSDT").
        data_dir: Root data directory for recorded events.
        state_file: Path to persist recorder state (for gap detection).
        stop_event: If provided, recorder stops when this event is set.
    """
    try:
        import websockets
    except ImportError:
        logger.error("websockets package required for live recording")
        return

    sym_lower = symbol.lower()
    streams = ["kline_1m", "aggTrade", "depth@100ms", "markPrice", "forceOrder"]
    ws_url = _build_stream_url(sym_lower, streams)

    state = RecorderState()
    state.start_ts = _now_ms()

    # Load previous state for gap detection
    if state_file and state_file.exists():
        try:
            prev_state = json.loads(state_file.read_text())
            state.restore(prev_state)
            logger.info(
                "Loaded recorder state: last events at %s",
                {k: v for k, v in state.last_event_ts.items()},
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Could not load recorder state: %s", e)

    buffer = EventBuffer(data_dir)
    backoff = 1.0

    while True:
        if stop_event and stop_event.is_set():
            logger.info("Stop event received, shutting down recorder")
            break

        try:
            logger.info("Connecting to %s", ws_url)
            async with websockets.connect(ws_url, ping_interval=20) as ws:  # type: ignore[attr-defined]
                logger.info("Connected to Binance websocket")
                backoff = 1.0  # Reset backoff on successful connection

                # Check for gaps since last run
                reconnect_ts = _now_ms()
                for stream_name, last_ts in state.last_event_ts.items():
                    gap_ms = reconnect_ts - last_ts
                    if gap_ms > 120_000:  # More than 2 minutes
                        state.gaps_detected += 1
                        logger.warning(
                            "GAP DETECTED on %s: %d ms since last event (%.1f min)",
                            stream_name, gap_ms, gap_ms / 60_000,
                        )

                async for raw_msg in ws:
                    if stop_event and stop_event.is_set():
                        break

                    recv_ts = _now_ms()
                    try:
                        msg = json.loads(raw_msg)
                    except json.JSONDecodeError:
                        logger.warning("Invalid JSON from websocket")
                        continue

                    # Combined stream format: {"stream": "...", "data": {...}}
                    stream_name = msg.get("stream", "unknown")
                    data = msg.get("data", msg)
                    stream_type = _classify_stream(stream_name)
                    exchange_ts = _extract_exchange_ts(stream_type, data)

                    event = RawEvent(
                        stream=stream_type,
                        exchange_ts=exchange_ts,
                        recv_ts=recv_ts,
                        data=data,
                    )
                    buffer.add(event)
                    state.last_event_ts[stream_type] = exchange_ts
                    state.total_events += 1

                    # Periodic flush
                    if buffer.should_flush():
                        flushed = buffer.flush()
                        logger.info(
                            "Flushed %d events (total: %d, gaps: %d)",
                            flushed, state.total_events, state.gaps_detected,
                        )
                        # Persist state
                        if state_file:
                            state_file.parent.mkdir(parents=True, exist_ok=True)
                            state_file.write_text(json.dumps(state.snapshot(), default=str))

        except Exception as e:
            # Flush any remaining events before reconnect
            remaining = buffer.flush()
            if remaining:
                logger.info("Flushed %d events before reconnect", remaining)

            # Persist state
            if state_file:
                state_file.parent.mkdir(parents=True, exist_ok=True)
                state_file.write_text(json.dumps(state.snapshot(), default=str))

            logger.warning(
                "Websocket disconnected: %s. Reconnecting in %.1fs",
                e, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, MAX_BACKOFF)

    # Final flush
    remaining = buffer.flush()
    if remaining:
        logger.info("Final flush: %d events", remaining)
    if state_file:
        state_file.parent.mkdir(parents=True, exist_ok=True)
        state_file.write_text(json.dumps(state.snapshot(), default=str))
    logger.info(
        "Recorder stopped. Total events: %d, Gaps detected: %d",
        state.total_events, state.gaps_detected,
    )
