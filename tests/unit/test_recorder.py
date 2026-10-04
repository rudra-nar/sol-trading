"""Tests for the recorder module."""

from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq

from sol_ew.data.recorder import (
    EventBuffer,
    RawEvent,
    RecorderState,
    _build_stream_url,
    _classify_stream,
    _extract_exchange_ts,
)


class TestRawEvent:
    """Test RawEvent data structure."""

    def test_creation(self) -> None:
        event = RawEvent(
            stream="kline_1m",
            exchange_ts=1700000000000,
            recv_ts=1700000000080,
            data={"k": {"o": "100.0"}},
        )
        assert event.stream == "kline_1m"
        assert event.recv_ts - event.exchange_ts == 80  # ~80ms latency

    def test_frozen(self) -> None:
        event = RawEvent(stream="test", exchange_ts=0, recv_ts=0, data={})
        try:
            event.stream = "other"  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


class TestRecorderState:
    """Test recorder state snapshot/restore."""

    def test_snapshot_restore(self) -> None:
        state = RecorderState()
        state.last_event_ts = {"kline_1m": 1000, "aggTrade": 2000}
        state.total_events = 500
        state.gaps_detected = 2

        snap = state.snapshot()

        state2 = RecorderState()
        state2.restore(snap)
        assert state2.last_event_ts == {"kline_1m": 1000, "aggTrade": 2000}
        assert state2.total_events == 500
        assert state2.gaps_detected == 2

    def test_gap_detection_on_restore(self) -> None:
        """When restoring state, gaps should be detectable."""
        state = RecorderState()
        state.last_event_ts = {"kline_1m": 1_000_000}
        snap = state.snapshot()

        # Simulate restart after 5 minutes
        state2 = RecorderState()
        state2.restore(snap)
        # Gap detection happens in run_recorder, but we can verify state is preserved
        assert state2.last_event_ts["kline_1m"] == 1_000_000


class TestEventBuffer:
    """Test event buffering and Parquet flush."""

    def test_add_and_flush(self, tmp_path: Path) -> None:
        buf = EventBuffer(tmp_path)
        event = RawEvent(
            stream="kline_1m",
            exchange_ts=1700000000000,
            recv_ts=1700000000080,
            data={"test": True},
        )
        buf.add(event)
        flushed = buf.flush()
        assert flushed == 1

        # Verify Parquet was written
        pq_files = list(tmp_path.rglob("*.parquet"))
        assert len(pq_files) == 1

        df = pq.read_table(pq_files[0]).to_pandas()
        assert len(df) == 1
        assert df["exchange_ts"].iloc[0] == 1700000000000

    def test_multiple_streams(self, tmp_path: Path) -> None:
        buf = EventBuffer(tmp_path)
        for stream in ["kline_1m", "aggTrade", "depth"]:
            buf.add(RawEvent(
                stream=stream,
                exchange_ts=1700000000000,
                recv_ts=1700000000080,
                data={"stream": stream},
            ))
        flushed = buf.flush()
        assert flushed == 3

        # Should have separate directories per stream
        pq_files = list(tmp_path.rglob("*.parquet"))
        assert len(pq_files) == 3

    def test_append_to_existing(self, tmp_path: Path) -> None:
        buf = EventBuffer(tmp_path)
        # Same hour timestamp for both events
        for i in range(2):
            buf.add(RawEvent(
                stream="test",
                exchange_ts=1700000000000 + i,
                recv_ts=1700000000000 + i,
                data={"i": i},
            ))
            buf.flush()

        # Should have appended to same file
        pq_files = list(tmp_path.rglob("*.parquet"))
        assert len(pq_files) == 1

        df = pq.ParquetFile(pq_files[0]).read().to_pandas()
        assert len(df) == 2

    def test_should_flush_by_size(self, tmp_path: Path) -> None:
        buf = EventBuffer(tmp_path)
        assert not buf.should_flush()

        # Fill beyond threshold
        for i in range(5001):
            buf.add(RawEvent(stream="test", exchange_ts=i, recv_ts=i, data={}))
        assert buf.should_flush()


class TestHelperFunctions:
    """Test utility functions."""

    def test_build_stream_url(self) -> None:
        url = _build_stream_url("solusdt", ["kline_1m", "aggTrade"])
        assert "solusdt@kline_1m" in url
        assert "solusdt@aggTrade" in url
        assert url.startswith("wss://")

    def test_classify_stream(self) -> None:
        assert _classify_stream("solusdt@kline_1m") == "kline_1m"
        assert _classify_stream("solusdt@aggTrade") == "aggTrade"
        assert _classify_stream("solusdt@depth@100ms") == "depth@100ms"
        assert _classify_stream("unknown") == "unknown"

    def test_extract_exchange_ts(self) -> None:
        assert _extract_exchange_ts("kline", {"E": 1700000000000}) == 1700000000000
        assert _extract_exchange_ts("trade", {"T": 1700000000001}) == 1700000000001

    def test_extract_exchange_ts_missing(self) -> None:
        # Should return current time (non-zero)
        ts = _extract_exchange_ts("unknown", {})
        assert ts > 0
