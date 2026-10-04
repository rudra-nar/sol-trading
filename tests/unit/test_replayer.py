"""Tests for the replayer module."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from sol_ew.data.recorder import RawEvent
from sol_ew.data.replayer import EventReplayer, KlineReplayer


def _write_recorded_events(data_dir: Path, stream: str, events: list[dict[str, object]]) -> None:
    """Helper: write synthetic recorded events as Parquet."""
    df = pd.DataFrame(events)
    path = data_dir / "recorded" / stream / "year=2023" / "month=01" / "day=01" / "hour=00" / "part.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


def _write_kline_parquet(data_dir: Path, symbol: str, interval: str, klines: list[dict[str, object]]) -> None:
    """Helper: write synthetic kline data as partitioned Parquet."""
    df = pd.DataFrame(klines)
    path = (
        data_dir / "raw" / "klines"
        / f"symbol={symbol}" / f"interval={interval}"
        / "year=2023" / "month=01"
        / "part.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(table, path)


class TestEventReplayer:
    """Test replay of recorded events."""

    def test_replay_single_stream(self, tmp_path: Path) -> None:
        events = [
            {"exchange_ts": 1000, "recv_ts": 1010, "data": json.dumps({"v": 1})},
            {"exchange_ts": 2000, "recv_ts": 2010, "data": json.dumps({"v": 2})},
        ]
        _write_recorded_events(tmp_path, "kline_1m", events)

        collected: list[RawEvent] = []
        replayer = EventReplayer(tmp_path, handler=collected.append)
        count = replayer.replay()

        assert count == 2
        assert collected[0].exchange_ts == 1000
        assert collected[1].exchange_ts == 2000

    def test_replay_multiple_streams(self, tmp_path: Path) -> None:
        _write_recorded_events(tmp_path, "kline_1m", [
            {"exchange_ts": 1000, "recv_ts": 1010, "data": json.dumps({"s": "kline"})},
        ])
        _write_recorded_events(tmp_path, "aggTrade", [
            {"exchange_ts": 500, "recv_ts": 505, "data": json.dumps({"s": "trade"})},
        ])

        collected: list[RawEvent] = []
        replayer = EventReplayer(tmp_path, handler=collected.append)
        count = replayer.replay()

        assert count == 2
        # Should be sorted by recv_ts
        assert collected[0].recv_ts <= collected[1].recv_ts

    def test_replay_with_stream_filter(self, tmp_path: Path) -> None:
        _write_recorded_events(tmp_path, "kline_1m", [
            {"exchange_ts": 1000, "recv_ts": 1010, "data": json.dumps({})},
        ])
        _write_recorded_events(tmp_path, "aggTrade", [
            {"exchange_ts": 2000, "recv_ts": 2010, "data": json.dumps({})},
        ])

        collected: list[RawEvent] = []
        replayer = EventReplayer(tmp_path, handler=collected.append, streams=["kline_1m"])
        count = replayer.replay()

        assert count == 1
        assert collected[0].stream == "kline_1m"

    def test_replay_with_time_filter(self, tmp_path: Path) -> None:
        events = [
            {"exchange_ts": 1000, "recv_ts": 1000, "data": json.dumps({})},
            {"exchange_ts": 2000, "recv_ts": 2000, "data": json.dumps({})},
            {"exchange_ts": 3000, "recv_ts": 3000, "data": json.dumps({})},
        ]
        _write_recorded_events(tmp_path, "test", events)

        collected: list[RawEvent] = []
        replayer = EventReplayer(tmp_path, handler=collected.append, start_ts=1500, end_ts=2500)
        count = replayer.replay()

        assert count == 1
        assert collected[0].exchange_ts == 2000

    def test_replay_empty_dir(self, tmp_path: Path) -> None:
        collected: list[RawEvent] = []
        replayer = EventReplayer(tmp_path, handler=collected.append)
        count = replayer.replay()
        assert count == 0

    def test_replay_preserves_data(self, tmp_path: Path) -> None:
        original_data = {"key": "value", "nested": {"a": 1}}
        events = [
            {"exchange_ts": 1000, "recv_ts": 1010, "data": json.dumps(original_data)},
        ]
        _write_recorded_events(tmp_path, "test", events)

        collected: list[RawEvent] = []
        replayer = EventReplayer(tmp_path, handler=collected.append)
        replayer.replay()

        assert collected[0].data == original_data


class TestKlineReplayer:
    """Test replay of historical kline data."""

    def _sample_klines(self) -> list[dict[str, object]]:
        """Generate sample kline rows."""
        return [
            {
                "open_time": 1700000000000 + i * 60000,
                "open": 100.0 + i,
                "high": 105.0 + i,
                "low": 98.0 + i,
                "close": 103.0 + i,
                "volume": 1000.0,
                "close_time": 1700000000000 + (i + 1) * 60000 - 1,
                "quote_volume": 50000.0,
                "count": 500,
                "taker_buy_volume": 600.0,
                "taker_buy_quote_volume": 30000.0,
            }
            for i in range(10)
        ]

    def test_replay_klines(self, tmp_path: Path) -> None:
        klines = self._sample_klines()
        _write_kline_parquet(tmp_path, "SOLUSDT", "1m", klines)

        collected: list[RawEvent] = []
        replayer = KlineReplayer(tmp_path, "SOLUSDT", "1m", handler=collected.append)
        count = replayer.replay()

        assert count == 10
        # Events should be in chronological order
        for i in range(1, len(collected)):
            assert collected[i].exchange_ts >= collected[i - 1].exchange_ts

    def test_kline_event_format(self, tmp_path: Path) -> None:
        klines = self._sample_klines()[:1]
        _write_kline_parquet(tmp_path, "SOLUSDT", "1m", klines)

        collected: list[RawEvent] = []
        replayer = KlineReplayer(tmp_path, "SOLUSDT", "1m", handler=collected.append)
        replayer.replay()

        event = collected[0]
        assert event.stream == "kline_1m"
        k = event.data["k"]
        assert k["x"] is True  # is closed bar
        assert k["o"] == "100.0"
        assert k["s"] == "SOLUSDT"

    def test_kline_emitted_at_close(self, tmp_path: Path) -> None:
        """Kline events should be emitted at close_ts, not open_ts."""
        klines = self._sample_klines()[:1]
        _write_kline_parquet(tmp_path, "SOLUSDT", "1m", klines)

        collected: list[RawEvent] = []
        replayer = KlineReplayer(tmp_path, "SOLUSDT", "1m", handler=collected.append)
        replayer.replay()

        event = collected[0]
        # exchange_ts should be close_time, not open_time
        assert event.exchange_ts == klines[0]["close_time"]

    def test_kline_time_filter(self, tmp_path: Path) -> None:
        klines = self._sample_klines()
        _write_kline_parquet(tmp_path, "SOLUSDT", "1m", klines)

        # Filter to only middle klines
        start = int(klines[3]["close_time"])  # type: ignore[arg-type]
        end = int(klines[6]["close_time"])  # type: ignore[arg-type]

        collected: list[RawEvent] = []
        replayer = KlineReplayer(
            tmp_path, "SOLUSDT", "1m", handler=collected.append,
            start_ts=start, end_ts=end,
        )
        count = replayer.replay()

        assert 0 < count < 10  # Should filter some out

    def test_kline_no_data(self, tmp_path: Path) -> None:
        collected: list[RawEvent] = []
        replayer = KlineReplayer(tmp_path, "SOLUSDT", "1m", handler=collected.append)
        count = replayer.replay()
        assert count == 0
