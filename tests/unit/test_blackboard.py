"""Tests for the Blackboard messaging system."""

import json
from pathlib import Path

from sol_ew.core.blackboard import Blackboard
from sol_ew.core.types import Bar


class TestBlackboard:
    """Test Blackboard read/write/log behavior."""

    def test_write_and_read(self) -> None:
        bb = Blackboard()
        bb.write(1000, "test_topic", {"value": 42})
        result = bb.read("test_topic")
        assert result == {"value": 42}

    def test_read_missing_topic(self) -> None:
        bb = Blackboard()
        assert bb.read("nonexistent") is None

    def test_latest_overwrites(self) -> None:
        bb = Blackboard()
        bb.write(1000, "topic", {"v": 1})
        bb.write(2000, "topic", {"v": 2})
        assert bb.read("topic") == {"v": 2}

    def test_log_preserves_all(self) -> None:
        bb = Blackboard()
        bb.write(1000, "topic", {"v": 1})
        bb.write(2000, "topic", {"v": 2})
        bb.write(3000, "other", {"x": 99})
        log = bb.read_log()
        assert len(log) == 3
        assert log[0].ts == 1000
        assert log[2].topic == "other"

    def test_log_filter_by_topic(self) -> None:
        bb = Blackboard()
        bb.write(1000, "a", {"v": 1})
        bb.write(2000, "b", {"v": 2})
        bb.write(3000, "a", {"v": 3})
        filtered = bb.read_log("a")
        assert len(filtered) == 2
        assert all(e.topic == "a" for e in filtered)

    def test_topics_list(self) -> None:
        bb = Blackboard()
        bb.write(1000, "alpha", {})
        bb.write(2000, "beta", {})
        assert sorted(bb.topics) == ["alpha", "beta"]

    def test_snapshot(self) -> None:
        bb = Blackboard()
        bb.write(1000, "t", {"x": 1})
        snap = bb.snapshot()
        assert snap["log_len"] == 1
        assert "t" in snap["latest"]

    def test_write_dataclass(self) -> None:
        bb = Blackboard()
        bar = Bar(
            tf="5m", open_ts=0, close_ts=300000,
            o=100.0, h=105.0, l=98.0, c=103.0,
            volume=1000.0, taker_buy_volume=500.0, n_trades=100,
        )
        bb.write(300000, "bar_5m", bar)
        # The log should have serialized it to a dict
        log = bb.read_log("bar_5m")
        assert len(log) == 1
        assert log[0].payload["tf"] == "5m"
        assert log[0].payload["o"] == 100.0

    def test_dump_log(self, tmp_path: Path) -> None:
        bb = Blackboard()
        bb.write(1000, "topic", {"val": 1})
        bb.write(2000, "topic", {"val": 2})

        log_file = tmp_path / "test_log.jsonl"
        bb.dump_log(log_file)

        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["ts"] == 1000
        assert first["topic"] == "topic"
