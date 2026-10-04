"""Blackboard — typed topic-based message bus for inter-agent communication.

Every write is appended to a run log with (ts, topic, payload) so any decision
can be reconstructed. Agents read inputs and write outputs through the blackboard.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class LogEntry:
    """A single entry in the blackboard's append-only log."""

    ts: int  # UTC ms when the write occurred
    topic: str
    payload: dict[str, Any]


class Blackboard:
    """Simple in-process typed topic store with append-only logging.

    - latest[topic]: the most recent value for that topic
    - log: chronological list of all writes
    """

    def __init__(self) -> None:
        self._latest: dict[str, Any] = {}
        self._log: list[LogEntry] = []

    def write(self, ts: int, topic: str, payload: Any) -> None:
        """Write a value to a topic. Appends to the log."""
        self._latest[topic] = payload
        serializable = _to_dict(payload)
        entry = LogEntry(ts=ts, topic=topic, payload=serializable)
        self._log.append(entry)

    def read(self, topic: str) -> Any | None:
        """Read the latest value for a topic, or None if not written yet."""
        return self._latest.get(topic)

    def read_log(self, topic: str | None = None) -> list[LogEntry]:
        """Read log entries, optionally filtered by topic."""
        if topic is None:
            return list(self._log)
        return [e for e in self._log if e.topic == topic]

    @property
    def topics(self) -> list[str]:
        """List all topics that have been written to."""
        return list(self._latest.keys())

    def snapshot(self) -> dict[str, Any]:
        """Capture current state for replayability."""
        return {
            "latest": dict(self._latest),
            "log_len": len(self._log),
        }

    def dump_log(self, path: Path) -> None:
        """Write the full log to a JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for entry in self._log:
                line = {
                    "ts": entry.ts,
                    "topic": entry.topic,
                    "payload": entry.payload,
                }
                f.write(json.dumps(line, default=str) + "\n")


def _to_dict(obj: Any) -> dict[str, Any]:
    """Convert a dataclass or dict to a plain dict for logging."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)  # type: ignore[arg-type]
    if isinstance(obj, dict):
        return obj
    return {"value": str(obj)}
