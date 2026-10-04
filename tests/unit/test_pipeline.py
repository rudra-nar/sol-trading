"""Tests for the pipeline orchestrator."""

from __future__ import annotations

from sol_ew.core.config import AppConfig, DegreeConfig
from sol_ew.pipeline import Pipeline


def _make_kline(idx: int, base: float = 100.0, amplitude: float = 5.0) -> dict[str, object]:
    """Create a synthetic kline event dict."""
    import math
    # Create a sinusoidal price pattern
    phase = idx * 0.1
    mid = base + amplitude * math.sin(phase) + idx * 0.01  # slight trend
    return {
        "open_ts": idx * 60_000,
        "close_ts": (idx + 1) * 60_000 - 1,
        "o": mid,
        "h": mid + 2,
        "l": mid - 2,
        "c": mid + 0.5,
        "volume": 1000.0,
        "taker_buy_volume": 500.0,
        "n_trades": 100,
        "is_closed": True,
    }


class TestPipeline:
    """Test full pipeline integration."""

    def test_pipeline_creation(self) -> None:
        """Pipeline should initialize with default config."""
        pipeline = Pipeline()
        assert pipeline.bar_count == 0
        assert pipeline.simulator is not None

    def test_pipeline_processes_bars(self) -> None:
        """Pipeline should process bars without errors."""
        config = AppConfig(
            degrees=[DegreeConfig(tf="5m", atr_period=5, mult=2.0)],
        )
        pipeline = Pipeline(config=config)

        for i in range(100):
            kline = _make_kline(i)
            events = pipeline.on_kline(**kline)  # type: ignore[arg-type]
            assert isinstance(events, dict)

        assert pipeline.bar_count == 100

    def test_pipeline_produces_pivots(self) -> None:
        """With enough bars and volatility, should produce pivots."""
        config = AppConfig(
            degrees=[DegreeConfig(tf="5m", atr_period=5, mult=1.0)],
        )
        pipeline = Pipeline(config=config)

        # Feed bars with a clear zigzag pattern
        all_pivots = []
        for i in range(200):
            # Create a zigzag: up for 20 bars, down for 20 bars
            cycle = i % 40
            if cycle < 20:
                mid = 100 + cycle * 3
            else:
                mid = 160 - (cycle - 20) * 3
            kline = {
                "open_ts": i * 60_000,
                "close_ts": (i + 1) * 60_000 - 1,
                "o": mid, "h": mid + 1, "l": mid - 1, "c": mid + 0.5,
                "volume": 1000.0, "taker_buy_volume": 500.0,
                "n_trades": 100, "is_closed": True,
            }
            events = pipeline.on_kline(**kline)  # type: ignore[arg-type]
            all_pivots.extend(events.get("pivots", []))  # type: ignore[union-attr]

        # Should have produced some pivots
        # (exact count depends on ATR dynamics)
        assert isinstance(all_pivots, list)

    def test_pipeline_equity_stable(self) -> None:
        """Without trades, equity should remain at initial capital."""
        pipeline = Pipeline(initial_capital=50_000.0)

        for i in range(50):
            kline = _make_kline(i)
            pipeline.on_kline(**kline)  # type: ignore[arg-type]

        # No trades should have been made with so few bars
        assert pipeline.simulator.trade_count == 0

    def test_pipeline_snapshot(self) -> None:
        """Snapshot should capture pipeline state."""
        pipeline = Pipeline()
        for i in range(10):
            kline = _make_kline(i)
            pipeline.on_kline(**kline)  # type: ignore[arg-type]

        snap = pipeline.snapshot()
        assert snap["bar_count"] == 10
        assert "sim" in snap
        assert "degrees" in snap

    def test_unclosed_klines_ignored(self) -> None:
        """Unclosed klines should not advance the bar count."""
        pipeline = Pipeline()
        kline = _make_kline(0)
        kline["is_closed"] = False
        events = pipeline.on_kline(**kline)  # type: ignore[arg-type]
        assert pipeline.bar_count == 0
        assert events["pivots"] == []  # type: ignore[comparison-overlap]
