"""Configuration loader using Pydantic v2.

All config values are starting points fixed from priors, not optimized.
Every field is logged with each run. Fee values are placeholders — the system
prints a warning at startup reminding the user to set real fee tiers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator


class DegreeConfig(BaseModel):
    """Configuration for a single wave degree (timeframe)."""

    tf: str
    atr_period: int = 14
    mult: float = 3.0


class FibConfig(BaseModel):
    """Fibonacci ratio sets."""

    retr: list[float] = Field(default=[0.236, 0.382, 0.5, 0.618, 0.786])
    ext: list[float] = Field(default=[1.272, 1.618, 2.618, 4.236])
    confluence_tol_atr: float = 0.4


class SetupConfig(BaseModel):
    """Setup generation parameters."""

    min_rr: float = 2.0
    stop_buffer_atr: float = 0.5
    max_setup_age_bars: int = 24


class RiskConfig(BaseModel):
    """Risk management parameters."""

    risk_fraction: float = 0.005  # 0.5% of equity per trade
    max_leverage: float = 3.0
    max_daily_loss_r: float = 3.0
    max_drawdown_halt: float = 0.10
    max_consecutive_losses: int = 5


class LatencyConfig(BaseModel):
    """Latency distribution parameters."""

    dist: str = "lognormal"
    median: float = 80.0
    sigma: float = 0.5


class SimConfig(BaseModel):
    """Simulator parameters. Fee values are PLACEHOLDERS."""

    latency_ms: LatencyConfig = Field(default_factory=LatencyConfig)
    fee_maker: float = 0.0002  # PLACEHOLDER
    fee_taker: float = 0.0005  # PLACEHOLDER
    slippage_stress_multiplier: float = 1.0
    stop_trigger_price: str = "mark"  # mark|last


class LabelConfig(BaseModel):
    """Labeling parameters."""

    horizon_bars: int = 96  # 8 hours of 5m bars
    stop_first_on_tie: bool = True


class CVConfig(BaseModel):
    """Cross-validation parameters."""

    train_bars: int = 210_000
    test_bars: int = 26_000
    embargo_bars: int = 200


class AppConfig(BaseModel):
    """Top-level application configuration.

    All values below are starting points to be fixed from priors, not optimized.
    Every field is logged with each run.
    """

    instrument: str = "SOLUSDT"
    venue: str = "binance_usdm"
    bar_tf: str = "5m"
    degrees: list[DegreeConfig] = Field(
        default=[
            DegreeConfig(tf="15m", atr_period=14, mult=3.0),
            DegreeConfig(tf="1h", atr_period=14, mult=3.0),
            DegreeConfig(tf="4h", atr_period=14, mult=2.5),
            DegreeConfig(tf="1D", atr_period=14, mult=2.0),
        ]
    )
    pivot_window: int = 12
    top_k_hypotheses: int = 5
    weight_ema_alpha: float = 0.8
    min_hyp_weight: float = 0.25
    min_weight_margin: float = 0.10
    fib: FibConfig = Field(default_factory=FibConfig)
    setup: SetupConfig = Field(default_factory=SetupConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    sim: SimConfig = Field(default_factory=SimConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    seed: int = 42

    @model_validator(mode="after")
    def _warn_placeholder_fees(self) -> AppConfig:
        """Record that fee values need user verification."""
        # Warning is printed at startup in CLI, not here (keeps model pure)
        return self


# Placeholder fee values that trigger the startup warning
_PLACEHOLDER_FEES = {"fee_maker": 0.0002, "fee_taker": 0.0005}


def is_placeholder_fee(config: AppConfig) -> bool:
    """Check if fee values are still at placeholder defaults."""
    return (
        config.sim.fee_maker == _PLACEHOLDER_FEES["fee_maker"]
        and config.sim.fee_taker == _PLACEHOLDER_FEES["fee_taker"]
    )


def load_config(path: Path | str) -> AppConfig:
    """Load configuration from a YAML file.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        Validated AppConfig instance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return AppConfig(**raw)


def default_config() -> AppConfig:
    """Return the default configuration with all priors."""
    return AppConfig()
