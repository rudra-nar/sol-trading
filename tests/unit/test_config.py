"""Tests for configuration loading."""

from pathlib import Path
from textwrap import dedent

from sol_ew.core.config import AppConfig, default_config, is_placeholder_fee, load_config


class TestDefaultConfig:
    """Test default configuration values match the spec."""

    def test_defaults(self) -> None:
        cfg = default_config()
        assert cfg.instrument == "SOLUSDT"
        assert cfg.venue == "binance_usdm"
        assert cfg.bar_tf == "5m"
        assert cfg.seed == 42
        assert len(cfg.degrees) == 4
        assert cfg.degrees[0].tf == "15m"
        assert cfg.degrees[3].tf == "1D"

    def test_fib_defaults(self) -> None:
        cfg = default_config()
        assert 0.618 in cfg.fib.retr
        assert 1.618 in cfg.fib.ext
        assert cfg.fib.confluence_tol_atr == 0.4

    def test_risk_defaults(self) -> None:
        cfg = default_config()
        assert cfg.risk.risk_fraction == 0.005
        assert cfg.risk.max_leverage == 3.0
        assert cfg.risk.max_drawdown_halt == 0.10

    def test_sim_defaults(self) -> None:
        cfg = default_config()
        assert cfg.sim.latency_ms.dist == "lognormal"
        assert cfg.sim.stop_trigger_price == "mark"

    def test_placeholder_fees(self) -> None:
        cfg = default_config()
        assert is_placeholder_fee(cfg) is True

    def test_custom_fees_not_placeholder(self) -> None:
        cfg = AppConfig(sim={"latency_ms": {}, "fee_maker": 0.0001, "fee_taker": 0.0003})  # type: ignore[arg-type]
        assert is_placeholder_fee(cfg) is False


class TestLoadConfig:
    """Test config loading from YAML files."""

    def test_load_default_yaml(self, tmp_path: Path) -> None:
        yaml_content = dedent("""\
            instrument: SOLUSDT
            venue: binance_usdm
            seed: 123
        """)
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        cfg = load_config(cfg_file)
        assert cfg.instrument == "SOLUSDT"
        assert cfg.seed == 123

    def test_load_with_overrides(self, tmp_path: Path) -> None:
        yaml_content = dedent("""\
            instrument: BTCUSDT
            risk:
              risk_fraction: 0.01
              max_leverage: 5.0
        """)
        cfg_file = tmp_path / "test.yaml"
        cfg_file.write_text(yaml_content)

        cfg = load_config(cfg_file)
        assert cfg.instrument == "BTCUSDT"
        assert cfg.risk.risk_fraction == 0.01
        assert cfg.risk.max_leverage == 5.0
        # Other defaults preserved
        assert cfg.risk.max_drawdown_halt == 0.10

    def test_load_missing_file(self) -> None:
        try:
            load_config(Path("nonexistent.yaml"))
            raise AssertionError("Should raise FileNotFoundError")
        except FileNotFoundError:
            pass

    def test_empty_yaml(self, tmp_path: Path) -> None:
        cfg_file = tmp_path / "empty.yaml"
        cfg_file.write_text("")
        cfg = load_config(cfg_file)
        # All defaults should be used
        assert cfg.instrument == "SOLUSDT"
