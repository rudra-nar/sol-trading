"""Tests for the risk gate.

Acceptance: each rejection reason is independently triggered.
"""

from __future__ import annotations

from sol_ew.agents.gate.risk_gate import GateConfig, evaluate
from sol_ew.core.types import Setup, Zone


def _setup(
    entry: float = 100.0,
    stop: float = 95.0,
    targets: tuple[float, ...] = (110.0,),
    rule_score: float = 0.8,
    zone_strength: float = 1.0,
) -> Setup:
    """Create a test setup."""
    zone = Zone(lo=99.0, hi=101.0, strength=zone_strength, n_degrees=2, sources=("test",))
    return Setup(
        ts=1000, hyp_id="test:0:impulse", side=1,
        entry=entry, stop=stop, targets=targets,
        zone=zone,
        features={"rule_score": rule_score},
    )


class TestRiskGate:
    """Test each rejection reason independently."""

    def test_all_pass(self) -> None:
        """A well-formed setup should pass all gate checks."""
        setup = _setup(entry=100.0, stop=95.0, targets=(110.0,))
        config = GateConfig(max_position_pct=50.0)  # Permissive for this test
        decision = evaluate(setup, equity=10_000.0, p_success=0.6, config=config)
        assert decision.approved is True

    def test_reject_low_rr(self) -> None:
        """RR ratio below minimum."""
        # Risk = 5, reward = 2, RR = 0.4
        setup = _setup(entry=100.0, stop=95.0, targets=(102.0,))
        config = GateConfig(min_rr_ratio=1.5)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6, config=config)
        assert decision.approved is False
        assert any("RR" in r for r in decision.reasons)

    def test_reject_zero_risk(self) -> None:
        """Stop == entry (zero risk)."""
        setup = _setup(entry=100.0, stop=100.0)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6)
        assert decision.approved is False
        assert any("zero risk" in r for r in decision.reasons)

    def test_reject_daily_loss(self) -> None:
        """Daily loss limit exceeded."""
        setup = _setup()
        config = GateConfig(max_daily_loss_pct=5.0)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6,
                            daily_pnl_pct=-6.0, config=config)
        assert decision.approved is False
        assert any("daily loss" in r for r in decision.reasons)

    def test_reject_max_concurrent(self) -> None:
        """Too many open positions."""
        setup = _setup()
        config = GateConfig(max_concurrent=2)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6,
                            n_open_positions=2, config=config)
        assert decision.approved is False
        assert any("open positions" in r for r in decision.reasons)

    def test_reject_low_score(self) -> None:
        """Hypothesis rule_score below threshold."""
        setup = _setup(rule_score=0.2)
        config = GateConfig(min_score=0.5)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6, config=config)
        assert decision.approved is False
        assert any("rule_score" in r for r in decision.reasons)

    def test_reject_no_zone(self) -> None:
        """No confluence zone."""
        setup = Setup(
            ts=1000, hyp_id="test", side=1,
            entry=100.0, stop=95.0, targets=(110.0,),
            zone=None,
            features={"rule_score": 0.8},
        )
        decision = evaluate(setup, equity=10_000.0, p_success=0.6)
        assert decision.approved is False
        assert any("no confluence" in r for r in decision.reasons)

    def test_reject_weak_zone(self) -> None:
        """Zone strength below minimum."""
        setup = _setup(zone_strength=0.1)
        config = GateConfig(min_zone_strength=0.5)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6, config=config)
        assert decision.approved is False
        assert any("zone strength" in r for r in decision.reasons)

    def test_reject_cooldown(self) -> None:
        """Too soon after last trade."""
        setup = _setup()
        config = GateConfig(cooldown_bars=10)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6,
                            bars_since_last_trade=3, config=config)
        assert decision.approved is False
        assert any("cooldown" in r for r in decision.reasons)

    def test_exp_r_net_computed(self) -> None:
        """Expected return is computed correctly."""
        setup = _setup(entry=100.0, stop=95.0, targets=(110.0,))
        decision = evaluate(setup, equity=10_000.0, p_success=0.6)
        # exp_r = 0.6 * 10 - 0.4 * 5 = 4.0
        assert abs(decision.exp_r_net - 4.0) < 0.01

    def test_multiple_rejections(self) -> None:
        """Multiple rules can fail simultaneously."""
        setup = _setup(rule_score=0.1, zone_strength=0.01)
        config = GateConfig(min_score=0.5, min_zone_strength=0.5, cooldown_bars=10)
        decision = evaluate(setup, equity=10_000.0, p_success=0.6,
                            bars_since_last_trade=1, config=config)
        assert decision.approved is False
        assert len(decision.reasons) >= 3
