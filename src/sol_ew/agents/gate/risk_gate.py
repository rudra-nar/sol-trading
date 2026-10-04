"""Risk gate -- deterministic hard-rule filter for trade setups.

Enforces non-negotiable risk constraints. If any rule fails, the setup
is rejected with a reason. This is the last line of defense before the
simulator executes a trade.

Each rule is an independent pure function returning (pass, reason).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sol_ew.core.types import Decision, Setup

logger = logging.getLogger(__name__)


@dataclass
class GateConfig:
    """Risk gate configuration."""

    max_risk_pct: float = 2.0        # Max % of equity risked per trade
    min_rr_ratio: float = 1.0        # Min reward:risk ratio
    max_position_pct: float = 100.0  # Max % of equity in a single position (paper trading)
    max_daily_loss_pct: float = 5.0  # Max daily drawdown before halting
    max_concurrent: int = 3          # Max concurrent open positions
    min_score: float = 0.3           # Min hypothesis rule_score
    min_zone_strength: float = 0.3   # Min confluence zone strength
    cooldown_bars: int = 5           # Min bars between trades


def _check_risk_pct(setup: Setup, equity: float, config: GateConfig) -> tuple[bool, str]:
    """Check that risk per trade doesn't exceed max_risk_pct of equity."""
    if equity <= 0:
        return False, "equity <= 0"
    risk_amount = abs(setup.entry - setup.stop) * _implied_qty(setup, equity, config)
    risk_pct = (risk_amount / equity) * 100
    if risk_pct > config.max_risk_pct:
        return False, f"risk {risk_pct:.1f}% exceeds max {config.max_risk_pct}%"
    return True, ""


def _check_reward_risk(setup: Setup, config: GateConfig) -> tuple[bool, str]:
    """Check minimum reward:risk ratio."""
    risk = abs(setup.entry - setup.stop)
    if risk < 1e-10:
        return False, "stop == entry (zero risk)"
    if not setup.targets:
        return False, "no targets defined"
    # Use the first target for RR calculation
    reward = abs(setup.targets[0] - setup.entry)
    rr = reward / risk
    if rr < config.min_rr_ratio:
        return False, f"RR {rr:.2f} below min {config.min_rr_ratio}"
    return True, ""


def _check_position_size(setup: Setup, equity: float, config: GateConfig) -> tuple[bool, str]:
    """Check that position doesn't exceed max_position_pct of equity."""
    if equity <= 0:
        return False, "equity <= 0"
    position_value = setup.entry * _implied_qty(setup, equity, config)
    position_pct = (position_value / equity) * 100
    if position_pct > config.max_position_pct:
        return False, f"position {position_pct:.1f}% exceeds max {config.max_position_pct}%"
    return True, ""


def _check_daily_loss(daily_pnl_pct: float, config: GateConfig) -> tuple[bool, str]:
    """Check if daily loss limit has been hit."""
    if daily_pnl_pct < -config.max_daily_loss_pct:
        return False, f"daily loss {daily_pnl_pct:.1f}% exceeds max {config.max_daily_loss_pct}%"
    return True, ""


def _check_concurrent(n_open: int, config: GateConfig) -> tuple[bool, str]:
    """Check max concurrent positions."""
    if n_open >= config.max_concurrent:
        return False, f"{n_open} open positions, max is {config.max_concurrent}"
    return True, ""


def _check_score(setup: Setup, config: GateConfig) -> tuple[bool, str]:
    """Check minimum hypothesis rule score."""
    score = setup.features.get("rule_score", 0.0)
    if score < config.min_score:
        return False, f"rule_score {score:.2f} below min {config.min_score}"
    return True, ""


def _check_zone_strength(setup: Setup, config: GateConfig) -> tuple[bool, str]:
    """Check minimum confluence zone strength."""
    if setup.zone is None:
        return False, "no confluence zone"
    if setup.zone.strength < config.min_zone_strength:
        return False, f"zone strength {setup.zone.strength:.2f} below min {config.min_zone_strength}"
    return True, ""


def _check_cooldown(bars_since_last: int, config: GateConfig) -> tuple[bool, str]:
    """Check cooldown period between trades."""
    if bars_since_last < config.cooldown_bars:
        return False, f"cooldown: {bars_since_last} bars since last trade, need {config.cooldown_bars}"
    return True, ""


def _implied_qty(setup: Setup, equity: float, config: GateConfig) -> float:
    """Compute implied quantity from risk parameters.

    Position sized so that hitting the stop loses at most max_risk_pct of equity,
    capped so position value does not exceed max_position_pct of equity.
    """
    risk_per_unit = abs(setup.entry - setup.stop)
    if risk_per_unit < 1e-10:
        return 0.0
    max_loss = equity * config.max_risk_pct / 100.0
    risk_qty = max_loss / risk_per_unit
    max_notional = equity * config.max_position_pct / 100.0
    max_qty = max_notional / setup.entry if setup.entry > 0 else risk_qty
    return min(risk_qty, max_qty)


def evaluate(
    setup: Setup,
    equity: float,
    p_success: float,
    daily_pnl_pct: float = 0.0,
    n_open_positions: int = 0,
    bars_since_last_trade: int = 999,
    config: GateConfig | None = None,
) -> Decision:
    """Evaluate a setup through all risk gate rules.

    Args:
        setup: The candidate trade setup.
        equity: Current account equity.
        p_success: Predicted probability of success (from meta-labeler).
        daily_pnl_pct: Today's PnL as percentage of equity.
        n_open_positions: Number of currently open positions.
        bars_since_last_trade: Bars since the last trade was opened.
        config: Gate configuration (uses defaults if None).

    Returns:
        Decision with approved=True if all rules pass, False otherwise.
    """
    if config is None:
        config = GateConfig()

    reasons: list[str] = []
    all_pass = True

    checks = [
        _check_reward_risk(setup, config),
        _check_risk_pct(setup, equity, config),
        _check_position_size(setup, equity, config),
        _check_daily_loss(daily_pnl_pct, config),
        _check_concurrent(n_open_positions, config),
        _check_score(setup, config),
        _check_zone_strength(setup, config),
        _check_cooldown(bars_since_last_trade, config),
    ]

    for passed, reason in checks:
        if not passed:
            all_pass = False
            reasons.append(reason)

    # Compute expected return net of fees
    risk = abs(setup.entry - setup.stop)
    reward = abs(setup.targets[0] - setup.entry) if setup.targets else 0.0
    exp_r_net = p_success * reward - (1 - p_success) * risk

    if all_pass:
        reasons.append("all checks passed")
        logger.info("Setup %s APPROVED: exp_r=%.2f", setup.hyp_id, exp_r_net)
    else:
        logger.info("Setup %s REJECTED: %s", setup.hyp_id, "; ".join(reasons))

    return Decision(
        setup=setup,
        p_success=p_success,
        exp_r_net=exp_r_net,
        approved=all_pass,
        reasons=tuple(reasons),
    )
