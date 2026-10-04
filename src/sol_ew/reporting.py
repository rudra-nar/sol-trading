"""Reporting -- equity curve, trade log, and run summary.

Generates reports from simulator state. Saves to runs/ directory
with a trial log entry in runs/trials.jsonl.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from sol_ew.core.config import AppConfig
from sol_ew.sim.simulator import Simulator

logger = logging.getLogger(__name__)


def generate_trade_log(sim: Simulator) -> list[dict[str, object]]:
    """Export all fills as a list of dicts."""
    return [
        {
            "order_id": f.order_id,
            "fill_ts": f.fill_ts,
            "side": f.side.value,
            "qty": f.qty,
            "price": f.price,
            "fee": f.fee,
            "is_maker": f.is_maker,
            "reason": f.reason,
        }
        for f in sim.fills
    ]


def generate_equity_csv(sim: Simulator) -> str:
    """Export equity curve as CSV string."""
    lines = ["ts,equity,drawdown,position_value"]
    for ep in sim.equity_curve:
        lines.append(f"{ep.ts},{ep.equity:.2f},{ep.drawdown:.6f},{ep.position_value:.2f}")
    return "\n".join(lines)


def generate_summary(sim: Simulator, config: AppConfig, elapsed_s: float) -> dict[str, object]:
    """Generate a run summary dict."""
    equity_curve = sim.equity_curve
    final_equity = equity_curve[-1].equity if equity_curve else sim.cash

    return {
        "instrument": config.instrument,
        "bar_tf": config.bar_tf,
        "seed": config.seed,
        "initial_capital": sim._initial_capital,
        "final_equity": round(final_equity, 2),
        "total_return_pct": round((final_equity / sim._initial_capital - 1) * 100, 2),
        "max_drawdown_pct": round(sim.max_drawdown * 100, 2),
        "total_trades": sim.trade_count,
        "total_fees": round(sim.total_fees, 4),
        "total_funding": round(sim.total_funding, 4),
        "bars_processed": len(equity_curve),
        "elapsed_seconds": round(elapsed_s, 2),
        "fee_maker": config.sim.fee_maker,
        "fee_taker": config.sim.fee_taker,
    }


def save_run(
    sim: Simulator,
    config: AppConfig,
    elapsed_s: float,
    run_dir: Path | None = None,
) -> Path:
    """Save a complete run to disk.

    Creates:
      runs/<timestamp>/summary.json
      runs/<timestamp>/trades.jsonl
      runs/<timestamp>/equity.csv
      runs/<timestamp>/config.json

    Also appends to runs/trials.jsonl.

    Args:
        sim: The simulator after a backtest/live run.
        config: The config used for the run.
        elapsed_s: Wall-clock time elapsed.
        run_dir: Override for the run output directory.

    Returns:
        Path to the run directory.
    """
    if run_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_dir = Path("runs") / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)

    # Summary
    summary = generate_summary(sim, config, elapsed_s)
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8",
    )

    # Trade log
    trades = generate_trade_log(sim)
    with (run_dir / "trades.jsonl").open("w", encoding="utf-8") as f:
        for trade in trades:
            f.write(json.dumps(trade, default=str) + "\n")

    # Equity curve
    (run_dir / "equity.csv").write_text(
        generate_equity_csv(sim), encoding="utf-8",
    )

    # Config snapshot
    (run_dir / "config.json").write_text(
        config.model_dump_json(indent=2), encoding="utf-8",
    )

    # Append to trials log
    trials_path = Path("runs") / "trials.jsonl"
    trials_path.parent.mkdir(parents=True, exist_ok=True)
    trial_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_dir": str(run_dir),
        **summary,
    }
    with trials_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(trial_entry, default=str) + "\n")

    logger.info("Run saved to %s", run_dir)
    return run_dir
