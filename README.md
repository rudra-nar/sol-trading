# SOL Elliott Wave / Fibonacci Multi-Agent Paper Trading System

A research and paper-trading system for Solana (SOL) that:

- Builds multi-timeframe candles (5m/15m/1h/4h/1D) from a single data stream
- Detects swing pivots and enumerates Elliott Wave hypotheses at several degrees
- Computes Fibonacci retracement/extension grids and multi-degree confluence zones
- Scores candidate trades with a machine-learning meta-labeler
- Enforces hard risk rules in a deterministic gate
- Sizes positions from wave invalidation levels
- Executes trades in a realistic paper-trading simulator

## ⚠️ Important

This is a **paper trading and research system only**. It never places real orders.
Nothing here is evidence of profitability. The deliverable is an honest, reproducible
measurement of whether Elliott Wave and Fibonacci features add predictive value for
SOL beyond a no-wave baseline.

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run CLI
sol_ew --help

# Dry-run replay
sol_ew replay --dry-run

# Run tests
pytest

# Lint
ruff check src/ tests/
mypy src/sol_ew/
```

## Project Structure

See `docs/SPEC.md` for the full specification.

```
sol_ew/
  src/sol_ew/
    core/         # types, messages, blackboard, config, clock
    data/         # downloaders, recorder, event log, replayer, validators
    bars/         # bar builder, resampler
    agents/       # pivot_wave, fib, context, setup, risk_exec
    models/       # hsmm, meta-labeler, optional RL
    sim/          # paper trading simulator
    research/     # labeling, CV, ablations, reports
    cli.py        # CLI entry point
  tests/
    unit/ property/ integration/ leakage/
  configs/
    default.yaml
  docs/
    SPEC.md
    DECISIONS.md
```
