"""CLI entry point for the SOL Elliott Wave system.

Commands:
    sol_ew --help           Show help
    sol_ew replay --dry-run No-op replay, exits 0
    sol_ew download         Download and validate historical data (Phase 1)
    sol_ew record           Run live data recorder (Phase 1)
    sol_ew backtest         Run backtest with walk-forward (Phase 6+)
    sol_ew paper --live     Run live paper trader (Phase 12)
    sol_ew explain <id>     Explain a trade decision (Phase 11)
"""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import typer
from rich.console import Console

from sol_ew.core.config import AppConfig, is_placeholder_fee, load_config

app = typer.Typer(
    name="sol_ew",
    help="SOL Elliott Wave / Fibonacci Multi-Agent Paper Trading System.\n\n"
    "A research and paper-trading system. PAPER TRADING ONLY -- never places real orders.",
    add_completion=False,
)
console = Console()


def _load_cfg(config: Path) -> AppConfig:
    """Load and validate config, printing fee warnings."""
    cfg = load_config(config)

    if is_placeholder_fee(cfg):
        console.print(
            "[bold yellow]WARNING:[/bold yellow] Fee values are at PLACEHOLDER defaults "
            f"(maker={cfg.sim.fee_maker}, taker={cfg.sim.fee_taker}). "
            "Set your actual exchange fee tier in the config file before trusting any results.",
            highlight=False,
        )

    return cfg


@app.command()
def replay(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="No-op replay: validate config and exit.",
    ),
) -> None:
    """Replay historical data through the agent pipeline."""
    cfg = _load_cfg(config)

    if dry_run:
        console.print(f"[green]OK[/green] Config loaded: instrument={cfg.instrument}, bar_tf={cfg.bar_tf}")
        console.print(f"[green]OK[/green] Degrees: {[d.tf for d in cfg.degrees]}")
        console.print(f"[green]OK[/green] Seed: {cfg.seed}")
        console.print("[green]OK[/green] Dry run complete. All systems nominal.")
        raise typer.Exit(code=0)

    # Full replay — implemented in Phase 2+
    console.print("[yellow]Full replay not yet implemented. Use --dry-run for now.[/yellow]")
    raise typer.Exit(code=0)


@app.command()
def download(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",
    ),
    symbols: str | None = typer.Option(
        None,
        "--symbols",
        "-s",
        help="Comma-separated symbols to download (default: from config).",
    ),
    start: str | None = typer.Option(
        None,
        "--start",
        help="Start date YYYY-MM-DD.",
    ),
    end: str | None = typer.Option(
        None,
        "--end",
        help="End date YYYY-MM-DD.",
    ),
) -> None:
    """Download and validate historical kline data."""
    cfg = _load_cfg(config)

    import asyncio as _asyncio

    from sol_ew.data.download import download_all

    sym_list = [s.strip() for s in symbols.split(",")] if symbols else [cfg.instrument]
    start_date = date.fromisoformat(start) if start else date(2020, 1, 1)
    end_date = date.fromisoformat(end) if end else date.today()
    data_dir = Path("data")

    console.print(
        f"Downloading {sym_list} {cfg.bar_tf} from {start_date} to {end_date}"
    )

    logging.basicConfig(level=logging.INFO)
    results = _asyncio.run(
        download_all(sym_list, "1m", start_date, end_date, data_dir)
    )

    for sym, (dl, sk, errs) in results.items():
        console.print(f"  {sym}: {dl} downloaded, {sk} skipped, {len(errs)} errors")
        for err in errs:
            console.print(f"    [red]{err}[/red]")

    raise typer.Exit(code=0)


@app.command()
def record(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",
    ),
) -> None:
    """Run the live data recorder (public websocket feeds)."""
    cfg = _load_cfg(config)

    import asyncio as _asyncio

    from sol_ew.data.recorder import run_recorder

    data_dir = Path("data")
    state_file = data_dir / "recorder_state.json"

    console.print(f"Starting live recorder for {cfg.instrument}...")
    console.print("Press Ctrl+C to stop.")

    logging.basicConfig(level=logging.INFO)
    try:
        _asyncio.run(
            run_recorder(cfg.instrument, data_dir, state_file=state_file)
        )
    except KeyboardInterrupt:
        console.print("\nRecorder stopped by user.")

    raise typer.Exit(code=0)


@app.command()
def backtest(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",
    ),
    data_dir: Path = typer.Option(
        Path("data"),
        "--data-dir",
        "-d",
        help="Path to data directory with parquet files.",
    ),
    capital: float = typer.Option(
        10_000.0,
        "--capital",
        help="Initial capital for the simulation.",
    ),
) -> None:
    """Run backtest: feed historical klines through the full pipeline."""
    import time as _time

    import numpy as np
    import pyarrow.parquet as pq
    from rich.progress import Progress

    from sol_ew.agents.gate.risk_gate import GateConfig
    from sol_ew.pipeline import Pipeline
    from sol_ew.reporting import save_run

    cfg = _load_cfg(config)
    logging.basicConfig(level=logging.INFO)

    # Find parquet files
    kline_dir = data_dir / "raw" / "klines"
    if not kline_dir.exists():
        console.print(f"[red]No kline data found at {kline_dir}[/red]")
        console.print("Run 'sol_ew download' first to fetch historical data.")
        raise typer.Exit(code=1)

    pq_files = sorted(kline_dir.rglob("*.parquet"))
    if not pq_files:
        console.print(f"[red]No parquet files found in {kline_dir}[/red]")
        raise typer.Exit(code=1)

    console.print(f"Found {len(pq_files)} parquet files")
    console.print(f"Config: {cfg.instrument}, capital=${capital:,.0f}, seed={cfg.seed}")

    # Initialize pipeline
    pipeline = Pipeline(
        config=cfg,
        gate_config=GateConfig(),
        initial_capital=capital,
        seed=cfg.seed,
    )

    start_time = _time.time()
    total_bars = 0
    total_pivots = 0
    total_trades = 0

    # Read and sort all klines by timestamp
    console.print("Loading kline data...")
    all_rows = []
    for pf in pq_files:
        try:
            table = pq.read_table(pf)
            df = table.to_pandas()
            all_rows.append(df)
        except Exception as e:
            console.print(f"  [yellow]Skipping {pf.name}: {e}[/yellow]")

    if not all_rows:
        console.print("[red]No valid kline data loaded[/red]")
        raise typer.Exit(code=1)

    import pandas as pd
    combined = pd.concat(all_rows, ignore_index=True)
    combined = combined.sort_values("open_time").drop_duplicates(subset=["open_time"])
    n_rows = len(combined)
    console.print(f"Loaded {n_rows:,} klines")

    # Convert to numpy arrays for fast iteration (iterrows is ~100x slower)
    col_open_ts = combined["open_time"].values
    col_close_ts = combined["close_time"].values
    col_open = combined["open"].values
    col_high = combined["high"].values
    col_low = combined["low"].values
    col_close = combined["close"].values
    col_volume = combined["volume"].values if "volume" in combined.columns else np.zeros(n_rows)
    col_tbv = combined["taker_buy_base_volume"].values if "taker_buy_base_volume" in combined.columns else np.zeros(n_rows)
    col_ntrades = combined["n_trades"].values if "n_trades" in combined.columns else np.zeros(n_rows, dtype=int)

    # Reduce logging noise during backtest
    logging.getLogger("sol_ew.agents.gate.risk_gate").setLevel(logging.WARNING)

    # Feed through pipeline
    with Progress() as progress:
        task = progress.add_task("Backtesting...", total=n_rows)
        for i in range(n_rows):
            events = pipeline.on_kline(
                open_ts=int(col_open_ts[i]),
                close_ts=int(col_close_ts[i]),
                o=float(col_open[i]),
                h=float(col_high[i]),
                l=float(col_low[i]),
                c=float(col_close[i]),
                volume=float(col_volume[i]),
                taker_buy_volume=float(col_tbv[i]),
                n_trades=int(col_ntrades[i]),
                is_closed=True,
            )
            total_bars += 1
            total_pivots += len(events.get("pivots", []))  # type: ignore[arg-type]
            total_trades += len([
                d for d in events.get("decisions", [])  # type: ignore[union-attr]
                if hasattr(d, "approved") and d.approved
            ])
            if i % 5000 == 0:
                progress.update(task, completed=i)

    elapsed = _time.time() - start_time
    sim = pipeline.simulator

    # Print results
    console.print(f"\n[bold]Backtest Complete[/bold] ({elapsed:.1f}s)")
    console.print(f"  Bars processed: {total_bars:,}")
    console.print(f"  Pivots detected: {total_pivots}")
    console.print(f"  Trades executed: {sim.trade_count}")
    console.print(f"  Final equity: ${sim.equity:,.2f}")
    console.print(f"  Total return: {(sim.equity / capital - 1) * 100:.2f}%")
    console.print(f"  Max drawdown: {sim.max_drawdown * 100:.2f}%")
    console.print(f"  Total fees: ${sim.total_fees:,.4f}")
    console.print(f"  Total funding: ${sim.total_funding:,.4f}")

    # Save run
    run_dir = save_run(sim, cfg, elapsed)
    console.print(f"\n  Run saved to: {run_dir}")

    raise typer.Exit(code=0)


@app.command()
def paper(
    config: Path = typer.Option(
        Path("configs/default.yaml"),
        "--config",
        "-c",
        help="Path to YAML config file.",
    ),
    capital: float = typer.Option(
        10_000.0,
        "--capital",
        help="Initial capital for the simulation.",
    ),
) -> None:
    """Run live paper trader: connect to websocket and feed through pipeline.

    PAPER TRADING ONLY -- no real orders are placed.
    Uses the same Pipeline.on_kline() as backtest (invariant 4).
    """
    import asyncio as _asyncio
    import json as _json
    import time as _time

    from sol_ew.agents.gate.risk_gate import GateConfig
    from sol_ew.pipeline import Pipeline
    from sol_ew.reporting import save_run

    cfg = _load_cfg(config)
    logging.basicConfig(level=logging.INFO)

    pipeline = Pipeline(
        config=cfg,
        gate_config=GateConfig(),
        initial_capital=capital,
        seed=cfg.seed,
    )

    symbol = cfg.instrument.lower()
    stream = f"{symbol}@kline_1m"
    url = f"wss://fstream.binance.com/ws/{stream}"

    console.print("[bold]Live Paper Trader[/bold] -- PAPER TRADING ONLY")
    console.print(f"  Symbol: {cfg.instrument}")
    console.print(f"  Capital: ${capital:,.0f}")
    console.print(f"  Stream: {url}")
    console.print("  Press Ctrl+C to stop.\n")

    start_time = _time.time()
    bar_count = 0

    async def _run() -> None:
        nonlocal bar_count
        try:
            import websockets
        except ImportError:
            console.print("[red]websockets package required. Install with: pip install websockets[/red]")
            return

        async with websockets.connect(url) as ws:
            console.print("[green]Connected to Binance websocket[/green]")
            async for msg in ws:
                data = _json.loads(msg)
                k = data.get("k")
                if k is None:
                    continue

                events = pipeline.on_kline(
                    open_ts=int(k["t"]),
                    close_ts=int(k["T"]),
                    o=float(k["o"]),
                    h=float(k["h"]),
                    l=float(k["l"]),
                    c=float(k["c"]),
                    volume=float(k["v"]),
                    taker_buy_volume=float(k.get("V", 0)),
                    n_trades=int(k.get("n", 0)),
                    is_closed=bool(k.get("x", False)),
                )

                if bool(k.get("x", False)):
                    bar_count += 1
                    sim = pipeline.simulator
                    if bar_count % 10 == 0:
                        console.print(
                            f"  Bar {bar_count} | "
                            f"Equity: ${sim.equity:,.2f} | "
                            f"Trades: {sim.trade_count} | "
                            f"DD: {sim.max_drawdown*100:.1f}%"
                        )

                # Log pivots and trades
                for p in events.get("pivots", []):  # type: ignore[union-attr]
                    console.print(f"  [cyan]PIVOT[/cyan] {p.tf} {'HIGH' if p.kind == 1 else 'LOW'} @ {p.price:.2f}")
                for d in events.get("decisions", []):  # type: ignore[union-attr]
                    status = "[green]APPROVED[/green]" if d.approved else "[red]REJECTED[/red]"
                    console.print(f"  [bold]DECISION[/bold] {d.setup.hyp_id} {status}")

    try:
        _asyncio.run(_run())
    except KeyboardInterrupt:
        elapsed = _time.time() - start_time
        sim = pipeline.simulator
        console.print(f"\n[bold]Paper Trading Stopped[/bold] ({elapsed:.0f}s)")
        console.print(f"  Bars: {bar_count} | Trades: {sim.trade_count} | Equity: ${sim.equity:,.2f}")

        if bar_count > 0:
            run_dir = save_run(sim, cfg, elapsed)
            console.print(f"  Run saved to: {run_dir}")

    raise typer.Exit(code=0)


@app.command()
def explain(
    trade_id: str = typer.Argument(..., help="Trade ID to explain."),
    run_dir: Path = typer.Option(
        Path("runs/latest"),
        "--run-dir",
        help="Path to the run directory.",
    ),
) -> None:
    """Print the full decision chain for a trade."""
    console.print(f"[yellow]Explain not yet implemented (Phase 11). Trade ID: {trade_id}[/yellow]")
    raise typer.Exit(code=0)


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
