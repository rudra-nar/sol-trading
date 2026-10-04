"""Pipeline orchestrator -- wires all agents together.

Receives bars from the data layer, feeds them through the full agent
pipeline: bars -> zigzag -> wave rules -> hypothesis -> fib -> gate -> sim.

Same code for live and replay (invariant 4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sol_ew.agents.fib.confluence import compute_confluence_zones
from sol_ew.agents.gate.risk_gate import GateConfig, evaluate
from sol_ew.agents.pivot_wave.hypothesis import HypothesisTracker
from sol_ew.agents.pivot_wave.zigzag import ZigZag
from sol_ew.bars.builder import BarBuilder
from sol_ew.bars.resample import MultiTimeframeResampler
from sol_ew.core.blackboard import Blackboard
from sol_ew.core.config import AppConfig, DegreeConfig
from sol_ew.core.types import Bar, Decision, Hypothesis, Setup, Zone
from sol_ew.sim.simulator import Side, Simulator

logger = logging.getLogger(__name__)


@dataclass
class DegreeState:
    """State for a single wave degree (timeframe)."""

    config: DegreeConfig
    zigzag: ZigZag
    tracker: HypothesisTracker
    bar_count: int = 0


class Pipeline:
    """Full agent pipeline orchestrator.

    Wires together:
    1. BarBuilder (1m kline -> Bar)
    2. MultiTimeframeResampler (1m -> 5m/15m/1h/4h/1D)
    3. ZigZag per degree (each tf -> Pivots)
    4. HypothesisTracker per degree (Pivots -> Hypotheses)
    5. Fibonacci confluence (Hypotheses -> Zones)
    6. Risk gate (Setup -> Decision)
    7. Simulator (Decision -> Fills)

    Same code for live and replay.
    """

    def __init__(
        self,
        config: AppConfig | None = None,
        gate_config: GateConfig | None = None,
        initial_capital: float = 10_000.0,
        seed: int = 42,
    ) -> None:
        if config is None:
            config = AppConfig()
        if gate_config is None:
            gate_config = GateConfig()

        self._config = config
        self._gate_config = gate_config

        # Bar builder for 1m klines
        self._bar_builder = BarBuilder(tf="1m")

        # Multi-timeframe resampler
        target_tfs = [d.tf for d in config.degrees]
        if config.bar_tf not in target_tfs:
            target_tfs = [config.bar_tf, *target_tfs]
        self._resampler = MultiTimeframeResampler(
            source_tf="1m", target_tfs=target_tfs,
        )

        # Per-degree state
        self._degrees: dict[str, DegreeState] = {}
        for d in config.degrees:
            self._degrees[d.tf] = DegreeState(
                config=d,
                zigzag=ZigZag(tf=d.tf, atr_period=d.atr_period, mult=d.mult),
                tracker=HypothesisTracker(tf=d.tf),
            )

        # Simulator
        self._sim = Simulator(
            initial_capital=initial_capital,
            fee_maker=config.sim.fee_maker,
            fee_taker=config.sim.fee_taker,
            seed=seed,
        )

        # Blackboard for inter-agent communication
        self._blackboard = Blackboard()

        # Pipeline state
        self._bar_count = 0
        self._bars_since_last_trade = 999
        self._last_zones: list[Zone] = []

    def on_kline(
        self,
        open_ts: int,
        close_ts: int,
        o: float,
        h: float,
        l: float,
        c: float,
        volume: float,
        taker_buy_volume: float = 0.0,
        n_trades: int = 0,
        is_closed: bool = True,
    ) -> dict[str, object]:
        """Process a kline event through the full pipeline.

        This is the main entry point for both live and replay.

        Returns:
            Dict of events that occurred: pivots, hypotheses, zones,
            decisions, fills.
        """
        events: dict[str, object] = {
            "pivots": [],
            "hypotheses": [],
            "zones": [],
            "decisions": [],
            "fills": [],
        }

        # 1. Build 1m bar
        bar = self._bar_builder.on_kline(
            open_ts=open_ts, close_ts=close_ts,
            o=o, h=h, l=l, c=c,
            volume=volume, taker_buy_volume=taker_buy_volume,
            n_trades=n_trades, is_closed=is_closed,
        )
        if bar is None:
            return events

        self._bar_count += 1
        self._bars_since_last_trade += 1

        # 2. Simulator processes bar (fills pending orders from previous bar, checks stops/TP/funding)
        fills = self._sim.on_bar(bar)
        if fills:
            events["fills"] = fills  # type: ignore[assignment]

        # 3. Resample to higher timeframes
        completed = self._resampler.on_bar(bar)

        # 4. Process each completed HTF bar through its degree
        all_active_hyps: list[Hypothesis] = []
        for tf, htf_bar in completed.items():
            if tf in self._degrees:
                degree_events = self._process_degree(tf, htf_bar)
                events["pivots"].extend(degree_events.get("pivots", []))  # type: ignore[union-attr]
                events["hypotheses"].extend(degree_events.get("hypotheses", []))  # type: ignore[union-attr]

        # Collect all active hypotheses across degrees
        for ds in self._degrees.values():
            all_active_hyps.extend(ds.tracker.active_hypotheses)

        # 5. Compute confluence zones
        if all_active_hyps:
            zones = compute_confluence_zones(all_active_hyps)
            if zones:
                self._last_zones = zones
                events["zones"] = zones  # type: ignore[assignment]
                self._blackboard.write(bar.close_ts, "zones", zones)

        # 6. Generate setups and evaluate through gate
        has_pos = self._sim.position is not None or len(self._sim.pending_orders) > 0
        for hyp in all_active_hyps:
            if hyp.status != "active" or len(hyp.pivots) < 3:
                continue
            setup = self._generate_setup(hyp, bar)
            if setup is not None:
                decision = evaluate(
                    setup=setup,
                    equity=self._sim.equity,
                    p_success=hyp.rule_score,
                    n_open_positions=1 if has_pos else 0,
                    bars_since_last_trade=self._bars_since_last_trade,
                    config=self._gate_config,
                )
                events["decisions"].append(decision)  # type: ignore[union-attr]

                if decision.approved and not has_pos:
                    self._execute_decision(decision, bar)
                    self._bars_since_last_trade = 0
                    has_pos = True

        return events

    def _process_degree(self, tf: str, bar: Bar) -> dict[str, list[object]]:
        """Process a bar through a single degree's zigzag + hypothesis tracker."""
        degree = self._degrees[tf]
        degree.bar_count += 1
        degree.tracker.on_bar(degree.bar_count)
        result: dict[str, list[object]] = {"pivots": [], "hypotheses": []}

        pivot = degree.zigzag.on_bar(bar)
        if pivot is not None:
            result["pivots"].append(pivot)
            self._blackboard.write(bar.close_ts, f"pivot:{tf}", pivot)

            updated = degree.tracker.on_pivot(pivot, degree.zigzag.pivots)
            result["hypotheses"].extend(updated)
            for hyp in updated:
                self._blackboard.write(bar.close_ts, f"hyp:{tf}:{hyp.id}", hyp)

        return result

    def _generate_setup(self, hyp: Hypothesis, current_bar: Bar) -> Setup | None:
        """Generate a trade setup from a hypothesis if conditions are met."""
        if not hyp.targets:
            return None

        # Find the best confluence zone near the current price
        best_zone: Zone | None = None
        for zone in self._last_zones:
            # Check if bar overlaps or is within 0.25% of zone
            zone_buffer = (zone.lo + zone.hi) * 0.0025
            z_lo = zone.lo - zone_buffer
            z_hi = zone.hi + zone_buffer
            if current_bar.l <= z_hi and current_bar.h >= z_lo:
                best_zone = zone
                break

        if best_zone is None:
            return None

        # Determine entry/stop/targets
        entry = current_bar.c

        # Use the most recent opposing swing for a tighter stop
        buffer = entry * 0.005  # 0.5% buffer
        stop = hyp.invalidation
        if hyp.direction == 1:
            # Bullish: stop below recent swing low
            troughs = [p.price for p in hyp.pivots if p.kind == -1]
            if troughs:
                stop = min(troughs) - buffer
        else:
            # Bearish: stop above recent swing high
            peaks = [p.price for p in hyp.pivots if p.kind == 1]
            if peaks:
                stop = max(peaks) + buffer

        # Invalidation check: entry must be valid relative to stop
        if hyp.direction == 1 and stop >= entry:
            return None
        if hyp.direction == -1 and stop <= entry:
            return None

        # Filter targets: only those on our side of entry
        if hyp.direction == 1:
            valid_targets = tuple(t for t in sorted(hyp.targets.values()) if t > entry)
        else:
            valid_targets = tuple(t for t in sorted(hyp.targets.values(), reverse=True) if t < entry)

        if not valid_targets or abs(entry - stop) < 1e-10:
            return None

        return Setup(
            ts=current_bar.close_ts,
            hyp_id=hyp.id,
            side=hyp.direction,
            entry=entry,
            stop=stop,
            targets=valid_targets,
            zone=best_zone,
            features={
                "rule_score": hyp.rule_score,
                "hsmm_p": hyp.hsmm_p,
                "weight": hyp.weight,
                "zone_strength": best_zone.strength,
                "n_degrees": best_zone.n_degrees,
            },
        )

    def _execute_decision(self, decision: Decision, bar: Bar) -> None:
        """Execute an approved trade decision."""
        setup = decision.setup
        side = Side.LONG if setup.side == 1 else Side.SHORT

        # Size position from invalidation level
        risk = abs(setup.entry - setup.stop)
        if risk < 1e-10:
            return
        max_loss = self._sim.equity * self._gate_config.max_risk_pct / 100
        qty = max_loss / risk

        # Cap qty at max_position_pct of equity
        max_notional = self._sim.equity * (self._gate_config.max_position_pct / 100)
        max_qty = max_notional / setup.entry if setup.entry > 0 else qty
        qty = min(qty, max_qty)

        order_id = f"EW-{self._bar_count}"
        self._sim.set_stop_loss(setup.stop)
        if setup.targets:
            self._sim.set_take_profit(setup.targets[0])
        self._sim.submit_market_order(order_id, side, qty)

        logger.info(
            "Trade executed: %s %.4f @ ~%.2f | SL=%.2f | TP=%s",
            side.value, qty, setup.entry, setup.stop,
            f"{setup.targets[0]:.2f}" if setup.targets else "none",
        )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def simulator(self) -> Simulator:
        return self._sim

    @property
    def blackboard(self) -> Blackboard:
        return self._blackboard

    @property
    def bar_count(self) -> int:
        return self._bar_count

    @property
    def degrees(self) -> dict[str, DegreeState]:
        return dict(self._degrees)

    def snapshot(self) -> dict[str, object]:
        """Capture full pipeline state."""
        return {
            "bar_count": self._bar_count,
            "bars_since_last_trade": self._bars_since_last_trade,
            "sim": self._sim.snapshot(),
            "degrees": {
                tf: {
                    "zigzag": ds.zigzag.snapshot(),
                    "tracker": ds.tracker.snapshot(),
                    "bar_count": ds.bar_count,
                }
                for tf, ds in self._degrees.items()
            },
        }
