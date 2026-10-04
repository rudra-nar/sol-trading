"""Research labels — future-looking labels for training only.

THIS MODULE MUST NEVER BE IMPORTED BY AGENT OR FEATURE CODE.
It uses future bars to compute whether a hypothetical trade would have
succeeded, which is only valid in a research/training context.

Invariant: labels live in research/ only — never imported by feature or agent code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import IntEnum

from sol_ew.core.types import Bar

logger = logging.getLogger(__name__)


class Label(IntEnum):
    """Trade outcome label."""

    LOSS = -1
    NEUTRAL = 0
    WIN = 1


@dataclass(frozen=True)
class LabeledSetup:
    """A labeled trade setup for training data.

    Attributes:
        entry_ts: Timestamp of entry bar.
        entry_price: Entry price.
        stop: Stop-loss price.
        target: Take-profit price.
        side: +1 long, -1 short.
        label: WIN, LOSS, or NEUTRAL.
        exit_price: Price at which the trade exited.
        exit_ts: Timestamp of exit.
        bars_held: Number of bars the trade was open.
        max_favorable: Maximum favorable excursion (MFE).
        max_adverse: Maximum adverse excursion (MAE).
        r_multiple: Realized reward/risk ratio.
    """

    entry_ts: int
    entry_price: float
    stop: float
    target: float
    side: int
    label: Label
    exit_price: float
    exit_ts: int
    bars_held: int
    max_favorable: float
    max_adverse: float
    r_multiple: float


def label_trade(
    entry_bar_idx: int,
    entry_price: float,
    stop: float,
    target: float,
    side: int,
    future_bars: list[Bar],
    max_hold_bars: int = 100,
) -> LabeledSetup:
    """Label a hypothetical trade by walking through future bars.

    RESEARCH ONLY — uses future data. Never call from live/agent code.

    Args:
        entry_bar_idx: Index of the entry bar in the full bar list.
        entry_price: Entry price.
        stop: Stop-loss price.
        target: Take-profit price.
        side: +1 for long, -1 for short.
        future_bars: Bars AFTER the entry bar (future-looking!).
        max_hold_bars: Maximum bars to hold before expiring as NEUTRAL.

    Returns:
        LabeledSetup with the outcome.
    """
    risk = abs(entry_price - stop)
    if risk < 1e-10:
        return LabeledSetup(
            entry_ts=0, entry_price=entry_price, stop=stop, target=target,
            side=side, label=Label.NEUTRAL, exit_price=entry_price,
            exit_ts=0, bars_held=0, max_favorable=0, max_adverse=0, r_multiple=0,
        )

    max_favorable = 0.0
    max_adverse = 0.0
    exit_price = entry_price
    exit_ts = 0
    bars_held = 0
    label = Label.NEUTRAL

    for i, bar in enumerate(future_bars[:max_hold_bars]):
        bars_held = i + 1

        if side == 1:  # Long
            # Check stop hit (bar low)
            if bar.l <= stop:
                exit_price = stop
                exit_ts = bar.close_ts
                label = Label.LOSS
                max_adverse = max(max_adverse, entry_price - bar.l)
                break
            # Check target hit (bar high)
            if bar.h >= target:
                exit_price = target
                exit_ts = bar.close_ts
                label = Label.WIN
                max_favorable = max(max_favorable, bar.h - entry_price)
                break
            # Track excursions
            max_favorable = max(max_favorable, bar.h - entry_price)
            max_adverse = max(max_adverse, entry_price - bar.l)

        else:  # Short
            # Check stop hit (bar high)
            if bar.h >= stop:
                exit_price = stop
                exit_ts = bar.close_ts
                label = Label.LOSS
                max_adverse = max(max_adverse, bar.h - entry_price)
                break
            # Check target hit (bar low)
            if bar.l <= target:
                exit_price = target
                exit_ts = bar.close_ts
                label = Label.WIN
                max_favorable = max(max_favorable, entry_price - bar.l)
                break
            # Track excursions
            max_favorable = max(max_favorable, entry_price - bar.l)
            max_adverse = max(max_adverse, bar.h - entry_price)

    # If neither stop nor target hit, close at last bar's close
    if label == Label.NEUTRAL and future_bars:
        last_bar = future_bars[min(max_hold_bars - 1, len(future_bars) - 1)]
        exit_price = last_bar.c
        exit_ts = last_bar.close_ts

    # Compute R-multiple
    pnl = (exit_price - entry_price) * side
    r_multiple = pnl / risk if risk > 0 else 0.0

    return LabeledSetup(
        entry_ts=future_bars[0].open_ts if future_bars else 0,
        entry_price=entry_price,
        stop=stop,
        target=target,
        side=side,
        label=label,
        exit_price=exit_price,
        exit_ts=exit_ts,
        bars_held=bars_held,
        max_favorable=max_favorable,
        max_adverse=max_adverse,
        r_multiple=r_multiple,
    )


def label_series(
    setups: list[dict[str, object]],
    all_bars: list[Bar],
    max_hold_bars: int = 100,
) -> list[LabeledSetup]:
    """Label a list of setups against a bar series.

    RESEARCH ONLY.

    Args:
        setups: List of dicts with keys: bar_idx, entry_price, stop, target, side.
        all_bars: Full bar series.
        max_hold_bars: Max bars to hold.

    Returns:
        List of LabeledSetup.
    """
    results = []
    for s in setups:
        bar_idx = int(s["bar_idx"])  # type: ignore[arg-type]
        future = all_bars[bar_idx + 1:]
        labeled = label_trade(
            entry_bar_idx=bar_idx,
            entry_price=float(s["entry_price"]),  # type: ignore[arg-type]
            stop=float(s["stop"]),  # type: ignore[arg-type]
            target=float(s["target"]),  # type: ignore[arg-type]
            side=int(s["side"]),  # type: ignore[arg-type]
            future_bars=future,
            max_hold_bars=max_hold_bars,
        )
        results.append(labeled)
    return results
