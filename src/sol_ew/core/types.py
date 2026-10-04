"""Core data contracts.

All price fields are float, all times are UTC milliseconds (int).
These types form the communication contracts between agents.
Every type that agents exchange must be defined here.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Bar:
    """A completed OHLCV bar. Only usable at/after close_ts."""

    tf: str  # "1m","5m","15m","1h","4h","1D"
    open_ts: int  # UTC ms, bar open time
    close_ts: int  # UTC ms, bar close time — bar is only usable at/after this time
    o: float
    h: float
    l: float
    c: float
    volume: float
    taker_buy_volume: float
    n_trades: int
    degraded: bool = False  # True if built from incomplete data (gap)


@dataclass(frozen=True)
class Pivot:
    """A confirmed swing point. Downstream logic must use confirm_idx/confirm_ts."""

    tf: str
    idx: int  # bar index of the price extreme
    ts: int  # open_ts of the extreme bar
    confirm_idx: int  # bar index at which the pivot became knowable
    confirm_ts: int  # close_ts of that bar — USE THIS for any decision
    price: float
    kind: int  # +1 swing high, -1 swing low


@dataclass
class Hypothesis:
    """An Elliott Wave hypothesis at a specific degree/timeframe.

    Attributes:
        id: Stable identifier f"{tf}:{anchor_pivot_ts}:{kind}"
        status: active|invalidated|superseded|expired|completed
    """

    id: str
    tf: str
    kind: str  # "impulse_W3","impulse_W5","abc_C","impulse_W1W2", ...
    direction: int  # +1 / -1
    pivots: tuple[Pivot, ...]
    invalidation: float
    targets: dict[float, float]  # ratio -> price
    rule_score: float  # 0..1 Fib-ratio fit
    hsmm_p: float = 0.0
    weight: float = 0.0  # normalized among live hypotheses of that tf
    born_ts: int = 0
    status: str = "active"  # active|invalidated|superseded|expired|completed
    meta: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class Zone:
    """A Fibonacci confluence zone aggregating multiple levels across degrees."""

    lo: float
    hi: float
    strength: float
    n_degrees: int
    sources: tuple[str, ...]  # hypothesis ids + level labels


@dataclass(frozen=True)
class Setup:
    """A candidate trade setup generated from hypothesis + confluence zone.

    Attributes:
        ts: Timestamp (UTC ms) when the setup was generated
        features: Exactly what the meta-labeler sees — available_at <= ts
    """

    ts: int
    hyp_id: str
    side: int  # +1 long, -1 short
    entry: float
    stop: float  # = hypothesis invalidation +/- buffer
    targets: tuple[float, ...]
    zone: Zone | None
    features: dict[str, float]


@dataclass(frozen=True)
class Decision:
    """Output of the gate: approved or rejected with reasons."""

    setup: Setup
    p_success: float
    exp_r_net: float
    approved: bool
    reasons: tuple[str, ...]  # why approved or rejected (gate log)
