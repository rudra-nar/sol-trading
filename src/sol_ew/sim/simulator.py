"""Fill simulator for paper trading.

Tracks open positions, applies fees (maker/taker), funding (8h snapshots),
slippage model, stop-loss/take-profit execution against bar high/low,
and equity curve / drawdown tracking.

PAPER TRADING ONLY -- never places real orders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from sol_ew.core.types import Bar

logger = logging.getLogger(__name__)


class Side(str, Enum):
    """Trade side."""
    LONG = "LONG"
    SHORT = "SHORT"


class OrderType(str, Enum):
    """Order type."""
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"


@dataclass
class Order:
    """A pending or filled order."""
    order_id: str
    side: Side
    qty: float
    price: float  # limit/stop price; 0 for market
    order_type: OrderType
    reduce_only: bool = False


@dataclass
class Fill:
    """Record of a fill execution."""
    order_id: str
    fill_ts: int       # UTC ms
    side: Side
    qty: float
    price: float       # actual fill price (after slippage)
    fee: float          # fee charged
    is_maker: bool
    reason: str = ""    # "entry", "stop_loss", "take_profit", "liquidation"


@dataclass
class Position:
    """An open position."""
    side: Side
    qty: float
    entry_price: float
    entry_ts: int
    stop_loss: float = 0.0
    take_profit: float = 0.0
    unrealized_pnl: float = 0.0
    funding_paid: float = 0.0


@dataclass
class EquityPoint:
    """A point on the equity curve."""
    ts: int
    equity: float
    drawdown: float
    position_value: float


class Simulator:
    """Deterministic paper-trading fill simulator.

    Processes bars and executes orders with realistic fills, fees,
    slippage, and funding payments. Tracks equity curve and drawdown.

    All state is explicit (snapshot/restore for replayability).
    Same code live and in replay.
    """

    def __init__(
        self,
        initial_capital: float = 10_000.0,
        fee_maker: float = 0.0002,
        fee_taker: float = 0.0005,
        slippage_bps: float = 1.0,
        funding_interval_ms: int = 8 * 3_600_000,  # 8 hours
        seed: int = 42,
    ) -> None:
        self._capital = initial_capital
        self._initial_capital = initial_capital
        self._fee_maker = fee_maker
        self._fee_taker = fee_taker
        self._slippage_bps = slippage_bps
        self._funding_interval_ms = funding_interval_ms
        self._seed = seed

        # State
        self._cash: float = initial_capital
        self._position: Position | None = None
        self._pending_orders: list[Order] = []
        self._pending_stop_loss: float = 0.0
        self._pending_take_profit: float = 0.0
        self._fills: list[Fill] = []
        self._equity_curve: list[EquityPoint] = []
        self._peak_equity: float = initial_capital
        self._last_funding_ts: int = 0
        self._total_fees: float = 0.0
        self._total_funding: float = 0.0
        self._trade_count: int = 0

    # ------------------------------------------------------------------
    # Order placement
    # ------------------------------------------------------------------

    def submit_market_order(
        self,
        order_id: str,
        side: Side,
        qty: float,
        reduce_only: bool = False,
    ) -> Order:
        """Submit a market order for immediate fill on next bar."""
        order = Order(
            order_id=order_id, side=side, qty=qty, price=0.0,
            order_type=OrderType.MARKET, reduce_only=reduce_only,
        )
        self._pending_orders.append(order)
        return order

    def submit_limit_order(
        self,
        order_id: str,
        side: Side,
        qty: float,
        price: float,
        reduce_only: bool = False,
    ) -> Order:
        """Submit a limit order."""
        order = Order(
            order_id=order_id, side=side, qty=qty, price=price,
            order_type=OrderType.LIMIT, reduce_only=reduce_only,
        )
        self._pending_orders.append(order)
        return order

    def set_stop_loss(self, price: float) -> None:
        """Set or update stop-loss for current position or pending fill."""
        self._pending_stop_loss = price
        if self._position is not None:
            self._position.stop_loss = price

    def set_take_profit(self, price: float) -> None:
        """Set or update take-profit for current position or pending fill."""
        self._pending_take_profit = price
        if self._position is not None:
            self._position.take_profit = price

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order by ID."""
        for i, o in enumerate(self._pending_orders):
            if o.order_id == order_id:
                self._pending_orders.pop(i)
                return True
        return False

    # ------------------------------------------------------------------
    # Bar processing
    # ------------------------------------------------------------------

    def on_bar(self, bar: Bar) -> list[Fill]:
        """Process a closed bar: check stops/TP, fill pending orders, apply funding.

        Order of operations within a bar:
        1. Check stop-loss against bar low (long) / bar high (short)
        2. Check take-profit against bar high (long) / bar low (short)
        3. Fill pending limit orders if price touched
        4. Fill pending market orders at bar open + slippage
        5. Apply funding if crossing an 8h boundary
        6. Update equity curve

        Args:
            bar: A closed bar.

        Returns:
            List of fills that occurred on this bar.
        """
        bar_fills: list[Fill] = []

        # 1. Check stop-loss first (worst case first)
        if self._position is not None and self._position.stop_loss > 0:
            sl_fill = self._check_stop_loss(bar)
            if sl_fill is not None:
                bar_fills.append(sl_fill)

        # 2. Check take-profit
        if self._position is not None and self._position.take_profit > 0:
            tp_fill = self._check_take_profit(bar)
            if tp_fill is not None:
                bar_fills.append(tp_fill)

        # 3. Fill limit orders
        for order in list(self._pending_orders):
            if order.order_type == OrderType.LIMIT:
                fill = self._try_limit_fill(order, bar)
                if fill is not None:
                    bar_fills.append(fill)
                    self._pending_orders.remove(order)

        # 4. Fill market orders at bar open
        for order in list(self._pending_orders):
            if order.order_type == OrderType.MARKET:
                fill = self._fill_market_order(order, bar)
                bar_fills.append(fill)
                self._pending_orders.remove(order)

        # 5. Apply funding
        self._apply_funding(bar)

        # 6. Update equity
        self._update_equity(bar)

        return bar_fills

    # ------------------------------------------------------------------
    # Fill execution (private)
    # ------------------------------------------------------------------

    def _apply_slippage(self, price: float, side: Side) -> float:
        """Apply slippage to a fill price.

        Long buys slip up, short sells slip down (adversarial).
        """
        slip = price * self._slippage_bps / 10_000
        if side == Side.LONG:
            return price + slip
        return price - slip

    def _compute_fee(self, qty: float, price: float, is_maker: bool) -> float:
        """Compute fee for a fill."""
        rate = self._fee_maker if is_maker else self._fee_taker
        return qty * price * rate

    def _fill_market_order(self, order: Order, bar: Bar) -> Fill:
        """Fill a market order at bar open + slippage."""
        fill_price = self._apply_slippage(bar.o, order.side)
        fee = self._compute_fee(order.qty, fill_price, is_maker=False)
        self._execute_fill(order, fill_price, fee, bar.open_ts)

        return Fill(
            order_id=order.order_id,
            fill_ts=bar.open_ts,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=fee,
            is_maker=False,
            reason="entry",
        )

    def _try_limit_fill(self, order: Order, bar: Bar) -> Fill | None:
        """Try to fill a limit order if bar touched the limit price."""
        if (order.side == Side.LONG and bar.l <= order.price) or (order.side == Side.SHORT and bar.h >= order.price):
            fill_price = order.price
        else:
            return None

        fee = self._compute_fee(order.qty, fill_price, is_maker=True)
        self._execute_fill(order, fill_price, fee, bar.open_ts)

        return Fill(
            order_id=order.order_id,
            fill_ts=bar.open_ts,
            side=order.side,
            qty=order.qty,
            price=fill_price,
            fee=fee,
            is_maker=True,
            reason="entry",
        )

    def _check_stop_loss(self, bar: Bar) -> Fill | None:
        """Check if stop-loss was hit on this bar."""
        pos = self._position
        if pos is None:
            return None

        hit = False
        # For long: stop hit if bar low <= stop price
        if pos.side == Side.LONG and bar.l <= pos.stop_loss:
            hit = True
            # Gap-through: fill at worst of stop price or bar open
            fill_price = min(pos.stop_loss, bar.o)
        # For short: stop hit if bar high >= stop price
        elif pos.side == Side.SHORT and bar.h >= pos.stop_loss:
            hit = True
            fill_price = max(pos.stop_loss, bar.o)
        else:
            return None

        if not hit:
            return None

        fill_price = self._apply_slippage(fill_price, self._close_side(pos.side))
        fee = self._compute_fee(pos.qty, fill_price, is_maker=False)
        self._close_position(fill_price, fee, bar.close_ts)

        return Fill(
            order_id=f"SL-{self._trade_count}",
            fill_ts=bar.close_ts,
            side=self._close_side(pos.side),
            qty=pos.qty,
            price=fill_price,
            fee=fee,
            is_maker=False,
            reason="stop_loss",
        )

    def _check_take_profit(self, bar: Bar) -> Fill | None:
        """Check if take-profit was hit on this bar."""
        pos = self._position
        if pos is None:
            return None

        # For long: TP hit if bar high >= TP price
        if (pos.side == Side.LONG and bar.h >= pos.take_profit) or (pos.side == Side.SHORT and bar.l <= pos.take_profit):
            fill_price = pos.take_profit
        else:
            return None

        fee = self._compute_fee(pos.qty, fill_price, is_maker=True)
        self._close_position(fill_price, fee, bar.close_ts)

        return Fill(
            order_id=f"TP-{self._trade_count}",
            fill_ts=bar.close_ts,
            side=self._close_side(pos.side),
            qty=pos.qty,
            price=fill_price,
            fee=fee,
            is_maker=True,
            reason="take_profit",
        )

    def _execute_fill(self, order: Order, price: float, fee: float, ts: int) -> None:
        """Execute a fill: update cash and position."""
        self._cash -= fee
        self._total_fees += fee
        self._fills.append(Fill(
            order_id=order.order_id, fill_ts=ts,
            side=order.side, qty=order.qty, price=price,
            fee=fee, is_maker=(order.order_type == OrderType.LIMIT),
        ))

        if order.reduce_only and self._position is not None:
            self._close_position(price, 0, ts)
            return

        if self._position is None:
            self._position = Position(
                side=order.side, qty=order.qty,
                entry_price=price, entry_ts=ts,
                stop_loss=self._pending_stop_loss,
                take_profit=self._pending_take_profit,
            )
            self._trade_count += 1
            self._pending_stop_loss = 0.0
            self._pending_take_profit = 0.0
        elif self._position.side == order.side:
            # Add to position (average entry)
            total_qty = self._position.qty + order.qty
            avg_price = (
                (self._position.entry_price * self._position.qty + price * order.qty)
                / total_qty
            )
            self._position.qty = total_qty
            self._position.entry_price = avg_price
        else:
            # Opposite side: reduce or flip
            if order.qty >= self._position.qty:
                self._close_position(price, 0, ts)
                remainder = order.qty - self._position.qty if self._position else order.qty
                if remainder > 0 and self._position is None:
                    self._position = Position(
                        side=order.side, qty=remainder,
                        entry_price=price, entry_ts=ts,
                        stop_loss=self._pending_stop_loss,
                        take_profit=self._pending_take_profit,
                    )
                    self._pending_stop_loss = 0.0
                    self._pending_take_profit = 0.0
            else:
                self._position.qty -= order.qty
                pnl = self._compute_pnl(
                    self._position.side, order.qty,
                    self._position.entry_price, price,
                )
                self._cash += pnl

    def _close_position(self, price: float, fee: float, ts: int) -> None:
        """Close the current position and realize PnL."""
        if self._position is None:
            return

        pnl = self._compute_pnl(
            self._position.side, self._position.qty,
            self._position.entry_price, price,
        )
        self._cash += pnl - fee
        self._total_fees += fee

        logger.info(
            "Position closed: %s %.4f @ %.2f -> %.2f | PnL=%.2f | Fee=%.4f",
            self._position.side.value, self._position.qty,
            self._position.entry_price, price, pnl, fee,
        )
        self._position = None
        self._pending_stop_loss = 0.0
        self._pending_take_profit = 0.0

    @staticmethod
    def _compute_pnl(side: Side, qty: float, entry: float, exit_price: float) -> float:
        """Compute PnL for a position."""
        if side == Side.LONG:
            return qty * (exit_price - entry)
        return qty * (entry - exit_price)

    @staticmethod
    def _close_side(side: Side) -> Side:
        """Get the closing side for a position."""
        return Side.SHORT if side == Side.LONG else Side.LONG

    # ------------------------------------------------------------------
    # Funding
    # ------------------------------------------------------------------

    def _apply_funding(self, bar: Bar) -> None:
        """Apply funding payments at 8h intervals.

        Funding = position_size * mark_price * funding_rate.
        Uses a default rate of 0.01% per 8h interval (can be overridden).
        """
        if self._position is None:
            return

        if self._last_funding_ts == 0:
            self._last_funding_ts = bar.open_ts
            return

        # Check if we've crossed a funding boundary
        elapsed = bar.close_ts - self._last_funding_ts
        if elapsed < self._funding_interval_ms:
            return

        # Number of funding periods crossed
        periods = elapsed // self._funding_interval_ms
        self._last_funding_ts += periods * self._funding_interval_ms

        # Default funding rate (can be parameterized later with recorded data)
        funding_rate = 0.0001  # 0.01% per 8h

        for _ in range(periods):
            payment = self._position.qty * bar.c * funding_rate
            # Longs pay when rate is positive, shorts receive
            if self._position.side == Side.LONG:
                self._cash -= payment
                self._position.funding_paid += payment
            else:
                self._cash += payment
                self._position.funding_paid -= payment
            self._total_funding += payment

    # ------------------------------------------------------------------
    # Equity tracking
    # ------------------------------------------------------------------

    def _update_equity(self, bar: Bar) -> None:
        """Update equity curve after processing a bar."""
        equity = self._cash
        position_value = 0.0

        if self._position is not None:
            unrealized = self._compute_pnl(
                self._position.side, self._position.qty,
                self._position.entry_price, bar.c,
            )
            self._position.unrealized_pnl = unrealized
            equity += unrealized
            position_value = self._position.qty * bar.c

        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (self._peak_equity - equity) / self._peak_equity if self._peak_equity > 0 else 0.0

        self._equity_curve.append(EquityPoint(
            ts=bar.close_ts,
            equity=equity,
            drawdown=drawdown,
            position_value=position_value,
        ))

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def equity(self) -> float:
        if self._equity_curve:
            return self._equity_curve[-1].equity
        return self._cash

    @property
    def position(self) -> Position | None:
        return self._position

    @property
    def fills(self) -> list[Fill]:
        return list(self._fills)

    @property
    def equity_curve(self) -> list[EquityPoint]:
        return list(self._equity_curve)

    @property
    def max_drawdown(self) -> float:
        if not self._equity_curve:
            return 0.0
        return max(ep.drawdown for ep in self._equity_curve)

    @property
    def total_fees(self) -> float:
        return self._total_fees

    @property
    def total_funding(self) -> float:
        return self._total_funding

    @property
    def trade_count(self) -> int:
        return self._trade_count

    @property
    def pending_orders(self) -> list[Order]:
        return list(self._pending_orders)

    # ------------------------------------------------------------------
    # Snapshot / Restore
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, object]:
        """Capture full state for replayability."""
        return {
            "cash": self._cash,
            "position": {
                "side": self._position.side.value,
                "qty": self._position.qty,
                "entry_price": self._position.entry_price,
                "entry_ts": self._position.entry_ts,
                "stop_loss": self._position.stop_loss,
                "take_profit": self._position.take_profit,
                "funding_paid": self._position.funding_paid,
            } if self._position else None,
            "pending_stop_loss": self._pending_stop_loss,
            "pending_take_profit": self._pending_take_profit,
            "peak_equity": self._peak_equity,
            "last_funding_ts": self._last_funding_ts,
            "total_fees": self._total_fees,
            "total_funding": self._total_funding,
            "trade_count": self._trade_count,
        }

    def restore(self, state: dict[str, object]) -> None:
        """Restore from snapshot."""
        self._cash = float(state["cash"])  # type: ignore[arg-type]
        self._peak_equity = float(state["peak_equity"])  # type: ignore[arg-type]
        self._last_funding_ts = int(state["last_funding_ts"])  # type: ignore[arg-type]
        self._total_fees = float(state["total_fees"])  # type: ignore[arg-type]
        self._total_funding = float(state["total_funding"])  # type: ignore[arg-type]
        self._trade_count = int(state["trade_count"])  # type: ignore[arg-type]
        self._pending_stop_loss = float(state.get("pending_stop_loss", 0.0))  # type: ignore[arg-type]
        self._pending_take_profit = float(state.get("pending_take_profit", 0.0))  # type: ignore[arg-type]

        pos_data = state.get("position")
        if pos_data is not None:
            self._position = Position(
                side=Side(pos_data["side"]),  # type: ignore[index]
                qty=float(pos_data["qty"]),  # type: ignore[index]
                entry_price=float(pos_data["entry_price"]),  # type: ignore[index]
                entry_ts=int(pos_data["entry_ts"]),  # type: ignore[index]
                stop_loss=float(pos_data.get("stop_loss", 0)),  # type: ignore[union-attr]
                take_profit=float(pos_data.get("take_profit", 0)),  # type: ignore[union-attr]
                funding_paid=float(pos_data.get("funding_paid", 0)),  # type: ignore[union-attr]
            )
        else:
            self._position = None
