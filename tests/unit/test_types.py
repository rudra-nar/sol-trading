"""Tests for core types."""

from sol_ew.core.types import Bar, Decision, Hypothesis, Pivot, Setup, Zone


class TestBar:
    """Test Bar data contract."""

    def test_bar_creation(self) -> None:
        bar = Bar(
            tf="5m",
            open_ts=1000000,
            close_ts=1300000,
            o=100.0,
            h=105.0,
            l=98.0,
            c=103.0,
            volume=1000.0,
            taker_buy_volume=600.0,
            n_trades=500,
        )
        assert bar.tf == "5m"
        assert bar.degraded is False
        assert bar.l <= bar.o <= bar.h
        assert bar.l <= bar.c <= bar.h

    def test_bar_is_frozen(self) -> None:
        bar = Bar(
            tf="1m", open_ts=0, close_ts=60000,
            o=1.0, h=2.0, l=0.5, c=1.5,
            volume=100.0, taker_buy_volume=50.0, n_trades=10,
        )
        try:
            bar.o = 999.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass

    def test_bar_degraded(self) -> None:
        bar = Bar(
            tf="1m", open_ts=0, close_ts=60000,
            o=1.0, h=2.0, l=0.5, c=1.5,
            volume=100.0, taker_buy_volume=50.0, n_trades=10,
            degraded=True,
        )
        assert bar.degraded is True


class TestPivot:
    """Test Pivot data contract."""

    def test_pivot_creation(self) -> None:
        p = Pivot(
            tf="1h", idx=10, ts=1000000,
            confirm_idx=15, confirm_ts=1500000,
            price=150.0, kind=1,
        )
        assert p.kind == 1
        assert p.confirm_ts > p.ts  # confirmation always after the extreme

    def test_pivot_is_frozen(self) -> None:
        p = Pivot(tf="1h", idx=0, ts=0, confirm_idx=5, confirm_ts=5000, price=1.0, kind=-1)
        try:
            p.price = 999.0  # type: ignore[misc]
            raise AssertionError("Should be frozen")
        except AttributeError:
            pass


class TestHypothesis:
    """Test Hypothesis data contract."""

    def test_hypothesis_defaults(self) -> None:
        h = Hypothesis(
            id="1h:1000:impulse_W3",
            tf="1h",
            kind="impulse_W3",
            direction=1,
            pivots=(),
            invalidation=90.0,
            targets={1.618: 200.0},
            rule_score=0.85,
        )
        assert h.status == "active"
        assert h.hsmm_p == 0.0
        assert h.weight == 0.0
        assert h.born_ts == 0
        assert h.meta == {}

    def test_hypothesis_is_mutable(self) -> None:
        h = Hypothesis(
            id="test", tf="1h", kind="abc_C", direction=-1,
            pivots=(), invalidation=200.0, targets={}, rule_score=0.5,
        )
        h.status = "invalidated"
        h.weight = 0.7
        assert h.status == "invalidated"
        assert h.weight == 0.7


class TestZone:
    """Test Zone data contract."""

    def test_zone_creation(self) -> None:
        z = Zone(lo=99.0, hi=101.0, strength=3.5, n_degrees=2, sources=("h1", "h2"))
        assert z.hi > z.lo
        assert z.n_degrees == 2


class TestSetup:
    """Test Setup data contract."""

    def test_setup_creation(self) -> None:
        s = Setup(
            ts=5000000,
            hyp_id="1h:1000:impulse_W3",
            side=1,
            entry=100.0,
            stop=95.0,
            targets=(110.0, 115.0),
            zone=None,
            features={"weight": 0.8, "rule_score": 0.9},
        )
        assert s.side == 1
        assert len(s.targets) == 2


class TestDecision:
    """Test Decision data contract."""

    def test_decision_approved(self) -> None:
        setup = Setup(
            ts=5000000, hyp_id="test", side=1, entry=100.0,
            stop=95.0, targets=(110.0,), zone=None, features={},
        )
        d = Decision(
            setup=setup, p_success=0.65, exp_r_net=1.2,
            approved=True, reasons=("all checks passed",),
        )
        assert d.approved is True

    def test_decision_rejected(self) -> None:
        setup = Setup(
            ts=5000000, hyp_id="test", side=-1, entry=100.0,
            stop=105.0, targets=(90.0,), zone=None, features={},
        )
        d = Decision(
            setup=setup, p_success=0.3, exp_r_net=-0.5,
            approved=False, reasons=("leverage exceeds max", "model score below threshold"),
        )
        assert d.approved is False
        assert len(d.reasons) == 2
