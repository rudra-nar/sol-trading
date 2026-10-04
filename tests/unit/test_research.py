"""Tests for research labels, meta-labeler, and walk-forward CV."""

from __future__ import annotations

import numpy as np

from sol_ew.core.types import Bar
from sol_ew.research.labels import Label, label_series, label_trade
from sol_ew.research.meta_labeler import MetaLabeler
from sol_ew.research.walk_forward import (
    purged_walk_forward_split,
    walk_forward_cv,
)


def _bar(idx: int, h: float, l: float, c: float, tf: str = "1h") -> Bar:
    return Bar(
        tf=tf, open_ts=idx * 3600000, close_ts=(idx + 1) * 3600000 - 1,
        o=c, h=h, l=l, c=c,
        volume=1000, taker_buy_volume=500, n_trades=100,
    )


# ============================================================================
# Phase 8: Labels
# ============================================================================


class TestLabelTrade:
    """Test the label_trade function."""

    def test_long_win(self) -> None:
        """Long trade that hits target."""
        future = [_bar(i, h=105 + i * 2, l=99, c=103 + i) for i in range(5)]
        result = label_trade(0, 100.0, 95.0, 110.0, 1, future)
        assert result.label == Label.WIN
        assert result.exit_price == 110.0
        assert result.r_multiple > 0

    def test_long_loss(self) -> None:
        """Long trade that hits stop."""
        future = [_bar(i, h=101, l=90 - i * 2, c=95 - i) for i in range(5)]
        result = label_trade(0, 100.0, 95.0, 110.0, 1, future)
        assert result.label == Label.LOSS
        assert result.exit_price == 95.0
        assert result.r_multiple < 0

    def test_short_win(self) -> None:
        """Short trade that hits target."""
        future = [_bar(i, h=101, l=95 - i * 2, c=97 - i) for i in range(5)]
        result = label_trade(0, 100.0, 105.0, 90.0, -1, future)
        assert result.label == Label.WIN
        assert result.exit_price == 90.0

    def test_short_loss(self) -> None:
        """Short trade that hits stop."""
        future = [_bar(i, h=102 + i * 2, l=99, c=101 + i) for i in range(5)]
        result = label_trade(0, 100.0, 105.0, 90.0, -1, future)
        assert result.label == Label.LOSS
        assert result.exit_price == 105.0

    def test_neutral_timeout(self) -> None:
        """Trade that neither hits target nor stop within max_hold."""
        future = [_bar(i, h=101, l=99, c=100) for i in range(10)]
        result = label_trade(0, 100.0, 90.0, 120.0, 1, future, max_hold_bars=5)
        assert result.label == Label.NEUTRAL
        assert result.bars_held == 5

    def test_mfe_mae_tracked(self) -> None:
        """Max favorable and adverse excursions are tracked."""
        future = [
            _bar(0, h=108, l=97, c=105),  # MFE=8, MAE=3
            _bar(1, h=106, l=94, c=95),   # MAE=6
            _bar(2, h=104, l=89, c=90),   # Stop hit at 90
        ]
        result = label_trade(0, 100.0, 90.0, 115.0, 1, future)
        assert result.max_favorable >= 8.0
        assert result.max_adverse >= 3.0

    def test_zero_risk(self) -> None:
        """Zero risk (stop == entry) returns NEUTRAL."""
        result = label_trade(0, 100.0, 100.0, 110.0, 1, [])
        assert result.label == Label.NEUTRAL

    def test_empty_future(self) -> None:
        """No future bars returns NEUTRAL."""
        result = label_trade(0, 100.0, 95.0, 110.0, 1, [])
        assert result.label == Label.NEUTRAL


class TestLabelSeries:
    """Test batch labeling."""

    def test_batch_labeling(self) -> None:
        bars = [_bar(i, h=100 + i, l=95 + i, c=98 + i) for i in range(20)]
        setups = [
            {"bar_idx": 2, "entry_price": 100, "stop": 90, "target": 115, "side": 1},
            {"bar_idx": 5, "entry_price": 105, "stop": 95, "target": 120, "side": 1},
        ]
        results = label_series(setups, bars)
        assert len(results) == 2
        for r in results:
            assert r.label in (Label.WIN, Label.LOSS, Label.NEUTRAL)


# ============================================================================
# Phase 9: Meta-labeler
# ============================================================================


class TestMetaLabeler:
    """Test the meta-labeler."""

    def _make_data(self, n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data."""
        rng = np.random.RandomState(seed)
        X = rng.randn(n, 8)
        # Label based on first two features (so model can learn)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        return X, y

    def test_unfitted_returns_prior(self) -> None:
        ml = MetaLabeler()
        features = {f: 0.5 for f in ml.feature_names}
        assert ml.predict_proba(features) == 0.5

    def test_fit_and_predict(self) -> None:
        ml = MetaLabeler()
        X, y = self._make_data()
        metrics = ml.fit(X, y)
        assert metrics["accuracy"] > 0.5
        assert metrics["auc"] > 0.5
        assert metrics["n_samples"] == 200

        # Predict on a single sample
        features = dict(zip(ml.feature_names, X[0], strict=False))
        p = ml.predict_proba(features)
        assert 0 <= p <= 1

    def test_batch_predict(self) -> None:
        ml = MetaLabeler()
        X, y = self._make_data()
        ml.fit(X, y)
        probs = ml.predict_batch(X[:10])
        assert len(probs) == 10
        assert all(0 <= p <= 1 for p in probs)

    def test_feature_importances(self) -> None:
        ml = MetaLabeler()
        X, y = self._make_data()
        ml.fit(X, y)
        imp = ml.feature_importances
        assert len(imp) == 8
        assert sum(imp.values()) > 0

    def test_save_load(self, tmp_path: object) -> None:
        from pathlib import Path
        save_dir = Path(str(tmp_path))
        ml = MetaLabeler()
        X, y = self._make_data()
        ml.fit(X, y)
        p_before = ml.predict_proba(dict(zip(ml.feature_names, X[0], strict=False)))

        ml.save(save_dir / "model.pkl")

        ml2 = MetaLabeler()
        ml2.load(save_dir / "model.pkl")
        p_after = ml2.predict_proba(dict(zip(ml2.feature_names, X[0], strict=False)))

        assert abs(p_before - p_after) < 1e-6

    def test_snapshot(self) -> None:
        ml = MetaLabeler()
        snap = ml.snapshot()
        assert snap["is_fitted"] is False

        X, y = self._make_data(50)
        ml.fit(X, y)
        snap = ml.snapshot()
        assert snap["is_fitted"] is True


# ============================================================================
# Phase 10: Walk-forward CV
# ============================================================================


class TestWalkForwardSplit:
    """Test purged walk-forward splitting."""

    def test_no_overlap(self) -> None:
        """Train and test indices must not overlap."""
        splits = purged_walk_forward_split(100, n_folds=3)
        for train_idx, test_idx in splits:
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Overlap: {overlap}"

    def test_purge_gap(self) -> None:
        """There should be a gap between train end and test start."""
        splits = purged_walk_forward_split(1000, n_folds=3, purge_pct=0.02, embargo_pct=0.02)
        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                gap = test_idx[0] - train_idx[-1]
                assert gap > 1, f"No purge/embargo gap: {gap}"

    def test_temporal_order(self) -> None:
        """Train indices should come before test indices."""
        splits = purged_walk_forward_split(500, n_folds=4)
        for train_idx, test_idx in splits:
            if len(train_idx) > 0 and len(test_idx) > 0:
                assert train_idx[-1] < test_idx[0]

    def test_produces_folds(self) -> None:
        splits = purged_walk_forward_split(200, n_folds=5)
        assert len(splits) >= 2  # At least some folds should be valid


class TestWalkForwardCV:
    """Test the full walk-forward CV pipeline."""

    def test_cv_runs(self) -> None:
        """CV should run and produce results."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        features = ["f1", "f2", "f3", "f4"]

        result = walk_forward_cv(X, y, features, n_folds=3, seed=42)
        assert len(result.folds) >= 1
        assert 0 <= result.mean_accuracy <= 1
        assert 0 <= result.mean_auc <= 1

    def test_cv_above_random(self) -> None:
        """With learnable signal, CV should beat random."""
        rng = np.random.RandomState(123)
        X = rng.randn(300, 4)
        # Strong signal in first feature
        y = (X[:, 0] > 0).astype(int)
        features = ["f1", "f2", "f3", "f4"]

        result = walk_forward_cv(X, y, features, n_folds=3, seed=123)
        assert result.mean_auc > 0.55, f"AUC {result.mean_auc} not above random"
