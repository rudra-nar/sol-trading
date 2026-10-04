"""Meta-labeler — classical ML model for probability estimation.

Uses gradient-boosted trees (LightGBM) to predict the probability that
a setup will hit its target. Trained on labeled data from research/labels.py.

Features (all available_at <= setup.ts):
- rule_score: hypothesis Fib-ratio fit score
- zone_strength: confluence zone strength
- n_degrees: number of degrees contributing to the zone
- hsmm_p: HSMM state probability
- atr_ratio: current ATR / historical ATR
- volume_ratio: current volume / average volume
- bars_since_pivot: bars since last pivot confirmation
- trend_alignment: degree trend agreement score

Fitted parameters come from the TRAINING slice only (invariant 3).
"""

from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_NAMES = [
    "rule_score",
    "zone_strength",
    "n_degrees",
    "hsmm_p",
    "weight",
    "atr_ratio",
    "volume_ratio",
    "bars_since_pivot",
]


@dataclass
class MetaLabeler:
    """Gradient-boosted meta-labeler for trade setup probability.

    Uses a simple sklearn GradientBoostingClassifier (no LightGBM dependency
    required) for portability. Can be swapped to LightGBM for production.

    Attributes:
        model: The fitted sklearn model (None before training).
        threshold: Minimum predicted probability to consider a setup.
        feature_names: Ordered list of feature names.
    """

    threshold: float = 0.5
    feature_names: list[str] = field(default_factory=lambda: list(FEATURE_NAMES))
    _model: object = field(default=None, repr=False)
    _is_fitted: bool = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 4,
        learning_rate: float = 0.1,
        seed: int = 42,
    ) -> dict[str, float]:
        """Train the meta-labeler on labeled data.

        Args:
            X: Feature matrix (n_samples, n_features).
            y: Binary labels (1=win, 0=loss).
            n_estimators: Number of boosting rounds.
            max_depth: Max tree depth.
            learning_rate: Learning rate.
            seed: Random seed for determinism.

        Returns:
            Dict of training metrics.
        """
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import (
            accuracy_score,
            log_loss,
            roc_auc_score,
        )

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed,
        )
        model.fit(X, y)
        self._model = model
        self._is_fitted = True

        # Training metrics
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        metrics = {
            "accuracy": float(accuracy_score(y, y_pred)),
            "auc": float(roc_auc_score(y, y_proba)) if len(set(y)) > 1 else 0.0,
            "log_loss": float(log_loss(y, y_proba)),
            "n_samples": len(y),
            "n_positive": int(sum(y)),
            "n_features": int(X.shape[1]),
        }

        logger.info(
            "MetaLabeler trained: acc=%.3f, auc=%.3f, n=%d",
            metrics["accuracy"], metrics["auc"], metrics["n_samples"],
        )
        return metrics

    def predict_proba(self, features: dict[str, float]) -> float:
        """Predict probability of success for a single setup.

        Args:
            features: Feature dict (must contain all feature_names).

        Returns:
            Probability of success (0 to 1).
        """
        if not self._is_fitted:
            return 0.5  # Prior when not trained

        X = np.array([[features.get(f, 0.0) for f in self.feature_names]])
        proba = self._model.predict_proba(X)[0, 1]  # type: ignore[union-attr]
        return float(proba)

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for a batch of feature rows.

        Args:
            X: Feature matrix (n_samples, n_features).

        Returns:
            Array of probabilities.
        """
        if not self._is_fitted:
            return np.full(len(X), 0.5)
        return self._model.predict_proba(X)[:, 1]  # type: ignore[union-attr]

    @property
    def feature_importances(self) -> dict[str, float]:
        """Feature importances from the fitted model."""
        if not self._is_fitted:
            return {}
        importances = self._model.feature_importances_  # type: ignore[union-attr]
        return dict(zip(self.feature_names, importances, strict=False))

    def save(self, path: Path) -> None:
        """Save model to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "model": self._model,
                "threshold": self.threshold,
                "feature_names": self.feature_names,
                "is_fitted": self._is_fitted,
            }, f)
        logger.info("MetaLabeler saved to %s", path)

    def load(self, path: Path) -> None:
        """Load model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._model = data["model"]
        self.threshold = data["threshold"]
        self.feature_names = data["feature_names"]
        self._is_fitted = data["is_fitted"]
        logger.info("MetaLabeler loaded from %s", path)

    def snapshot(self) -> dict[str, object]:
        """Capture state."""
        return {
            "is_fitted": self._is_fitted,
            "threshold": self.threshold,
            "n_features": len(self.feature_names),
        }
