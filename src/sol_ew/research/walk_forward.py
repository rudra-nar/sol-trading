"""Walk-forward cross-validation with purging and embargo.

Implements purged walk-forward CV to prevent lookahead leakage in
time-series model evaluation. Each fold:
1. Train on the training window.
2. Purge: remove samples near the train/test boundary.
3. Embargo: skip a gap after purge.
4. Test on the out-of-sample window.

Fitted parameters come from the training slice only (invariant 3).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FoldResult:
    """Result of a single walk-forward fold."""

    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    n_train: int
    n_test: int
    accuracy: float
    auc: float
    log_loss_val: float
    feature_importances: dict[str, float]


@dataclass(frozen=True)
class WalkForwardResult:
    """Aggregate result of walk-forward cross-validation."""

    folds: tuple[FoldResult, ...]
    mean_accuracy: float
    mean_auc: float
    mean_log_loss: float
    std_accuracy: float
    std_auc: float


def purged_walk_forward_split(
    n_samples: int,
    n_folds: int = 5,
    train_pct: float = 0.6,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Generate purged walk-forward train/test splits.

    Args:
        n_samples: Total number of samples.
        n_folds: Number of walk-forward folds.
        train_pct: Fraction of each fold used for training.
        purge_pct: Fraction to purge from end of training.
        embargo_pct: Fraction to skip after purge.

    Returns:
        List of (train_indices, test_indices) tuples.
    """
    splits: list[tuple[np.ndarray, np.ndarray]] = []

    fold_size = n_samples // n_folds
    if fold_size < 10:
        logger.warning("Very small fold size (%d). Results may be unreliable.", fold_size)

    for i in range(n_folds):
        # Each fold shifts the window forward
        fold_start = i * (n_samples - fold_size) // max(n_folds - 1, 1)
        fold_end = min(fold_start + fold_size, n_samples)

        train_end_raw = fold_start + int((fold_end - fold_start) * train_pct)

        # Purge: remove samples near boundary
        purge_n = max(1, int(n_samples * purge_pct))
        train_end = train_end_raw - purge_n

        # Embargo: skip after purge
        embargo_n = max(1, int(n_samples * embargo_pct))
        test_start = train_end_raw + embargo_n

        test_end = fold_end

        if train_end <= fold_start or test_start >= test_end:
            continue

        train_idx = np.arange(fold_start, train_end)
        test_idx = np.arange(test_start, test_end)
        splits.append((train_idx, test_idx))

    return splits


def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_folds: int = 5,
    train_pct: float = 0.6,
    purge_pct: float = 0.01,
    embargo_pct: float = 0.01,
    n_estimators: int = 100,
    max_depth: int = 4,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> WalkForwardResult:
    """Run purged walk-forward cross-validation.

    Args:
        X: Feature matrix (n_samples, n_features).
        y: Binary labels.
        feature_names: Feature names for importance tracking.
        n_folds: Number of folds.
        train_pct: Training fraction per fold.
        purge_pct: Purge fraction.
        embargo_pct: Embargo fraction.
        n_estimators: GBM estimators.
        max_depth: GBM max depth.
        learning_rate: GBM learning rate.
        seed: Random seed.

    Returns:
        WalkForwardResult with per-fold and aggregate metrics.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

    splits = purged_walk_forward_split(
        len(X), n_folds, train_pct, purge_pct, embargo_pct,
    )

    fold_results: list[FoldResult] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        # Skip folds with no variation in labels
        if len(set(y_train)) < 2 or len(set(y_test)) < 2:
            logger.warning("Fold %d skipped: insufficient label variation", fold_idx)
            continue

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=seed + fold_idx,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        acc = float(accuracy_score(y_test, y_pred))
        auc = float(roc_auc_score(y_test, y_proba))
        ll = float(log_loss(y_test, y_proba))
        importances = dict(zip(feature_names, model.feature_importances_, strict=False))

        fold_results.append(FoldResult(
            fold_idx=fold_idx,
            train_start=int(train_idx[0]),
            train_end=int(train_idx[-1]),
            test_start=int(test_idx[0]),
            test_end=int(test_idx[-1]),
            n_train=len(train_idx),
            n_test=len(test_idx),
            accuracy=acc,
            auc=auc,
            log_loss_val=ll,
            feature_importances=importances,
        ))

        logger.info(
            "Fold %d: acc=%.3f, auc=%.3f, train=%d, test=%d",
            fold_idx, acc, auc, len(train_idx), len(test_idx),
        )

    if not fold_results:
        return WalkForwardResult(
            folds=(), mean_accuracy=0, mean_auc=0,
            mean_log_loss=0, std_accuracy=0, std_auc=0,
        )

    accs = [f.accuracy for f in fold_results]
    aucs = [f.auc for f in fold_results]
    lls = [f.log_loss_val for f in fold_results]

    result = WalkForwardResult(
        folds=tuple(fold_results),
        mean_accuracy=float(np.mean(accs)),
        mean_auc=float(np.mean(aucs)),
        mean_log_loss=float(np.mean(lls)),
        std_accuracy=float(np.std(accs)),
        std_auc=float(np.std(aucs)),
    )

    logger.info(
        "Walk-forward CV: acc=%.3f±%.3f, auc=%.3f±%.3f (%d folds)",
        result.mean_accuracy, result.std_accuracy,
        result.mean_auc, result.std_auc, len(fold_results),
    )

    return result


def save_cv_results(result: WalkForwardResult, path: Path) -> None:
    """Save CV results to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "mean_accuracy": result.mean_accuracy,
        "mean_auc": result.mean_auc,
        "mean_log_loss": result.mean_log_loss,
        "std_accuracy": result.std_accuracy,
        "std_auc": result.std_auc,
        "n_folds": len(result.folds),
        "folds": [
            {
                "fold_idx": f.fold_idx,
                "accuracy": f.accuracy,
                "auc": f.auc,
                "log_loss": f.log_loss_val,
                "n_train": f.n_train,
                "n_test": f.n_test,
                "feature_importances": f.feature_importances,
            }
            for f in result.folds
        ],
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    logger.info("CV results saved to %s", path)
