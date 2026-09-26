"""Walk-forward (expanding window) evaluation for direction classifiers."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

from .models import make_model

_METRICS = {"accuracy": accuracy_score, "balanced_accuracy": balanced_accuracy_score}


def ranking_scores(model, X) -> np.ndarray:
    """Continuous scores for AUC: probabilities if available, else margins."""
    if hasattr(model, "predict_proba"):
        try:
            return model.predict_proba(X)[:, 1]
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return model.predict(X)


def best_threshold(y_true, prob, metric: str = "accuracy", grid=None) -> float:
    """Probability cut-off that maximizes ``metric`` on the given (training) data."""
    grid = np.linspace(0.20, 0.80, 61) if grid is None else grid
    score_fn = _METRICS[metric]
    scores = [score_fn(y_true, (prob >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def walk_forward_evaluate(
    df: pd.DataFrame,
    feature_cols: list[str],
    model_name: str,
    n_splits: int = 5,
    threshold_metric: str | None = None,
    balanced: bool = False,
) -> dict:
    """Train on each expanding past window, predict the next block, pool the predictions.

    Rows are sorted by ``timestamp_utc`` so every test block lies strictly after
    its training data. With ``threshold_metric`` set, the decision threshold is
    tuned on the training slice of each fold (never on the test slice).
    """
    data = df.sort_values("timestamp_utc").reset_index(drop=True)
    y = data["target_direction"].astype(int).to_numpy()
    if feature_cols:
        X = data[feature_cols].replace([np.inf, -np.inf], np.nan)
    else:
        X = pd.DataFrame({"constant": np.zeros(len(data))})

    y_true, y_pred, y_score, thresholds = [], [], [], []
    for train_idx, test_idx in TimeSeriesSplit(n_splits=n_splits).split(X):
        y_train = y[train_idx]
        if len(np.unique(y_train)) < 2:
            continue

        model = make_model(model_name, balanced=balanced)
        model.fit(X.iloc[train_idx], y_train)
        X_test = X.iloc[test_idx]

        if model_name == "Baseline":
            pred = model.predict(X_test)
            score = pred.astype(float)
        elif threshold_metric is not None:
            train_prob = ranking_scores(model, X.iloc[train_idx])
            score = ranking_scores(model, X_test)
            t = best_threshold(y_train, train_prob, metric=threshold_metric)
            pred = (score >= t).astype(int)
            thresholds.append(t)
        else:
            pred = model.predict(X_test)
            score = ranking_scores(model, X_test)

        y_true.extend(y[test_idx])
        y_pred.extend(pred)
        y_score.extend(score)

    y_true, y_pred, y_score = map(np.asarray, (y_true, y_pred, y_score))
    auc = roc_auc_score(y_true, y_score) if len(np.unique(y_true)) == 2 else np.nan

    return {
        "model": model_name,
        "n_features": len(feature_cols),
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": auc,
        "up_rate_true": y_true.mean(),
        "up_rate_pred": y_pred.mean(),
        "avg_threshold": float(np.mean(thresholds)) if thresholds else np.nan,
        "n_eval": len(y_true),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def run_grid(
    df: pd.DataFrame,
    feature_sets: dict[str, list[str]],
    model_names: list[str],
    **kwargs,
) -> pd.DataFrame:
    """Evaluate every (feature set, model) pair; best rows first."""
    rows = []
    for set_name, cols in feature_sets.items():
        if not cols:
            continue
        for model_name in model_names:
            row = walk_forward_evaluate(df, cols, model_name, **kwargs)
            row["feature_set"] = set_name
            rows.append(row)
    results = pd.DataFrame(rows)
    ranked = results.sort_values(["auc", "balanced_accuracy", "accuracy"], ascending=False)
    return ranked.reset_index(drop=True)
