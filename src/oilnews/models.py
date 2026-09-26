"""Model factory. Names encode hyperparameters, e.g. ``LinearSVC_C0.05`` or ``XGB_d2_lr0.03_n100``."""

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _scaled(clf) -> Pipeline:
    # Imputer and scaler live inside the pipeline so each walk-forward fold fits
    # them on its own training slice only.
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def _unscaled(clf) -> Pipeline:
    return Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])


def _param(name: str, key: str) -> float:
    return float(name.split(key)[-1])


def make_model(name: str, balanced: bool = False):
    """Build an unfitted estimator from its name.

    ``balanced`` switches logistic regression to ``class_weight="balanced"``.
    """
    class_weight = "balanced" if balanced else None

    if name == "Baseline":
        return DummyClassifier(strategy="most_frequent")
    if name.startswith("Logit_L1"):
        return _scaled(
            LogisticRegression(
                penalty="l1",
                C=_param(name, "_C"),
                solver="liblinear",
                max_iter=5000,
                class_weight=class_weight,
            )
        )
    if name.startswith("Logit_L2"):
        return _scaled(
            LogisticRegression(
                penalty="l2",
                C=_param(name, "_C"),
                solver="lbfgs",
                max_iter=5000,
                class_weight=class_weight,
            )
        )
    if name.startswith("LinearSVC"):
        return _scaled(LinearSVC(C=_param(name, "_C"), max_iter=10000, random_state=42))
    if name.startswith("Ridge"):
        return _scaled(RidgeClassifier(alpha=_param(name, "_alpha")))
    if name == "RF":
        return _unscaled(
            RandomForestClassifier(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=5,
                max_features="sqrt",
                random_state=42,
                n_jobs=-1,
            )
        )
    if name.startswith("XGB"):
        from xgboost import XGBClassifier  # optional dependency

        _, depth, lr, n_est = name.split("_")
        return _unscaled(
            XGBClassifier(
                n_estimators=int(n_est[1:]),
                max_depth=int(depth[1:]),
                learning_rate=float(lr[2:]),
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
            )
        )
    if name == "LGBM":
        from lightgbm import LGBMClassifier  # optional dependency

        return _unscaled(
            LGBMClassifier(
                n_estimators=150,
                max_depth=2,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        )
    raise ValueError(f"unknown model: {name}")


def linear_grid() -> list[str]:
    """The regularized linear models compared in the final ablation."""
    names = []
    for c in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0]:
        names += [f"Logit_L1_C{c}", f"Logit_L2_C{c}"]
    names += [f"LinearSVC_C{c}" for c in [0.01, 0.05, 0.1, 0.5]]
    names += [f"Ridge_alpha{a}" for a in [0.5, 1.0, 2.0, 5.0, 10.0]]
    return names
