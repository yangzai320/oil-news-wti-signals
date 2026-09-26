import numpy as np
import pandas as pd
import pytest

from oilnews import evaluation
from oilnews.evaluation import best_threshold, walk_forward_evaluate
from oilnews.models import linear_grid, make_model


def _synthetic(n=300, signal=2.0, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = (signal * x + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "timestamp_utc": pd.date_range("2026-01-01", periods=n, freq="4h", tz="UTC"),
            "signal": x,
            "noise": rng.normal(size=n),
            "target_direction": y,
        }
    )


def test_learns_an_informative_feature():
    res = walk_forward_evaluate(_synthetic(), ["signal"], "LinearSVC_C0.05")
    assert res["auc"] > 0.85
    assert res["n_eval"] == 300 - 300 // 6  # first block is training-only


def test_noise_feature_is_near_chance():
    res = walk_forward_evaluate(_synthetic(signal=0.0), ["noise"], "Logit_L2_C0.1")
    assert 0.35 < res["auc"] < 0.65


def test_baseline_predicts_the_majority_class():
    df = _synthetic()
    df["target_direction"] = (np.arange(len(df)) % 5 != 0).astype(int)  # 80% up
    res = walk_forward_evaluate(df, [], "Baseline")
    assert res["up_rate_pred"] == 1.0


def test_folds_never_train_on_the_future(monkeypatch):
    df = _synthetic().sample(frac=1, random_state=0)  # shuffled input
    seen = []

    class Spy:
        def __init__(self):
            self.model = make_model("Logit_L2_C1.0")

        def fit(self, X, y):
            seen.append(("train", X.index.max()))
            return self.model.fit(X, y)

        def predict(self, X):
            seen.append(("test", X.index.min()))
            return self.model.predict(X)

        def predict_proba(self, X):
            return self.model.predict_proba(X)

    monkeypatch.setattr(evaluation, "make_model", lambda name, balanced=False: Spy())
    walk_forward_evaluate(df, ["signal"], "Spy")

    trains = [i for kind, i in seen if kind == "train"]
    tests = [i for kind, i in seen if kind == "test"]
    assert all(t_max < t_min for t_max, t_min in zip(trains, tests, strict=True))


def test_threshold_is_tuned_on_training_scores():
    y = np.array([0, 0, 1, 1])
    prob = np.array([0.30, 0.40, 0.60, 0.70])
    t = best_threshold(y, prob, metric="accuracy")
    assert 0.40 < t <= 0.60


@pytest.mark.parametrize("name", linear_grid() + ["Baseline", "RF"])
def test_every_model_name_builds(name):
    assert make_model(name) is not None


def test_unknown_model_is_rejected():
    with pytest.raises(ValueError):
        make_model("GPT")
