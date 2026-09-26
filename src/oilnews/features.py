"""Feature engineering: headline aggregation, price features, shocks, pressure indices."""

import numpy as np
import pandas as pd

from .config import CORE_TOPICS, NON_FEATURE_COLS

# ---------------------------------------------------------------------------
# Text features
# ---------------------------------------------------------------------------


def _is_log_candidate(col: str) -> bool:
    return col.endswith("_count") or col.endswith("_sum")


def aggregate_text_features(news: pd.DataFrame, freq: str = "4h") -> pd.DataFrame:
    """Aggregate labelled headlines into per-window, per-topic features.

    For every UTC window of length ``freq`` and every topic, emits the headline
    count, distinct-source count, relevance sum/mean, plus ``log1p_`` versions of
    count and sum columns. Windows are labelled by their start time.
    """
    temp = news.copy()
    temp["bucket"] = temp["published_utc"].dt.floor(freq)

    all_agg = temp.groupby("bucket").agg(
        all_news_count=("title_clean", "size"),
        all_source_count=("source", "nunique"),
        all_relevance_sum=("relevance_score", "sum"),
        all_relevance_mean=("relevance_score", "mean"),
    )

    topic_agg = temp.groupby(["bucket", "topic"]).agg(
        count=("title_clean", "size"),
        source_count=("source", "nunique"),
        relevance_sum=("relevance_score", "sum"),
        relevance_mean=("relevance_score", "mean"),
    )
    topic_wide = topic_agg.unstack("topic")
    topic_wide.columns = [f"{topic}_{metric}" for metric, topic in topic_wide.columns]

    features = all_agg.join(topic_wide, how="outer").fillna(0)
    for col in list(features.columns):
        if _is_log_candidate(col):
            features[f"log1p_{col}"] = np.log1p(features[col])

    features.index.name = "timestamp_utc"
    return features.reset_index()


# ---------------------------------------------------------------------------
# Price features and target
# ---------------------------------------------------------------------------


def make_price_features(prices: pd.DataFrame, freq: str = "4h") -> pd.DataFrame:
    """Build WTI return/volatility features, market controls, and the target.

    ``prices`` is an hourly close-price frame indexed by UTC timestamp with a
    ``WTI`` column plus optional control series. The target is the log return
    from the close of window t to the close of window t+1.
    """
    p = prices.resample(freq, label="left", closed="left").last()
    p = p.dropna(subset=["WTI"]).ffill()

    feat = pd.DataFrame(index=p.index)
    feat["wti_close"] = p["WTI"]
    feat["wti_ret_0"] = np.log(p["WTI"]).diff()
    for lag in [1, 2, 3, 6, 12]:
        feat[f"wti_ret_lag{lag}"] = feat["wti_ret_0"].shift(lag)
    for window in [3, 6, 12]:
        feat[f"wti_vol_{window}"] = feat["wti_ret_0"].rolling(window).std()

    for col in p.columns:
        if col == "WTI":
            continue
        ret = np.log(p[col]).diff()
        name = col.lower()
        feat[f"{name}_ret_0"] = ret
        for lag in [1, 2, 3]:
            feat[f"{name}_ret_lag{lag}"] = ret.shift(lag)

    feat["hour"] = feat.index.hour
    feat["dayofweek"] = feat.index.dayofweek
    feat["hour_sin"] = np.sin(2 * np.pi * feat["hour"] / 24)
    feat["hour_cos"] = np.cos(2 * np.pi * feat["hour"] / 24)

    feat["future_ret_next"] = np.log(p["WTI"].shift(-1)) - np.log(p["WTI"])
    feat["target_direction"] = (feat["future_ret_next"] > 0).astype(int)

    feat.index.name = "timestamp_utc"
    return feat.reset_index()


def build_direction_dataset(text: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    """Left-join text features onto price windows; windows with no news get zeros."""
    text = text.assign(timestamp_utc=pd.to_datetime(text["timestamp_utc"], utc=True))
    price = price.assign(timestamp_utc=pd.to_datetime(price["timestamp_utc"], utc=True))

    df = price.merge(text, on="timestamp_utc", how="left")
    text_cols = [c for c in text.columns if c != "timestamp_utc"]
    df[text_cols] = df[text_cols].fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=["future_ret_next", "target_direction"])
    return df.sort_values("timestamp_utc").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------


def _belongs_to(col: str, topic: str) -> bool:
    return col.startswith(f"{topic}_") or col.startswith(f"log1p_{topic}_")


def core_topic_columns(df: pd.DataFrame, topics: list[str] = CORE_TOPICS) -> list[str]:
    """Current-window numeric features for the core topics (no lags or shocks)."""
    derived_suffixes = ("_lag1", "_lag2", "_lag3", "_shock")
    return [
        c
        for c in df.columns
        if c not in NON_FEATURE_COLS
        and pd.api.types.is_numeric_dtype(df[c])
        and any(_belongs_to(c, t) for t in topics)
        and not c.endswith(derived_suffixes)
    ]


def _is_intensity(col: str) -> bool:
    return (
        col.endswith("_count")
        or col.endswith("_source_count")
        or col.endswith("_relevance_sum")
        or col.startswith("log1p_")
    )


def add_lag_features(df: pd.DataFrame, cols: list[str], lags=(1, 2, 3)) -> tuple[pd.DataFrame, list[str]]:
    """Add shifted copies of the intensity columns among ``cols``."""
    out = df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    lag_cols = []
    for col in (c for c in cols if _is_intensity(c)):
        for lag in lags:
            name = f"{col}_lag{lag}"
            out[name] = out[col].shift(lag).fillna(0)
            lag_cols.append(name)
    return out, lag_cols


def rolling_zscore_past(series: pd.Series, window: int = 12, min_periods: int = 4) -> pd.Series:
    """Z-score of each value against the preceding ``window`` values only.

    The rolling statistics are shifted by one step so the current and future
    windows never leak into their own baseline. With 4h windows, 12 = ~48h.
    """
    past = series.shift(1).rolling(window=window, min_periods=min_periods)
    z = (series - past.mean()) / past.std().replace(0, np.nan)
    return z.replace([np.inf, -np.inf], np.nan).fillna(0)


def add_shock_features(df: pd.DataFrame, cols: list[str], window: int = 12) -> tuple[pd.DataFrame, list[str]]:
    """Add ``<col>_shock``: how abnormal the current news intensity is."""
    out = df.copy()
    shock_cols = []
    for col in (c for c in cols if _is_intensity(c)):
        name = f"{col}_shock"
        out[name] = rolling_zscore_past(out[col], window=window)
        shock_cols.append(name)
    return out, shock_cols


def _topic_signal(df: pd.DataFrame, topic: str) -> pd.Series:
    candidates = [
        f"log1p_{topic}_count_shock",
        f"log1p_{topic}_source_count_shock",
        f"log1p_{topic}_relevance_sum_shock",
        f"{topic}_count_shock",
        f"{topic}_source_count_shock",
        f"{topic}_relevance_sum_shock",
    ]
    existing = [c for c in candidates if c in df.columns]
    if not existing:
        return pd.Series(0.0, index=df.index)
    return df[existing].mean(axis=1)


PRESSURE_COLS = [
    "military_signal",
    "hormuz_signal",
    "sanctions_signal",
    "diplomacy_signal",
    "inventory_signal",
    "geo_risk_pressure",
    "diplomacy_relief_pressure",
    "inventory_shock_pressure",
    "net_geo_oil_pressure",
    "core_news_attention",
]


def add_pressure_features(df: pd.DataFrame, core_cols: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Combine topic shocks into signed, economically motivated pressure indices.

    Military escalation, Hormuz shipping risk, and sanctions push oil up;
    diplomacy/ceasefire news relieves the risk premium. Requires shock columns.
    """
    out = df.copy()
    out["military_signal"] = _topic_signal(out, "military_escalation")
    out["hormuz_signal"] = _topic_signal(out, "hormuz_shipping")
    out["sanctions_signal"] = _topic_signal(out, "sanctions_exports")
    out["diplomacy_signal"] = _topic_signal(out, "diplomacy_ceasefire")
    out["inventory_signal"] = _topic_signal(out, "inventory_eia")

    out["geo_risk_pressure"] = (out["military_signal"] + out["hormuz_signal"] + out["sanctions_signal"]) / 3
    out["diplomacy_relief_pressure"] = out["diplomacy_signal"]
    out["inventory_shock_pressure"] = out["inventory_signal"]
    out["net_geo_oil_pressure"] = out["geo_risk_pressure"] - out["diplomacy_relief_pressure"]

    attention_cols = [c for c in core_cols if c.startswith("log1p_") and c.endswith("_count")]
    out["core_news_attention"] = out[attention_cols].sum(axis=1) if attention_cols else 0.0
    return out, list(PRESSURE_COLS)


def add_interaction_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Let news pressure interact with market state (volatility, recent returns)."""
    out = df.copy()
    cols = []
    for market_col, suffix in [
        ("wti_vol_3", "wti_vol3"),
        ("wti_vol_6", "wti_vol6"),
        ("wti_ret_lag1", "wti_ret_lag1"),
        ("wti_ret_lag2", "wti_ret_lag2"),
    ]:
        if market_col in out.columns:
            name = f"net_geo_pressure_x_{suffix}"
            out[name] = out["net_geo_oil_pressure"] * out[market_col]
            cols.append(name)
    return out, cols


def build_feature_sets(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    """Add every engineered feature and return the ablation feature sets."""
    core = core_topic_columns(df)
    df, shock = add_shock_features(df, core)
    df, pressure = add_pressure_features(df, core)
    df, interaction = add_interaction_features(df)
    feature_sets = {
        "Core count only": core,
        "Topic shock only": shock,
        "Signed pressure only": pressure,
        "Signed pressure + interaction": pressure + interaction,
        "Core count + shock": core + shock,
        "Core count + pressure": core + pressure,
        "Core count + shock + pressure": core + shock + pressure,
        "Core count + shock + pressure + interaction": core + shock + pressure + interaction,
    }
    return df, feature_sets


# ---------------------------------------------------------------------------
# Sample definition
# ---------------------------------------------------------------------------


def meaningful_move_sample(df: pd.DataFrame, filter_q: float) -> tuple[pd.DataFrame, float]:
    """Drop the ``filter_q`` fraction of windows with the smallest |future return|.

    This conditions on the realized move size, so it is an evaluation lens
    ("is news informative when the market actually moves?"), not a tradable rule.
    """
    out = df.copy()
    threshold = 0.0
    if filter_q > 0:
        threshold = float(out["future_ret_next"].abs().quantile(filter_q))
        out = out[out["future_ret_next"].abs() >= threshold].copy()
    out["target_direction"] = (out["future_ret_next"] > 0).astype(int)
    return out, threshold
