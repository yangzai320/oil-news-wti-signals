import numpy as np
import pandas as pd

from oilnews.cleaning import clean_headlines
from oilnews.features import (
    aggregate_text_features,
    build_direction_dataset,
    build_feature_sets,
    make_price_features,
    meaningful_move_sample,
    rolling_zscore_past,
)
from oilnews.topics import label_topics


def _dataset(headlines, hourly_prices):
    news = label_topics(clean_headlines(headlines, "2026-03-01", "2026-03-11"))
    text = aggregate_text_features(news, freq="4h")
    return build_direction_dataset(text, make_price_features(hourly_prices, freq="4h"))


def test_headlines_land_in_their_utc_bucket(headlines):
    news = label_topics(clean_headlines(headlines, "2026-03-01", "2026-03-02"))
    text = aggregate_text_features(news, freq="4h").set_index("timestamp_utc")

    first = pd.Timestamp("2026-03-01 00:00", tz="UTC")
    second = pd.Timestamp("2026-03-01 04:00", tz="UTC")
    assert text.loc[first, "all_news_count"] == 2
    assert text.loc[first, "hormuz_shipping_count"] == 1
    assert text.loc[second, "inventory_eia_count"] == 1
    assert text.loc[first, "log1p_all_news_count"] == np.log1p(2)


def test_target_is_next_window_log_return(hourly_prices):
    feat = make_price_features(hourly_prices, freq="4h").set_index("timestamp_utc")
    closes = hourly_prices["WTI"].resample("4h").last()

    t0, t1 = closes.index[0], closes.index[1]
    expected = np.log(closes[t1]) - np.log(closes[t0])
    assert np.isclose(feat.loc[t0, "future_ret_next"], expected)
    assert feat.loc[t0, "target_direction"] == int(expected > 0)


def test_windows_without_news_get_zero_counts(headlines, hourly_prices):
    df = _dataset(headlines, hourly_prices)
    assert len(df) == 10 * 6 - 1  # last window has no next close
    assert (df["all_news_count"] > 0).sum() == 2
    assert df["all_news_count"].isna().sum() == 0


def test_rolling_zscore_uses_only_the_past():
    s = pd.Series(np.arange(30, dtype=float))
    base = rolling_zscore_past(s, window=12)

    tampered = s.copy()
    tampered.iloc[20:] = 1_000.0  # change the future
    after = rolling_zscore_past(tampered, window=12)

    pd.testing.assert_series_equal(base.iloc[:20], after.iloc[:20])


def test_feature_sets_cover_the_ablation(headlines, hourly_prices):
    df, sets = build_feature_sets(_dataset(headlines, hourly_prices))
    full = sets["Core count + shock + pressure + interaction"]

    assert len(sets) == 8
    assert "net_geo_oil_pressure" in full
    assert all(col in df.columns for col in full)
    assert not {"future_ret_next", "target_direction", "wti_close"} & set(full)


def test_meaningful_move_sample_drops_smallest_moves(hourly_prices):
    df = make_price_features(hourly_prices).dropna(subset=["future_ret_next"])
    kept, threshold = meaningful_move_sample(df, filter_q=0.4)

    assert len(kept) < len(df)
    assert (kept["future_ret_next"].abs() >= threshold).all()
    assert np.isclose(len(kept) / len(df), 0.6, atol=0.03)
