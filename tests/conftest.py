import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def hourly_prices() -> pd.DataFrame:
    """Ten days of synthetic hourly closes for WTI and one control series."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-03-01", periods=24 * 10, freq="1h", tz="UTC")
    wti = 70 * np.exp(np.cumsum(rng.normal(0, 0.003, len(idx))))
    spy = 500 * np.exp(np.cumsum(rng.normal(0, 0.001, len(idx))))
    return pd.DataFrame({"WTI": wti, "SPY": spy}, index=idx)


@pytest.fixture
def headlines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "published_utc": [
                "2026-03-01T01:10:00Z",
                "2026-03-01T02:30:00Z",
                "2026-03-01T05:00:00Z",
                "2026-03-01T05:05:00Z",
            ],
            "title": [
                "Tanker traffic through the Strait of Hormuz slows sharply",
                "Missile strike hits oil facility near the Gulf coast",
                "EIA reports surprise crude inventory drawdown this week",
                "EIA reports surprise crude inventory drawdown this week!",
            ],
            "source": ["reuters.com", "apnews.com", "cnbc.com", "wsj.com"],
            "relevance_score": [1.0, 2.0, 1.5, 1.5],
        }
    )
