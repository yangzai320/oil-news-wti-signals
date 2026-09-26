import pandas as pd
import pytest

from oilnews.cleaning import clean_headlines, normalize_for_dedup
from oilnews.config import DEFAULT_TOPIC
from oilnews.topics import assign_topic


def test_normalize_strips_urls_punctuation_and_case():
    assert normalize_for_dedup("  OPEC+ Cuts Output!  http://x.co/a ") == "opec cuts output"


def test_clean_headlines_dedups_filters_and_windows(headlines):
    extra = pd.DataFrame(
        {
            "published_utc": ["2026-03-01T06:00:00Z", "2026-03-01T07:00:00Z", "2026-05-01T00:00:00Z"],
            "title": [
                "Olive oil prices climb across southern Europe markets",
                "Short title",
                "Crude futures rally after OPEC meeting ends early",
            ],
        }
    )
    out = clean_headlines(pd.concat([headlines, extra]), "2026-03-01", "2026-04-01")

    titles = out["title_clean"].tolist()
    assert len(titles) == 3  # near-duplicate EIA headline collapsed
    assert not any("Olive oil" in t for t in titles)  # irrelevant "oil" filtered
    assert "Short title" not in titles  # below minimum length
    assert out["published_utc"].max() < pd.Timestamp("2026-04-01", tz="UTC")  # end exclusive
    assert out["published_utc"].is_monotonic_increasing


def test_clean_headlines_requires_columns():
    with pytest.raises(ValueError, match="published_utc"):
        clean_headlines(pd.DataFrame({"title": ["x"]}), "2026-01-01", "2026-02-01")


@pytest.mark.parametrize(
    ("title", "topic"),
    [
        ("Tanker seized near Strait of Hormuz", "hormuz_shipping"),
        # matches both shipping and military keywords: priority order wins
        ("Drone strike hits tanker in Gulf waters", "hormuz_shipping"),
        ("Ceasefire talks resume in Doha", "diplomacy_ceasefire"),
        ("OPEC+ weighs spare capacity", "opec_gulf_supply"),
        ("Crude inventories fall, EIA says", "inventory_eia"),
        ("Oil edges higher in quiet trade", DEFAULT_TOPIC),
    ],
)
def test_assign_topic(title, topic):
    assert assign_topic(title) == topic


def test_assign_topic_matches_whole_words_only():
    # "api" must not fire inside "capital", nor "war" inside "award"
    assert assign_topic("Energy capital award announced") == DEFAULT_TOPIC
