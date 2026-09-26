"""Deterministic, rule-based topic assignment for oil headlines."""

import re

import pandas as pd

from .config import DEFAULT_TOPIC, TOPIC_KEYWORDS, TOPIC_PRIORITY


def _compile(keywords: list[str]) -> re.Pattern:
    # (?<!\w)/(?!\w) instead of \b so keywords ending in "+" (e.g. "opec+") still match.
    alternatives = "|".join(re.escape(k.lower()) for k in sorted(keywords, key=len, reverse=True))
    return re.compile(rf"(?<!\w)(?:{alternatives})(?!\w)")


# One compiled pattern per topic, checked in priority order.
_TOPIC_PATTERNS: list[tuple[str, re.Pattern]] = [
    (topic, _compile(TOPIC_KEYWORDS[topic])) for topic in TOPIC_PRIORITY
]


def assign_topic(title: object) -> str:
    """Return the highest-priority topic whose keywords appear in ``title``."""
    text = str(title).lower()
    for topic, pattern in _TOPIC_PATTERNS:
        if pattern.search(text):
            return topic
    return DEFAULT_TOPIC


def label_topics(news: pd.DataFrame, title_col: str = "title_clean") -> pd.DataFrame:
    """Add a ``topic`` column to a cleaned headline frame."""
    out = news.copy()
    out["topic"] = out[title_col].map(assign_topic)
    return out
