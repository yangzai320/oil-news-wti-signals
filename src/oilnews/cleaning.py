"""Headline cleaning: normalization, window filtering, and de-duplication."""

import re

import numpy as np
import pandas as pd

_WHITESPACE_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"http\S+")
_NON_ALNUM_RE = re.compile(r"[^a-z0-9\s]")

# Headlines that mention "oil" but have nothing to do with crude.
IRRELEVANT_RE = re.compile(
    r"\bolive oil\b|\boil painting\b|\bcooking oil\b"
    r"|\bessential oil\b|\bhair oil\b|\bmassage oil\b",
    re.IGNORECASE,
)


def clean_title_text(title: object) -> str:
    """Collapse newlines and repeated whitespace."""
    text = str(title).replace("\n", " ").replace("\r", " ")
    return _WHITESPACE_RE.sub(" ", text).strip()


def normalize_for_dedup(title: object) -> str:
    """Lower-case, strip URLs and punctuation; used as the de-duplication key."""
    text = clean_title_text(title).lower()
    text = _URL_RE.sub("", text)
    text = _NON_ALNUM_RE.sub(" ", text)
    return _WHITESPACE_RE.sub(" ", text).strip()


def clean_headlines(
    news: pd.DataFrame,
    start: str,
    end: str,
    min_title_len: int = 20,
) -> pd.DataFrame:
    """Return cleaned, de-duplicated headlines published in [start, end) UTC.

    Expects at least ``published_utc`` and ``title``. Adds ``title_clean`` and
    ``title_norm``; fills ``source``/``relevance_score`` when absent.
    """
    df = news.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = {"published_utc", "title"} - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns: {sorted(missing)}")

    df["published_utc"] = pd.to_datetime(df["published_utc"], errors="coerce", utc=True)
    df = df.dropna(subset=["published_utc", "title"])
    df = df[
        (df["published_utc"] >= pd.Timestamp(start, tz="UTC"))
        & (df["published_utc"] < pd.Timestamp(end, tz="UTC"))
    ].copy()

    for col in ["source", "url", "query_bucket"]:
        if col not in df.columns:
            df[col] = np.nan
    if "relevance_score" not in df.columns:
        df["relevance_score"] = 1
    df["relevance_score"] = pd.to_numeric(df["relevance_score"], errors="coerce").fillna(1)

    df["title_clean"] = df["title"].map(clean_title_text)
    df["title_norm"] = df["title_clean"].map(normalize_for_dedup)

    df = df[df["title_clean"].str.len() >= min_title_len]
    # Keep the earliest copy of each headline so syndication does not inflate counts.
    df = df.sort_values("published_utc").drop_duplicates(subset=["title_norm"], keep="first")
    df = df[~df["title_clean"].str.contains(IRRELEVANT_RE, na=False)]

    return df.reset_index(drop=True)
