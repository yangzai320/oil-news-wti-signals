"""Hourly market prices from Yahoo Finance."""

import pandas as pd

from .config import TICKERS


def extract_close(raw: pd.DataFrame, tickers: dict[str, str] = TICKERS) -> pd.DataFrame:
    """Pull one close-price column per ticker out of a ``yfinance.download`` frame."""
    close = pd.DataFrame(index=raw.index)
    for name, ticker in tickers.items():
        if isinstance(raw.columns, pd.MultiIndex):
            for key in [(ticker, "Close"), ("Close", ticker)]:
                if key in raw.columns:
                    close[name] = raw[key]
                    break
        elif "Close" in raw.columns:
            close[name] = raw["Close"]

    close.index = pd.to_datetime(close.index)
    close.index = close.index.tz_localize("UTC") if close.index.tz is None else close.index.tz_convert("UTC")
    return close.dropna(how="all")


def download_hourly_closes(start: str, end: str, tickers: dict[str, str] = TICKERS) -> pd.DataFrame:
    """Download hourly closes. Yahoo only serves ~730 days of hourly history."""
    import yfinance as yf  # optional dependency

    raw = yf.download(
        list(tickers.values()),
        start=start,
        end=end,
        interval="1h",
        auto_adjust=False,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    return extract_close(raw, tickers)
