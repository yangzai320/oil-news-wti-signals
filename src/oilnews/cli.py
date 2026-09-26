"""Command-line entry points.

oilnews build-dataset --news headlines.csv --prices prices_1h.csv --out data/direction_4h.csv
oilnews ablate --dataset data/direction_4h.csv --filter-q 0.4 --out results/ablation.csv
"""

import argparse
from pathlib import Path

import pandas as pd

from . import config
from .cleaning import clean_headlines
from .evaluation import run_grid
from .features import (
    aggregate_text_features,
    build_direction_dataset,
    build_feature_sets,
    make_price_features,
    meaningful_move_sample,
)
from .models import linear_grid
from .topics import label_topics


def build_dataset(args: argparse.Namespace) -> None:
    news = clean_headlines(pd.read_csv(args.news, low_memory=False), args.start, args.end)
    news = label_topics(news)
    text = aggregate_text_features(news, freq=args.freq)

    if args.prices:
        prices = pd.read_csv(args.prices, index_col=0, parse_dates=True)
        prices.index = pd.to_datetime(prices.index, utc=True)
    else:
        from .prices import download_hourly_closes

        prices = download_hourly_closes(args.start, args.end)

    dataset = build_direction_dataset(text, make_price_features(prices, freq=args.freq))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(args.out, index=False)
    print(f"{len(news):,} headlines -> {len(dataset):,} windows -> {args.out}")
    print(news["topic"].value_counts().to_string())


def ablate(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.dataset)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    df, feature_sets = build_feature_sets(df)
    sample, threshold = meaningful_move_sample(df, args.filter_q)
    print(f"filter_q={args.filter_q}: {len(sample)} windows with |return| >= {threshold:.6f}")

    results = run_grid(sample, feature_sets, linear_grid(), n_splits=args.splits)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    cols = ["feature_set", "model", "accuracy", "balanced_accuracy", "auc"]
    print(results[cols].head(10).to_string(index=False, float_format="%.3f"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="oilnews")
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("build-dataset", help="clean + label headlines and join them to WTI windows")
    p.add_argument(
        "--news", required=True, help="headline CSV: published_utc, title, source, relevance_score"
    )
    p.add_argument("--prices", help="hourly close CSV (UTC index, WTI column); Yahoo download if omitted")
    p.add_argument("--start", default=config.EVENT_START)
    p.add_argument("--end", default=config.EVENT_END)
    p.add_argument("--freq", default=config.FREQ)
    p.add_argument("--out", default="data/direction_4h.csv")
    p.set_defaults(func=build_dataset)

    p = sub.add_parser("ablate", help="walk-forward feature ablation over linear models")
    p.add_argument("--dataset", required=True)
    p.add_argument("--filter-q", type=float, default=0.4, help="drop this fraction of smallest moves")
    p.add_argument("--splits", type=int, default=5)
    p.add_argument("--out", default="results/ablation.csv")
    p.set_defaults(func=ablate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
