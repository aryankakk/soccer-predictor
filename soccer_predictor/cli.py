from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .dataset import normalize_schedule_df
from .fbref import (
    DEFAULT_COMPETITIONS,
    Competition,
    fetch_scores_and_fixtures,
    list_seasons,
    resolve_requested_competitions,
    season_schedule_url,
)
from .features import add_rolling_team_features
from .model import TrainedModel, load_model, predict_proba, save_model, train_multiclass
from .stats import enrich_fixtures_with_fbref_teamlogs
from .utils import HttpCache, ensure_dir


def cmd_fetch(args: argparse.Namespace) -> int:
    cache = HttpCache(
        cache_dir=Path(args.cache_dir),
        min_delay_s=args.min_delay_s,
        use_playwright=args.use_playwright,
    )
    if args.comp_ids:
        default_names = {c.comp_id: c.name for c in DEFAULT_COMPETITIONS}
        comps = [Competition(name=default_names.get(i, f"comp_{i}"), comp_id=i) for i in args.comp_ids]
    else:
        comps = resolve_requested_competitions(cache)
    if not comps:
        raise SystemExit("Could not resolve competitions from FBref.")

    out_rows = []
    out_rows_enriched = []
    for comp in comps:
        seasons = list_seasons(cache, comp, max_seasons=args.max_seasons)
        for season in seasons:
            url = season_schedule_url(cache, comp, season)
            if not url:
                continue
            raw = fetch_scores_and_fixtures(cache, url)
            norm = normalize_schedule_df(raw, competition=comp.name, season=season)
            out_rows.append(norm)
            if args.with_stats:
                enriched = enrich_fixtures_with_fbref_teamlogs(
                    cache,
                    fixtures=norm,
                    comp=comp,
                    season=season,
                    max_teams=args.max_teams,
                )
                out_rows_enriched.append(enriched)

    if not out_rows:
        raise SystemExit("No schedule data fetched.")

    df = pd.concat(out_rows, ignore_index=True)
    ensure_dir(Path(args.out_csv).parent)
    df.to_csv(args.out_csv, index=False)
    print(f"Wrote {len(df):,} rows to {args.out_csv}")

    if args.with_stats:
        if not out_rows_enriched:
            print("Warning: --with-stats requested but no enriched rows were produced.")
        else:
            df_enriched = pd.concat(out_rows_enriched, ignore_index=True)
            ensure_dir(Path(args.out_enriched_csv).parent)
            df_enriched.to_csv(args.out_enriched_csv, index=False)
            print(f"Wrote {len(df_enriched):,} rows to {args.out_enriched_csv}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data_csv, parse_dates=["date"])

    df_feat = add_rolling_team_features(df, windows=(5, 10))

    # Feature set: only safe pre-match features
    feature_cols = [
        "competition",
        "season",
    ] + [c for c in df_feat.columns if c.startswith("diff_") or c.startswith("home_matches_played") or c.startswith("away_matches_played")]

    categorical_cols = ["competition", "season"]

    model, metrics = train_multiclass(
        df_feat,
        feature_cols=feature_cols,
        categorical_cols=categorical_cols,
        test_fraction=args.test_fraction,
    )

    ensure_dir(Path(args.model_path).parent)
    save_model(model, args.model_path)

    print("Saved model to", args.model_path)
    print("Accuracy:", metrics["accuracy"])
    print("Log loss:", metrics["log_loss"])
    print(metrics["report"])

    # Optional QML demo
    if args.quantum:
        from .quantum import try_quantum_baseline

        played = df_feat.dropna(subset=["result"]).sort_values("date")
        # binary label (home win vs not)
        y = (played["result"].astype(str) == "H").astype(int).to_numpy()
        # use a small numeric slice
        num = played[[c for c in feature_cols if c.startswith("diff_")]].to_numpy(dtype=float)
        # subsample
        n = min(len(num), 256)
        num = num[:n]
        y = y[:n]
        res = try_quantum_baseline(num, y)
        print(res.info)

    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    df = pd.read_csv(args.data_csv, parse_dates=["date"])
    model = load_model(args.model_path)

    df_feat = add_rolling_team_features(df, windows=(5, 10))

    feature_cols = [
        "competition",
        "season",
    ] + [c for c in df_feat.columns if c.startswith("diff_") or c.startswith("home_matches_played") or c.startswith("away_matches_played")]

    upcoming = df_feat[df_feat["result"].isna()].copy()
    if args.after_date:
        upcoming = upcoming[upcoming["date"] >= pd.to_datetime(args.after_date)]

    if upcoming.empty:
        print("No upcoming fixtures found (rows without scores).")
        return 0

    X = upcoming[feature_cols]
    proba = predict_proba(model, X)
    out = pd.concat(
        [
            upcoming[["date", "competition", "home_team", "away_team"]].reset_index(drop=True),
            proba.reset_index(drop=True),
        ],
        axis=1,
    )

    out = out.sort_values(["date", "competition", "home_team"]) 

    if args.out_csv:
        ensure_dir(Path(args.out_csv).parent)
        out.to_csv(args.out_csv, index=False)
        print(f"Wrote predictions to {args.out_csv}")
    else:
        with pd.option_context("display.max_rows", 50, "display.width", 160):
            print(out.head(50).to_string(index=False))

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="soccer-predictor")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="Fetch schedules from FBref")
    f.add_argument("--out-csv", default="data/matches_raw.csv")
    f.add_argument("--out-enriched-csv", default="data/matches_enriched.csv")
    f.add_argument("--cache-dir", default="data/cache")
    f.add_argument("--max-seasons", type=int, default=6)
    f.add_argument("--min-delay-s", type=float, default=3.0)
    f.add_argument(
        "--comp-ids",
        type=int,
        nargs="+",
        default=None,
        help="Optional: explicit FBref competition IDs (skips discovery).",
    )
    f.add_argument(
        "--use-playwright",
        action="store_true",
        help="Use a headless browser for fetching (helps with Cloudflare/403).",
    )
    f.add_argument(
        "--with-stats",
        action="store_true",
        help="Also scrape team matchlogs (shots/possession/passing/etc where available) and merge into fixtures.",
    )
    f.add_argument(
        "--max-teams",
        type=int,
        default=None,
        help="Limit teams scraped per comp/season (useful for testing).",
    )
    f.set_defaults(func=cmd_fetch)

    t = sub.add_parser("train", help="Train outcome model")
    t.add_argument("--data-csv", default="data/matches_raw.csv")
    t.add_argument("--model-path", default="models/outcome_model.joblib")
    t.add_argument("--test-fraction", type=float, default=0.2)
    t.add_argument("--quantum", action="store_true", help="Run optional tiny QML demo")
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict", help="Predict upcoming fixtures")
    pr.add_argument("--data-csv", default="data/matches_raw.csv")
    pr.add_argument("--model-path", default="models/outcome_model.joblib")
    pr.add_argument("--after-date", default=None)
    pr.add_argument("--out-csv", default="data/predictions.csv")
    pr.set_defaults(func=cmd_predict)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
