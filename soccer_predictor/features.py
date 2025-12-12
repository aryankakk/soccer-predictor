from __future__ import annotations

import numpy as np
import pandas as pd


def add_rolling_team_features(
    matches: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Add pre-match rolling features for home and away teams.

    Expects columns: date, home_team, away_team, home_goals, away_goals,
    optionally home_xg, away_xg.

    Rolling features are computed from *prior* matches only.
    """
    df = matches.copy()
    df = df.sort_values(["date", "competition", "season"], kind="stable").reset_index(drop=True)

    # Identify stat pairs present as home_<stat> / away_<stat>
    ignore_bases = {
        "team",
        "goals",  # we use goals via home_goals/away_goals below
    }
    stat_bases: list[str] = []
    for c in df.columns:
        if not c.startswith("home_"):
            continue
        base = c[len("home_") :]
        if base in ignore_bases:
            continue
        if f"away_{base}" in df.columns:
            stat_bases.append(base)

    # Build a long-form table of team-match rows
    base_cols = ["competition", "season", "date", "home_team", "away_team", "home_goals", "away_goals"]
    df2 = df[base_cols + [f"home_{b}" for b in stat_bases] + [f"away_{b}" for b in stat_bases]].copy()

    home_rows = pd.DataFrame(
        {
            "competition": df2["competition"],
            "season": df2["season"],
            "date": df2["date"],
            "team": df2["home_team"],
            "is_home": 1,
            "gf": df2["home_goals"],
            "ga": df2["away_goals"],
        }
    )
    away_rows = pd.DataFrame(
        {
            "competition": df2["competition"],
            "season": df2["season"],
            "date": df2["date"],
            "team": df2["away_team"],
            "is_home": 0,
            "gf": df2["away_goals"],
            "ga": df2["home_goals"],
        }
    )

    # Add generic stat_for/stat_against from detected pairs
    for b in stat_bases:
        home_rows[f"{b}_for"] = df2[f"home_{b}"]
        home_rows[f"{b}_against"] = df2[f"away_{b}"]
        away_rows[f"{b}_for"] = df2[f"away_{b}"]
        away_rows[f"{b}_against"] = df2[f"home_{b}"]

    long = pd.concat([home_rows, away_rows], ignore_index=True)
    long = long.sort_values(["competition", "season", "team", "date"], kind="stable")

    # Points from match result (from team perspective)
    long["pts"] = np.where(long["gf"] > long["ga"], 3, np.where(long["gf"] == long["ga"], 1, 0))

    grp = long.groupby(["competition", "season", "team"], sort=False)

    # Prior-match rolling stats (shift by 1 to avoid leakage)
    for w in windows:
        for col in [c for c in long.columns if c not in {"competition", "season", "date", "team", "is_home"}]:
            if col in {"matches_played"}:
                continue
            # Only roll numeric-like columns
            if long[col].dtype.kind not in {"i", "u", "f"}:
                continue
            long[f"{col}_r{w}"] = grp[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())

    # Also include simple match counts
    long["matches_played"] = grp.cumcount()

    # Split back into home/away feature tables aligned to original match order
    long_home = long[long["is_home"] == 1].copy().reset_index(drop=True)
    long_away = long[long["is_home"] == 0].copy().reset_index(drop=True)

    # They should align 1:1 with df after sorting by date/comp/season (same as we built them)
    # Reconstruct by merging on keys to be safe.
    # Create stable row ids in df
    df["_row_id"] = np.arange(len(df))

    # Merge home features
    home_key = pd.DataFrame(
        {
            "_row_id": df["_row_id"],
            "competition": df["competition"],
            "season": df["season"],
            "date": df["date"],
            "team": df["home_team"],
        }
    )
    away_key = pd.DataFrame(
        {
            "_row_id": df["_row_id"],
            "competition": df["competition"],
            "season": df["season"],
            "date": df["date"],
            "team": df["away_team"],
        }
    )

    feat_cols = [c for c in long.columns if c.endswith(tuple([f"_r{w}" for w in windows])) or c in {"matches_played"}]
    home_feats = long_home[["competition", "season", "date", "team"] + feat_cols]
    away_feats = long_away[["competition", "season", "date", "team"] + feat_cols]

    home_join = home_key.merge(home_feats, on=["competition", "season", "date", "team"], how="left")
    home_join = home_join[["_row_id"] + feat_cols].rename(columns={c: f"home_{c}" for c in feat_cols})
    df = df.merge(home_join, on="_row_id", how="left")

    away_join = away_key.merge(away_feats, on=["competition", "season", "date", "team"], how="left")
    away_join = away_join[["_row_id"] + feat_cols].rename(columns={c: f"away_{c}" for c in feat_cols})
    df = df.merge(away_join, on="_row_id", how="left")

    # Differences (home - away)
    for c in feat_cols:
        hc, ac = f"home_{c}", f"away_{c}"
        if hc in df.columns and ac in df.columns:
            df[f"diff_{c}"] = df[hc] - df[ac]

    df = df.drop(columns=["_row_id"])
    return df
