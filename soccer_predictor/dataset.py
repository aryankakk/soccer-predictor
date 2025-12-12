from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


SCORE_RE = re.compile(r"^(\d+)\D+(\d+)$")


def _parse_score(score: object) -> tuple[Optional[int], Optional[int]]:
    if score is None or (isinstance(score, float) and np.isnan(score)):
        return (None, None)
    s = str(score).strip()
    if not s or s.lower() in {"nan", ""}:
        return (None, None)
    m = SCORE_RE.match(s)
    if not m:
        return (None, None)
    return (int(m.group(1)), int(m.group(2)))


def normalize_schedule_df(df: pd.DataFrame, competition: str, season: str) -> pd.DataFrame:
    """Normalize FBref schedule dataframe to a consistent schema."""
    out = df.copy()

    # Standardize column names (FBref uses slightly different variants)
    rename = {
        "Home": "home_team",
        "Away": "away_team",
        "Date": "date",
        "Time": "time",
        "Score": "score",
        "xG": "home_xg",
        "xG.1": "away_xg",
        "Notes": "notes",
        "Venue": "venue",
        "Attendance": "attendance",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})

    for col in ["date", "home_team", "away_team"]:
        if col not in out.columns:
            out[col] = np.nan

    out["competition"] = competition
    out["season"] = season

    # Parse score -> goals
    if "score" not in out.columns:
        out["score"] = np.nan

    goals = out["score"].apply(_parse_score)
    out["home_goals"] = [g[0] for g in goals]
    out["away_goals"] = [g[1] for g in goals]

    # Parse date
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Target label: H/D/A for played matches only
    def outcome(row: pd.Series) -> Optional[str]:
        hg, ag = row.get("home_goals"), row.get("away_goals")
        if pd.isna(hg) or pd.isna(ag):
            return None
        if hg > ag:
            return "H"
        if hg < ag:
            return "A"
        return "D"

    out["result"] = out.apply(outcome, axis=1)

    # Basic cleanup
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()

    # Drop obvious junk rows
    out = out[~out["home_team"].isin({"nan", "None", ""})].copy()
    out = out[~out["away_team"].isin({"nan", "None", ""})].copy()

    # Keep relevant columns if present
    keep = [
        "competition",
        "season",
        "date",
        "time",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_xg",
        "away_xg",
        "venue",
        "attendance",
        "notes",
        "result",
    ]
    cols = [c for c in keep if c in out.columns]
    return out[cols].copy()
