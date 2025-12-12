from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .fbref import Competition, FBREF_BASE, _soup_with_uncommented_tables
from .utils import HttpCache


def _norm_team(s: str) -> str:
    s = "" if s is None else str(s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s


def _pick_matchlog_table(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    # Prefer a table that looks like a match log: has Date + Opponent.
    for df in dfs:
        cols = {str(c) for c in df.columns}
        if "Date" in cols and ("Opponent" in cols or "Opp" in cols):
            return df
    # fallback to the first table
    return dfs[0]


def fetch_team_matchlogs(cache: HttpCache, url: str, team_name: str) -> pd.DataFrame:
    """Fetch a team matchlog schedule/stats table and normalize key columns."""
    html = cache.get(url)
    soup = _soup_with_uncommented_tables(html)
    tables = soup.find_all("table")
    if not tables:
        raise ValueError(f"No tables found at {url}")

    dfs = pd.read_html(str(tables[0]))
    # Some pages have multiple tables in the HTML; fallback to full page if needed.
    if not dfs:
        dfs = pd.read_html(html)
    if not dfs:
        raise ValueError(f"Unable to parse any tables at {url}")

    df = _pick_matchlog_table(dfs).copy()

    # Drop repeated header rows
    if "Date" in df.columns:
        df = df[df["Date"].astype(str) != "Date"].copy()

    rename = {
        "Date": "date",
        "Venue": "venue",
        "Opponent": "opponent",
        "Opp": "opponent",
        "Comp": "comp",
        "GF": "gf",
        "GA": "ga",
        "xG": "xg",
        "xGA": "xga",
        "npxG": "npxg",
        "npxGA": "npxga",
        "Sh": "sh",
        "SoT": "sot",
        "Poss": "poss",
        "CrdY": "crdy",
        "CrdR": "crdr",
        "PK": "pk",
        "PKatt": "pkatt",
        "Cmp": "pass_cmp",
        "Att": "pass_att",
        "Cmp%": "pass_cmp_pct",
        "PrgP": "prgp",
        "PrgC": "prgc",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    df["team"] = team_name
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for c in [
        "gf",
        "ga",
        "xg",
        "xga",
        "npxg",
        "npxga",
        "sh",
        "sot",
        "poss",
        "crdy",
        "crdr",
        "pk",
        "pkatt",
        "pass_cmp",
        "pass_att",
        "pass_cmp_pct",
        "prgp",
        "prgc",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Keep only useful columns
    keep = [
        "team",
        "date",
        "venue",
        "opponent",
        "comp",
        "gf",
        "ga",
        "xg",
        "xga",
        "npxg",
        "npxga",
        "sh",
        "sot",
        "poss",
        "crdy",
        "crdr",
        "pk",
        "pkatt",
        "pass_cmp",
        "pass_att",
        "pass_cmp_pct",
        "prgp",
        "prgc",
    ]
    cols = [c for c in keep if c in df.columns]
    return df[cols].copy()


def list_squad_links_for_comp_season(cache: HttpCache, comp: Competition, season: str) -> list[tuple[str, str]]:
    """Return (team_name, team_stats_url) pairs for a competition season."""
    html = cache.get(comp.season_page(season))
    soup = _soup_with_uncommented_tables(html)

    out: list[tuple[str, str]] = []
    seen: set[str] = set()

    for a in soup.select('a[href^="/en/squads/"]'):
        href = a.get("href") or ""
        if not href.startswith("/en/squads/"):
            continue
        # Prefer season-qualified squad stats pages: /en/squads/<id>/<season>/...
        if f"/en/squads/" in href and f"/{season}/" in href:
            team = a.get_text(strip=True)
            if not team:
                continue
            full = f"{FBREF_BASE}{href}"
            if full in seen:
                continue
            seen.add(full)
            out.append((team, full))

    return out


def find_matchlogs_url(cache: HttpCache, team_stats_url: str, comp_id: int) -> Optional[str]:
    """From a squad stats page, find a matchlogs schedule/stats URL (best-effort)."""
    html = cache.get(team_stats_url)
    soup = _soup_with_uncommented_tables(html)

    # Prefer competition-specific matchlogs links.
    cand: list[str] = []
    for a in soup.select('a[href*="/matchlogs/"]'):
        href = a.get("href") or ""
        if not href.startswith("/en/"):
            continue
        if "schedule" not in href:
            continue
        cand.append(href)

    # Heuristics: prefer links containing /c<comp_id>/
    for href in cand:
        if f"/c{comp_id}/" in href:
            return f"{FBREF_BASE}{href}"

    # Next best: any matchlogs schedule
    if cand:
        return f"{FBREF_BASE}{cand[0]}"

    return None


def build_match_level_stats_from_teamlogs(
    teamlogs: pd.DataFrame, competition: str, season: str
) -> pd.DataFrame:
    """Convert team-perspective matchlogs to match-perspective home/away rows."""
    df = teamlogs.copy()

    if "date" not in df.columns or "opponent" not in df.columns or "team" not in df.columns:
        return pd.DataFrame()

    # Normalize venue into Home/Away/Neutral if present
    if "venue" not in df.columns:
        df["venue"] = np.nan

    out_rows: list[dict[str, object]] = []
    stat_cols = [c for c in df.columns if c not in {"team", "date", "venue", "opponent", "comp"}]

    for _, r in df.iterrows():
        date = r.get("date")
        team = r.get("team")
        opp = r.get("opponent")
        venue = str(r.get("venue") or "").strip().lower()

        if pd.isna(date) or not team or not opp:
            continue

        if venue == "home":
            home_team, away_team = str(team), str(opp)
            side = "home"
        elif venue == "away":
            home_team, away_team = str(opp), str(team)
            side = "away"
        else:
            # Neutral/unknown: keep team as home-like for determinism.
            home_team, away_team = str(team), str(opp)
            side = "home"

        row: dict[str, object] = {
            "competition": competition,
            "season": season,
            "date": pd.to_datetime(date),
            "home_team": home_team,
            "away_team": away_team,
        }

        for c in stat_cols:
            val = r.get(c)
            if side == "home":
                row[f"home_{c}"] = val
            else:
                row[f"away_{c}"] = val

        out_rows.append(row)

    if not out_rows:
        return pd.DataFrame()

    out = pd.DataFrame(out_rows)

    # If we created only home_* or only away_* columns, fill missing counterpart with NaN
    for c in list(out.columns):
        if c.startswith("home_"):
            base = c[len("home_") :]
            out.setdefault(f"away_{base}", np.nan)
        if c.startswith("away_"):
            base = c[len("away_") :]
            out.setdefault(f"home_{base}", np.nan)

    return out


def merge_match_stats(fixtures: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Merge match-level stats into fixtures using normalized keys."""
    if fixtures.empty or stats.empty:
        return fixtures

    fx = fixtures.copy()
    st = stats.copy()

    fx["_date"] = pd.to_datetime(fx["date"], errors="coerce")
    st["_date"] = pd.to_datetime(st["date"], errors="coerce")

    fx["_hn"] = fx["home_team"].map(_norm_team)
    fx["_an"] = fx["away_team"].map(_norm_team)
    st["_hn"] = st["home_team"].map(_norm_team)
    st["_an"] = st["away_team"].map(_norm_team)

    key = ["competition", "season", "_date", "_hn", "_an"]

    # Deduplicate stats on key (keep first non-null rows)
    st = st.sort_values(["_date"]).groupby(key, as_index=False).first()

    merged = fx.merge(st.drop(columns=["date"], errors="ignore"), on=key, how="left", suffixes=("", "_st"))

    # Prefer existing fixture xG if present; otherwise fill from stats gf/ga/xg.
    # (This is conservative; you can change precedence if you prefer matchlogs.)
    for col in list(merged.columns):
        if col.endswith("_st"):
            base = col[:-3]
            if base in merged.columns:
                merged[base] = merged[base].combine_first(merged[col])
                merged = merged.drop(columns=[col])
            else:
                merged = merged.rename(columns={col: base})

    merged = merged.drop(columns=["_date", "_hn", "_an"], errors="ignore")
    return merged


def enrich_fixtures_with_fbref_teamlogs(
    cache: HttpCache,
    fixtures: pd.DataFrame,
    comp: Competition,
    season: str,
    max_teams: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch team matchlogs for a comp/season and merge into fixtures."""
    squads = list_squad_links_for_comp_season(cache, comp, season)
    if max_teams is not None:
        squads = squads[: max_teams]

    all_stats: list[pd.DataFrame] = []
    for team_name, team_stats_url in squads:
        murl = find_matchlogs_url(cache, team_stats_url, comp.comp_id)
        if not murl:
            continue
        tlog = fetch_team_matchlogs(cache, murl, team_name=team_name)
        if tlog.empty:
            continue
        mstats = build_match_level_stats_from_teamlogs(tlog, competition=comp.name, season=season)
        if not mstats.empty:
            all_stats.append(mstats)

    if not all_stats:
        return fixtures

    stats_df = pd.concat(all_stats, ignore_index=True)
    return merge_match_stats(fixtures, stats_df)
