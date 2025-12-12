from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
from bs4 import BeautifulSoup, Comment

from .utils import HttpCache


FBREF_BASE = "https://fbref.com"

DEFAULT_COMPETITIONS: list[Competition] = [
    Competition(name="Premier League", comp_id=9),
    Competition(name="La Liga", comp_id=12),
    Competition(name="Serie A", comp_id=11),
    Competition(name="Bundesliga", comp_id=20),
    Competition(name="Ligue 1", comp_id=13),
    Competition(name="Champions League", comp_id=8),
    Competition(name="Europa League", comp_id=19),
    # Best-effort; if FBref changes this ID, pass --comp-ids explicitly.
    Competition(name="Europa Conference League", comp_id=882),
]


@dataclass(frozen=True)
class Competition:
    name: str
    comp_id: int

    def season_page(self, season: str) -> str:
        # season examples: "2024-2025" (leagues) or "2024-2025" (UEFA comps)
        return f"{FBREF_BASE}/en/comps/{self.comp_id}/{season}"


def _soup_with_uncommented_tables(html: str) -> BeautifulSoup:
    """FBref often wraps tables in HTML comments; unwrap them."""
    soup = BeautifulSoup(html, "lxml")
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        txt = str(c)
        if "<table" in txt and "</table>" in txt:
            try:
                c.replace_with(BeautifulSoup(txt, "lxml"))
            except Exception:
                # If parsing fails, keep original comment
                continue
    return soup


def discover_competitions(cache: HttpCache) -> dict[str, Competition]:
    """Return a map of competition name -> Competition discovered from /en/comps/."""
    html = cache.get(f"{FBREF_BASE}/en/comps/")
    soup = _soup_with_uncommented_tables(html)

    comps: dict[str, Competition] = {}
    # Match /en/comps/<id>/ links that have a nearby text label
    for a in soup.select('a[href^="/en/comps/"]'):
        href = a.get("href") or ""
        m = re.match(r"^/en/comps/(\d+)(?:/|$)", href)
        if not m:
            continue
        comp_id = int(m.group(1))
        name = a.get_text(strip=True)
        if not name:
            continue
        # Keep first seen name for id; avoid duplicates like "9" shown in many places
        if name not in comps:
            comps[name] = Competition(name=name, comp_id=comp_id)

    return comps


def resolve_requested_competitions(cache: HttpCache) -> list[Competition]:
    """Resolve Top 5 leagues + UCL/UEL/UECL by name from discovery."""
    try:
        comps = discover_competitions(cache)
    except Exception:
        comps = {}

    wanted = [
        "Premier League",
        "La Liga",
        "Serie A",
        "Bundesliga",
        "Ligue 1",
        "Champions League",
        "Europa League",
        "Europa Conference League",
    ]

    # Strategy: exact match first, then substring match (case-insensitive)
    out: list[Competition] = []
    used_ids: set[int] = set()

    def pick(label: str) -> Optional[Competition]:
        if label in comps:
            return comps[label]
        ll = label.lower()
        for k, v in comps.items():
            if ll == k.lower():
                return v
        for k, v in comps.items():
            if ll in k.lower():
                return v
        return None

    for w in wanted:
        c = pick(w)
        if c is None:
            continue
        if c.comp_id in used_ids:
            continue
        out.append(c)
        used_ids.add(c.comp_id)

    if len(out) == len(wanted):
        return out

    # Fallback to known IDs for any missing competitions
    default_by_name = {c.name: c for c in DEFAULT_COMPETITIONS}
    for w in wanted:
        if any(x.name == w for x in out):
            continue
        c = default_by_name.get(w)
        if c and c.comp_id not in used_ids:
            out.append(c)
            used_ids.add(c.comp_id)

    return out


def list_seasons(cache: HttpCache, comp: Competition, max_seasons: int = 8) -> list[str]:
    """List seasons available for a competition (best-effort)."""
    html = cache.get(f"{FBREF_BASE}/en/comps/{comp.comp_id}/")
    soup = _soup_with_uncommented_tables(html)

    seasons: list[str] = []
    # Look for season-like links under the competition page
    for a in soup.select('a[href^="/en/comps/"]'):
        href = a.get("href") or ""
        if f"/en/comps/{comp.comp_id}/" not in href:
            continue

        # season in path
        m = re.search(rf"/en/comps/{comp.comp_id}/(\d{{4}}-\d{{4}}|\d{{4}})(?:/|$)", href)
        if not m:
            continue
        season = m.group(1)
        if season not in seasons:
            seasons.append(season)

    # Prefer most recent-looking seasons first
    def season_key(s: str) -> tuple[int, int]:
        if "-" in s:
            a, b = s.split("-", 1)
            return (int(a), int(b))
        return (int(s), int(s))

    seasons = sorted(seasons, key=season_key, reverse=True)
    return seasons[:max_seasons]


def season_schedule_url(cache: HttpCache, comp: Competition, season: str) -> Optional[str]:
    """Find the "Scores and Fixtures" schedule link for a specific season."""
    html = cache.get(comp.season_page(season))
    soup = _soup_with_uncommented_tables(html)

    # Many pages include a nav link containing "Scores and Fixtures"
    for a in soup.select('a[href*="schedule"]'):
        text = a.get_text(" ", strip=True)
        href = a.get("href") or ""
        if "Scores and Fixtures" in text and href.startswith("/en/"):
            return f"{FBREF_BASE}{href}"

    # Fallback: first schedule link for the season
    for a in soup.select('a[href*="/schedule/"]'):
        href = a.get("href") or ""
        if href.startswith("/en/") and f"/en/comps/{comp.comp_id}/" in href:
            return f"{FBREF_BASE}{href}"

    return None


def fetch_scores_and_fixtures(cache: HttpCache, schedule_url: str) -> pd.DataFrame:
    """Fetch Scores & Fixtures table from a schedule URL."""
    html = cache.get(schedule_url)
    soup = _soup_with_uncommented_tables(html)

    # Prefer the standard schedule table id if present
    table = soup.select_one("table#sched_all") or soup.select_one("table#sched")
    if table is None:
        # fallback to first table
        tables = soup.find_all("table")
        if not tables:
            raise ValueError(f"No tables found at {schedule_url}")
        table = tables[0]

    df_list = pd.read_html(str(table))
    if not df_list:
        raise ValueError(f"Unable to parse schedule table at {schedule_url}")

    df = df_list[0].copy()
    # Remove repeated header rows
    if "Wk" in df.columns:
        df = df[df["Wk"].astype(str) != "Wk"].copy()

    return df
