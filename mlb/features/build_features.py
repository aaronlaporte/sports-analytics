"""
mlb/features/build_features.py — Build batter/pitcher/matchup feature tables in mlb.db.

Reads from statcast_raw, computes rolling stats, rebound tiers, and season
aggregates, then writes to batter_stats, pitcher_stats, matchup_stats tables.

Usage:
    python mlb/features/build_features.py
    python mlb/features/build_features.py --start 2025-03-18
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn, create_schema

VALID_EVENTS = [
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "grounded_into_double_play",
    "strikeout", "walk", "hit_by_pitch",
]

AB_EVENTS = [
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "grounded_into_double_play", "strikeout",
]


def load_statcast(conn, start: str) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT * FROM statcast_raw WHERE game_date >= ?",
        conn, params=(start,)
    )
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df[df["events"].isin(VALID_EVENTS)].copy()


def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    df["hit_flag"] = df["events"].isin(["single", "double", "triple", "home_run"]).astype(int)
    df["hr_flag"]  = (df["events"] == "home_run").astype(int)
    df["bb_flag"]  = (df["events"] == "walk").astype(int)
    df["hbp_flag"] = (df["events"] == "hit_by_pitch").astype(int)
    df["so_flag"]  = (df["events"] == "strikeout").astype(int)
    df["ab_flag"]  = df["events"].isin(AB_EVENTS).astype(int)
    df["tb"] = df["events"].map(
        {"single": 1, "double": 2, "triple": 3, "home_run": 4}
    ).fillna(0)
    return df


def build_batter_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_flags(df)

    daily = df.groupby(["game_date", "batter", "pitcher"]).agg(
        pa=("events", "count"),
        ab=("ab_flag", "sum"),
        hits=("hit_flag", "sum"),
        hr=("hr_flag", "sum"),
        bb=("bb_flag", "sum"),
        hbp=("hbp_flag", "sum"),
        so=("so_flag", "sum"),
        tb=("tb", "sum"),
    ).reset_index()

    # Collapse to one row per batter per game date
    batter_daily = daily.groupby(["game_date", "batter"]).agg(
        pa=("pa", "sum"),
        ab=("ab", "sum"),
        hits=("hits", "sum"),
        hr=("hr", "sum"),
        bb=("bb", "sum"),
        hbp=("hbp", "sum"),
        so=("so", "sum"),
        tb=("tb", "sum"),
    ).reset_index()

    batter_daily["avg"] = (batter_daily["hits"] / batter_daily["ab"]).round(3).fillna(0)
    batter_daily["obp"] = (
        (batter_daily["hits"] + batter_daily["bb"] + batter_daily["hbp"]) /
        (batter_daily["ab"] + batter_daily["bb"] + batter_daily["hbp"])
    ).round(3).fillna(0)
    batter_daily["slg"] = (batter_daily["tb"] / batter_daily["ab"]).round(3).fillna(0)

    # Sort chronologically for rolling calculations
    batter_daily = batter_daily.sort_values(["batter", "game_date"])

    # Rolling hit counts (looking backward from each game)
    for window in [1, 2, 3, 5, 10, 20]:
        batter_daily[f"hits_pg_{window}"] = (
            batter_daily.groupby("batter")["hits"]
            .shift(1)
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # Current streak (positive = hot, negative = cold)
    def compute_streak(series: pd.Series) -> int:
        vals = list(series)
        if not vals:
            return 0
        streak = 1 if vals[-1] > 0 else -1
        for v in reversed(vals[:-1]):
            if (v > 0) == (vals[-1] > 0):
                streak += (1 if vals[-1] > 0 else -1)
            else:
                break
        return streak

    batter_daily["current_streak"] = (
        batter_daily.groupby("batter")["hits"]
        .transform(lambda s: s.expanding().apply(lambda x: compute_streak(x), raw=True))
        .astype(int)
    )

    # Season aggregates (cumulative up to but not including current game)
    batter_daily["season_hits"] = (
        batter_daily.groupby("batter")["hits"].cumsum() - batter_daily["hits"]
    )
    batter_daily["season_pa"] = (
        batter_daily.groupby("batter")["pa"].cumsum() - batter_daily["pa"]
    )
    batter_daily["season_ab"] = (
        batter_daily.groupby("batter")["ab"].cumsum() - batter_daily["ab"]
    )
    batter_daily["season_hit_pct"] = (
        batter_daily["season_hits"] / batter_daily["season_ab"]
    ).round(3).fillna(0)

    # Qualification (>=3.1 PA/game)
    pa_per_game = (
        batter_daily.groupby("batter")
        .apply(lambda g: g["pa"].sum() / g["game_date"].nunique())
        .reset_index(name="pa_per_game")
    )
    batter_daily = batter_daily.merge(pa_per_game, on="batter", how="left")
    batter_daily["qualified"] = batter_daily["pa_per_game"] >= 3.1

    # Tier by season hit pct
    batter_daily["hit_tier"] = pd.cut(
        batter_daily["season_hit_pct"],
        bins=[-0.01, 0.199, 0.249, 0.299, 1.0],
        labels=["Poor", "Below Avg", "Above Avg", "Elite"],
    ).astype(str)
    batter_daily.loc[~batter_daily["qualified"], "hit_tier"] = "Unqualified"

    # Rebound tier
    def rebound_tier(row):
        tier = row.get("hit_tier")
        def g(i): return row.get(f"hits_pg_{i}", 0) or 0
        if tier == "Elite":
            if g(1) == 0 and g(2) >= 1 and g(3) >= 2: return "Very Likely Rebound"
            if g(1) == 0 and g(2) >= 1: return "Likely Rebound"
            if g(1) == 0: return "Potential Rebound"
        elif tier == "Above Avg":
            if g(1) == 0 and g(2) == 0 and g(3) >= 1 and g(5) >= 2: return "Very Likely Rebound"
            if g(1) == 0 and g(2) == 0 and g(3) >= 1: return "Likely Rebound"
            if g(1) == 0 and g(2) == 0: return "Potential Rebound"
        elif tier == "Below Avg":
            if all(g(i) == 0 for i in [1,2,3]) and g(5) >= 2: return "Very Likely Rebound"
            if all(g(i) == 0 for i in [1,2,3]) and g(5) >= 1: return "Likely Rebound"
            if all(g(i) == 0 for i in [1,2,3]): return "Potential Rebound"
        elif tier == "Poor":
            if all(g(i) == 0 for i in [1,2,3,4,5]) and g(10) >= 1: return "Very Likely Rebound"
            if all(g(i) == 0 for i in [1,2,3,4]) and g(5) >= 1: return "Likely Rebound"
            if all(g(i) == 0 for i in [1,2,3]): return "Potential Rebound"
        return None

    batter_daily["rebound_tier"] = batter_daily.apply(rebound_tier, axis=1)

    return batter_daily


def build_pitcher_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_flags(df)
    return df.groupby(["game_date", "pitcher"]).agg(
        batters_faced=("events", "count"),
        hits_allowed=("hit_flag", "sum"),
        hr_allowed=("hr_flag", "sum"),
        bb_allowed=("bb_flag", "sum"),
        so=("so_flag", "sum"),
    ).reset_index().assign(
        avg_against=lambda d: (d["hits_allowed"] / d["batters_faced"]).round(3)
    )


def build_matchup_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = _add_flags(df)
    m = df.groupby(["batter", "pitcher"]).agg(
        pa=("events", "count"),
        hits=("hit_flag", "sum"),
        hr=("hr_flag", "sum"),
        bb=("bb_flag", "sum"),
        so=("so_flag", "sum"),
        ab=("ab_flag", "sum"),
        tb=("tb", "sum"),
        hbp=("hbp_flag", "sum"),
    ).reset_index()
    m["avg"] = (m["hits"] / m["ab"]).round(3).fillna(0)
    m["obp"] = ((m["hits"] + m["bb"] + m["hbp"]) / (m["ab"] + m["bb"] + m["hbp"])).round(3).fillna(0)
    m["slg"] = (m["tb"] / m["ab"]).round(3).fillna(0)
    return m


def write_batter_stats(conn, df: pd.DataFrame):
    conn.execute("DELETE FROM batter_stats")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["game_date"])[:10], int(r["batter"]), None, None,
            int(r.get("pa", 0)), int(r.get("ab", 0)), int(r.get("hits", 0)),
            int(r.get("hr", 0)), int(r.get("bb", 0)), int(r.get("so", 0)),
            float(r.get("avg", 0)), float(r.get("obp", 0)), float(r.get("slg", 0)),
            float(r.get("hits_pg_1", 0)), float(r.get("hits_pg_2", 0)),
            float(r.get("hits_pg_3", 0)), float(r.get("hits_pg_5", 0)),
            float(r.get("hits_pg_10", 0)), float(r.get("hits_pg_20", 0)),
            int(r.get("current_streak", 0)), r.get("rebound_tier"),
            int(r.get("season_hits", 0)), int(r.get("season_pa", 0)),
            float(r.get("season_hit_pct", 0)),
        ))
    conn.executemany("""
        INSERT OR REPLACE INTO batter_stats
            (game_date, batter_id, batter_name, team,
             pa, ab, hits, hr, bb, so, avg, obp, slg,
             hits_pg_1, hits_pg_2, hits_pg_3, hits_pg_5, hits_pg_10, hits_pg_20,
             current_streak, rebound_tier,
             season_hits, season_pa, season_hit_pct)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    print(f"[features] Wrote {len(rows)} batter_stats rows")


def build_all(start: str):
    create_schema("mlb")
    conn = get_conn("mlb")
    print(f"[features] Loading statcast from {start} ...")
    df = load_statcast(conn, start)
    if df.empty:
        print("[features] No statcast data found. Run pull_statcast.py first.")
        conn.close()
        return

    print(f"[features] Building features on {len(df):,} pitch rows ...")
    batter_stats  = build_batter_stats(df)
    pitcher_stats = build_pitcher_stats(df)
    matchup_stats = build_matchup_stats(df)

    write_batter_stats(conn, batter_stats)
    conn.close()
    print("[features] Done.")
    return batter_stats, pitcher_stats, matchup_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-03-18")
    args = parser.parse_args()
    build_all(args.start)
