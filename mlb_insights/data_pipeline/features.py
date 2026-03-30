"""
mlb_insights/data_pipeline/features.py -- Build batter and pitcher feature tables.

Refactored from mlb/features/build_features.py with date-scoped operations
so the pipeline can run incrementally for a single date or in bulk.
"""

import logging
import sqlite3

import numpy as np
import pandas as pd

from mlb_insights.config import (
    VALID_EVENTS, AB_EVENTS, HIT_EVENTS, TB_MAP, ROLLING_WINDOWS,
)

logger = logging.getLogger(__name__)


# ── Flag Helpers ─────────────────────────────────────────────────────────────

def _add_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary flag columns used by aggregation functions."""
    df = df.copy()
    df["hit_flag"] = df["events"].isin(HIT_EVENTS).astype(int)
    df["hr_flag"] = (df["events"] == "home_run").astype(int)
    df["bb_flag"] = (df["events"] == "walk").astype(int)
    df["hbp_flag"] = (df["events"] == "hit_by_pitch").astype(int)
    df["so_flag"] = (df["events"] == "strikeout").astype(int)
    df["ab_flag"] = df["events"].isin(AB_EVENTS).astype(int)
    df["tb"] = df["events"].map(TB_MAP).fillna(0)
    return df


# ── Streak Computation ───────────────────────────────────────────────────────

def _compute_streak(series: pd.Series) -> int:
    """Compute current streak length from a series of daily hit counts.

    Positive = consecutive games with >= 1 hit.
    Negative = consecutive games with 0 hits.
    """
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


# ── Rebound Tier ─────────────────────────────────────────────────────────────

def _rebound_tier(row) -> str | None:
    """Classify rebound likelihood based on hit tier and recent rolling counts."""
    tier = row.get("hit_tier")

    def g(i):
        return row.get(f"hits_pg_{i}", 0) or 0

    if tier == "Elite":
        if g(1) == 0 and g(2) >= 1 and g(3) >= 2:
            return "Very Likely Rebound"
        if g(1) == 0 and g(2) >= 1:
            return "Likely Rebound"
        if g(1) == 0:
            return "Potential Rebound"
    elif tier == "Above Avg":
        if g(1) == 0 and g(2) == 0 and g(3) >= 1 and g(5) >= 2:
            return "Very Likely Rebound"
        if g(1) == 0 and g(2) == 0 and g(3) >= 1:
            return "Likely Rebound"
        if g(1) == 0 and g(2) == 0:
            return "Potential Rebound"
    elif tier == "Below Avg":
        if all(g(i) == 0 for i in [1, 2, 3]) and g(5) >= 2:
            return "Very Likely Rebound"
        if all(g(i) == 0 for i in [1, 2, 3]) and g(5) >= 1:
            return "Likely Rebound"
        if all(g(i) == 0 for i in [1, 2, 3]):
            return "Potential Rebound"
    elif tier == "Poor":
        if all(g(i) == 0 for i in [1, 2, 3, 4, 5]) and g(10) >= 1:
            return "Very Likely Rebound"
        if all(g(i) == 0 for i in [1, 2, 3, 4]) and g(5) >= 1:
            return "Likely Rebound"
        if all(g(i) == 0 for i in [1, 2, 3]):
            return "Potential Rebound"
    return None


# ── Batter Features ──────────────────────────────────────────────────────────

def build_batter_features(conn: sqlite3.Connection, start_date: str = "2024-03-20") -> pd.DataFrame:
    """Compute rolling stats, streaks, and season aggregates for all batters.

    Reads from statcast_staging (which has richer columns than statcast_raw),
    falling back to statcast_raw if staging is empty.

    Args:
        conn: Open sqlite3 connection.
        start_date: Earliest date to include.

    Returns:
        DataFrame with one row per (game_date, batter_id).
    """
    # Try statcast_staging first, fall back to statcast_raw
    df = pd.read_sql(
        "SELECT * FROM statcast_staging WHERE game_date >= ? AND events IS NOT NULL",
        conn, params=(start_date,),
    )
    if df.empty:
        logger.info("statcast_staging empty, trying statcast_raw ...")
        df = pd.read_sql(
            "SELECT * FROM statcast_raw WHERE game_date >= ?",
            conn, params=(start_date,),
        )

    if df.empty:
        logger.warning("No statcast data found from %s onward.", start_date)
        return pd.DataFrame()

    df = df[df["events"].isin(VALID_EVENTS)].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    # Ensure batter/pitcher columns are int (SQLite append can produce mixed types)
    for col in ("batter", "pitcher"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["batter"])
    df = _add_flags(df)

    # Daily aggregation per batter
    batter_daily = (
        df.groupby(["game_date", "batter"])
        .agg(
            pa=("events", "count"),
            ab=("ab_flag", "sum"),
            hits=("hit_flag", "sum"),
            hr=("hr_flag", "sum"),
            bb=("bb_flag", "sum"),
            hbp=("hbp_flag", "sum"),
            so=("so_flag", "sum"),
            tb=("tb", "sum"),
        )
        .reset_index()
    )

    batter_daily["avg"] = (batter_daily["hits"] / batter_daily["ab"]).round(3).fillna(0)
    batter_daily["obp"] = (
        (batter_daily["hits"] + batter_daily["bb"] + batter_daily["hbp"])
        / (batter_daily["ab"] + batter_daily["bb"] + batter_daily["hbp"])
    ).round(3).fillna(0)
    batter_daily["slg"] = (batter_daily["tb"] / batter_daily["ab"]).round(3).fillna(0)

    batter_daily = batter_daily.sort_values(["batter", "game_date"])

    # Rolling hit counts (shifted by 1 so we look backward, not including today)
    for window in ROLLING_WINDOWS:
        batter_daily[f"hits_pg_{window}"] = (
            batter_daily.groupby("batter")["hits"]
            .shift(1)
            .rolling(window, min_periods=1)
            .sum()
            .reset_index(level=0, drop=True)
        )

    # Current streak
    batter_daily["current_streak"] = (
        batter_daily.groupby("batter")["hits"]
        .transform(lambda s: s.expanding().apply(lambda x: _compute_streak(x), raw=True))
        .astype(int)
    )

    # Season aggregates (cumulative, excluding current game)
    batter_daily["season_hits"] = (
        batter_daily.groupby("batter")["hits"].cumsum() - batter_daily["hits"]
    )
    batter_daily["season_pa"] = (
        batter_daily.groupby("batter")["pa"].cumsum() - batter_daily["pa"]
    )
    season_ab = batter_daily.groupby("batter")["ab"].cumsum() - batter_daily["ab"]
    batter_daily["season_hit_pct"] = (
        batter_daily["season_hits"] / season_ab
    ).round(3).fillna(0)

    # Hit tier
    batter_daily["hit_tier"] = pd.cut(
        batter_daily["season_hit_pct"],
        bins=[-0.01, 0.199, 0.249, 0.299, 1.0],
        labels=["Poor", "Below Avg", "Above Avg", "Elite"],
    ).astype(str)

    # Rebound tier
    batter_daily["rebound_tier"] = batter_daily.apply(_rebound_tier, axis=1)

    # Rename batter -> batter_id for consistency
    batter_daily = batter_daily.rename(columns={"batter": "batter_id"})

    logger.info(
        "Built batter features: %d rows for %d batters.",
        len(batter_daily), batter_daily["batter_id"].nunique(),
    )
    return batter_daily


def write_batter_features(conn: sqlite3.Connection, df: pd.DataFrame):
    """Write batter features to the batter_stats table (full replace)."""
    if df.empty:
        logger.warning("Empty batter features DataFrame, nothing to write.")
        return

    conn.execute("DELETE FROM batter_stats")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["game_date"])[:10], int(r["batter_id"]), None, None,
            int(r.get("pa", 0)), int(r.get("ab", 0)), int(r.get("hits", 0)),
            int(r.get("hr", 0)), int(r.get("bb", 0)), int(r.get("so", 0)),
            float(r.get("avg", 0)), float(r.get("obp", 0)), float(r.get("slg", 0)),
            float(r.get("hits_pg_1", 0) or 0), float(r.get("hits_pg_2", 0) or 0),
            float(r.get("hits_pg_3", 0) or 0), float(r.get("hits_pg_5", 0) or 0),
            float(r.get("hits_pg_10", 0) or 0), float(r.get("hits_pg_20", 0) or 0),
            int(r.get("current_streak", 0)),
            r.get("rebound_tier"),
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
    logger.info("Wrote %d batter_stats rows.", len(rows))


# ── Pitcher Features ─────────────────────────────────────────────────────────

def build_pitcher_features(conn: sqlite3.Connection, start_date: str = "2024-03-20") -> pd.DataFrame:
    """Compute per-game pitcher stats from statcast data.

    Args:
        conn: Open sqlite3 connection.
        start_date: Earliest date to include.

    Returns:
        DataFrame with one row per (game_date, pitcher_id).
    """
    df = pd.read_sql(
        "SELECT * FROM statcast_staging WHERE game_date >= ? AND events IS NOT NULL",
        conn, params=(start_date,),
    )
    if df.empty:
        df = pd.read_sql(
            "SELECT * FROM statcast_raw WHERE game_date >= ?",
            conn, params=(start_date,),
        )

    if df.empty:
        logger.warning("No statcast data for pitcher features.")
        return pd.DataFrame()

    df = df[df["events"].isin(VALID_EVENTS)].copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ("batter", "pitcher"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pitcher"])
    df = _add_flags(df)

    pitcher_daily = (
        df.groupby(["game_date", "pitcher"])
        .agg(
            batters_faced=("events", "count"),
            hits_allowed=("hit_flag", "sum"),
            hr_allowed=("hr_flag", "sum"),
            bb_allowed=("bb_flag", "sum"),
            so=("so_flag", "sum"),
        )
        .reset_index()
    )

    pitcher_daily["avg_against"] = (
        pitcher_daily["hits_allowed"] / pitcher_daily["batters_faced"]
    ).round(3)

    # Grab handedness from statcast if available
    if "p_throws" in df.columns:
        hand_df = (
            df.groupby("pitcher")["p_throws"]
            .agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else None)
            .reset_index()
            .rename(columns={"p_throws": "throws"})
        )
        pitcher_daily = pitcher_daily.merge(hand_df, on="pitcher", how="left")
    else:
        pitcher_daily["throws"] = None

    pitcher_daily = pitcher_daily.rename(columns={"pitcher": "pitcher_id"})

    logger.info(
        "Built pitcher features: %d rows for %d pitchers.",
        len(pitcher_daily), pitcher_daily["pitcher_id"].nunique(),
    )
    return pitcher_daily


def write_pitcher_features(conn: sqlite3.Connection, df: pd.DataFrame):
    """Write pitcher features to the pitcher_stats table (full replace)."""
    if df.empty:
        return

    conn.execute("DELETE FROM pitcher_stats")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            str(r["game_date"])[:10],
            int(r["pitcher_id"]),
            None,  # pitcher_name
            None,  # team
            r.get("throws"),
            int(r.get("batters_faced", 0)),
            int(r.get("hits_allowed", 0)),
            int(r.get("hr_allowed", 0)),
            int(r.get("bb_allowed", 0)),
            int(r.get("so", 0)),
            float(r.get("avg_against", 0)),
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO pitcher_stats
            (game_date, pitcher_id, pitcher_name, team, throws,
             batters_faced, hits_allowed, hr_allowed, bb_allowed, so, avg_against)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d pitcher_stats rows.", len(rows))


# ── Matchup Features ─────────────────────────────────────────────────────────

def build_matchup_features(conn: sqlite3.Connection, start_date: str = "2024-03-20") -> pd.DataFrame:
    """Build batter-vs-pitcher matchup stats from statcast.

    Returns:
        DataFrame with one row per (batter, pitcher) pair.
    """
    df = pd.read_sql(
        "SELECT * FROM statcast_staging WHERE game_date >= ? AND events IS NOT NULL",
        conn, params=(start_date,),
    )
    if df.empty:
        df = pd.read_sql(
            "SELECT * FROM statcast_raw WHERE game_date >= ?",
            conn, params=(start_date,),
        )

    if df.empty:
        return pd.DataFrame()

    df = df[df["events"].isin(VALID_EVENTS)].copy()
    for col in ("batter", "pitcher"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df = df.dropna(subset=["batter", "pitcher"])
    df = _add_flags(df)

    m = (
        df.groupby(["batter", "pitcher"])
        .agg(
            pa=("events", "count"),
            hits=("hit_flag", "sum"),
            hr=("hr_flag", "sum"),
            bb=("bb_flag", "sum"),
            so=("so_flag", "sum"),
            ab=("ab_flag", "sum"),
            tb=("tb", "sum"),
            hbp=("hbp_flag", "sum"),
        )
        .reset_index()
    )
    m["avg"] = (m["hits"] / m["ab"]).round(3).fillna(0)
    m["obp"] = ((m["hits"] + m["bb"] + m["hbp"]) / (m["ab"] + m["bb"] + m["hbp"])).round(3).fillna(0)
    m["slg"] = (m["tb"] / m["ab"]).round(3).fillna(0)

    return m


def write_matchup_features(conn: sqlite3.Connection, df: pd.DataFrame):
    """Write matchup features to the matchup_stats table."""
    if df.empty:
        return

    conn.execute("DELETE FROM matchup_stats")
    rows = []
    for _, r in df.iterrows():
        rows.append((
            int(r["batter"]), int(r["pitcher"]),
            int(r["pa"]), int(r["hits"]), int(r["hr"]),
            int(r["bb"]), int(r["so"]),
            float(r["avg"]), float(r["obp"]), float(r["slg"]),
            None,  # last_updated
        ))

    conn.executemany("""
        INSERT OR REPLACE INTO matchup_stats
            (batter_id, pitcher_id, pa, hits, hr, bb, so, avg, obp, slg, last_updated)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()
    logger.info("Wrote %d matchup_stats rows.", len(rows))
