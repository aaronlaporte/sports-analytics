"""
mlb/backfill_insights.py — One-time backfill of Statcast data, features, and insights
for 2024 + 2025 seasons.

Pulls raw Statcast data via pybaseball, builds feature tables in mlb.db,
then runs the insights pipeline for every game date to populate
daily_leaderboard, daily_signals, and prediction_tracking.

Usage:
    python mlb/backfill_insights.py
"""

import sys
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from pybaseball import statcast, cache as pb_cache

warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.db import get_conn, create_schema
from mlb.features.build_features import (
    build_batter_stats,
    build_pitcher_stats,
    build_matchup_stats,
    write_batter_stats,
)
from mlb.signals.engine import SignalEngine, signals_to_db_rows
from mlb.model.multi_outcome import compute_multi_outcomes, p_2plus_hits, p_home_run
from mlb.leaderboard.composite_score import score_all_batters
from mlb.leaderboard.rank import rank_leaderboard


# ── Season date ranges ───────────────────────────────────────────────────────

SEASONS = {
    2024: ("2024-03-20", "2024-10-30"),
    2025: ("2025-03-18", "2025-10-24"),
}

# pybaseball pulls in chunks to avoid timeouts
CHUNK_DAYS = 14


def pull_season_statcast(start: str, end: str) -> pd.DataFrame:
    """Pull Statcast data in chunks to avoid timeouts."""
    all_chunks = []
    current = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)

    while current <= end_dt:
        chunk_end = min(current + timedelta(days=CHUNK_DAYS - 1), end_dt)
        s = current.strftime("%Y-%m-%d")
        e = chunk_end.strftime("%Y-%m-%d")
        print(f"    Pulling {s} to {e} ...")
        try:
            chunk = statcast(s, e)
            if chunk is not None and not chunk.empty:
                all_chunks.append(chunk)
                print(f"      -> {len(chunk):,} rows")
            else:
                print(f"      -> 0 rows")
        except Exception as ex:
            print(f"      -> ERROR: {ex}")
        current = chunk_end + timedelta(days=1)

    if not all_chunks:
        return pd.DataFrame()

    combined = pd.concat(all_chunks, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.drop_duplicates(
        subset=["game_date", "game_pk", "at_bat_number", "pitch_number"],
    )
    return combined


def write_statcast_raw(conn, df: pd.DataFrame):
    """Write raw statcast to mlb.db statcast_raw table."""
    cols = [
        "game_pk", "game_date", "batter", "pitcher", "events", "description",
        "bb_type", "hit_distance_sc", "launch_speed", "launch_angle",
        "estimated_ba_using_speedangle",
    ]
    # Also keep fields needed for features that aren't in schema yet but are in the data
    extra_cols = [
        "stand", "p_throws", "home_team", "away_team", "inning_topbot",
        "at_bat_number", "pitch_number",
    ]
    keep_cols = [c for c in cols + extra_cols if c in df.columns]
    subset = df[keep_cols].copy()
    subset["game_date"] = subset["game_date"].dt.strftime("%Y-%m-%d")

    # Write to a staging table that has all columns we need
    subset.to_sql("statcast_staging", conn, if_exists="replace", index=False)
    count = conn.execute("SELECT COUNT(*) FROM statcast_staging").fetchone()[0]
    print(f"  Wrote {count:,} rows to statcast_staging")
    return count


def build_and_write_features(conn, raw_df: pd.DataFrame):
    """Build batter/pitcher/matchup features and write to mlb.db."""
    from mlb.features.build_features import _add_flags, VALID_EVENTS

    df = raw_df[raw_df["events"].isin(VALID_EVENTS)].copy()
    df = _add_flags(df)

    print("  Building batter stats ...")
    batter_stats = build_batter_stats(df)
    write_batter_stats(conn, batter_stats)

    print("  Building pitcher stats ...")
    pitcher_stats = build_pitcher_stats(df)
    conn.execute("DELETE FROM pitcher_stats")
    for _, r in pitcher_stats.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO pitcher_stats
                (game_date, pitcher_id, pitcher_name, team, throws,
                 batters_faced, hits_allowed, hr_allowed, bb_allowed, so, avg_against)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            str(r["game_date"])[:10],
            int(r["pitcher"]),
            None, None, None,
            int(r.get("batters_faced", 0)),
            int(r.get("hits_allowed", 0)),
            int(r.get("hr_allowed", 0)),
            int(r.get("bb_allowed", 0)),
            int(r.get("so", 0)),
            float(r.get("avg_against", 0)),
        ))
    conn.commit()
    print(f"  Wrote {len(pitcher_stats):,} pitcher_stats rows")

    print("  Building matchup stats ...")
    matchup_stats = build_matchup_stats(df)
    conn.execute("DELETE FROM matchup_stats")
    for _, r in matchup_stats.iterrows():
        conn.execute("""
            INSERT OR REPLACE INTO matchup_stats
                (batter_id, pitcher_id, pa, hits, hr, bb, so, avg, obp, slg, last_updated)
            VALUES (?,?,?,?,?,?,?,?,?,?,?)
        """, (
            int(r["batter"]),
            int(r["pitcher"]),
            int(r.get("pa", 0)),
            int(r.get("hits", 0)),
            int(r.get("hr", 0)),
            int(r.get("bb", 0)),
            int(r.get("so", 0)),
            float(r.get("avg", 0)),
            float(r.get("obp", 0)),
            float(r.get("slg", 0)),
            datetime.now().strftime("%Y-%m-%d"),
        ))
    conn.commit()
    print(f"  Wrote {len(matchup_stats):,} matchup_stats rows")

    return batter_stats, pitcher_stats, matchup_stats


def backfill_insights_for_dates(conn, game_dates: list[str]):
    """Run the insights pipeline for each game date."""
    engine = SignalEngine()

    # Load full tables once
    all_batters = pd.read_sql("SELECT * FROM batter_stats ORDER BY batter_id, game_date", conn)
    all_batters["game_date"] = pd.to_datetime(all_batters["game_date"])

    all_pitchers = pd.read_sql("SELECT * FROM pitcher_stats", conn)
    all_pitchers["game_date"] = pd.to_datetime(all_pitchers["game_date"])

    all_matchups = pd.read_sql("SELECT * FROM matchup_stats", conn)

    total_dates = len(game_dates)
    total_leaderboard_rows = 0
    total_signal_rows = 0

    for i, gd in enumerate(game_dates):
        gd_dt = pd.to_datetime(gd)

        # Get most recent batter stats BEFORE this date (simulating daily prediction)
        batter_mask = all_batters["game_date"] < gd_dt
        if not batter_mask.any():
            continue
        batters_before = all_batters[batter_mask]
        latest_batters = batters_before.loc[
            batters_before.groupby("batter_id")["game_date"].idxmax()
        ].copy()

        if latest_batters.empty:
            continue

        # Get pitcher stats up to this date
        pitcher_mask = all_pitchers["game_date"] <= gd_dt
        pitchers_before = all_pitchers[pitcher_mask]
        if not pitchers_before.empty:
            latest_pitchers = pitchers_before.loc[
                pitchers_before.groupby("pitcher_id")["game_date"].idxmax()
            ].copy()
        else:
            latest_pitchers = pd.DataFrame()

        # Simple P(1+ hit) estimate from season stats for backfill
        # (we don't have model_params for historical dates, so use a basic rate)
        latest_batters["p_1hit"] = latest_batters.apply(
            lambda r: _basic_hit_prob(r), axis=1
        )
        latest_batters["expected_pa"] = latest_batters["pa"].clip(lower=3.0, upper=5.0)
        latest_batters["model_prob"] = latest_batters["p_1hit"]

        # Multi-outcome
        for idx, row in latest_batters.iterrows():
            p1 = row["p_1hit"]
            epa = row["expected_pa"]
            if p1 > 0 and p1 < 1:
                p_per_pa = 1 - (1 - p1) ** (1 / epa)
            else:
                p_per_pa = 0.0
            latest_batters.at[idx, "p_2hit"] = p_2plus_hits(p_per_pa, epa)
            latest_batters.at[idx, "p_hr"] = p_home_run(row)

        # Signals
        all_signals = engine.evaluate_all_batters(latest_batters, latest_pitchers, all_matchups)

        # Write signals
        signal_rows = signals_to_db_rows(gd, all_signals)
        for row in signal_rows:
            conn.execute("""
                INSERT OR REPLACE INTO daily_signals
                    (signal_date, player_id, signal_type, confidence,
                     headline, reasons_json, interpretation)
                VALUES (?,?,?,?,?,?,?)
            """, row)

        # Composite scores
        scored = score_all_batters(latest_batters, all_signals)
        ranked = rank_leaderboard(scored, top_n=50)

        # Write leaderboard
        for _, r in ranked.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO daily_leaderboard
                    (prediction_date, player_id, player_name, team, opponent,
                     opp_pitcher, daily_rank, daily_score,
                     p_1hit, p_2hit, p_hr,
                     active_signal_count, top_signal, top_reason)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                gd,
                int(r.get("batter_id", 0)),
                r.get("batter_name"),
                r.get("team"),
                None, None,
                int(r.get("daily_rank", 0)),
                float(r.get("daily_score", 0)),
                float(r.get("p_1hit", 0) or 0),
                float(r.get("p_2hit", 0) or 0),
                float(r.get("p_hr", 0) or 0),
                int(r.get("active_signal_count", 0) or 0),
                r.get("top_signal", ""),
                r.get("top_reason", ""),
            ))

        # Write prediction tracking (actuals will be filled by Phase 2)
        for _, r in ranked.iterrows():
            conn.execute("""
                INSERT OR REPLACE INTO prediction_tracking
                    (prediction_date, player_id, player_name,
                     daily_rank, daily_score, p_1hit, p_2hit, p_hr)
                VALUES (?,?,?,?,?,?,?,?)
            """, (
                gd,
                int(r.get("batter_id", 0)),
                r.get("batter_name"),
                int(r.get("daily_rank", 0)),
                float(r.get("daily_score", 0)),
                float(r.get("p_1hit", 0) or 0),
                float(r.get("p_2hit", 0) or 0),
                float(r.get("p_hr", 0) or 0),
            ))

        total_leaderboard_rows += len(ranked)
        total_signal_rows += len(signal_rows)

        # Commit every 10 dates
        if (i + 1) % 10 == 0:
            conn.commit()
            print(f"  [{i+1}/{total_dates}] Processed through {gd}")

    conn.commit()
    print(f"  Backfill complete: {total_leaderboard_rows:,} leaderboard rows, "
          f"{total_signal_rows:,} signal rows across {total_dates} dates")


def _basic_hit_prob(row: pd.Series) -> float:
    """Basic P(1+ hit) estimate from season stats — used for backfill only."""
    season_hits = int(row.get("season_hits", 0) or 0)
    season_pa = int(row.get("season_pa", 0) or 0)
    league_rate = 0.245
    prior_pa = 100

    if season_pa > 0:
        rate = (season_hits + league_rate * prior_pa) / (season_pa + prior_pa)
    else:
        rate = league_rate

    expected_pa = float(row.get("pa", 3.5) or 3.5)
    expected_pa = min(max(expected_pa, 3.0), 5.0)
    return round(1 - (1 - rate) ** expected_pa, 4)


def main():
    print("=" * 60)
    print("MLB INSIGHTS BACKFILL — 2024 + 2025 Seasons")
    print("=" * 60)

    create_schema("mlb")
    conn = get_conn("mlb")

    all_raw = []

    for season, (start, end) in SEASONS.items():
        print(f"\n[{season}] Pulling Statcast data ({start} to {end}) ...")
        t0 = time.time()
        raw = pull_season_statcast(start, end)
        elapsed = time.time() - t0
        print(f"  [{season}] Got {len(raw):,} rows in {elapsed:.0f}s")
        if not raw.empty:
            all_raw.append(raw)

    if not all_raw:
        print("ERROR: No Statcast data retrieved. Check network/pybaseball.")
        conn.close()
        return

    combined = pd.concat(all_raw, ignore_index=True)
    combined["game_date"] = pd.to_datetime(combined["game_date"])
    combined = combined.drop_duplicates(
        subset=["game_date", "game_pk", "at_bat_number", "pitch_number"],
    )
    print(f"\nTotal raw Statcast: {len(combined):,} rows")
    print(f"Date range: {combined.game_date.min().date()} to {combined.game_date.max().date()}")

    # Write raw to staging
    print("\n[DB] Writing to statcast_staging ...")
    write_statcast_raw(conn, combined)

    # Build features
    print("\n[Features] Building batter/pitcher/matchup features ...")
    t0 = time.time()
    batter_stats, pitcher_stats, matchup_stats = build_and_write_features(conn, combined)
    print(f"  Features built in {time.time() - t0:.0f}s")

    # Get unique game dates for backfill
    game_dates = sorted(
        combined["game_date"].dt.strftime("%Y-%m-%d").unique().tolist()
    )
    # Skip the first 10 days (not enough prior data for meaningful signals)
    game_dates = game_dates[10:]
    print(f"\n[Insights] Backfilling insights for {len(game_dates)} game dates ...")

    t0 = time.time()
    backfill_insights_for_dates(conn, game_dates)
    print(f"  Insights backfill completed in {time.time() - t0:.0f}s")

    # Summary
    print("\n" + "=" * 60)
    print("BACKFILL SUMMARY")
    print("=" * 60)
    for table in ["batter_stats", "pitcher_stats", "matchup_stats",
                   "daily_leaderboard", "daily_signals", "prediction_tracking"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count:,} rows")

    conn.close()
    print("\nBackfill complete. mlb.db is ready.")


if __name__ == "__main__":
    main()
