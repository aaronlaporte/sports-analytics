"""
scripts/backfill_hr_features.py -- Backfill HR features for all historical dates.

Usage:
    cd sports-analytics
    /opt/sports-analytics-venv/bin/python scripts/backfill_hr_features.py

This script:
    1. Creates the hr_features and park_factors tables if they don't exist
    2. Computes park factors from all statcast_staging data
    3. Computes HR features for every date that has batter_stats
    4. Rebuilds the daily_leaderboard P(HR) for recent dates (2026-03-25 to 2026-03-29)
    5. Prints verification stats
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mlb_insights.utils.db import get_connection, ensure_tables
from mlb_insights.signal_engine.hr_model import (
    compute_hr_features, write_hr_features,
    compute_park_factors, write_park_factors,
)


def main():
    t0 = time.time()
    print("=" * 60)
    print("  HR FEATURES BACKFILL")
    print("=" * 60)

    # Step 0: Ensure tables exist
    print("\n[0/4] Ensuring tables exist ...")
    ensure_tables()

    conn = get_connection()

    # Step 1: Compute park factors
    print("\n[1/4] Computing park factors ...")
    park_factors = compute_park_factors(conn)
    write_park_factors(conn, park_factors)
    print(f"  Computed park factors for {len(park_factors)} teams:")
    for team in sorted(park_factors.keys()):
        pf = park_factors[team]
        label = ""
        if pf >= 1.1:
            label = " (HR-friendly)"
        elif pf <= 0.9:
            label = " (pitcher-friendly)"
        print(f"    {team}: {pf:.3f}{label}")

    # Step 2: Get all dates with batter_stats
    print("\n[2/4] Finding all dates with batter data ...")
    dates = conn.execute("""
        SELECT DISTINCT game_date FROM batter_stats
        ORDER BY game_date
    """).fetchall()
    dates_list = [r["game_date"] for r in dates]
    print(f"  Found {len(dates_list)} dates ({dates_list[0]} to {dates_list[-1]})")

    # Step 3: Compute HR features for each date
    print(f"\n[3/4] Computing HR features for {len(dates_list)} dates ...")
    total_features = 0
    errors = 0

    for idx, date_str in enumerate(dates_list):
        if idx % 50 == 0:
            elapsed = time.time() - t0
            pct = (idx / len(dates_list)) * 100 if dates_list else 0
            print(f"  [{idx + 1}/{len(dates_list)}] {date_str} ({pct:.0f}%, {elapsed:.0f}s elapsed)")

        try:
            hr_df = compute_hr_features(conn, date_str, park_factors=park_factors)
            if not hr_df.empty:
                write_hr_features(conn, date_str, hr_df)
                total_features += len(hr_df)
        except Exception as exc:
            errors += 1
            if errors <= 5:
                print(f"    ERROR on {date_str}: {exc}")

    print(f"\n  Done: {total_features} feature rows written, {errors} errors.")

    # Step 4: Verify
    print("\n[4/4] Verification ...")

    # Check feature distribution
    stats = conn.execute("""
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT feature_date) AS total_dates,
            COUNT(DISTINCT batter_id) AS total_batters,
            AVG(p_hr) AS avg_phr,
            MIN(p_hr) AS min_phr,
            MAX(p_hr) AS max_phr,
            AVG(barrel_rate) AS avg_barrel,
            AVG(avg_exit_velo) AS avg_velo
        FROM hr_features
    """).fetchone()

    print(f"  Total rows: {stats['total_rows']}")
    print(f"  Total dates: {stats['total_dates']}")
    print(f"  Total batters: {stats['total_batters']}")
    print(f"  P(HR) range: [{stats['min_phr']:.4f}, {stats['max_phr']:.4f}], mean={stats['avg_phr']:.4f}")
    print(f"  Avg barrel rate: {stats['avg_barrel']:.4f}")
    print(f"  Avg exit velo: {stats['avg_velo']:.1f}")

    # Check recent dates specifically
    print("\n  Recent date P(HR) samples:")
    recent = conn.execute("""
        SELECT h.feature_date, p.player_name, h.p_hr, h.barrel_rate,
               h.avg_exit_velo, h.park_factor
        FROM hr_features h
        LEFT JOIN player_lookup p ON h.batter_id = p.mlbam_id
        WHERE h.feature_date >= '2026-03-25'
        ORDER BY h.feature_date, h.p_hr DESC
    """).fetchall()

    if recent:
        current_date = None
        count = 0
        for r in recent:
            if r["feature_date"] != current_date:
                current_date = r["feature_date"]
                count = 0
                print(f"\n    {current_date}:")
            if count < 5:
                name = r["player_name"] or "Unknown"
                print(
                    f"      {name:20s}  P(HR)={r['p_hr']:.3f}  "
                    f"Barrel={r['barrel_rate']:.3f}  "
                    f"ExitV={r['avg_exit_velo']:.1f}  "
                    f"PF={r['park_factor']:.2f}"
                )
                count += 1
    else:
        print("    No data for recent dates (2026-03-25+).")

    # Check P(HR) differentiation
    differentiation = conn.execute("""
        SELECT feature_date,
               MIN(p_hr) AS min_phr, MAX(p_hr) AS max_phr,
               AVG(p_hr) AS avg_phr,
               COUNT(*) AS n
        FROM hr_features
        WHERE feature_date >= '2026-03-25'
        GROUP BY feature_date
        ORDER BY feature_date
    """).fetchall()

    if differentiation:
        print("\n  P(HR) differentiation by date:")
        for r in differentiation:
            spread = r["max_phr"] - r["min_phr"]
            print(
                f"    {r['feature_date']}: "
                f"min={r['min_phr']:.3f}, max={r['max_phr']:.3f}, "
                f"spread={spread:.3f}, n={r['n']}"
            )

    conn.close()

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"  Backfill complete in {elapsed:.1f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
