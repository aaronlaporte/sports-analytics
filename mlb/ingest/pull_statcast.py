"""
mlb/ingest/pull_statcast.py — Pull Statcast data into mlb.db.

Incrementally fetches missing dates from pybaseball and upserts into
the statcast_raw table. Run daily after games complete.

Usage:
    python mlb/ingest/pull_statcast.py
    python mlb/ingest/pull_statcast.py --start 2025-03-18 --end 2025-09-30
"""

import argparse
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning, module="pybaseball")

# Allow running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn, create_schema

MLB_SEASON_START = "2025-03-18"


def load_existing_dates(conn) -> set:
    rows = conn.execute("SELECT DISTINCT game_date FROM statcast_raw").fetchall()
    return {r["game_date"] for r in rows}


def find_missing_ranges(start: str, end: str, existing: set) -> list[tuple[str, str]]:
    full = pd.date_range(start, end, freq="D")
    missing = sorted(d for d in full if d.strftime("%Y-%m-%d") not in existing)
    if not missing:
        return []
    ranges, rs, prev = [], missing[0], missing[0]
    for d in missing[1:]:
        if (d - prev).days == 1:
            prev = d
        else:
            ranges.append((rs.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d")))
            rs = prev = d
    ranges.append((rs.strftime("%Y-%m-%d"), prev.strftime("%Y-%m-%d")))
    return ranges


def upsert_statcast(conn, df: pd.DataFrame):
    df = df[[
        "game_pk", "game_date", "batter", "pitcher",
        "events", "description", "bb_type",
        "hit_distance_sc", "launch_speed", "launch_angle",
        "estimated_ba_using_speedangle",
    ]].copy()
    df["game_date"] = df["game_date"].astype(str)
    df = df.dropna(subset=["game_pk", "batter", "pitcher"])

    conn.executemany("""
        INSERT OR IGNORE INTO statcast_raw
            (game_pk, game_date, batter, pitcher, events, description,
             bb_type, hit_distance_sc, launch_speed, launch_angle,
             estimated_ba_using_speedangle)
        VALUES (?,?,?,?,?,?,?,?,?,?,?)
    """, df[[
        "game_pk", "game_date", "batter", "pitcher", "events", "description",
        "bb_type", "hit_distance_sc", "launch_speed", "launch_angle",
        "estimated_ba_using_speedangle",
    ]].itertuples(index=False, name=None))
    conn.commit()
    return len(df)


def pull_statcast(start: str, end: str):
    from pybaseball import statcast

    create_schema("mlb")
    conn = get_conn("mlb")
    existing = load_existing_dates(conn)

    ranges = find_missing_ranges(start, end, existing)
    if not ranges:
        print("[statcast] Cache is up to date — nothing to pull.")
        conn.close()
        return

    total = 0
    for rs, re in ranges:
        print(f"[statcast] Pulling {rs} → {re} ...")
        try:
            df = statcast(rs, re)
        except Exception as e:
            print(f"[statcast] WARNING: pull failed for {rs}→{re}: {e}")
            continue
        if df.empty:
            print(f"[statcast] No data for {rs}→{re} (off day or no games)")
            continue
        df["game_date"] = pd.to_datetime(df["game_date"])
        n = upsert_statcast(conn, df)
        total += n
        print(f"[statcast] Upserted {n} rows")

    conn.close()
    print(f"[statcast] Done. Total rows upserted: {total}")

    # Validate freshness
    conn2 = get_conn("mlb")
    row = conn2.execute("SELECT MAX(game_date) as mx FROM statcast_raw").fetchone()
    conn2.close()
    max_date = row["mx"] if row else None
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    if max_date and max_date < (datetime.today() - timedelta(days=7)).strftime("%Y-%m-%d"):
        print(f"[statcast] WARNING: data is stale (max_date={max_date})")
    else:
        print(f"[statcast] Data current through {max_date}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=MLB_SEASON_START)
    parser.add_argument("--end",   default=(datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d"))
    args = parser.parse_args()
    pull_statcast(args.start, args.end)
