"""
nba/run_pipeline.py — Orchestrate the full NBA pipeline.

Steps:
  1. Pull player stats (nightly box scores)
  2. Pull schedules
  3. Train models (weekly / on demand)
  4. Pull live odds (on demand — conserves API quota)
  5. Generate daily picks

Usage:
    python nba/run_pipeline.py
    python nba/run_pipeline.py --skip-ingest
    python nba/run_pipeline.py --skip-model
    python nba/run_pipeline.py --no-odds
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nba.ingest.pull_player_stats import pull_player_stats
from nba.ingest.pull_schedules import pull_nba_schedules
from nba.model.train_model import train_all, PROP_STAT_MAP
from nba.odds.pull_live_odds import pull_nba_odds
from nba.predict.daily_picks import score_tonight
from shared.config import NBA_CURRENT_SEASON
from datetime import datetime, timedelta


def run(skip_ingest=False, skip_model=False, no_odds=False):
    print("\n" + "=" * 60)
    print("NBA PIPELINE")
    print("=" * 60)

    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    if not skip_ingest:
        print("\n[1/5] Pulling player stats ...")
        pull_player_stats("2024-10-22", yesterday)

        print("\n[2/5] Pulling schedules ...")
        pull_nba_schedules()
    else:
        print("\n[1/5] Skipping ingest (--skip-ingest)")
        print("\n[2/5] Skipping schedules (--skip-ingest)")

    if not skip_model:
        print("\n[3/5] Training models ...")
        train_all(list(PROP_STAT_MAP.keys()))
    else:
        print("\n[3/5] Skipping model training (--skip-model)")

    if not no_odds:
        print("\n[4/5] Pulling live odds ...")
        try:
            pull_nba_odds()
        except Exception as e:
            print(f"[nba/pipeline] WARN odds pull failed: {e}")
    else:
        print("\n[4/5] Skipping odds pull (--no-odds)")

    print("\n[5/5] Generating daily picks ...")
    score_tonight()

    print("\n[nba/pipeline] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--no-odds", action="store_true")
    args = parser.parse_args()
    run(skip_ingest=args.skip_ingest, skip_model=args.skip_model, no_odds=args.no_odds)
