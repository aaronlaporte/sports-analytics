"""
nfl/run_pipeline.py — Orchestrate the full NFL pipeline.

Steps:
  1. Pull schedule
  2. Pull player game stats (historical + current season)
  3. Pull injuries
  4. Train models (weekly / on demand)
  5. Pull live odds (on demand)
  6. Predict upcoming week

Usage:
    python nfl/run_pipeline.py
    python nfl/run_pipeline.py --skip-ingest
    python nfl/run_pipeline.py --skip-model
    python nfl/run_pipeline.py --no-odds
    python nfl/run_pipeline.py --week 14 --season 2025REG
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nfl.ingest.pull_schedule import pull_nfl_schedule
from nfl.ingest.pull_players import pull_nfl_players
from nfl.ingest.pull_player_game_stats import pull_nfl_player_game_stats
from nfl.ingest.pull_team_game_stats import pull_nfl_team_stats
from nfl.ingest.pull_injuries import pull_nfl_injuries
from nfl.model.train_model import main as train_model
from nfl.odds.pull_live_odds import pull_nfl_odds
from nfl.predict.predict_week import predict_week


def run(skip_ingest=False, skip_model=False, no_odds=False, week=None, season=None):
    print("\n" + "=" * 60)
    print("NFL PIPELINE")
    print("=" * 60)

    if not skip_ingest:
        print("\n[1/6] Pulling schedule ...")
        pull_nfl_schedule()

        print("\n[2/6] Pulling player roster ...")
        pull_nfl_players()

        print("\n[3/6] Pulling player game stats ...")
        pull_nfl_player_game_stats()

        print("\n[4/6] Pulling team stats ...")
        pull_nfl_team_stats()

        print("\n[5/6] Pulling injuries ...")
        pull_nfl_injuries()
    else:
        print("\n[1-5/6] Skipping ingest (--skip-ingest)")

    if not skip_model:
        print("\n[6/6a] Training models ...")
        train_model()
    else:
        print("\n[6/6a] Skipping model training (--skip-model)")

    if not no_odds:
        print("\n[6/6b] Pulling live odds ...")
        try:
            pull_nfl_odds()
        except Exception as e:
            print(f"[nfl/pipeline] WARN odds pull failed: {e}")
    else:
        print("\n[6/6b] Skipping odds pull (--no-odds)")

    print("\n[6/6c] Predicting this week ...")
    predict_week(week=week, season=season)

    print("\n[nfl/pipeline] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--no-odds", action="store_true")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--season", default=None)
    args = parser.parse_args()
    run(
        skip_ingest=args.skip_ingest,
        skip_model=args.skip_model,
        no_odds=args.no_odds,
        week=args.week,
        season=args.season,
    )
