"""
mlb/run_pipeline.py — Orchestrate the full MLB pipeline.

Steps:
  1. Pull player stats (Statcast)
  2. Build features (batter_stats table)
  3. Train hit-probability model
  4. Generate daily picks with live odds

Usage:
    python mlb/run_pipeline.py
    python mlb/run_pipeline.py --skip-ingest    # skip pull_statcast (uses existing DB)
    python mlb/run_pipeline.py --skip-model     # skip model retraining
    python mlb/run_pipeline.py --no-odds        # skip live odds pull
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mlb.ingest.pull_statcast import pull_statcast
from mlb.features.build_features import build_features
from mlb.model.train_model import main as train_model
from mlb.predict.daily_picks import score_today
from mlb.odds.pull_live_odds import pull_mlb_odds


def run(skip_ingest=False, skip_model=False, no_odds=False):
    print("\n" + "=" * 60)
    print("MLB PIPELINE")
    print("=" * 60)

    if not skip_ingest:
        print("\n[1/4] Pulling Statcast data ...")
        pull_statcast()
    else:
        print("\n[1/4] Skipping Statcast ingest (--skip-ingest)")

    print("\n[2/4] Building features ...")
    build_features()

    if not skip_model:
        print("\n[3/4] Training model ...")
        train_model()
    else:
        print("\n[3/4] Skipping model training (--skip-model)")

    if not no_odds:
        print("\n[4/4] Pulling live odds ...")
        try:
            pull_mlb_odds()
        except Exception as e:
            print(f"[mlb/pipeline] WARN odds pull failed: {e}")
    else:
        print("\n[4/4] Skipping odds pull (--no-odds)")

    print("\n[mlb/pipeline] Generating daily picks ...")
    score_today()

    print("\n[mlb/pipeline] Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ingest", action="store_true")
    parser.add_argument("--skip-model", action="store_true")
    parser.add_argument("--no-odds", action="store_true")
    args = parser.parse_args()
    run(skip_ingest=args.skip_ingest, skip_model=args.skip_model, no_odds=args.no_odds)
