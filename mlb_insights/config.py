"""
mlb_insights/config.py -- Central configuration for the MLB Insights Pipeline.

All thresholds, weights, paths, and constants live here so they can be
tuned in one place without touching business logic.
"""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "mlb.db"
CALIBRATION_MODEL_DIR = DATA_DIR / "calibration_models"

# ── Season Defaults ──────────────────────────────────────────────────────────

MLB_SEASON_START_2024 = "2024-03-20"
MLB_SEASON_END_2024 = "2024-10-24"
MLB_SEASON_START_2025 = "2025-03-18"

CALIBRATION_TRAIN_CUTOFF = "2025-01-01"  # Train on <2025, validate on >=2025

# ── Statcast Ingest ──────────────────────────────────────────────────────────

VALID_EVENTS = [
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "grounded_into_double_play",
    "strikeout", "walk", "hit_by_pitch",
]

AB_EVENTS = [
    "single", "double", "triple", "home_run",
    "field_out", "force_out", "grounded_into_double_play", "strikeout",
]

HIT_EVENTS = ["single", "double", "triple", "home_run"]

TB_MAP = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

# ── Feature Engineering ──────────────────────────────────────────────────────

ROLLING_WINDOWS = [1, 2, 3, 5, 10, 20]
MIN_SEASON_PA = 50       # Minimum PA for season stats to be meaningful
MIN_PITCHER_BF = 50      # Minimum batters faced for pitcher to be included

# ── Signal Thresholds ────────────────────────────────────────────────────────

# Signal 1: hot_streak_acceleration
HOT_STREAK_MIN_STREAK = 5
HOT_STREAK_MIN_HITS_PG5 = 7
HOT_STREAK_MIN_SEASON_AVG = 0.250
HOT_STREAK_STREAK_CAP = 15.0
HOT_STREAK_VOLUME_MIN = 5.0
HOT_STREAK_VOLUME_SCALE = 10.0
HOT_STREAK_SEASON_FLOOR = 0.250
HOT_STREAK_SEASON_SCALE = 0.100

# Signal 2: cold_streak_rebound
COLD_STREAK_MAX_STREAK = -5
COLD_STREAK_MIN_SEASON_AVG = 0.260
COLD_STREAK_MIN_SEASON_PA = 100
COLD_STREAK_MIN_EXIT_VELO = 88.0
COLD_STREAK_VELO_SCALE = 5.0
COLD_STREAK_GAP_SCALE = 0.200

# Signal 3: pitcher_vulnerability
PITCHER_VULN_PERCENTILE = 0.75     # Top 25% most hittable
PITCHER_VULN_RATE_SCALE = 0.050

# Signal 4: contact_quality_regression
CONTACT_QUALITY_MIN_SEASON_PA = 100
CONTACT_QUALITY_MIN_SEASON_AVG = 0.240
CONTACT_QUALITY_MIN_GAP = 0.050
CONTACT_QUALITY_MIN_EXIT_VELO = 87.0
CONTACT_QUALITY_GAP_SCALE = 0.150
CONTACT_QUALITY_VELO_SCALE = 6.0

# Signal 5: pitch_mix_advantage
PITCH_MIX_MIN_PA = 30
PITCH_MIX_MIN_AVG = 0.300
PITCH_MIX_MIN_ADVANTAGE = 0.020
PITCH_MIX_SAMPLE_SCALE = 100.0
PITCH_MIX_PERF_SCALE = 0.100
PITCH_MIX_ADV_SCALE = 0.080

# Signal 6: babip_regression
BABIP_MIN_SEASON_PA = 100
BABIP_MIN_GAP = 0.080
BABIP_GAP_SCALE = 0.200
BABIP_CONFIDENCE_SCALE = 0.8
BABIP_APPROX_HR_RATE = 0.08    # ~8% of hits are HR
BABIP_APPROX_SO_RATE = 0.22    # ~22% strikeout rate
BABIP_APPROX_AB_RATIO = 0.88   # AB/PA approximation

# ── Composite Score ──────────────────────────────────────────────────────────

SIGNAL_WEIGHTS = {
    "hot_streak_acceleration": 0.06,
    "cold_streak_rebound": 0.04,
    "pitcher_vulnerability": 0.10,
    "contact_quality_regression": 0.08,
    "pitch_mix_advantage": 0.07,
    "babip_regression": 0.05,
}

# Z-score scaling: z of -3 -> 0, z of +3 -> 100
ZSCORE_FLOOR = -3.0
ZSCORE_CEIL = 3.0
SCORE_MIN = 0.0
SCORE_MAX = 100.0

# ── Leaderboard ──────────────────────────────────────────────────────────────

LEADERBOARD_TOP_N = 50

# ── Tracking / Calibration ───────────────────────────────────────────────────

BRIER_ROLLING_WINDOWS = [7, 14, 30, 90]

# ── Launch Speed Cache ───────────────────────────────────────────────────────

LAUNCH_SPEED_N_GAMES = 10
LAUNCH_SPEED_BATTED_BALLS_PER_GAME = 4   # ~3-4 batted balls per game
LAUNCH_SPEED_MIN_SAMPLES = 5
