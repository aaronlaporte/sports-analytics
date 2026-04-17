"""
mlb_insights/signal_engine/calibration.py -- Isotonic regression calibration.

Extracted from phase2b_rebuild.py.  Provides train, save, load, and apply
functions for isotonic calibration of hit/2-hit/HR probabilities.
"""

import logging
import pickle
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from mlb_insights.config import (
    CALIBRATION_MODEL_DIR,
    CALIBRATION_TRAIN_CUTOFF,
    CALIBRATION_MAX_AGE_DAYS,
    CALIBRATION_BRIER_RETRAIN_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class CalibrationModels:
    """Container for the three isotonic regression models."""
    iso_1hit: object  # IsotonicRegression
    iso_2hit: object
    iso_hr: object


def _model_path() -> Path:
    """Path to the persisted calibration pickle."""
    CALIBRATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return CALIBRATION_MODEL_DIR / "isotonic_calibration.pkl"


def train_calibration(conn: sqlite3.Connection) -> CalibrationModels:
    """Train isotonic regression models on prediction_tracking data.

    Uses rows with prediction_date < CALIBRATION_TRAIN_CUTOFF (2025) for
    training.  The fitted models are saved to disk and returned.

    Args:
        conn: Open sqlite3 connection.

    Returns:
        CalibrationModels with fitted iso_1hit, iso_2hit, iso_hr.
    """
    from sklearn.isotonic import IsotonicRegression

    cur = conn.cursor()
    cur.execute("""
        SELECT prediction_date, player_id, p_1hit, p_2hit, p_hr,
               actual_hits, actual_hr
        FROM prediction_tracking
        WHERE actual_hits IS NOT NULL
        ORDER BY prediction_date
    """)
    all_rows = cur.fetchall()

    train_rows = [r for r in all_rows if r["prediction_date"] < CALIBRATION_TRAIN_CUTOFF]
    val_rows = [r for r in all_rows if r["prediction_date"] >= CALIBRATION_TRAIN_CUTOFF]

    if len(train_rows) < 100:
        logger.warning(
            "Only %d training rows (need 100+). Calibration may be unreliable.",
            len(train_rows),
        )

    logger.info(
        "Training calibration: %d train rows, %d validation rows.",
        len(train_rows), len(val_rows),
    )

    # Build arrays
    train_p1 = np.array([r["p_1hit"] for r in train_rows])
    train_y1 = np.array([1.0 if r["actual_hits"] >= 1 else 0.0 for r in train_rows])

    train_p2 = np.array([r["p_2hit"] for r in train_rows])
    train_y2 = np.array([1.0 if r["actual_hits"] >= 2 else 0.0 for r in train_rows])

    train_phr = np.array([r["p_hr"] for r in train_rows])
    train_yhr = np.array([1.0 if (r["actual_hr"] or 0) >= 1 else 0.0 for r in train_rows])

    # Fit models
    iso_1hit = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_1hit.fit(train_p1, train_y1)

    iso_2hit = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_2hit.fit(train_p2, train_y2)

    iso_hr = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_hr.fit(train_phr, train_yhr)

    models = CalibrationModels(iso_1hit=iso_1hit, iso_2hit=iso_2hit, iso_hr=iso_hr)

    # Log validation stats if we have validation data
    if val_rows:
        val_p1 = np.array([r["p_1hit"] for r in val_rows])
        val_y1 = np.array([1.0 if r["actual_hits"] >= 1 else 0.0 for r in val_rows])
        cal_p1 = iso_1hit.transform(val_p1)

        raw_brier = float(np.mean((val_p1 - val_y1) ** 2))
        cal_brier = float(np.mean((cal_p1 - val_y1) ** 2))
        logger.info(
            "Validation Brier (1hit): raw=%.6f, calibrated=%.6f",
            raw_brier, cal_brier,
        )

    # Save to disk
    save_calibration(models)

    return models


def save_calibration(models: CalibrationModels):
    """Persist calibration models to disk."""
    path = _model_path()
    with open(path, "wb") as f:
        pickle.dump(models, f)
    logger.info("Calibration models saved to %s", path)


def load_calibration() -> CalibrationModels | None:
    """Load calibration models from disk.

    Returns:
        CalibrationModels if found, None otherwise.
    """
    path = _model_path()
    if not path.exists():
        logger.warning("No calibration model found at %s", path)
        return None

    with open(path, "rb") as f:
        models = pickle.load(f)

    logger.info("Calibration models loaded from %s", path)
    return models


def calibration_needs_retrain(
    conn: sqlite3.Connection,
    max_age_days: int = CALIBRATION_MAX_AGE_DAYS,
    brier_threshold: float = CALIBRATION_BRIER_RETRAIN_THRESHOLD,
) -> bool:
    """Check whether calibration model should be retrained.

    Returns True if the pickle is older than *max_age_days* or the most
    recent 7-day brier_1hit score from calibration_summary exceeds
    *brier_threshold*.
    """
    import os
    from datetime import datetime, timedelta

    # 1. Check model age
    path = _model_path()
    if path.exists():
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
        age = datetime.now() - mtime
        if age > timedelta(days=max_age_days):
            logger.info(
                "Calibration model is %d days old (max %d). Retrain needed.",
                age.days, max_age_days,
            )
            return True

    # 2. Check recent Brier score
    try:
        row = conn.execute("""
            SELECT brier_1hit
            FROM calibration_summary
            WHERE window_days = 7
            ORDER BY as_of_date DESC
            LIMIT 1
        """).fetchone()
        if row is not None:
            recent_brier = row["brier_1hit"] if isinstance(row, sqlite3.Row) else row[0]
            if recent_brier > brier_threshold:
                logger.info(
                    "Recent 7-day brier_1hit=%.4f exceeds threshold %.4f. Retrain needed.",
                    recent_brier, brier_threshold,
                )
                return True
    except Exception as exc:
        logger.warning("Could not check calibration_summary: %s", exc)

    return False


def calibrate(
    models: CalibrationModels,
    raw_p1hit: np.ndarray,
    raw_p2hit: np.ndarray,
    raw_phr: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply isotonic calibration to raw probability arrays.

    Args:
        models: Fitted CalibrationModels.
        raw_p1hit: Raw P(1+ hit) array.
        raw_p2hit: Raw P(2+ hit) array.
        raw_phr: Raw P(HR) array.

    Returns:
        Tuple of (cal_p1hit, cal_p2hit, cal_phr) numpy arrays.
    """
    cal_p1 = models.iso_1hit.transform(raw_p1hit)
    cal_p2 = models.iso_2hit.transform(raw_p2hit)
    cal_phr = models.iso_hr.transform(raw_phr)
    return cal_p1, cal_p2, cal_phr


def calibrate_single(
    models: CalibrationModels,
    p1hit: float,
    p2hit: float,
    phr: float,
) -> tuple[float, float, float]:
    """Calibrate a single set of probabilities.

    Args:
        models: Fitted CalibrationModels.
        p1hit: Raw P(1+ hit).
        p2hit: Raw P(2+ hit).
        phr: Raw P(HR).

    Returns:
        Tuple of (cal_p1hit, cal_p2hit, cal_phr) floats.
    """
    arr_p1, arr_p2, arr_phr = calibrate(
        models,
        np.array([p1hit]),
        np.array([p2hit]),
        np.array([phr]),
    )
    return float(arr_p1[0]), float(arr_p2[0]), float(arr_phr[0])


# ── Brier Score Helpers ──────────────────────────────────────────────────────

def brier_score(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilities)."""
    return float(np.mean((pred - actual) ** 2))


def brier_skill_score(pred: np.ndarray, actual: np.ndarray) -> float:
    """Compute Brier Skill Score relative to climatological baseline.

    BSS > 0 means the model beats the base-rate forecast.
    """
    bs = brier_score(pred, actual)
    base_rate = float(np.mean(actual))
    bs_ref = base_rate * (1 - base_rate)
    return 1.0 - bs / bs_ref if bs_ref > 0 else 0.0
