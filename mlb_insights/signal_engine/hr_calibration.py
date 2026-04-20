"""
mlb_insights/signal_engine/hr_calibration.py -- Isotonic regression calibration for HR predictions.

Provides train, save, load, and apply functions for isotonic calibration
of HR probabilities, separate from the main hit calibration pipeline.
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
    HR_CALIBRATION_MAX_AGE_DAYS,
    HR_CALIBRATION_BRIER_RETRAIN_THRESHOLD,
)

logger = logging.getLogger(__name__)


@dataclass
class HRCalibrationModel:
    """Container for the HR isotonic regression model."""
    iso_hr: object  # IsotonicRegression


def _model_path() -> Path:
    """Path to the persisted HR calibration pickle."""
    CALIBRATION_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return CALIBRATION_MODEL_DIR / "hr_isotonic.pkl"


def train_hr_calibration(conn: sqlite3.Connection) -> HRCalibrationModel:
    """Train isotonic regression model on hr_prediction_tracking data.

    Uses rows with prediction_date < CALIBRATION_TRAIN_CUTOFF (2025) for
    training.  The fitted model is saved to disk and returned.

    Args:
        conn: Open sqlite3 connection.

    Returns:
        HRCalibrationModel with fitted iso_hr.
    """
    from sklearn.isotonic import IsotonicRegression

    cur = conn.cursor()
    cur.execute("""
        SELECT prediction_date, player_id, p_hr, actual_hr
        FROM hr_prediction_tracking
        WHERE actual_hr IS NOT NULL
        ORDER BY prediction_date
    """)
    all_rows = cur.fetchall()

    train_rows = [r for r in all_rows if r["prediction_date"] < CALIBRATION_TRAIN_CUTOFF]
    val_rows = [r for r in all_rows if r["prediction_date"] >= CALIBRATION_TRAIN_CUTOFF]

    if len(train_rows) < 100:
        logger.warning(
            "Only %d training rows (need 100+). HR calibration may be unreliable.",
            len(train_rows),
        )

    logger.info(
        "Training HR calibration: %d train rows, %d validation rows.",
        len(train_rows), len(val_rows),
    )

    # Build arrays
    train_phr = np.array([r["p_hr"] for r in train_rows])
    train_yhr = np.array([1.0 if (r["actual_hr"] or 0) >= 1 else 0.0 for r in train_rows])

    # Fit model
    iso_hr = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    iso_hr.fit(train_phr, train_yhr)

    model = HRCalibrationModel(iso_hr=iso_hr)

    # Log validation stats if we have validation data
    if val_rows:
        val_phr = np.array([r["p_hr"] for r in val_rows])
        val_yhr = np.array([1.0 if (r["actual_hr"] or 0) >= 1 else 0.0 for r in val_rows])
        cal_phr = iso_hr.transform(val_phr)

        raw_brier = float(np.mean((val_phr - val_yhr) ** 2))
        cal_brier = float(np.mean((cal_phr - val_yhr) ** 2))
        logger.info(
            "Validation Brier (HR): raw=%.6f, calibrated=%.6f",
            raw_brier, cal_brier,
        )

    # Save to disk
    save_hr_calibration(model)

    return model


def save_hr_calibration(model: HRCalibrationModel):
    """Persist HR calibration model to disk."""
    path = _model_path()
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("HR calibration model saved to %s", path)


def load_hr_calibration() -> HRCalibrationModel | None:
    """Load HR calibration model from disk.

    Returns:
        HRCalibrationModel if found, None otherwise.
    """
    path = _model_path()
    if not path.exists():
        logger.warning("No HR calibration model found at %s", path)
        return None

    with open(path, "rb") as f:
        model = pickle.load(f)

    logger.info("HR calibration model loaded from %s", path)
    return model


def calibrate_hr_single(model: HRCalibrationModel, p_hr: float) -> float:
    """Apply isotonic calibration to a single p_hr value.

    Args:
        model: Fitted HRCalibrationModel.
        p_hr: Raw P(HR).

    Returns:
        Calibrated P(HR) float.
    """
    cal_arr = model.iso_hr.transform(np.array([p_hr]))
    return float(cal_arr[0])


def hr_calibration_needs_retrain(
    conn: sqlite3.Connection,
    max_age_days: int = HR_CALIBRATION_MAX_AGE_DAYS,
    brier_threshold: float = HR_CALIBRATION_BRIER_RETRAIN_THRESHOLD,
) -> bool:
    """Check whether HR calibration model should be retrained.

    Returns True if the pickle is older than *max_age_days* or the most
    recent 7-day hr_brier score from calibration_summary exceeds
    *brier_threshold*.

    Note: HR brier threshold is much lower than hit brier (0.06 vs 0.22)
    because the HR base rate is ~4% vs ~65% for hits.
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
                "HR calibration model is %d days old (max %d). Retrain needed.",
                age.days, max_age_days,
            )
            return True
    else:
        logger.info("No HR calibration model found. Training needed.")
        return True

    # 2. Check recent Brier score
    try:
        row = conn.execute("""
            SELECT value
            FROM calibration_summary
            WHERE metric_type = 'hr_brier' AND window_days = 7
            ORDER BY summary_date DESC
            LIMIT 1
        """).fetchone()
        if row is not None:
            recent_brier = row["value"] if isinstance(row, sqlite3.Row) else row[0]
            if recent_brier > brier_threshold:
                logger.info(
                    "Recent 7-day hr_brier=%.4f exceeds threshold %.4f. Retrain needed.",
                    recent_brier, brier_threshold,
                )
                return True
    except Exception as exc:
        logger.warning("Could not check calibration_summary for HR: %s", exc)

    return False
