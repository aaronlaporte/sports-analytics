"""
mlb_insights/signal_engine/hr_retro.py -- HR retroactive training engine.

Analyzes HR prediction outcomes daily, comparing predicted signals/reasons
against actual game context. Accumulates findings in hr_retro_analysis table.
Weekly, uses accumulated findings to adjust HR signal weights.

Layer 1: Signal accuracy -- per-signal fire rate, HR rate, lift, false negatives, co-occurrence
Layer 2: Contextual matching -- statcast at-bat data for actual HRs vs predictions

Usage:
    cd sports-analytics
    python -m mlb_insights.signal_engine.hr_retro                    # daily analysis
    python -m mlb_insights.signal_engine.hr_retro --retune           # weekly weight adjust
    python -m mlb_insights.signal_engine.hr_retro --retune --dry-run # preview only
"""

import argparse
import json
import logging
import sqlite3
from pathlib import Path

from mlb_insights.config import HR_SIGNAL_WEIGHTS, DATA_DIR

logger = logging.getLogger(__name__)

TUNED_HR_WEIGHTS_PATH = DATA_DIR / "tuned_hr_weights.json"

# HR signal types (keys of HR_SIGNAL_WEIGHTS from config)
_HR_SIGNAL_TYPES = (
    "hr_power_signal", "hr_fly_ball_power", "hr_pitcher_flyball_tendency",
    "hr_power_streak", "hr_platoon_power",
)

# Damping and clamp for HR weight tuning
_DAMPING = 0.3
_WEIGHT_FLOOR = 0.03
_WEIGHT_CEIL = 0.20


# ── Layer 1: Signal Accuracy ─────────────────────────────────────────────────


def compute_hr_signal_accuracy(
    conn: sqlite3.Connection,
    lookback_days: int = 90,
) -> dict[str, dict]:
    """Compute per-HR-signal accuracy stats over the lookback window.

    For each HR signal type in daily_signals, joins with hr_prediction_tracking
    on (signal_date = prediction_date, player_id) to determine whether the
    player hit 1+ HR on the day the signal fired.

    Returns a dict keyed by signal_type with:
        fire_count, hr_rate_when_fired, avg_confidence, baseline_hr_rate,
        false_positive_rate
    """
    # Baseline HR rate: fraction of hr_prediction_tracking rows where
    # the player hit 1+ HR, regardless of signals.
    baseline_row = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN actual_hr >= 1 THEN 1 ELSE 0 END) AS hrs
        FROM hr_prediction_tracking
        WHERE prediction_date >= date('now', ?)
    """, (f"-{lookback_days} days",)).fetchone()

    total = baseline_row["total"] or 0
    baseline_hr_rate = (baseline_row["hrs"] / total) if total > 0 else 0.0

    # Per-signal stats (only HR signal types)
    placeholders = ",".join("?" for _ in _HR_SIGNAL_TYPES)
    rows = conn.execute(f"""
        SELECT
            ds.signal_type,
            COUNT(*) AS fire_count,
            SUM(CASE WHEN pt.actual_hr >= 1 THEN 1 ELSE 0 END) AS hr_count,
            AVG(ds.confidence) AS avg_confidence
        FROM daily_signals ds
        JOIN hr_prediction_tracking pt
            ON ds.signal_date = pt.prediction_date
            AND ds.player_id = pt.player_id
        WHERE ds.signal_date >= date('now', ?)
            AND ds.signal_type IN ({placeholders})
        GROUP BY ds.signal_type
    """, (f"-{lookback_days} days", *_HR_SIGNAL_TYPES)).fetchall()

    stats: dict[str, dict] = {}
    for r in rows:
        fire_count = r["fire_count"]
        hr_count = r["hr_count"] or 0
        hr_rate = hr_count / fire_count if fire_count > 0 else 0.0
        stats[r["signal_type"]] = {
            "fire_count": fire_count,
            "hr_rate_when_fired": round(hr_rate, 4),
            "avg_confidence": round(r["avg_confidence"], 4),
            "baseline_hr_rate": round(baseline_hr_rate, 4),
            "false_positive_rate": round(1.0 - hr_rate, 4),
        }

    # Ensure every HR signal type appears even if it never fired.
    for sig in _HR_SIGNAL_TYPES:
        if sig not in stats:
            stats[sig] = {
                "fire_count": 0,
                "hr_rate_when_fired": 0.0,
                "avg_confidence": 0.0,
                "baseline_hr_rate": round(baseline_hr_rate, 4),
                "false_positive_rate": 1.0,
            }

    return stats


def compute_false_negative_analysis(
    conn: sqlite3.Connection,
    lookback_days: int = 90,
) -> dict:
    """Analyze HRs that were missed (no HR signal fired) vs correctly signaled.

    Compares hr_features_v2 feature means between missed HRs and signaled HRs
    to identify what the signals are failing to capture.

    Returns dict with:
        missed_count, signaled_count, missed_feature_means, signaled_feature_means
    """
    placeholders = ",".join("?" for _ in _HR_SIGNAL_TYPES)

    # Missed HRs: actual_hr >= 1 but no HR signal fired that day for that player
    missed_rows = conn.execute(f"""
        SELECT pt.player_id, pt.prediction_date
        FROM hr_prediction_tracking pt
        WHERE pt.actual_hr >= 1
            AND pt.prediction_date >= date('now', ?)
            AND NOT EXISTS (
                SELECT 1 FROM daily_signals ds
                WHERE ds.player_id = pt.player_id
                    AND ds.signal_date = pt.prediction_date
                    AND ds.signal_type IN ({placeholders})
            )
    """, (f"-{lookback_days} days", *_HR_SIGNAL_TYPES)).fetchall()

    # Signaled HRs: actual_hr >= 1 AND at least one HR signal fired
    signaled_rows = conn.execute(f"""
        SELECT pt.player_id, pt.prediction_date
        FROM hr_prediction_tracking pt
        WHERE pt.actual_hr >= 1
            AND pt.prediction_date >= date('now', ?)
            AND EXISTS (
                SELECT 1 FROM daily_signals ds
                WHERE ds.player_id = pt.player_id
                    AND ds.signal_date = pt.prediction_date
                    AND ds.signal_type IN ({placeholders})
            )
    """, (f"-{lookback_days} days", *_HR_SIGNAL_TYPES)).fetchall()

    features_to_compare = (
        "barrel_rate", "avg_exit_velo", "hr_pa_rate",
        "fly_ball_rate", "hard_hit_fly_rate", "park_factor",
    )

    def _feature_means(player_date_pairs: list) -> dict[str, float]:
        """Compute mean feature values for a set of (player_id, date) pairs."""
        if not player_date_pairs:
            return {f: 0.0 for f in features_to_compare}

        # Build temp list for IN clause -- use individual queries to avoid
        # overly large SQL; aggregate in Python.
        totals = {f: 0.0 for f in features_to_compare}
        counts = {f: 0 for f in features_to_compare}

        for player_id, game_date in player_date_pairs:
            row = conn.execute("""
                SELECT barrel_rate, avg_exit_velo, hr_pa_rate,
                       fly_ball_rate, hard_hit_fly_rate, park_factor
                FROM hr_features_v2
                WHERE player_id = ? AND game_date = ?
                ORDER BY game_date DESC LIMIT 1
            """, (player_id, game_date)).fetchone()
            if row is None:
                continue
            for f in features_to_compare:
                val = row[f]
                if val is not None:
                    totals[f] += val
                    counts[f] += 1

        means = {}
        for f in features_to_compare:
            means[f] = round(totals[f] / counts[f], 4) if counts[f] > 0 else 0.0
        return means

    missed_pairs = [(r["player_id"], r["prediction_date"]) for r in missed_rows]
    signaled_pairs = [(r["player_id"], r["prediction_date"]) for r in signaled_rows]

    return {
        "missed_count": len(missed_pairs),
        "signaled_count": len(signaled_pairs),
        "missed_feature_means": _feature_means(missed_pairs),
        "signaled_feature_means": _feature_means(signaled_pairs),
    }


def compute_signal_cooccurrence(
    conn: sqlite3.Connection,
    lookback_days: int = 90,
) -> list[dict]:
    """Compute HR rates for combinations of HR signals firing together.

    Groups (prediction_date, player_id) pairs by the set of HR signal types
    that fired, then joins with hr_prediction_tracking to get the HR rate
    for each combination.

    Returns list of dicts sorted by HR rate desc:
        combo (tuple of signal names), fire_count, hr_rate, baseline_hr_rate
    Filtered to combos with fire_count >= 5.
    """
    placeholders = ",".join("?" for _ in _HR_SIGNAL_TYPES)

    # Get all (date, player, signal_type) tuples for HR signals
    rows = conn.execute(f"""
        SELECT ds.signal_date, ds.player_id, ds.signal_type
        FROM daily_signals ds
        WHERE ds.signal_date >= date('now', ?)
            AND ds.signal_type IN ({placeholders})
    """, (f"-{lookback_days} days", *_HR_SIGNAL_TYPES)).fetchall()

    # Group signals by (date, player)
    from collections import defaultdict
    combos_by_key: dict[tuple, set] = defaultdict(set)
    for r in rows:
        key = (r["signal_date"], r["player_id"])
        combos_by_key[key].add(r["signal_type"])

    # Group by frozenset of signal types
    combo_groups: dict[frozenset, list] = defaultdict(list)
    for key, sig_set in combos_by_key.items():
        combo_groups[frozenset(sig_set)].append(key)

    # Baseline HR rate
    baseline_row = conn.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN actual_hr >= 1 THEN 1 ELSE 0 END) AS hrs
        FROM hr_prediction_tracking
        WHERE prediction_date >= date('now', ?)
    """, (f"-{lookback_days} days",)).fetchone()

    total = baseline_row["total"] or 0
    baseline_hr_rate = (baseline_row["hrs"] / total) if total > 0 else 0.0

    # For each combo, compute HR rate
    results = []
    for combo_set, keys in combo_groups.items():
        fire_count = len(keys)
        if fire_count < 5:
            continue

        hr_count = 0
        for date_str, player_id in keys:
            row = conn.execute("""
                SELECT actual_hr FROM hr_prediction_tracking
                WHERE prediction_date = ? AND player_id = ?
            """, (date_str, player_id)).fetchone()
            if row and (row["actual_hr"] or 0) >= 1:
                hr_count += 1

        hr_rate = hr_count / fire_count if fire_count > 0 else 0.0
        results.append({
            "combo": tuple(sorted(combo_set)),
            "fire_count": fire_count,
            "hr_rate": round(hr_rate, 4),
            "baseline_hr_rate": round(baseline_hr_rate, 4),
        })

    results.sort(key=lambda x: x["hr_rate"], reverse=True)
    return results


# ── Layer 2: Contextual Matching ─────────────────────────────────────────────


def get_hr_atbat_context(
    conn: sqlite3.Connection,
    player_id: int,
    game_date: str,
) -> list[dict]:
    """Get statcast at-bat data for a player's HRs on a given date.

    Returns list of dicts with:
        pitch_type, release_speed, launch_speed, launch_angle, bb_type,
        p_throws, home_team
    """
    rows = conn.execute("""
        SELECT pitch_type, release_speed, launch_speed, launch_angle,
               bb_type, p_throws, home_team
        FROM statcast_staging
        WHERE batter = ? AND game_date = ? AND events = 'home_run'
    """, (player_id, game_date)).fetchall()

    results = []
    for r in rows:
        results.append({
            "pitch_type": r["pitch_type"],
            "release_speed": r["release_speed"],
            "launch_speed": r["launch_speed"],
            "launch_angle": r["launch_angle"],
            "bb_type": r["bb_type"],
            "p_throws": r["p_throws"],
            "home_team": r["home_team"],
        })
    return results


def build_contextual_findings(
    conn: sqlite3.Connection,
    date_str: str,
    lookback_days: int = 30,
) -> list[dict]:
    """Build Layer 2 contextual findings from statcast HR at-bat data.

    Aggregates pitch type distribution, launch conditions, signal-context
    cross-tabs, and park factor accuracy.

    Returns list of finding dicts with:
        finding_type, metric_name, metric_value, sample_size, narrative
    """
    # Get all HRs in lookback window
    hr_rows = conn.execute("""
        SELECT player_id, prediction_date
        FROM hr_prediction_tracking
        WHERE actual_hr >= 1
            AND prediction_date >= date(?, ?)
    """, (date_str, f"-{lookback_days} days")).fetchall()

    if not hr_rows:
        logger.debug("No HRs found in lookback window for contextual findings.")
        return []

    # Collect all at-bat contexts
    all_atbats = []
    for r in hr_rows:
        atbats = get_hr_atbat_context(conn, r["player_id"], r["prediction_date"])
        all_atbats.extend(atbats)

    if not all_atbats:
        logger.debug("No statcast at-bat data found for HR contextual analysis.")
        return []

    findings = []

    # (a) Pitch type distribution
    pitch_types = [ab["pitch_type"] for ab in all_atbats if ab["pitch_type"] is not None]
    if len(pitch_types) >= 5:
        from collections import Counter
        pitch_counts = Counter(pitch_types)
        total_pitches = len(pitch_types)
        top_pitches = pitch_counts.most_common(3)
        pct_str = ", ".join(
            f"{pt}: {ct / total_pitches * 100:.1f}%"
            for pt, ct in top_pitches
        )
        findings.append({
            "finding_type": "pitch_distribution",
            "metric_name": "hr_pitch_type_pct",
            "metric_value": {pt: round(ct / total_pitches, 4) for pt, ct in top_pitches},
            "sample_size": total_pitches,
            "narrative": f"HR pitch type distribution (n={total_pitches}): {pct_str}",
        })
    else:
        logger.debug("Skipping pitch type findings: all NULL or < 5 samples.")

    # (b) Launch conditions
    launch_speeds = [ab["launch_speed"] for ab in all_atbats if ab["launch_speed"] is not None]
    launch_angles = [ab["launch_angle"] for ab in all_atbats if ab["launch_angle"] is not None]
    if len(launch_speeds) >= 5 and len(launch_angles) >= 5:
        mean_ev = sum(launch_speeds) / len(launch_speeds)
        mean_la = sum(launch_angles) / len(launch_angles)
        findings.append({
            "finding_type": "launch_conditions",
            "metric_name": "mean_hr_launch",
            "metric_value": {"exit_velo": round(mean_ev, 1), "launch_angle": round(mean_la, 1)},
            "sample_size": len(launch_speeds),
            "narrative": f"Mean HR exit velo: {mean_ev:.1f} mph, mean launch angle: {mean_la:.1f} deg",
        })

    # (c) Signal-context cross-tab: pitcher_flyball signal vs actual fly balls
    placeholders = ",".join("?" for _ in _HR_SIGNAL_TYPES)
    flyball_signal_hrs = conn.execute(f"""
        SELECT pt.player_id, pt.prediction_date
        FROM hr_prediction_tracking pt
        WHERE pt.actual_hr >= 1
            AND pt.prediction_date >= date(?, ?)
            AND EXISTS (
                SELECT 1 FROM daily_signals ds
                WHERE ds.player_id = pt.player_id
                    AND ds.signal_date = pt.prediction_date
                    AND ds.signal_type = 'hr_pitcher_flyball_tendency'
            )
    """, (date_str, f"-{lookback_days} days")).fetchall()

    if len(flyball_signal_hrs) >= 5:
        flyball_count = 0
        total_checked = 0
        for r in flyball_signal_hrs:
            atbats = get_hr_atbat_context(conn, r["player_id"], r["prediction_date"])
            for ab in atbats:
                total_checked += 1
                if ab["bb_type"] == "fly_ball":
                    flyball_count += 1
        if total_checked > 0:
            flyball_pct = flyball_count / total_checked
            findings.append({
                "finding_type": "signal_context_cross",
                "metric_name": "pitcher_flyball_accuracy",
                "metric_value": round(flyball_pct, 4),
                "sample_size": total_checked,
                "narrative": (
                    f"pitcher_flyball signal correctly matched "
                    f"{flyball_pct * 100:.1f}% of fly ball HRs (n={total_checked})"
                ),
            })

    # (d) Park factor accuracy: group HRs by home_team
    home_teams = [ab["home_team"] for ab in all_atbats if ab["home_team"] is not None]
    if len(home_teams) >= 5:
        from collections import Counter
        park_counts = Counter(home_teams)
        top_parks = park_counts.most_common(3)
        park_names = ", ".join(f"{t[0]} ({t[1]})" for t in top_parks)
        findings.append({
            "finding_type": "park_factor",
            "metric_name": "top_hr_parks",
            "metric_value": {t[0]: t[1] for t in top_parks},
            "sample_size": len(home_teams),
            "narrative": f"Top 3 HR-producing parks: {park_names}",
        })

    return findings


# ── Daily Orchestration ──────────────────────────────────────────────────────


def write_retro_analysis(
    conn: sqlite3.Connection,
    date_str: str,
    signal_stats: dict[str, dict],
    false_neg: dict,
    cooccurrence: list[dict],
    contextual: list[dict],
) -> None:
    """Write all analysis results to hr_retro_analysis table.

    Deletes existing rows for analysis_date, then inserts fresh rows.
    """
    conn.execute(
        "DELETE FROM hr_retro_analysis WHERE analysis_date = ?", (date_str,)
    )

    # Signal accuracy: one row per signal
    for sig, st in signal_stats.items():
        for metric_name, metric_value in st.items():
            if metric_name == "fire_count":
                continue
            conn.execute("""
                INSERT INTO hr_retro_analysis
                    (analysis_date, analysis_type, signal_type, metric_name,
                     metric_value, sample_size, detail_json)
                VALUES (?, 'signal_accuracy', ?, ?, ?, ?, NULL)
            """, (date_str, sig, metric_name, metric_value, st["fire_count"]))

    # False negative: one row per feature
    for feature, mean_val in false_neg.get("missed_feature_means", {}).items():
        signaled_val = false_neg.get("signaled_feature_means", {}).get(feature, 0.0)
        detail = json.dumps({
            "missed_mean": mean_val,
            "signaled_mean": signaled_val,
            "missed_count": false_neg["missed_count"],
            "signaled_count": false_neg["signaled_count"],
        })
        conn.execute("""
            INSERT INTO hr_retro_analysis
                (analysis_date, analysis_type, signal_type, metric_name,
                 metric_value, sample_size, detail_json)
            VALUES (?, 'false_negative', NULL, ?, ?, ?, ?)
        """, (date_str, feature, mean_val, false_neg["missed_count"], detail))

    # Cooccurrence: one row per combo
    for combo_data in cooccurrence:
        detail = json.dumps(combo_data["combo"])
        conn.execute("""
            INSERT INTO hr_retro_analysis
                (analysis_date, analysis_type, signal_type, metric_name,
                 metric_value, sample_size, detail_json)
            VALUES (?, 'cooccurrence', NULL, 'hr_rate', ?, ?, ?)
        """, (date_str, combo_data["hr_rate"], combo_data["fire_count"], detail))

    # Contextual: one row per finding
    for finding in contextual:
        detail = json.dumps(finding["metric_value"])
        conn.execute("""
            INSERT INTO hr_retro_analysis
                (analysis_date, analysis_type, signal_type, metric_name,
                 metric_value, sample_size, detail_json)
            VALUES (?, 'contextual', NULL, ?, ?, ?, ?)
        """, (
            date_str, finding["metric_name"],
            finding["sample_size"],  # use sample_size as metric_value placeholder
            finding["sample_size"], detail,
        ))

    conn.commit()
    logger.info("Wrote hr_retro_analysis rows for %s.", date_str)


def run_daily_hr_retro(
    conn: sqlite3.Connection,
    date_str: str,
    lookback_days: int = 30,
) -> list[dict]:
    """Run the full daily HR retroactive analysis pipeline.

    Computes all Layer 1 + Layer 2 analyses, writes results to
    hr_retro_analysis table, and returns contextual findings.
    """
    logger.info("Running daily HR retro analysis for %s (lookback=%d).",
                date_str, lookback_days)

    signal_stats = compute_hr_signal_accuracy(conn, lookback_days=lookback_days)
    false_neg = compute_false_negative_analysis(conn, lookback_days=lookback_days)
    cooccurrence = compute_signal_cooccurrence(conn, lookback_days=lookback_days)
    contextual = build_contextual_findings(conn, date_str, lookback_days=lookback_days)

    write_retro_analysis(conn, date_str, signal_stats, false_neg, cooccurrence, contextual)

    # Print summary
    report = print_hr_retro_summary(signal_stats, false_neg, cooccurrence, contextual, date_str)
    print(report)

    return contextual


# ── Reporting ────────────────────────────────────────────────────────────────


def print_hr_retro_summary(
    signal_stats: dict[str, dict],
    false_neg: dict,
    cooccurrence: list[dict],
    contextual: list[dict],
    date_str: str,
) -> str:
    """Format a human-readable HR retro analysis summary.

    Returns the formatted string (for Zip integration later).
    """
    lines = []

    # Header
    lines.append("")
    lines.append("=" * 80)
    lines.append(f"  HR RETROACTIVE ANALYSIS REPORT -- {date_str}")
    lines.append("=" * 80)

    # Signal Accuracy table
    lines.append("")
    lines.append("  SIGNAL ACCURACY")
    lines.append("-" * 80)
    header = (
        f"{'signal_type':<35} {'fires':>6} {'hr_rate':>9} {'baseline':>9} "
        f"{'lift':>8} {'fp_rate':>8}"
    )
    lines.append(header)
    lines.append("-" * 80)

    for sig in sorted(signal_stats.keys()):
        st = signal_stats[sig]
        fires = st["fire_count"]
        hr_rate = st["hr_rate_when_fired"]
        baseline = st["baseline_hr_rate"]
        fp_rate = st["false_positive_rate"]
        if fires > 0 and baseline > 0:
            lift = (hr_rate - baseline) / baseline
        else:
            lift = 0.0
        lines.append(
            f"{sig:<35} {fires:>6} {hr_rate:>8.4f}  {baseline:>8.4f} "
            f"{lift:>+7.2f}  {fp_rate:>7.4f}"
        )
    lines.append("-" * 80)

    # Best Combos
    lines.append("")
    lines.append("  BEST SIGNAL COMBOS")
    lines.append("-" * 80)
    if cooccurrence:
        for combo_data in cooccurrence[:5]:
            combo_str = " + ".join(combo_data["combo"])
            lines.append(
                f"  {combo_str:<55} "
                f"n={combo_data['fire_count']:>4}  "
                f"hr_rate={combo_data['hr_rate']:.4f}"
            )
    else:
        lines.append("  (no combos with >= 5 fires)")
    lines.append("-" * 80)

    # Missed HR Profile
    lines.append("")
    lines.append("  MISSED HR PROFILE (false negatives)")
    lines.append("-" * 80)
    lines.append(
        f"  Missed HRs: {false_neg['missed_count']}  |  "
        f"Signaled HRs: {false_neg['signaled_count']}"
    )
    if false_neg["missed_count"] > 0:
        lines.append(f"  {'feature':<25} {'missed_mean':>12} {'signaled_mean':>14}")
        lines.append(f"  {'-'*25} {'-'*12} {'-'*14}")
        for feat in sorted(false_neg["missed_feature_means"].keys()):
            m_val = false_neg["missed_feature_means"][feat]
            s_val = false_neg["signaled_feature_means"].get(feat, 0.0)
            lines.append(f"  {feat:<25} {m_val:>12.4f} {s_val:>14.4f}")
    lines.append("-" * 80)

    # Contextual Insights
    lines.append("")
    lines.append("  CONTEXTUAL INSIGHTS")
    lines.append("-" * 80)
    if contextual:
        for finding in contextual:
            lines.append(f"  [{finding['finding_type']}] {finding['narrative']}")
    else:
        lines.append("  (no contextual findings with sufficient samples)")
    lines.append("=" * 80)

    return "\n".join(lines)


# ── Weekly Weight Tuning ─────────────────────────────────────────────────────


def suggest_hr_weights(
    signal_stats: dict[str, dict],
    current_weights: dict[str, float],
) -> dict[str, float]:
    """Suggest adjusted HR signal weights based on each signal's lift over baseline.

    Lift = (hr_rate_when_fired - baseline_hr_rate) / baseline_hr_rate

    new_weight = current_weight * (1 + DAMPING * lift)

    Weights are clamped to [WEIGHT_FLOOR, WEIGHT_CEIL], then normalized so
    the total weight budget matches the original sum.
    """
    original_budget = sum(current_weights.values())
    suggested: dict[str, float] = {}

    for sig, cur_w in current_weights.items():
        st = signal_stats.get(sig)
        if st is None or st["fire_count"] == 0 or st["baseline_hr_rate"] <= 0:
            # No data -- keep current weight unchanged.
            suggested[sig] = cur_w
            continue

        lift = (st["hr_rate_when_fired"] - st["baseline_hr_rate"]) / st["baseline_hr_rate"]
        new_w = cur_w * (1.0 + _DAMPING * lift)
        new_w = max(_WEIGHT_FLOOR, min(_WEIGHT_CEIL, new_w))
        suggested[sig] = new_w

    # Normalize to preserve original budget.
    raw_total = sum(suggested.values())
    if raw_total > 0:
        scale = original_budget / raw_total
        suggested = {k: round(v * scale, 6) for k, v in suggested.items()}

    return suggested


def save_tuned_hr_weights(weights: dict[str, float], path: Path | None = None) -> Path:
    """Persist suggested HR weights to JSON.

    Args:
        weights: signal_type -> weight mapping.
        path: Override output path (default: data/tuned_hr_weights.json).

    Returns:
        Path the file was written to.
    """
    out = path or TUNED_HR_WEIGHTS_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(weights, f, indent=2)
    print(f"\nTuned HR weights saved to {out}")
    return out


def run_weekly_hr_retune(
    conn: sqlite3.Connection,
    lookback_days: int = 90,
    dry_run: bool = False,
) -> None:
    """Run weekly HR signal weight retuning.

    Computes signal accuracy over the lookback window, suggests new weights,
    prints a report, and optionally saves to disk.
    """
    logger.info("Running weekly HR weight retune (lookback=%d, dry_run=%s).",
                lookback_days, dry_run)

    stats = compute_hr_signal_accuracy(conn, lookback_days=lookback_days)
    suggested = suggest_hr_weights(stats, HR_SIGNAL_WEIGHTS)

    # Print report (same pattern as weight_tuner.print_weight_report)
    header = (
        f"{'signal_type':<35} {'fires':>6} {'hr_rate':>9} {'baseline':>9} "
        f"{'lift':>8} {'cur_wt':>8} {'new_wt':>8}"
    )
    print("\n" + "=" * len(header))
    print("  HR SIGNAL WEIGHT TUNING REPORT")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for sig in sorted(HR_SIGNAL_WEIGHTS.keys()):
        st = stats.get(sig, {})
        fires = st.get("fire_count", 0)
        hr_rate = st.get("hr_rate_when_fired", 0.0)
        baseline = st.get("baseline_hr_rate", 0.0)
        if fires > 0 and baseline > 0:
            lift = (hr_rate - baseline) / baseline
        else:
            lift = 0.0
        cur_w = HR_SIGNAL_WEIGHTS.get(sig, 0.0)
        new_w = suggested.get(sig, cur_w)
        print(
            f"{sig:<35} {fires:>6} {hr_rate:>8.3f}  {baseline:>8.3f} "
            f"{lift:>+7.2f}  {cur_w:>7.4f}  {new_w:>7.4f}"
        )

    print("-" * len(header))
    print(
        f"{'TOTAL':<35} {'':>6} {'':>9} {'':>9} {'':>8} "
        f"{sum(HR_SIGNAL_WEIGHTS.values()):>7.4f}  "
        f"{sum(suggested.values()):>7.4f}"
    )
    print("=" * len(header))

    if not dry_run:
        save_tuned_hr_weights(suggested)
    else:
        print("\n(dry run -- HR weights not saved)")


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="HR retroactive training engine -- daily analysis and weekly weight tuning.",
        prog="python -m mlb_insights.signal_engine.hr_retro",
    )
    parser.add_argument(
        "--lookback",
        type=int,
        default=None,
        help="Number of days to look back (default: 30 for daily, 90 for retune).",
    )
    parser.add_argument(
        "--retune",
        action="store_true",
        help="Run weekly HR weight retuning instead of daily analysis.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print report but do not save weights to disk.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Analysis date in YYYY-MM-DD format (default: today).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from datetime import date as date_cls
    from mlb_insights.utils.db import get_connection

    analysis_date = args.date or date_cls.today().isoformat()

    if args.retune:
        lookback = args.lookback or 90
        conn = get_connection(readonly=True)
        try:
            run_weekly_hr_retune(conn, lookback_days=lookback, dry_run=args.dry_run)
        finally:
            conn.close()
    else:
        lookback = args.lookback or 30
        conn = get_connection(readonly=False)
        try:
            run_daily_hr_retro(conn, analysis_date, lookback_days=lookback)
        finally:
            conn.close()


if __name__ == "__main__":
    main()
