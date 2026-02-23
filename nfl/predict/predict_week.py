"""
nfl/predict/predict_week.py — Predict anytime TD probabilities for the upcoming NFL week.

Covers all 32 teams (not just a target subset).
Joins live odds from live_odds table and computes EV.
Writes to daily_picks in nfl.db.

Usage:
    python nfl/predict/predict_week.py
    python nfl/predict/predict_week.py --week 14 --season 2025REG
    python nfl/predict/predict_week.py --top 20
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import load

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn, create_schema
from nfl.features.build_training_table import FEATURE_COLS

MODELS_DIR = Path(__file__).parent.parent / "models"
DECAY_HALFLIFE_DAYS = 45


def _rolling_mean(series: pd.Series, w: int) -> float:
    if series.empty:
        return 0.0
    return float(series.tail(w).mean())


def _ewm_mean(series: pd.Series, span: int) -> float:
    if series.empty:
        return 0.0
    return float(series.ewm(span=span, adjust=False).mean().iloc[-1])


def _injury_flags(injuries_df: pd.DataFrame, player_id, player_name: str):
    if injuries_df.empty:
        return 0, 0, 0
    row = injuries_df[injuries_df["player_id"] == player_id]
    if row.empty:
        row = injuries_df[injuries_df["player_name"].str.lower() == str(player_name).lower()]
    if row.empty:
        return 0, 0, 0
    status = str(row.iloc[0].get("status", "")).lower()
    return int("out" in status), int("doubtful" in status), int("questionable" in status)


def _short_reason(row: dict) -> str:
    reasons = []
    if row.get("touches_rolling_3", 0) >= 10:
        reasons.append("high recent touches")
    if row.get("targets_rolling_3", 0) >= 4:
        reasons.append("strong target share")
    if row.get("season_td_rate", 0) >= 0.3:
        reasons.append("high TD rate")
    if row.get("opp_td_allowed_rolling_5", 0) >= 2:
        reasons.append("opponent allows TDs")
    if row.get("is_home", 0) == 1:
        reasons.append("home game")
    if not reasons:
        reasons.append("balanced matchup")
    return "; ".join(reasons[:3])


def predict_week(week: int = None, season: str = None, top: int = 30):
    create_schema("nfl")
    conn = get_conn("nfl")

    # Load historical player stats
    pstats = pd.read_sql("""
        SELECT player_id, player_name, team, opponent, game_key,
               season, week, game_date, position,
               rush_attempts, receiving_targets, receptions, rush_tds, receiving_tds, total_tds
        FROM player_game_stats
        ORDER BY player_id, game_date
    """, conn)

    sched = pd.read_sql("""
        SELECT game_key, season, week, game_date, home_team, away_team,
               home_score, away_score
        FROM schedules
        ORDER BY game_date
    """, conn)

    # Load injuries if available
    try:
        injuries_df = pd.read_sql(
            "SELECT player_id, player_name, status FROM injuries", conn
        )
    except Exception:
        injuries_df = pd.DataFrame()

    # Load live odds
    try:
        odds_df = pd.read_sql("""
            SELECT player, market, name, line, price
            FROM live_odds
            WHERE market = 'player_anytime_touchdown'
              AND name = 'Yes'
            ORDER BY fetched_at DESC
        """, conn)
    except Exception:
        odds_df = pd.DataFrame()

    conn.close()

    pstats["game_date"] = pd.to_datetime(pstats["game_date"])
    sched["game_date"]  = pd.to_datetime(sched["game_date"])
    for col in ["rush_attempts", "receiving_targets", "receptions", "rush_tds", "receiving_tds", "total_tds"]:
        pstats[col] = pstats[col].fillna(0)
    pstats["touches"] = pstats["rush_attempts"] + pstats["receiving_targets"]

    # Find target week
    if week and season:
        target_games = sched[(sched["season"] == season) & (sched["week"] == week)]
    else:
        # Next upcoming week (no scores yet)
        upcoming = sched[sched["home_score"].isna()].sort_values("game_date")
        if upcoming.empty:
            print("[nfl/predict] No upcoming games found.")
            return pd.DataFrame()
        target_date = upcoming.iloc[0]["game_date"]
        target_games = upcoming[upcoming["game_date"] == target_date]
        season = target_games.iloc[0]["season"] if not target_games.empty else "unknown"
        week = int(target_games.iloc[0]["week"]) if not target_games.empty else 0

    if target_games.empty:
        print(f"[nfl/predict] No games for {season} week {week}.")
        return pd.DataFrame()

    print(f"[nfl/predict] Scoring {len(target_games)} games — {season} Week {week}")

    # Load models
    models_ok = True
    for name in ["anytime_td_model.pkl", "td2plus_model.pkl"]:
        if not (MODELS_DIR / name).exists():
            print(f"[nfl/predict] Model not found: {name}. Run nfl/model/train_model.py first.")
            models_ok = False
    if not models_ok:
        return pd.DataFrame()

    anytime_bundle = load(MODELS_DIR / "anytime_td_model.pkl")
    td2plus_bundle = load(MODELS_DIR / "td2plus_model.pkl")
    anytime_model  = anytime_bundle["model"]
    anytime_by_pos = anytime_bundle.get("by_position", {})
    anytime_stacker = anytime_bundle.get("stacker", None)
    td2plus_model   = td2plus_bundle["model"]
    td2_features    = td2plus_bundle.get("features", FEATURE_COLS)

    # Build feature rows for each player in target games
    feature_rows = []
    target_date = target_games["game_date"].min().date()

    # Get players from historical stats (all teams)
    all_teams = set(target_games["home_team"]) | set(target_games["away_team"])
    recent_players = (
        pstats[pstats["team"].isin(all_teams) & (pstats["game_date"].dt.date < target_date)]
        .sort_values(["player_id", "game_date"])
        .groupby("player_id")
        .tail(20)  # last 20 games per player
    )

    if recent_players.empty:
        print("[nfl/predict] No recent player history found for target teams.")
        return pd.DataFrame()

    # For each player × game combo
    player_teams = recent_players.groupby("player_id")[["player_name", "team", "position"]].last().reset_index()
    player_teams = player_teams[player_teams["position"].isin(["RB", "WR", "TE", "QB", "FB"])]

    for _, game in target_games.iterrows():
        for team in [game["home_team"], game["away_team"]]:
            opponent = game["away_team"] if team == game["home_team"] else game["home_team"]
            is_home  = int(team == game["home_team"])
            denver   = int(game["home_team"] == "DEN")

            team_players = player_teams[player_teams["team"] == team]

            for _, p in team_players.iterrows():
                pid = p["player_id"]
                ph = pstats[
                    (pstats["player_id"] == pid)
                    & (pstats["game_date"].dt.date < target_date)
                ].sort_values("game_date")

                if len(ph) < 2:
                    continue

                rush = ph["rush_attempts"]
                tgts = ph["receiving_targets"]
                recs = ph["receptions"]
                tds  = ph["total_tds"]
                touches = ph["touches"]

                n_games = len(ph)
                season_td_rate = float(tds.sum() / n_games) if n_games else 0.0
                missed_last = 0  # can't know for future

                inj_out, inj_dbt, inj_qst = _injury_flags(injuries_df, pid, p["player_name"])

                # Team scoring context
                team_pts = pstats[
                    (pstats["team"] == team)
                    & (pstats["game_date"].dt.date < target_date)
                ].groupby("game_key")["total_tds"].sum()

                opp_pts = pstats[
                    (pstats["opponent"] == opponent)
                    & (pstats["game_date"].dt.date < target_date)
                ].groupby("game_key")["total_tds"].sum()

                # Shares (recent 5 games)
                team_recent_stats = pstats[
                    (pstats["team"] == team)
                    & (pstats["game_date"].dt.date < target_date)
                ].tail(5 * 50)  # approx
                team_recent_touches = team_recent_stats.groupby("game_key")["touches"].sum().tail(5).mean() if len(team_recent_stats) else 0
                team_recent_tgts    = team_recent_stats.groupby("game_key")["receiving_targets"].sum().tail(5).mean() if len(team_recent_stats) else 0
                team_recent_rush    = team_recent_stats.groupby("game_key")["rush_attempts"].sum().tail(5).mean() if len(team_recent_stats) else 0

                touch_share  = (_rolling_mean(touches, 5) / team_recent_touches) if team_recent_touches else 0.0
                target_share = (_rolling_mean(tgts, 5) / team_recent_tgts) if team_recent_tgts else 0.0
                rush_share   = (_rolling_mean(rush, 5) / team_recent_rush) if team_recent_rush else 0.0

                row = {
                    "player_id":   pid,
                    "player_name": p["player_name"],
                    "position":    p["position"],
                    "team":        team,
                    "opponent":    opponent,
                    "game_date":   str(target_date),
                    "is_home":     is_home,
                    "denver_game": denver,
                    "pos_wr": int(p["position"] == "WR"),
                    "pos_rb": int(p["position"] == "RB"),
                    "pos_te": int(p["position"] == "TE"),
                    "pos_qb": int(p["position"] == "QB"),
                    "rush_att_rolling_3": _rolling_mean(rush, 3),
                    "rush_att_rolling_5": _rolling_mean(rush, 5),
                    "rush_att_ewm_3": _ewm_mean(rush, 3),
                    "rush_att_ewm_5": _ewm_mean(rush, 5),
                    "targets_rolling_3": _rolling_mean(tgts, 3),
                    "targets_rolling_5": _rolling_mean(tgts, 5),
                    "targets_ewm_3": _ewm_mean(tgts, 3),
                    "targets_ewm_5": _ewm_mean(tgts, 5),
                    "receptions_rolling_3": _rolling_mean(recs, 3),
                    "receptions_rolling_5": _rolling_mean(recs, 5),
                    "receptions_ewm_3": _ewm_mean(recs, 3),
                    "receptions_ewm_5": _ewm_mean(recs, 5),
                    "touches_rolling_3": _rolling_mean(touches, 3),
                    "touches_rolling_5": _rolling_mean(touches, 5),
                    "touches_ewm_3": _ewm_mean(touches, 3),
                    "touches_ewm_5": _ewm_mean(touches, 5),
                    "total_td_rolling_3": _rolling_mean(tds, 3),
                    "total_td_rolling_5": _rolling_mean(tds, 5),
                    "total_td_ewm_3": _ewm_mean(tds, 3),
                    "total_td_ewm_5": _ewm_mean(tds, 5),
                    "season_td_rate": season_td_rate,
                    "points_scored_rolling_3": _rolling_mean(team_pts, 3),
                    "points_scored_rolling_5": _rolling_mean(team_pts, 5),
                    "opp_points_allowed_rolling_3": _rolling_mean(opp_pts, 3),
                    "opp_points_allowed_rolling_5": _rolling_mean(opp_pts, 5),
                    "opp_td_allowed_rolling_3": _rolling_mean(opp_pts, 3),
                    "opp_td_allowed_rolling_5": _rolling_mean(opp_pts, 5),
                    "td_allowed_pos_rolling_3": 0.0,
                    "td_allowed_pos_rolling_5": 0.0,
                    "touch_share": touch_share,
                    "target_share": target_share,
                    "rush_share": rush_share,
                    "missed_last_game": missed_last,
                    "injury_out": inj_out,
                    "injury_doubtful": inj_dbt,
                    "injury_questionable": inj_qst,
                }
                feature_rows.append(row)

    if not feature_rows:
        print("[nfl/predict] No feature rows built.")
        return pd.DataFrame()

    feat_df = pd.DataFrame(feature_rows)
    for col in FEATURE_COLS:
        if col not in feat_df.columns:
            feat_df[col] = 0.0
    feat_df[FEATURE_COLS] = feat_df[FEATURE_COLS].fillna(0.0)

    # Score anytime TD
    global_any = anytime_model.predict_proba(feat_df[FEATURE_COLS])[:, 1]
    pos_any = np.zeros(len(feat_df))
    for pos, sub in feat_df.groupby("position"):
        m = anytime_by_pos.get(pos, anytime_model)
        pos_any[sub.index] = m.predict_proba(sub[FEATURE_COLS])[:, 1]

    if anytime_stacker is not None:
        stack_X = np.vstack([global_any, pos_any]).T
        pred_any = anytime_stacker.predict_proba(stack_X)[:, 1]
    else:
        pred_any = pos_any
    feat_df["predicted_td_probability"] = pred_any

    # Score 2+ TD
    td2_feat_cols = [c for c in td2_features if c in feat_df.columns]
    lam = td2plus_model.predict(feat_df[td2_feat_cols]).clip(0)
    feat_df["predicted_2plus_td_prob"] = 1 - np.exp(-lam) * (1 + lam)

    # Short reason
    feat_df["short_reason"] = feat_df.apply(lambda r: _short_reason(r.to_dict()), axis=1)

    # Join live odds
    feat_df["best_odds"] = None
    feat_df["ev"] = None
    feat_df["value_bet"] = 0
    if not odds_df.empty:
        for i, row in feat_df.iterrows():
            last = row["player_name"].split()[-1].lower()
            match = odds_df[odds_df["player"].str.lower().str.contains(last, na=False)]
            if not match.empty:
                best = match.sort_values("price", ascending=False).iloc[0]
                amer = int(best["price"])
                dec = (1 + amer / 100) if amer > 0 else (1 + 100 / abs(amer))
                ev = row["predicted_td_probability"] * dec - 1
                feat_df.at[i, "best_odds"] = amer
                feat_df.at[i, "ev"] = round(ev, 4)
                feat_df.at[i, "value_bet"] = int(ev >= 0.05)

    # Save picks to DB
    pick_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    conn2 = get_conn("nfl")
    picks_rows = []
    for _, r in feat_df.iterrows():
        picks_rows.append((
            pick_date,
            r["player_id"],
            r["player_name"],
            r["team"],
            r["opponent"],
            r["position"],
            "anytime_td",
            round(float(r["predicted_td_probability"]), 4),
            "Yes",
            0.5,  # standard line
            int(r["best_odds"]) if pd.notna(r.get("best_odds")) and r.get("best_odds") is not None else None,
            None,  # best_book
            float(r["ev"]) if pd.notna(r.get("ev")) and r.get("ev") is not None else None,
            int(r["value_bet"]),
        ))
    conn2.executemany("""
        INSERT OR REPLACE INTO daily_picks
            (pick_date, player_id, player_name, team, opponent, position,
             prop_type, model_prob, direction, line, best_odds, best_book, ev, value_bet)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, picks_rows)
    conn2.commit()
    conn2.close()

    # Print top picks
    out = feat_df.sort_values("predicted_td_probability", ascending=False).head(top)
    print(f"\n=== NFL Week {week} — Top Anytime TD Picks ===")
    print(out[[
        "player_name", "position", "team", "opponent",
        "predicted_td_probability", "predicted_2plus_td_prob",
        "best_odds", "ev", "value_bet", "short_reason"
    ]].to_string(index=False))

    return feat_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--season", default=None)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()
    predict_week(week=args.week, season=args.season, top=args.top)
