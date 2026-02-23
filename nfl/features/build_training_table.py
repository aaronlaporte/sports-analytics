"""
nfl/features/build_training_table.py — Build NFL anytime-TD training table from nfl.db.

Reads player_game_stats and schedules from SQLite (all 32 teams),
computes rolling usage + team context features, and returns a DataFrame
ready for model training.

Usage:
    python nfl/features/build_training_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.db import get_conn

MODELS_DIR = Path(__file__).parent.parent / "models"

FEATURE_COLS = [
    "rush_att_rolling_3", "rush_att_rolling_5", "rush_att_ewm_3", "rush_att_ewm_5",
    "targets_rolling_3", "targets_rolling_5", "targets_ewm_3", "targets_ewm_5",
    "receptions_rolling_3", "receptions_rolling_5", "receptions_ewm_3", "receptions_ewm_5",
    "touches_rolling_3", "touches_rolling_5", "touches_ewm_3", "touches_ewm_5",
    "total_td_rolling_3", "total_td_rolling_5", "total_td_ewm_3", "total_td_ewm_5",
    "season_td_rate",
    "points_scored_rolling_3", "points_scored_rolling_5",
    "opp_points_allowed_rolling_3", "opp_points_allowed_rolling_5",
    "opp_td_allowed_rolling_3", "opp_td_allowed_rolling_5",
    "td_allowed_pos_rolling_3", "td_allowed_pos_rolling_5",
    "is_home", "denver_game",
    "injury_out", "injury_doubtful", "injury_questionable",
    "touch_share", "target_share", "rush_share",
    "missed_last_game",
    "pos_wr", "pos_rb", "pos_te", "pos_qb",
]


def _rolling(df: pd.DataFrame, group_col: str, date_col: str,
             value_cols: list, windows: list) -> pd.DataFrame:
    df = df.sort_values([group_col, date_col]).copy()
    for col in value_cols:
        for w in windows:
            df[f"{col}_rolling_{w}"] = (
                df.groupby(group_col)[col]
                .apply(lambda s: s.shift(1).rolling(w, min_periods=1).mean())
                .reset_index(level=0, drop=True)
            )
    return df


def _ewm(df: pd.DataFrame, group_col: str, date_col: str,
         value_cols: list, spans: list) -> pd.DataFrame:
    df = df.sort_values([group_col, date_col]).copy()
    for col in value_cols:
        for span in spans:
            df[f"{col}_ewm_{span}"] = (
                df.groupby(group_col)[col]
                .apply(lambda s: s.shift(1).ewm(span=span, adjust=False).mean())
                .reset_index(level=0, drop=True)
            )
    return df


def build_training_table(conn=None) -> pd.DataFrame:
    close_conn = conn is None
    if conn is None:
        conn = get_conn("nfl")

    pstats = pd.read_sql("""
        SELECT player_id, player_name, team, opponent, game_key, season, week,
               game_date, home_or_away, position,
               rush_attempts, receiving_targets, receptions, receiving_yards,
               rush_tds, receiving_tds, total_tds, injury_status
        FROM player_game_stats
        ORDER BY player_id, game_date
    """, conn)

    sched = pd.read_sql("""
        SELECT game_key, game_date, home_team, away_team, home_score, away_score
        FROM schedules
        ORDER BY game_date
    """, conn)

    if close_conn:
        conn.close()

    if pstats.empty or sched.empty:
        raise RuntimeError("Missing player_game_stats or schedules. Run ingest scripts first.")

    pstats["game_date"] = pd.to_datetime(pstats["game_date"])
    sched["game_date"]  = pd.to_datetime(sched["game_date"])

    # Fill nulls
    for col in ["rush_attempts", "receiving_targets", "receptions", "rush_tds", "receiving_tds", "total_tds"]:
        pstats[col] = pstats[col].fillna(0)

    pstats["touches"] = pstats["rush_attempts"] + pstats["receiving_targets"]

    # Labels
    pstats["anytime_td"] = (pstats["total_tds"] > 0).astype(int)

    # Position flags
    for pos in ["WR", "RB", "TE", "QB"]:
        pstats[f"pos_{pos.lower()}"] = (pstats["position"] == pos).astype(int)

    # Home / away / Denver
    pstats = pstats.merge(
        sched[["game_key", "home_team", "away_team"]],
        on="game_key", how="left",
    )
    pstats["is_home"] = (pstats["team"] == pstats["home_team"]).astype(int)
    pstats["opponent"] = np.where(
        pstats["team"] == pstats["home_team"],
        pstats["away_team"], pstats["home_team"]
    )
    pstats["denver_game"] = (pstats["home_team"] == "DEN").astype(int)

    # Injury flags
    pstats["injury_out"]          = (pstats["injury_status"].str.lower().str.contains("out", na=False)).astype(int)
    pstats["injury_doubtful"]     = (pstats["injury_status"].str.lower().str.contains("doubtful", na=False)).astype(int)
    pstats["injury_questionable"] = (pstats["injury_status"].str.lower().str.contains("questionable", na=False)).astype(int)

    # Rolling player usage
    usage_cols = ["rush_attempts", "receiving_targets", "receptions", "touches", "total_tds"]
    pstats = _rolling(pstats, "player_id", "game_date", usage_cols, [3, 5])
    pstats = _ewm(pstats, "player_id", "game_date", usage_cols, [3, 5])

    # Season TD rate
    td = pstats["total_tds"]
    cum_td = td.groupby(pstats["player_id"]).cumsum().shift(1).fillna(0)
    cum_g  = pstats.groupby("player_id").cumcount()
    pstats["season_td_rate"] = (cum_td / cum_g.replace(0, np.nan)).fillna(0)

    # Missed last game
    pstats = pstats.sort_values(["player_id", "game_date"]).copy()
    pstats["played_flag"] = 1  # historical games are all played
    pstats["missed_last_game"] = (
        pstats.groupby("player_id")["played_flag"].shift(1) == 0
    ).astype(int).fillna(0)

    # Team scoring context from schedule
    home_pts = sched[["game_key", "game_date", "home_team", "home_score", "away_score"]].copy()
    home_pts["team"] = home_pts["home_team"]
    home_pts["points_scored"] = home_pts["home_score"]
    home_pts["points_allowed"] = home_pts["away_score"]

    away_pts = sched[["game_key", "game_date", "away_team", "home_score", "away_score"]].copy()
    away_pts["team"] = away_pts["away_team"]
    away_pts["points_scored"] = away_pts["away_score"]
    away_pts["points_allowed"] = away_pts["home_score"]

    team_pts = pd.concat([home_pts, away_pts], ignore_index=True)
    team_pts = team_pts.dropna(subset=["points_scored"])
    team_pts = _rolling(team_pts, "team", "game_date",
                        ["points_scored", "points_allowed"], [3, 5])

    pstats = pstats.merge(
        team_pts[["game_key", "team", "points_scored_rolling_3", "points_scored_rolling_5"]],
        left_on=["game_key", "team"], right_on=["game_key", "team"], how="left",
    )

    # Opponent scoring context
    opp_pts = team_pts.rename(columns={
        "team": "opponent",
        "points_allowed_rolling_3": "opp_points_allowed_rolling_3",
        "points_allowed_rolling_5": "opp_points_allowed_rolling_5",
    })
    pstats = pstats.merge(
        opp_pts[["game_key", "opponent",
                 "opp_points_allowed_rolling_3", "opp_points_allowed_rolling_5"]],
        on=["game_key", "opponent"], how="left",
    )

    # Opponent TD allowed (total)
    td_by_opp = pstats.groupby(["game_key", "opponent"])["total_tds"].sum().reset_index()
    td_by_opp = td_by_opp.merge(sched[["game_key", "game_date"]], on="game_key", how="left")
    td_by_opp = _rolling(td_by_opp, "opponent", "game_date", ["total_tds"], [3, 5])
    td_by_opp = td_by_opp.rename(columns={
        "total_tds_rolling_3": "opp_td_allowed_rolling_3",
        "total_tds_rolling_5": "opp_td_allowed_rolling_5",
    })
    pstats = pstats.merge(
        td_by_opp[["game_key", "opponent",
                   "opp_td_allowed_rolling_3", "opp_td_allowed_rolling_5"]],
        on=["game_key", "opponent"], how="left",
    )

    # Opponent TD allowed by position
    td_pos = pstats.groupby(["game_key", "opponent", "position"])["total_tds"].sum().reset_index()
    td_pos = td_pos.merge(sched[["game_key", "game_date"]], on="game_key", how="left")
    td_pos["opp_pos_key"] = td_pos["opponent"] + "_" + td_pos["position"].astype(str)
    td_pos = _rolling(td_pos, "opp_pos_key", "game_date", ["total_tds"], [3, 5])
    td_pos = td_pos.rename(columns={
        "total_tds_rolling_3": "td_allowed_pos_rolling_3",
        "total_tds_rolling_5": "td_allowed_pos_rolling_5",
    })
    pstats = pstats.merge(
        td_pos[["game_key", "opponent", "position",
                "td_allowed_pos_rolling_3", "td_allowed_pos_rolling_5"]],
        on=["game_key", "opponent", "position"], how="left",
    )

    # Touch/target/rush share
    team_touches = pstats.groupby(["game_key", "team"])["touches"].sum().rename("team_touches").reset_index()
    team_tgts    = pstats.groupby(["game_key", "team"])["receiving_targets"].sum().rename("team_targets").reset_index()
    team_rush    = pstats.groupby(["game_key", "team"])["rush_attempts"].sum().rename("team_rush").reset_index()
    pstats = pstats.merge(team_touches, on=["game_key", "team"], how="left")
    pstats = pstats.merge(team_tgts, on=["game_key", "team"], how="left")
    pstats = pstats.merge(team_rush, on=["game_key", "team"], how="left")
    pstats["touch_share"]  = (pstats["touches"] / pstats["team_touches"].replace(0, np.nan)).fillna(0)
    pstats["target_share"] = (pstats["receiving_targets"] / pstats["team_targets"].replace(0, np.nan)).fillna(0)
    pstats["rush_share"]   = (pstats["rush_attempts"] / pstats["team_rush"].replace(0, np.nan)).fillna(0)

    # Fill feature NaNs
    for col in FEATURE_COLS:
        if col not in pstats.columns:
            pstats[col] = 0.0
    pstats[FEATURE_COLS] = pstats[FEATURE_COLS].fillna(0.0)

    keep = [
        "player_id", "player_name", "position", "team", "opponent",
        "game_date", "season", "week",
        "anytime_td", "total_tds", "receptions",
    ] + FEATURE_COLS

    result = pstats[[c for c in keep if c in pstats.columns]].copy()
    print(f"[nfl/features] Training table: {len(result):,} rows, {len(FEATURE_COLS)} features")
    return result


if __name__ == "__main__":
    df = build_training_table()
    print(df[["player_name", "position", "anytime_td"] + FEATURE_COLS[:5]].head(10).to_string())
