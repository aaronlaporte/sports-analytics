"""
MLB Player Insights Platform — Season Stats
Self-serve table: pick a metric, see season totals + last 15 games per player.
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime

st.set_page_config(page_title="MLB Insights — Season Stats", layout="wide")

DB_PATH = "data/mlb.db"

# ── Season boundaries ────────────────────────────────────────────────────────

SEASONS = {
    "2026": ("2026-03-01", "2026-12-31"),
    "2025": ("2025-03-01", "2025-10-31"),
    "2024": ("2024-03-01", "2024-10-31"),
}

# ── Metric config ────────────────────────────────────────────────────────────

METRICS = {
    "Hits":             {"source": "batter_stats", "col": "hits",   "label": "H"},
    "Home Runs":        {"source": "batter_stats", "col": "hr",     "label": "HR"},
    "Walks":            {"source": "batter_stats", "col": "bb",     "label": "BB"},
    "Strikeouts":       {"source": "batter_stats", "col": "so",     "label": "K"},
    "Plate Appearances":{"source": "batter_stats", "col": "pa",     "label": "PA"},
    "Doubles":          {"source": "statcast",     "col": "double", "label": "2B"},
    "Triples":          {"source": "statcast",     "col": "triple", "label": "3B"},
}

METRIC_NAMES = list(METRICS.keys())


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ── Data loaders ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300)
def load_batter_stats_metric(season: str, metric_col: str, min_pa: int) -> pd.DataFrame:
    """Pull per-game rows for a batter_stats metric in a season."""
    start, end = SEASONS[season]
    conn = get_connection()
    df = pd.read_sql(
        f"""
        SELECT
            b.batter_id,
            COALESCE(p.player_name, CAST(b.batter_id AS TEXT)) AS player_name,
            b.team,
            b.game_date,
            b.{metric_col}   AS metric_val,
            b.pa
        FROM batter_stats b
        LEFT JOIN player_lookup p ON b.batter_id = p.mlbam_id
        WHERE b.game_date >= ? AND b.game_date <= ?
        ORDER BY b.batter_id, b.game_date
        """,
        conn,
        params=(start, end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def load_statcast_metric(season: str, event_type: str, min_pa: int) -> pd.DataFrame:
    """Pull per-game event counts from statcast_staging for doubles/triples."""
    start, end = SEASONS[season]
    conn = get_connection()
    # Also pull PA from batter_stats for filtering
    df = pd.read_sql(
        f"""
        WITH sc AS (
            SELECT
                batter AS batter_id,
                game_date,
                SUM(CASE WHEN events = ? THEN 1 ELSE 0 END) AS metric_val
            FROM statcast_staging
            WHERE game_date >= ? AND game_date <= ?
            GROUP BY batter, game_date
        )
        SELECT
            sc.batter_id,
            COALESCE(p.player_name, CAST(sc.batter_id AS TEXT)) AS player_name,
            sc.game_date,
            sc.metric_val,
            COALESCE(b.pa, 0) AS pa,
            b.team
        FROM sc
        LEFT JOIN player_lookup p ON sc.batter_id = p.mlbam_id
        LEFT JOIN batter_stats b ON sc.batter_id = b.batter_id AND sc.game_date = b.game_date
        ORDER BY sc.batter_id, sc.game_date
        """,
        conn,
        params=(event_type, start, end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def build_season_table(
    season: str,
    metric_name: str,
    min_pa: int,
    top_n: int,
    team_filter: str,
) -> pd.DataFrame:
    """Build the pivot table: player | season total | g1 | g2 | ... | g15."""
    cfg = METRICS[metric_name]

    if cfg["source"] == "batter_stats":
        raw = load_batter_stats_metric(season, cfg["col"], min_pa)
    else:
        raw = load_statcast_metric(season, cfg["col"], min_pa)

    if raw.empty:
        return pd.DataFrame()

    # Season PA per player for min_pa filter
    pa_totals = raw.groupby("batter_id")["pa"].sum().reset_index()
    pa_totals.columns = ["batter_id", "total_pa"]

    # Season total of the metric
    season_totals = raw.groupby("batter_id")["metric_val"].sum().reset_index()
    season_totals.columns = ["batter_id", "season_total"]

    # Most recent team per player
    recent_team = (
        raw.sort_values("game_date")
        .groupby("batter_id")
        .last()[["team", "player_name"]]
        .reset_index()
    )

    # Last 15 games — per-player, sorted descending by date
    raw_sorted = raw.sort_values(["batter_id", "game_date"], ascending=[True, False])
    raw_sorted["game_num"] = raw_sorted.groupby("batter_id").cumcount() + 1
    last15 = raw_sorted[raw_sorted["game_num"] <= 15][["batter_id", "game_num", "metric_val"]]

    pivot = last15.pivot(index="batter_id", columns="game_num", values="metric_val")
    pivot.columns = [f"G{c}" for c in pivot.columns]
    pivot = pivot.reset_index()

    # Ensure all 15 columns exist
    for i in range(1, 16):
        col = f"G{i}"
        if col not in pivot.columns:
            pivot[col] = None

    game_cols = [f"G{i}" for i in range(1, 16)]
    pivot = pivot[["batter_id"] + game_cols]

    # Merge everything
    merged = (
        recent_team
        .merge(season_totals, on="batter_id")
        .merge(pa_totals, on="batter_id")
        .merge(pivot, on="batter_id", how="left")
    )

    # Apply filters
    merged = merged[merged["total_pa"] >= min_pa]
    if team_filter != "All":
        merged = merged[merged["team"] == team_filter]

    # Sort by season total desc
    merged = merged.sort_values("season_total", ascending=False).reset_index(drop=True)
    merged.insert(0, "rank", range(1, len(merged) + 1))

    # Title-case names
    merged["player_name"] = merged["player_name"].str.title()
    merged["team"] = merged["team"].fillna("--")

    # Keep top_n
    merged = merged.head(top_n)

    return merged


@st.cache_data(ttl=300)
def get_teams_for_season(season: str) -> list:
    start, end = SEASONS[season]
    conn = get_connection()
    df = pd.read_sql(
        "SELECT DISTINCT team FROM batter_stats "
        "WHERE game_date >= ? AND game_date <= ? AND team IS NOT NULL AND team <> '' "
        "ORDER BY team",
        conn,
        params=(start, end),
    )
    conn.close()
    return df["team"].tolist()


# ── Sidebar controls ─────────────────────────────────────────────────────────

st.sidebar.header("Season Stats Controls")

# Default to most recent season with data
available_seasons = list(SEASONS.keys())
season = st.sidebar.selectbox("Season", available_seasons, index=0)

metric = st.sidebar.selectbox("Metric", METRIC_NAMES, index=0)

teams = get_teams_for_season(season)
team_filter = st.sidebar.selectbox("Team", ["All"] + teams, index=0)

min_pa = st.sidebar.slider("Min Plate Appearances", min_value=0, max_value=300, value=50, step=10)

top_n = st.sidebar.select_slider("Show Top N", options=[25, 50, 100, 200], value=50)

# ── Page header ──────────────────────────────────────────────────────────────

st.title("Season Stats")
st.caption(
    "Season totals ranked by selected metric. "
    "G1 = most recent game, G2 = second most recent, … G15 = 15th most recent. "
    "RBIs and Stolen Bases are not tracked in this dataset."
)
st.markdown("---")

# ── Load & display ────────────────────────────────────────────────────────────

with st.spinner("Loading..."):
    df = build_season_table(season, metric, min_pa, top_n, team_filter)

if df.empty:
    st.warning(f"No data for {metric} in {season} with min PA ≥ {min_pa}.")
    st.stop()

cfg = METRICS[metric]
label = cfg["label"]

# Rename columns for display
game_cols = [f"G{i}" for i in range(1, 16)]
display_cols = ["rank", "player_name", "team", "season_total"] + game_cols
rename_map = {
    "rank": "#",
    "player_name": "Player",
    "team": "Team",
    "season_total": f"Season {label}",
}
rename_map.update({f"G{i}": f"G{i}" for i in range(1, 16)})

display_df = df[display_cols].copy()

# Convert game columns to int (0 for missing)
for col in game_cols:
    display_df[col] = display_df[col].fillna(0).astype(int)

display_df["season_total"] = display_df["season_total"].astype(int)
display_df = display_df.rename(columns=rename_map)

# Summary strip
c1, c2, c3 = st.columns(3)
c1.metric("Players shown", len(display_df))
c2.metric(
    f"Season leader ({label})",
    f"{display_df[f'Season {label}'].iloc[0]} — {display_df['Player'].iloc[0]}",
)
c3.metric(
    f"Avg {label} / player",
    f"{display_df[f'Season {label}'].mean():.1f}",
)

st.markdown("")


# Color coding for game columns
# Strikeouts: red = bad (high), green = good (low). Everything else: green = good (high).
is_inverse = metric == "Strikeouts"


def color_game_cell(val):
    """Color a single game cell based on value."""
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ""
    v = int(val)
    if v == 0:
        if is_inverse:
            return "background-color: rgba(0, 180, 0, 0.20)"
        return ""
    if is_inverse:
        # Strikeouts: 1 = yellow, 2 = orange, 3+ = red
        if v == 1:
            return "background-color: rgba(220, 180, 0, 0.25)"
        elif v == 2:
            return "background-color: rgba(230, 120, 0, 0.30)"
        else:
            return "background-color: rgba(200, 50, 50, 0.35)"
    else:
        # Positive metrics: 1 = light green, 2 = medium green, 3+ = bright green
        if v == 1:
            return "background-color: rgba(0, 180, 0, 0.15)"
        elif v == 2:
            return "background-color: rgba(0, 180, 0, 0.30)"
        elif v == 3:
            return "background-color: rgba(0, 180, 0, 0.45)"
        else:
            return "background-color: rgba(0, 180, 0, 0.60); font-weight: bold"


game_display_cols = [f"G{i}" for i in range(1, 16)]

styled = (
    display_df.style
    .map(color_game_cell, subset=game_display_cols)
    .map(
        lambda v: "background-color: rgba(77, 171, 247, 0.25); font-weight: bold",
        subset=[f"Season {label}"],
    )
    .format({f"Season {label}": "{:,d}"})
)

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=min(len(display_df) * 36 + 40, 900),
)

st.caption(
    f"Sorted by Season {label} descending. "
    f"Min PA filter: {min_pa}. "
    f"Season: {season}. "
    "Game columns show raw count per game (0 = played but no occurrence)."
)
