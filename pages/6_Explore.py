"""
MLB Player Insights Platform — Explore

Dynamic scatter plots: pick any two metrics for X and Y axes,
optionally color and size by additional metrics.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import sqlite3
from datetime import datetime, timedelta

st.set_page_config(page_title="MLB Insights — Explore", layout="wide", page_icon="\U0001f50d")

DB_PATH = "data/mlb.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ── Metric definitions ──────────────────────────────────────────────────────
# Each metric: (label, SQL expression, format, description)
# These get computed from season-aggregated batter_stats + hr_features.

METRICS = {
    # Counting stats (season totals)
    "Hits": {
        "sql": "SUM(b.hits)",
        "fmt": ",.0f",
        "desc": "Total hits on the season",
    },
    "Home Runs": {
        "sql": "SUM(b.hr)",
        "fmt": ",.0f",
        "desc": "Total home runs on the season",
    },
    "Walks": {
        "sql": "SUM(b.bb)",
        "fmt": ",.0f",
        "desc": "Total walks on the season",
    },
    "Strikeouts": {
        "sql": "SUM(b.so)",
        "fmt": ",.0f",
        "desc": "Total strikeouts on the season",
    },
    "Plate Appearances": {
        "sql": "SUM(b.pa)",
        "fmt": ",.0f",
        "desc": "Total plate appearances on the season",
    },
    "At Bats": {
        "sql": "SUM(b.ab)",
        "fmt": ",.0f",
        "desc": "Total at bats on the season",
    },
    "Games": {
        "sql": "COUNT(DISTINCT b.game_date)",
        "fmt": ",.0f",
        "desc": "Games played on the season",
    },
    # Rate stats
    "AVG": {
        "sql": "ROUND(CAST(SUM(b.hits) AS REAL) / NULLIF(SUM(b.ab), 0), 3)",
        "fmt": ".3f",
        "desc": "Batting average (H/AB)",
    },
    "OBP": {
        "sql": "ROUND(CAST(SUM(b.hits) + SUM(b.bb) AS REAL) / NULLIF(SUM(b.pa), 0), 3)",
        "fmt": ".3f",
        "desc": "On-base percentage",
    },
    "HR Rate": {
        "sql": "ROUND(CAST(SUM(b.hr) AS REAL) / NULLIF(SUM(b.pa), 0), 4)",
        "fmt": ".4f",
        "desc": "Home runs per plate appearance",
    },
    "K Rate": {
        "sql": "ROUND(CAST(SUM(b.so) AS REAL) / NULLIF(SUM(b.pa), 0), 3)",
        "fmt": ".3f",
        "desc": "Strikeout rate (SO/PA)",
    },
    "BB Rate": {
        "sql": "ROUND(CAST(SUM(b.bb) AS REAL) / NULLIF(SUM(b.pa), 0), 3)",
        "fmt": ".3f",
        "desc": "Walk rate (BB/PA)",
    },
    "Hits / Game": {
        "sql": "ROUND(CAST(SUM(b.hits) AS REAL) / NULLIF(COUNT(DISTINCT b.game_date), 0), 2)",
        "fmt": ".2f",
        "desc": "Average hits per game played",
    },
    "HR / Game": {
        "sql": "ROUND(CAST(SUM(b.hr) AS REAL) / NULLIF(COUNT(DISTINCT b.game_date), 0), 3)",
        "fmt": ".3f",
        "desc": "Average home runs per game played",
    },
    # Power metrics (from latest hr_features)
    "Barrel Rate": {
        "sql": "h.barrel_rate",
        "fmt": ".1%",
        "desc": "% of batted balls at optimal speed/angle (latest)",
        "source": "hr_features",
    },
    "Avg Exit Velo": {
        "sql": "h.avg_exit_velo",
        "fmt": ".1f",
        "desc": "Average exit velocity in mph (latest)",
        "source": "hr_features",
    },
    "P(HR)": {
        "sql": "h.p_hr",
        "fmt": ".1%",
        "desc": "Model's latest home run probability",
        "source": "hr_features",
    },
    "Park Factor": {
        "sql": "h.park_factor",
        "fmt": ".2f",
        "desc": "Ballpark HR factor (>1.0 = HR-friendly)",
        "source": "hr_features",
    },
    "Power Trend": {
        "sql": "h.power_trend",
        "fmt": ".3f",
        "desc": "Recent exit velo trend vs season baseline",
        "source": "hr_features",
    },
}

# Metrics available for color/size (subset that makes visual sense)
COLOR_METRICS = ["None"] + list(METRICS.keys())


# ── Data loading ────────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def get_scatter_data(year: int, min_pa: int):
    """Build a single dataframe with all metrics for the scatter plot."""
    conn = get_connection()
    season_start = f"{year}-01-01"
    season_end = f"{year}-12-31"

    # Get latest hr_features date
    latest_hf = conn.execute(
        "SELECT MAX(feature_date) FROM hr_features WHERE feature_date BETWEEN ? AND ?",
        (season_start, season_end),
    ).fetchone()[0]

    # Build aggregated batter stats
    agg_cols = []
    agg_aliases = []
    hr_cols = []
    for name, cfg in METRICS.items():
        alias = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_per_")
        if cfg.get("source") == "hr_features":
            hr_cols.append((name, cfg["sql"], alias))
        else:
            agg_cols.append(f"{cfg['sql']} AS {alias}")
            agg_aliases.append((name, alias))

    agg_sql = ", ".join(agg_cols)

    query = f"""
        SELECT b.batter_id,
               p.player_name,
               b.team,
               {agg_sql}
        FROM batter_stats b
        LEFT JOIN player_lookup p ON b.batter_id = p.mlbam_id
        WHERE b.game_date BETWEEN ? AND ?
          AND b.pa > 0
        GROUP BY b.batter_id
        HAVING SUM(b.pa) >= ?
    """
    df = pd.read_sql(query, conn, params=(season_start, season_end, min_pa))

    # Get team from most recent game
    team_df = pd.read_sql("""
        SELECT batter_id, team
        FROM batter_stats
        WHERE game_date = (SELECT MAX(game_date) FROM batter_stats WHERE game_date BETWEEN ? AND ?)
          AND team IS NOT NULL
    """, conn, params=(season_start, season_end))
    if not team_df.empty:
        team_map = dict(zip(team_df["batter_id"], team_df["team"]))
        df["team"] = df["batter_id"].map(team_map).fillna(df["team"])

    # Join hr_features if available
    if latest_hf and hr_cols:
        hf_df = pd.read_sql(
            "SELECT batter_id, barrel_rate, avg_exit_velo, p_hr, park_factor, power_trend "
            "FROM hr_features WHERE feature_date = ?",
            conn,
            params=(latest_hf,),
        )
        if not hf_df.empty:
            df = df.merge(hf_df, on="batter_id", how="left")

    conn.close()

    # Rename columns to metric names for easy access
    col_map = {}
    for name, alias in agg_aliases:
        col_map[alias] = name
    for name, sql_expr, alias in hr_cols:
        # These come in as the raw column name from hr_features
        pass  # Already named correctly from the merge

    df = df.rename(columns=col_map)

    # Map hr_features columns to metric names
    hf_rename = {
        "barrel_rate": "Barrel Rate",
        "avg_exit_velo": "Avg Exit Velo",
        "p_hr": "P(HR)",
        "park_factor": "Park Factor",
        "power_trend": "Power Trend",
    }
    df = df.rename(columns=hf_rename)

    df["player_name"] = df["player_name"].fillna("Unknown")
    df["team"] = df["team"].fillna("--")

    return df


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Explore Controls")

# Year picker
current_year = datetime.today().year
years = list(range(current_year, current_year - 3, -1))
year = st.sidebar.selectbox("Season", years, index=0)

min_pa = st.sidebar.slider("Min Plate Appearances", 10, 200, 50, step=10)

# Load data
df = get_scatter_data(year, min_pa)

if df.empty:
    st.error(f"No data for {year} with min PA >= {min_pa}.")
    st.stop()

# Get available metrics (only those that exist in the dataframe)
available_metrics = [m for m in METRICS.keys() if m in df.columns]

# Axis selectors
st.sidebar.markdown("---")
st.sidebar.subheader("Axes")
x_metric = st.sidebar.selectbox("X Axis", available_metrics, index=available_metrics.index("Avg Exit Velo") if "Avg Exit Velo" in available_metrics else 0)
y_default = available_metrics.index("Home Runs") if "Home Runs" in available_metrics else min(1, len(available_metrics) - 1)
y_metric = st.sidebar.selectbox("Y Axis", available_metrics, index=y_default)

st.sidebar.markdown("---")
st.sidebar.subheader("Style")
available_color = ["None"] + available_metrics
color_metric = st.sidebar.selectbox("Color by", available_color, index=0)
size_metric = st.sidebar.selectbox("Size by", available_color, index=0)

# Team filter
teams = sorted(df["team"].unique())
team_filter = st.sidebar.selectbox("Team Filter", ["All"] + teams, index=0)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <h1 style="margin-bottom: 0.2rem;">Explore</h1>
        <p style="color: #888; font-size: 1.1rem; margin-top: 0;">
            Pick any two metrics. See what shows up.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"**{year} Season** | Min PA: {min_pa} | {len(df)} players")
st.markdown("---")


# ── Apply filters ────────────────────────────────────────────────────────────

plot_df = df.copy()
if team_filter != "All":
    plot_df = plot_df[plot_df["team"] == team_filter]

if plot_df.empty:
    st.warning("No players match the current filters.")
    st.stop()

# Drop rows where either axis metric is null
plot_df = plot_df.dropna(subset=[x_metric, y_metric])


# ── Scatter plot ─────────────────────────────────────────────────────────────

x_fmt = METRICS[x_metric]["fmt"]
y_fmt = METRICS[y_metric]["fmt"]

hover_data = {
    "team": True,
    x_metric: f":{x_fmt}",
    y_metric: f":{y_fmt}",
}

scatter_kwargs = dict(
    data_frame=plot_df,
    x=x_metric,
    y=y_metric,
    hover_name="player_name",
    hover_data=hover_data,
    labels={x_metric: x_metric, y_metric: y_metric},
)

if color_metric != "None" and color_metric in plot_df.columns:
    scatter_kwargs["color"] = color_metric
    scatter_kwargs["color_continuous_scale"] = "magma"
    hover_data[color_metric] = f":{METRICS[color_metric]['fmt']}"
else:
    scatter_kwargs["color_discrete_sequence"] = ["#fc8961"]

if size_metric != "None" and size_metric in plot_df.columns:
    scatter_kwargs["size"] = size_metric
    scatter_kwargs["size_max"] = 25
    hover_data[size_metric] = f":{METRICS[size_metric]['fmt']}"

scatter_kwargs["hover_data"] = hover_data

fig = px.scatter(**scatter_kwargs)

# Add median reference lines
x_med = plot_df[x_metric].median()
y_med = plot_df[y_metric].median()
fig.add_hline(y=y_med, line_dash="dot", line_color="#555", opacity=0.4)
fig.add_vline(x=x_med, line_dash="dot", line_color="#555", opacity=0.4)

fig.update_layout(
    height=600,
    template="plotly_dark",
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=20, b=40),
)

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Dotted lines = median. "
    f"X median: {x_med:{x_fmt}} | Y median: {y_med:{y_fmt}} | "
    f"{len(plot_df)} players shown."
)

# ── Metric descriptions ─────────────────────────────────────────────────────

with st.expander("Metric Definitions"):
    for name in available_metrics:
        cfg = METRICS[name]
        st.markdown(f"**{name}** — {cfg['desc']}")

st.markdown("---")
st.caption("Explore | MLB Player Insights Platform")
