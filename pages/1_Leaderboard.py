"""
MLB Player Insights Platform — Daily Leaderboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3

st.set_page_config(page_title="MLB Insights — Leaderboard", layout="wide")

DB_PATH = "data/mlb.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ── Cached queries ───────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def get_available_dates():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT DISTINCT prediction_date FROM daily_leaderboard "
        "ORDER BY prediction_date DESC",
        conn,
    )
    conn.close()
    return df["prediction_date"].tolist()


@st.cache_data(ttl=300)
def get_signal_types():
    conn = get_connection()
    df = pd.read_sql("SELECT DISTINCT signal_type FROM daily_signals ORDER BY signal_type", conn)
    conn.close()
    return df["signal_type"].tolist()


@st.cache_data(ttl=300)
def get_leaderboard(selected_date: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM daily_leaderboard WHERE prediction_date = ? ORDER BY daily_rank",
        conn,
        params=(selected_date,),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_signals_for_date(selected_date: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM daily_signals WHERE signal_date = ?",
        conn,
        params=(selected_date,),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_calibration(selected_date: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM calibration_summary WHERE summary_date = ?",
        conn,
        params=(selected_date,),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_yesterdays_results(selected_date: str, available_dates: list):
    """Get the most recent date before selected_date that has actuals."""
    idx = available_dates.index(selected_date) if selected_date in available_dates else -1
    if idx < 0 or idx + 1 >= len(available_dates):
        return pd.DataFrame()

    # Look through prior dates for one with actual results
    conn = get_connection()
    for prev_date in available_dates[idx + 1 : idx + 10]:
        df = pd.read_sql(
            "SELECT * FROM prediction_tracking "
            "WHERE prediction_date = ? AND actual_hits IS NOT NULL "
            "ORDER BY daily_rank",
            conn,
            params=(prev_date,),
        )
        if len(df) > 0:
            conn.close()
            return df
    conn.close()
    return pd.DataFrame()


@st.cache_data(ttl=300)
def get_teams_for_date(selected_date: str):
    """Get teams from batter_stats for players on the leaderboard."""
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT DISTINCT b.team
        FROM batter_stats b
        JOIN daily_leaderboard l ON b.batter_id = l.player_id
        WHERE l.prediction_date = ? AND b.team IS NOT NULL AND b.team <> ''
        ORDER BY b.team
        """,
        conn,
        params=(selected_date,),
    )
    conn.close()
    return df["team"].tolist()


# ── Score color helper ───────────────────────────────────────────────────────


def score_color(val):
    """Return background color CSS for a score value."""
    if pd.isna(val):
        return ""
    if val >= 80:
        return "background-color: rgba(0, 180, 0, 0.3)"
    elif val >= 60:
        return "background-color: rgba(220, 180, 0, 0.3)"
    else:
        return "background-color: rgba(200, 50, 50, 0.3)"


def style_score_col(df_styled, col_name="daily_score"):
    """Apply conditional color to a score column."""
    return df_styled.map(score_color, subset=[col_name])


# ── Sidebar controls ────────────────────────────────────────────────────────

st.sidebar.header("Leaderboard Controls")

available_dates = get_available_dates()
if not available_dates:
    st.error("No leaderboard data available.")
    st.stop()

selected_date = st.sidebar.selectbox("Date", available_dates, index=0)

signal_types = get_signal_types()
signal_filter = st.sidebar.selectbox(
    "Signal Type Filter", ["All"] + signal_types, index=0
)

teams = get_teams_for_date(selected_date)
team_filter = st.sidebar.selectbox("Team Filter", ["All"] + teams, index=0)

min_confidence = st.sidebar.slider(
    "Minimum Score Threshold", 0, 100, 0, step=5
)

top_n = st.sidebar.select_slider("Show Top N", options=[10, 25, 50], value=25)

# ── Load data ────────────────────────────────────────────────────────────────

lb = get_leaderboard(selected_date)
signals_df = get_signals_for_date(selected_date)
cal_df = get_calibration(selected_date)
yesterday_df = get_yesterdays_results(selected_date, available_dates)

if lb.empty:
    st.warning(f"No leaderboard data for {selected_date}.")
    st.stop()

# ── Apply filters ────────────────────────────────────────────────────────────

# Signal type filter: keep only players who have this signal type
if signal_filter != "All":
    player_ids_with_signal = signals_df.loc[
        signals_df["signal_type"] == signal_filter, "player_id"
    ].unique()
    lb = lb[lb["player_id"].isin(player_ids_with_signal)]

# Team filter — join from batter_stats if teams exist
if team_filter != "All":
    conn = get_connection()
    team_players = pd.read_sql(
        "SELECT DISTINCT batter_id FROM batter_stats WHERE team = ?",
        conn,
        params=(team_filter,),
    )
    conn.close()
    lb = lb[lb["player_id"].isin(team_players["batter_id"])]

# Confidence threshold
lb = lb[lb["daily_score"] >= min_confidence]

# Top N
lb = lb.head(top_n)

# ── Header ───────────────────────────────────────────────────────────────────

# The selected_date is the data date — predictions are for the NEXT day's games
from datetime import datetime, timedelta

data_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
prediction_target = data_date + timedelta(days=1)

st.title(f"Predictions for {prediction_target.strftime('%Y-%m-%d')}")
st.caption(f"Based on data through {selected_date}")
st.markdown("---")

# ── Summary metrics ──────────────────────────────────────────────────────────

mc1, mc2, mc3 = st.columns(3)

# Yesterday's top-10 accuracy
if not yesterday_df.empty:
    top10 = yesterday_df[yesterday_df["daily_rank"] <= 10]
    if len(top10) > 0:
        acc = top10["hit_1_correct"].mean() * 100
        mc1.metric("Prior Top-10 Accuracy (1+ Hit)", f"{acc:.1f}%")
    else:
        mc1.metric("Prior Top-10 Accuracy (1+ Hit)", "N/A")
else:
    mc1.metric("Prior Top-10 Accuracy (1+ Hit)", "N/A")

mc2.metric("Total Signals Fired", len(signals_df))

# BSS from calibration
bss_row = cal_df[(cal_df["metric_type"] == "bss_1hit") & (cal_df["window_days"] == 30)]
if not bss_row.empty:
    mc3.metric("Brier Skill Score (1-Hit, 30d)", f"{bss_row['value'].iloc[0]:.3f}")
else:
    mc3.metric("Brier Skill Score (1-Hit, 30d)", "N/A")

st.caption(
    "Top-10 Accuracy = % of top-10 ranked players who recorded 1+ hit. "
    "BSS > 0 means the model outperforms a naive baseline."
)

st.markdown("---")

# ── Main leaderboard table ───────────────────────────────────────────────────

st.subheader("Rankings")

# Join actuals from prediction_tracking for past dates
@st.cache_data(ttl=300)
def get_actuals_for_date(dt: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT player_id, actual_hits AS actual_h, actual_hr AS actual_hr_, actual_pa AS actual_pa "
        "FROM prediction_tracking "
        "WHERE prediction_date = ? AND actual_hits IS NOT NULL",
        conn,
        params=(dt,),
    )
    conn.close()
    return df

actuals_df = get_actuals_for_date(selected_date)
has_actuals = not actuals_df.empty

# Build display columns
base_cols = [
    "daily_rank",
    "player_name",
    "daily_score",
    "p_1hit",
    "p_2hit",
    "p_hr",
    "active_signal_count",
    "top_signal",
    "top_reason",
]

display_df = lb[base_cols].copy()

# Merge actuals if available
if has_actuals:
    display_df = display_df.merge(
        actuals_df,
        left_on=lb["player_id"].values,
        right_on="player_id",
        how="left",
    ).drop(columns=["key_0", "player_id"], errors="ignore")

display_col_names = [
    "Rank",
    "Player",
    "Score",
    "P(1+ Hit)",
    "P(2+ Hit)",
    "P(HR)",
    "# Signals",
    "Top Signal",
    "Top Reason",
]
if has_actuals:
    display_col_names += ["H", "HR", "PA"]

display_df.columns = display_col_names

# Format probabilities as percentages
for col in ["P(1+ Hit)", "P(2+ Hit)", "P(HR)"]:
    display_df[col] = display_df[col].apply(
        lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "—"
    )

display_df["Score"] = display_df["Score"].round(1)

# Format actuals as integers
if has_actuals:
    for col in ["H", "HR", "PA"]:
        display_df[col] = display_df[col].apply(
            lambda x: int(x) if pd.notna(x) else "—"
        )

# Style the dataframe
styled = display_df.style.map(
    score_color, subset=["Score"]
).format({"Score": "{:.1f}"})

st.dataframe(
    styled,
    use_container_width=True,
    hide_index=True,
    height=min(len(display_df) * 38 + 40, 900),
)

if has_actuals:
    st.caption("H / HR / PA columns show actual results for this date.")
else:
    st.caption("Actuals will appear after games are played and data is refreshed.")

st.markdown("---")

# ── Charts ───────────────────────────────────────────────────────────────────

chart_col1, chart_col2 = st.columns(2)

# Signal breakdown bar chart
with chart_col1:
    st.subheader("Signal Breakdown")
    if not signals_df.empty:
        sig_counts = (
            signals_df["signal_type"]
            .value_counts()
            .reset_index()
        )
        sig_counts.columns = ["Signal Type", "Count"]
        fig_sig = px.bar(
            sig_counts,
            x="Signal Type",
            y="Count",
            color="Count",
            color_continuous_scale=["#ff6b6b", "#ffd93d", "#6bcb77"],
            title="Signals Fired Today by Type",
        )
        fig_sig.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=400,
            margin=dict(t=40, b=40),
        )
        st.plotly_chart(fig_sig, use_container_width=True)
    else:
        st.info("No signals fired for this date.")

# Score distribution histogram
with chart_col2:
    st.subheader("Score Distribution")
    full_lb = get_leaderboard(selected_date)
    if not full_lb.empty:
        fig_hist = px.histogram(
            full_lb,
            x="daily_score",
            nbins=20,
            title="Distribution of Daily Scores",
            labels={"daily_score": "Daily Score"},
            color_discrete_sequence=["#4dabf7"],
        )
        fig_hist.update_layout(
            height=400,
            margin=dict(t=40, b=40),
            yaxis_title="Count",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("No score data available.")

st.markdown("---")

# ── Yesterday's Results ──────────────────────────────────────────────────────

with st.expander("Previous Day's Results (Prediction vs Actual)", expanded=False):
    if not yesterday_df.empty:
        prev_date = yesterday_df["prediction_date"].iloc[0]
        st.markdown(f"**Results from: {prev_date}**")

        top25 = yesterday_df.head(25).copy()

        # Summary
        if len(top25) > 0:
            acc_1hit = top25["hit_1_correct"].mean() * 100
            acc_2hit = top25["hit_2_correct"].mean() * 100
            r1, r2, r3 = st.columns(3)
            r1.metric("Top-25 Accuracy (1+ Hit)", f"{acc_1hit:.1f}%")
            r2.metric("Top-25 Accuracy (2+ Hit)", f"{acc_2hit:.1f}%")
            avg_hits = top25["actual_hits"].mean()
            r3.metric("Avg Actual Hits (Top 25)", f"{avg_hits:.2f}")

        results_display = top25[
            [
                "daily_rank",
                "player_name",
                "daily_score",
                "p_1hit",
                "actual_hits",
                "actual_hr",
                "hit_1_correct",
            ]
        ].copy()
        results_display.columns = [
            "Rank",
            "Player",
            "Predicted Score",
            "P(1+ Hit)",
            "Actual Hits",
            "Actual HR",
            "Correct (1+ Hit)",
        ]
        results_display["P(1+ Hit)"] = results_display["P(1+ Hit)"].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "—"
        )
        results_display["Correct (1+ Hit)"] = results_display["Correct (1+ Hit)"].map(
            {1: "Yes", 0: "No"}
        )
        results_display["Predicted Score"] = results_display["Predicted Score"].round(1)

        st.dataframe(results_display, use_container_width=True, hide_index=True)
    else:
        st.info(
            "No prior-day results with actuals available near the selected date."
        )
