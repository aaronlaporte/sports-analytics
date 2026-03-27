"""
MLB Player Insights Platform — Home / Overview
"""

import streamlit as st
import pandas as pd
import sqlite3
from datetime import date

st.set_page_config(
    page_title="MLB Insights",
    page_icon="baseball",
    layout="wide",
    initial_sidebar_state="expanded",
)

DB_PATH = "data/mlb.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


@st.cache_data(ttl=300)
def get_latest_date():
    conn = get_connection()
    row = pd.read_sql(
        "SELECT MAX(prediction_date) AS latest FROM daily_leaderboard", conn
    )
    conn.close()
    return row["latest"].iloc[0]


@st.cache_data(ttl=300)
def get_overview_stats(latest_date: str):
    conn = get_connection()

    total_players = pd.read_sql(
        "SELECT COUNT(DISTINCT player_id) AS n FROM daily_leaderboard "
        "WHERE prediction_date = ?",
        conn,
        params=(latest_date,),
    )["n"].iloc[0]

    total_signals = pd.read_sql(
        "SELECT COUNT(*) AS n FROM daily_signals WHERE signal_date = ?",
        conn,
        params=(latest_date,),
    )["n"].iloc[0]

    # Model accuracy — Brier Skill Score for 1-hit, 30-day window (latest available)
    bss = pd.read_sql(
        "SELECT value FROM calibration_summary "
        "WHERE metric_type = 'bss_1hit' AND window_days = 30 "
        "ORDER BY summary_date DESC LIMIT 1",
        conn,
    )
    bss_val = round(bss["value"].iloc[0], 3) if len(bss) > 0 else None

    # Yesterday's top-10 accuracy
    prev_dates = pd.read_sql(
        "SELECT DISTINCT prediction_date FROM prediction_tracking "
        "WHERE actual_hits IS NOT NULL "
        "ORDER BY prediction_date DESC LIMIT 1",
        conn,
    )
    top10_acc = None
    if len(prev_dates) > 0:
        prev = prev_dates["prediction_date"].iloc[0]
        acc = pd.read_sql(
            "SELECT AVG(hit_1_correct) AS acc FROM prediction_tracking "
            "WHERE prediction_date = ? AND daily_rank <= 10 AND actual_hits IS NOT NULL",
            conn,
            params=(prev,),
        )
        if len(acc) > 0 and acc["acc"].iloc[0] is not None:
            top10_acc = round(acc["acc"].iloc[0] * 100, 1)

    conn.close()
    return total_players, total_signals, bss_val, top10_acc


# ── Page Content ─────────────────────────────────────────────────────────────

st.title("MLB Player Insights Platform")
st.markdown("---")

latest_date = get_latest_date()

if latest_date is None:
    st.error("No data found in the database. Run the daily pipeline first.")
    st.stop()

col_date1, col_date2 = st.columns(2)
with col_date1:
    st.markdown(f"**Today:** {date.today().isoformat()}")
with col_date2:
    st.markdown(f"**Latest data:** {latest_date}")

st.markdown("")

total_players, total_signals, bss_val, top10_acc = get_overview_stats(latest_date)

# Quick stats row
m1, m2, m3, m4 = st.columns(4)
m1.metric("Players Tracked", total_players)
m2.metric("Signals Fired (Latest)", total_signals)
m3.metric("Brier Skill Score (1-Hit, 30d)", bss_val if bss_val is not None else "N/A")
m4.metric(
    "Last Top-10 Accuracy",
    f"{top10_acc}%" if top10_acc is not None else "N/A",
)

st.markdown("---")

st.subheader("What is this?")
st.markdown(
    """
The **MLB Player Insights Platform** combines statistical signals, regression
models, and real-time data to produce daily player rankings focused on
predicting batting performance (hits, multi-hit games, home runs).

Each day the pipeline:
1. Ingests the latest box scores and Statcast data.
2. Runs six signal detectors (BABIP regression, contact quality, hot/cold
   streaks, pitcher vulnerability, pitch-mix advantage).
3. Produces a composite score (0--100) for every qualifying batter.
4. Tracks predictions against actual outcomes for continuous calibration.
"""
)

st.markdown("---")

st.subheader("Navigation")
st.page_link("pages/1_Leaderboard.py", label="Leaderboard -- Daily ranked players", icon=":material/format_list_numbered:")
st.page_link("pages/2_Player_Page.py", label="Player Page -- Individual drill-down", icon=":material/person:")

st.markdown("---")
st.caption("MLB Player Insights Platform v0.1 | Data refreshed daily")
