"""
MLB Player Insights Platform — Daily Leaderboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta

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


@st.cache_data(ttl=300)
def get_hr_features(selected_date: str):
    """Get HR features for the selected date."""
    conn = get_connection()
    try:
        df = pd.read_sql(
            """
            SELECT h.*, p.player_name, b.team
            FROM hr_features h
            LEFT JOIN player_lookup p ON h.batter_id = p.mlbam_id
            LEFT JOIN (
                SELECT DISTINCT batter_id, team FROM batter_stats
                WHERE game_date = ?
            ) b ON h.batter_id = b.batter_id
            WHERE h.feature_date = ?
            ORDER BY h.p_hr DESC
            """,
            conn,
            params=(selected_date, selected_date),
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


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


@st.cache_data(ttl=300)
def get_opp_pitcher_names(selected_date: str):
    """Get opposing pitcher names for HR Watch display."""
    conn = get_connection()
    try:
        df = pd.read_sql(
            """
            SELECT l.player_id, l.opp_pitcher
            FROM daily_leaderboard l
            WHERE l.prediction_date = ? AND l.opp_pitcher IS NOT NULL
            """,
            conn,
            params=(selected_date,),
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


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


def phr_color(val):
    """Return background color CSS for a P(HR) percentage value."""
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ""
    if val >= 20:
        return "background-color: rgba(0, 180, 0, 0.3)"
    elif val >= 12:
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
hr_features_df = get_hr_features(selected_date)

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

# ── Tabs: Hit Predictions | HR Watch ─────────────────────────────────────────

tab_hits, tab_hr = st.tabs(["Hit Predictions", "HR Watch"])

# ── Hit Predictions Tab ──────────────────────────────────────────────────────

with tab_hits:
    st.subheader("Rankings")

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
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "--"
        )

    display_df["Score"] = display_df["Score"].round(1)

    # Format actuals as integers
    if has_actuals:
        for col in ["H", "HR", "PA"]:
            display_df[col] = display_df[col].apply(
                lambda x: int(x) if pd.notna(x) else "--"
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

    # ── Charts ───────────────────────────────────────────────────────────────

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

# ── HR Watch Tab ─────────────────────────────────────────────────────────────

with tab_hr:
    st.subheader("HR Watch")

    with st.popover("?"):
        st.markdown(
            "**HR Watch** ranks batters by their probability of hitting a home run today, "
            "based on power metrics, park factors, and pitcher matchups.\n\n"
            "**Column Definitions:**\n"
            "- **P(HR)**: Probability of hitting at least one home run in today's game\n"
            "- **Barrel Rate**: % of batted balls hit at optimal speed (98+ mph) and angle (26-30 deg). "
            "Barrels become HRs about 70% of the time. League avg ~6.5%\n"
            "- **Avg Exit Velo**: How hard the batter hits the ball on average (mph). League avg ~88 mph\n"
            "- **Park Factor**: How the ballpark affects HR rates. 1.0 = neutral, >1.0 = HR-friendly\n"
            "- **Pitcher HR Vuln**: Opposing pitcher's HR rate vs league average. >1.0 = more HR-prone"
        )

    if hr_features_df.empty:
        st.info("No HR feature data available for this date. Run the pipeline to generate HR features.")
    else:
        hr_display = hr_features_df.head(25).copy()

        # Add rank column
        hr_display.insert(0, "rank", range(1, len(hr_display) + 1))

        # Get actuals for this date
        actuals_df_hr = get_actuals_for_date(selected_date)

        # Merge actuals
        if not actuals_df_hr.empty:
            hr_display = hr_display.merge(
                actuals_df_hr,
                left_on="batter_id",
                right_on="player_id",
                how="left",
            ).drop(columns=["player_id"], errors="ignore")

        # Get opp pitcher info
        opp_df = get_opp_pitcher_names(selected_date)
        if not opp_df.empty:
            hr_display = hr_display.merge(
                opp_df,
                left_on="batter_id",
                right_on="player_id",
                how="left",
            ).drop(columns=["player_id"], errors="ignore")

        # Build display columns
        display_cols = ["rank", "player_name", "p_hr", "barrel_rate", "avg_exit_velo",
                        "park_factor", "pitcher_hr_vuln"]
        if "opp_pitcher" in hr_display.columns:
            display_cols.append("opp_pitcher")

        col_names = ["Rank", "Player", "P(HR)", "Barrel Rate", "Avg Exit Velo",
                     "Park Factor", "Pitcher HR Vuln"]
        if "opp_pitcher" in hr_display.columns:
            col_names.append("Opp Pitcher")

        has_hr_actuals = "actual_hr_" in hr_display.columns
        if has_hr_actuals:
            display_cols += ["actual_h", "actual_hr_", "actual_pa"]
            col_names += ["H", "HR", "PA"]

        hr_table = hr_display[display_cols].copy()
        hr_table.columns = col_names

        # Format values
        hr_table["P(HR) val"] = hr_table["P(HR)"] * 100  # Keep numeric for coloring
        hr_table["P(HR)"] = hr_table["P(HR)"].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "--"
        )
        hr_table["Barrel Rate"] = hr_table["Barrel Rate"].apply(
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "--"
        )
        hr_table["Avg Exit Velo"] = hr_table["Avg Exit Velo"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "--"
        )
        hr_table["Park Factor"] = hr_table["Park Factor"].apply(
            lambda x: f"{x:.2f}" if pd.notna(x) else "--"
        )
        hr_table["Pitcher HR Vuln"] = hr_table["Pitcher HR Vuln"].apply(
            lambda x: f"{x:.2f}x" if pd.notna(x) else "--"
        )

        if has_hr_actuals:
            for col in ["H", "HR", "PA"]:
                hr_table[col] = hr_table[col].apply(
                    lambda x: int(x) if pd.notna(x) else "--"
                )

        # Remove the numeric helper column before display
        hr_table_display = hr_table.drop(columns=["P(HR) val"], errors="ignore")

        st.dataframe(
            hr_table_display,
            use_container_width=True,
            hide_index=True,
            height=min(len(hr_table_display) * 38 + 40, 900),
        )

        if has_hr_actuals:
            st.caption("H / HR / PA columns show actual results for this date.")

        # HR distribution chart
        st.markdown("---")
        st.subheader("P(HR) Distribution")

        fig_hr = px.histogram(
            hr_features_df,
            x="p_hr",
            nbins=25,
            title="Distribution of P(HR) Across All Batters",
            labels={"p_hr": "P(HR)"},
            color_discrete_sequence=["#ff6b6b"],
        )
        fig_hr.update_layout(
            height=350,
            margin=dict(t=40, b=40),
            yaxis_title="Count",
            xaxis_tickformat=".0%",
        )
        st.plotly_chart(fig_hr, use_container_width=True)

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
            lambda x: f"{x * 100:.1f}%" if pd.notna(x) else "--"
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
