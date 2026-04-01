"""
MLB Player Insights Platform — Player Page (Drill-Down)
"""

import json
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
from datetime import datetime, timedelta

st.set_page_config(page_title="MLB Insights — Player", layout="wide")

DB_PATH = "data/mlb.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ── Cached queries ───────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def get_all_players():
    """Return all players from lookup table, sorted by name."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT mlbam_id, player_name, current_team FROM player_lookup ORDER BY player_name", conn
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_player_page_data(player_id: int, date_start: str, date_end: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM player_pages "
        "WHERE player_id = ? AND page_date BETWEEN ? AND ? "
        "ORDER BY page_date DESC",
        conn,
        params=(player_id, date_start, date_end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_batter_stats(player_id: int, date_start: str, date_end: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM batter_stats "
        "WHERE batter_id = ? AND game_date BETWEEN ? AND ? "
        "ORDER BY game_date",
        conn,
        params=(player_id, date_start, date_end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_player_signals(player_id: int, date_start: str, date_end: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM daily_signals "
        "WHERE player_id = ? AND signal_date BETWEEN ? AND ? "
        "ORDER BY signal_date DESC",
        conn,
        params=(player_id, date_start, date_end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_player_predictions(player_id: int, date_start: str, date_end: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM prediction_tracking "
        "WHERE player_id = ? AND prediction_date BETWEEN ? AND ? "
        "ORDER BY prediction_date",
        conn,
        params=(player_id, date_start, date_end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_player_leaderboard_history(player_id: int, date_start: str, date_end: str):
    conn = get_connection()
    df = pd.read_sql(
        "SELECT * FROM daily_leaderboard "
        "WHERE player_id = ? AND prediction_date BETWEEN ? AND ? "
        "ORDER BY prediction_date",
        conn,
        params=(player_id, date_start, date_end),
    )
    conn.close()
    return df


# ── Score color helper ───────────────────────────────────────────────────────


def score_color_css(val):
    if pd.isna(val):
        return ""
    if val >= 80:
        return "background-color: rgba(0, 180, 0, 0.3)"
    elif val >= 60:
        return "background-color: rgba(220, 180, 0, 0.3)"
    else:
        return "background-color: rgba(200, 50, 50, 0.3)"


def streak_indicator(streak: int):
    """Return a visual streak indicator."""
    if streak >= 5:
        return f"🔥 {streak}-game hitting streak"
    elif streak > 0:
        return f"✅ {streak}-game hitting streak"
    elif streak == 0:
        return "— No active streak"
    else:
        return f"❄️ {abs(streak)}-game hitless streak"


# ── Sidebar controls ────────────────────────────────────────────────────────

st.sidebar.header("Player Selection")

players_df = get_all_players()
if players_df.empty:
    st.error("No players found in the database.")
    st.stop()

# Build name -> id mapping
player_names = players_df["player_name"].tolist()
player_id_map = dict(zip(players_df["player_name"], players_df["mlbam_id"]))

selected_player = st.sidebar.selectbox(
    "Search Player",
    player_names,
    index=0,
    placeholder="Type to search...",
)

st.sidebar.markdown("---")
st.sidebar.subheader("Date Range")

# Default date range: last 90 days from most recent data across all tables
conn_tmp = get_connection()
max_date_row = pd.read_sql("""
    SELECT MAX(d) AS md FROM (
        SELECT MAX(page_date) AS d FROM player_pages
        UNION ALL
        SELECT MAX(prediction_date) FROM daily_leaderboard
        UNION ALL
        SELECT MAX(game_date) FROM batter_stats
    )
""", conn_tmp)
conn_tmp.close()

if max_date_row["md"].iloc[0]:
    end_date = datetime.strptime(max_date_row["md"].iloc[0], "%Y-%m-%d").date()
else:
    end_date = datetime.today().date()
start_date = end_date - timedelta(days=90)

date_range = st.sidebar.date_input(
    "Date range",
    value=(start_date, end_date),
    max_value=end_date,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d_start, d_end = date_range[0].isoformat(), date_range[1].isoformat()
else:
    d_start = date_range[0].isoformat() if isinstance(date_range, tuple) else date_range.isoformat()
    d_end = end_date.isoformat()

# ── Load data ────────────────────────────────────────────────────────────────

player_id = player_id_map[selected_player]
page_data = get_player_page_data(player_id, d_start, d_end)
batter_data = get_batter_stats(player_id, d_start, d_end)
signals_data = get_player_signals(player_id, d_start, d_end)
predictions = get_player_predictions(player_id, d_start, d_end)
lb_history = get_player_leaderboard_history(player_id, d_start, d_end)

# ── Player header ────────────────────────────────────────────────────────────

# Get team from player_lookup
player_team = None
if "current_team" in players_df.columns:
    team_row = players_df[players_df["mlbam_id"] == player_id]
    if not team_row.empty and pd.notna(team_row.iloc[0].get("current_team")):
        player_team = team_row.iloc[0]["current_team"]

if player_team:
    st.title(f"{selected_player.title()} — {player_team}")
else:
    st.title(selected_player.title())

# Always use batter_stats as source of truth for current form stats (most up to date).
# Use page_data/lb_history only for daily score and rank.
if not batter_data.empty:
    latest_bs = batter_data.iloc[-1]

    # Get latest score/rank from page_data or lb_history
    score_val = "—"
    rank_delta = None
    if not page_data.empty:
        latest_pg = page_data.iloc[0]
        if pd.notna(latest_pg.get("daily_score")):
            score_val = f"{latest_pg['daily_score']:.1f}"
        if pd.notna(latest_pg.get("daily_rank")):
            rank_delta = f"Rank #{int(latest_pg['daily_rank'])}"
    elif not lb_history.empty:
        latest_lb = lb_history.iloc[-1]
        if pd.notna(latest_lb.get("daily_score")):
            score_val = f"{latest_lb['daily_score']:.1f}"
        if pd.notna(latest_lb.get("daily_rank")):
            rank_delta = f"Rank #{int(latest_lb['daily_rank'])}"

    hdr1, hdr2, hdr3, hdr4 = st.columns(4)
    hdr1.metric("Team", latest_bs.get("team", "—") or "—")
    hdr2.metric(
        "Current Streak",
        f"{int(latest_bs['current_streak'])} games" if pd.notna(latest_bs.get("current_streak")) else "—",
    )
    hdr3.metric(
        "Season AVG",
        f"{latest_bs['season_hit_pct']:.3f}" if pd.notna(latest_bs.get("season_hit_pct")) else "—",
    )
    hdr4.metric("Daily Score", score_val, delta=rank_delta)

    st.markdown("")
    streak_val = int(latest_bs["current_streak"]) if pd.notna(latest_bs.get("current_streak")) else 0
    st.info(streak_indicator(streak_val))
else:
    st.warning(f"No data found for {selected_player.title()} in the selected date range.")
    st.stop()

st.markdown("---")

# ── Two-column layout ───────────────────────────────────────────────────────

left_col, right_col = st.columns(2)

# ── Left column: Current Form ────────────────────────────────────────────────

with left_col:
    st.subheader("Current Form")

    if not batter_data.empty:
        latest_bs = batter_data.iloc[-1]

        # Season stats
        stat1, stat2, stat3, stat4 = st.columns(4)
        stat1.metric(
            "AVG",
            f"{latest_bs['avg']:.3f}" if pd.notna(latest_bs.get("avg")) else "—",
        )
        stat2.metric(
            "OBP",
            f"{latest_bs['obp']:.3f}" if pd.notna(latest_bs.get("obp")) else "—",
        )
        stat3.metric(
            "SLG",
            f"{latest_bs['slg']:.3f}" if pd.notna(latest_bs.get("slg")) else "—",
        )
        stat4.metric(
            "Hit%",
            f"{latest_bs['season_hit_pct']:.1%}" if pd.notna(latest_bs.get("season_hit_pct")) else "—",
        )

        # Rolling hits line chart
        rolling_cols = {
            "hits_pg_1": "Last 1",
            "hits_pg_3": "Last 3",
            "hits_pg_5": "Last 5",
            "hits_pg_10": "Last 10",
            "hits_pg_20": "Last 20",
        }
        available_rolling = [c for c in rolling_cols if c in batter_data.columns]

        if available_rolling:
            rolling_df = batter_data[["game_date"] + available_rolling].copy()
            rolling_df = rolling_df.melt(
                id_vars="game_date",
                value_vars=available_rolling,
                var_name="Window",
                value_name="Hits/Game",
            )
            rolling_df["Window"] = rolling_df["Window"].map(rolling_cols)

            fig_rolling = px.line(
                rolling_df,
                x="game_date",
                y="Hits/Game",
                color="Window",
                title="Rolling Hits Per Game",
                labels={"game_date": "Date"},
            )
            fig_rolling.update_layout(
                height=350,
                margin=dict(t=40, b=40),
                legend=dict(orientation="h", y=-0.2),
            )
            st.plotly_chart(fig_rolling, use_container_width=True)
        else:
            st.info("No rolling hit data available.")
    else:
        st.info("No batter stats found for the selected date range.")

# ── Right column: Active Signals ─────────────────────────────────────────────

with right_col:
    st.subheader("Active Signals")

    # Show signals from the most recent date in the range
    if not page_data.empty:
        latest_page = page_data.iloc[0]
        active_json = latest_page.get("active_signals_json")
        if active_json and active_json != "[]":
            try:
                active_signals = json.loads(active_json)
            except (json.JSONDecodeError, TypeError):
                active_signals = []

            if active_signals:
                st.caption(f"Signals as of {latest_page['page_date']}")
                for sig in active_signals:
                    sig_type = sig.get("signal_type", "Unknown")
                    confidence = sig.get("confidence", 0)
                    headline = sig.get("headline", "")
                    interpretation = sig.get("interpretation", "")
                    reasons = sig.get("reasons", [])

                    # Confidence bar color
                    if confidence >= 0.8:
                        bar_color = "green"
                    elif confidence >= 0.6:
                        bar_color = "orange"
                    else:
                        bar_color = "red"

                    st.markdown(f"**{sig_type.replace('_', ' ').title()}** — Confidence: {confidence:.0%}")
                    st.progress(confidence)
                    st.markdown(f"*{headline}*")

                    if reasons:
                        with st.expander("Reasons"):
                            for r in reasons:
                                st.markdown(f"- {r}")

                    if interpretation:
                        st.caption(interpretation)

                    st.markdown("---")
            else:
                st.info("No active signals for this player on the latest date.")
        else:
            st.info("No active signals for this player on the latest date.")
    elif not signals_data.empty:
        # Fallback: show most recent signals from daily_signals table
        latest_sig_date = signals_data["signal_date"].iloc[0]
        latest_sigs = signals_data[signals_data["signal_date"] == latest_sig_date]
        st.caption(f"Signals as of {latest_sig_date}")

        for _, sig in latest_sigs.iterrows():
            confidence = sig["confidence"]
            st.markdown(
                f"**{sig['signal_type'].replace('_', ' ').title()}** — "
                f"Confidence: {confidence:.0%}"
            )
            st.progress(confidence)
            st.markdown(f"*{sig['headline']}*")

            if sig["reasons_json"]:
                try:
                    reasons = json.loads(sig["reasons_json"])
                    with st.expander("Reasons"):
                        for r in reasons:
                            st.markdown(f"- {r}")
                except (json.JSONDecodeError, TypeError):
                    pass

            if sig["interpretation"]:
                st.caption(sig["interpretation"])

            st.markdown("---")
    else:
        st.info("No signals found for this player in the selected date range.")

# ── Power Profile ────────────────────────────────────────────────────────────

st.markdown("---")
st.subheader("Power Profile")


@st.cache_data(ttl=300)
def get_hr_features_for_player(pid: int, date_start: str, date_end: str):
    conn = get_connection()
    try:
        df = pd.read_sql(
            "SELECT * FROM hr_features "
            "WHERE batter_id = ? AND feature_date BETWEEN ? AND ? "
            "ORDER BY feature_date DESC LIMIT 1",
            conn,
            params=(pid, date_start, date_end),
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_barrel_rate_history(pid: int, end_date: str):
    """Compute barrel rate and exit velo history from statcast_staging."""
    conn = get_connection()
    try:
        # Load raw BBE data
        df = pd.read_sql(
            """
            SELECT game_date, launch_speed, launch_angle
            FROM statcast_staging
            WHERE batter = ?
              AND launch_speed IS NOT NULL
              AND launch_speed > 0
              AND game_date <= ?
            ORDER BY game_date, at_bat_number
            """,
            conn,
            params=(pid, end_date),
        )
    except Exception:
        df = pd.DataFrame()
    conn.close()

    if df.empty:
        return pd.DataFrame()

    # Compute rolling barrel rate
    BARREL_MIN_SPEED = 98.0
    BARREL_BASE_ANGLE_MIN = 26.0
    BARREL_BASE_ANGLE_MAX = 30.0
    BARREL_ANGLE_MAX_CAP = 50.0
    ROLLING_WINDOW = 50

    def is_barrel(speed, angle):
        if pd.isna(speed) or pd.isna(angle) or speed < BARREL_MIN_SPEED:
            return False
        extra = speed - BARREL_MIN_SPEED
        return BARREL_BASE_ANGLE_MIN <= angle <= min(BARREL_BASE_ANGLE_MAX + extra, BARREL_ANGLE_MAX_CAP)

    df["is_barrel"] = df.apply(lambda r: is_barrel(r["launch_speed"], r["launch_angle"]), axis=1)

    # Group by game date and compute rolling metrics
    game_groups = df.groupby("game_date")
    all_bbes = []
    results = []

    for date in sorted(df["game_date"].unique()):
        game_df = game_groups.get_group(date)
        for _, row in game_df.iterrows():
            all_bbes.append({
                "speed": row["launch_speed"],
                "barrel": row["is_barrel"],
            })

        window = all_bbes[-ROLLING_WINDOW:]
        if len(window) >= 10:
            br = sum(1 for b in window if b["barrel"]) / len(window)
            avg_velo = sum(b["speed"] for b in window) / len(window)
            results.append({
                "game_date": date,
                "barrel_rate": br,
                "avg_exit_velo": avg_velo,
            })

    return pd.DataFrame(results)


hr_feat_df = get_hr_features_for_player(player_id, d_start, d_end)
barrel_history = get_barrel_rate_history(player_id, d_end)

if not barrel_history.empty or not hr_feat_df.empty:
    power_col1, power_col2 = st.columns(2)

    # a) Barrel Rate Trend
    with power_col1:
        header_col, info_col = st.columns([8, 1])
        with header_col:
            st.markdown("**Barrel Rate Trend**")
        with info_col:
            with st.popover("?"):
                st.markdown(
                    "**Barrel Rate** measures the percentage of batted balls hit at optimal "
                    "speed (98+ mph) and angle (26-30 deg). Barrels become HRs about 70% of "
                    "the time. League average is ~6.5%."
                )

        if not barrel_history.empty:
            fig_br = go.Figure()
            fig_br.add_trace(go.Scatter(
                x=barrel_history["game_date"],
                y=barrel_history["barrel_rate"],
                mode="lines",
                name="Barrel Rate",
                line=dict(color="#ff6b6b", width=2),
            ))
            fig_br.add_hline(
                y=0.065,
                line_dash="dash",
                line_color="gray",
                annotation_text="League Avg (6.5%)",
                annotation_position="top right",
            )
            fig_br.update_layout(
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis_title="Barrel Rate",
                yaxis_tickformat=".1%",
                xaxis_title="Date",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_br, use_container_width=True)
        else:
            st.info("Not enough batted ball data for barrel rate chart.")

    # b) Exit Velocity Trend
    with power_col2:
        header_col, info_col = st.columns([8, 1])
        with header_col:
            st.markdown("**Exit Velocity Trend**")
        with info_col:
            with st.popover("?"):
                st.markdown(
                    "**Average Exit Velocity** measures how hard a batter hits the ball. "
                    "Higher exit velocity correlates strongly with home runs and extra-base "
                    "hits. League average is ~88 mph."
                )

        if not barrel_history.empty:
            fig_ev = go.Figure()
            fig_ev.add_trace(go.Scatter(
                x=barrel_history["game_date"],
                y=barrel_history["avg_exit_velo"],
                mode="lines",
                name="Avg Exit Velo",
                line=dict(color="#4dabf7", width=2),
            ))
            fig_ev.add_hline(
                y=88.0,
                line_dash="dash",
                line_color="gray",
                annotation_text="League Avg (88 mph)",
                annotation_position="top right",
            )
            fig_ev.update_layout(
                height=350,
                margin=dict(t=20, b=40, l=40, r=20),
                yaxis_title="Avg Exit Velo (mph)",
                xaxis_title="Date",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_ev, use_container_width=True)
        else:
            st.info("Not enough batted ball data for exit velocity chart.")

    # c) Power Profile vs League
    if not hr_feat_df.empty:
        latest_hr = hr_feat_df.iloc[0]

        profile_col, park_col = st.columns([3, 1])

        with profile_col:
            header_col, info_col = st.columns([8, 1])
            with header_col:
                st.markdown("**Power Profile vs League**")
            with info_col:
                with st.popover("?"):
                    st.markdown(
                        "This chart compares the batter's power metrics to league averages. "
                        "Values above 1.0x mean the batter is above average in that category."
                    )

            # Compute ratios vs league
            metrics = {
                "Barrel Rate": (latest_hr.get("barrel_rate", 0) or 0) / 0.065 if 0.065 > 0 else 1.0,
                "Exit Velo": (latest_hr.get("avg_exit_velo", 88) or 88) / 88.0,
                "HR/PA Rate": (latest_hr.get("hr_pa_rate", 0.035) or 0.035) / 0.035 if 0.035 > 0 else 1.0,
            }

            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())

            colors = []
            for v in metric_values:
                if v >= 1.3:
                    colors.append("#6bcb77")
                elif v >= 1.0:
                    colors.append("#4dabf7")
                else:
                    colors.append("#ff6b6b")

            fig_profile = go.Figure()
            fig_profile.add_trace(go.Bar(
                y=metric_names,
                x=metric_values,
                orientation="h",
                marker_color=colors,
                text=[f"{v:.2f}x" for v in metric_values],
                textposition="outside",
            ))
            fig_profile.add_vline(
                x=1.0,
                line_dash="dash",
                line_color="gray",
                annotation_text="League Avg",
            )
            fig_profile.update_layout(
                height=200,
                margin=dict(t=10, b=20, l=80, r=40),
                xaxis_title="vs League Average",
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_profile, use_container_width=True)

        # d) Park Factor indicator
        with park_col:
            header_col, info_col = st.columns([3, 1])
            with header_col:
                st.markdown("**Park Factor**")
            with info_col:
                with st.popover("?"):
                    st.markdown(
                        "**Park Factor** measures how a ballpark affects home run rates. "
                        "1.0 is neutral. Values above 1.0 mean the park boosts HRs "
                        "(e.g., Coors Field ~1.3). Below 1.0 means the park suppresses "
                        "HRs (e.g., Oracle Park ~0.7)."
                    )

            pf_val = latest_hr.get("park_factor", None)
            if pf_val and pd.notna(pf_val):
                pf_val = float(pf_val)
                if pf_val >= 1.1:
                    pf_label = "HR-Friendly"
                elif pf_val >= 0.95:
                    pf_label = "Neutral"
                else:
                    pf_label = "Pitcher-Friendly"

                st.metric(
                    "Today's Park",
                    f"{pf_val:.2f}",
                    delta=pf_label,
                    delta_color="normal" if pf_val >= 1.0 else "inverse",
                )
            else:
                st.metric("Today's Park", "--")

            p_hr_val = latest_hr.get("p_hr", None)
            if p_hr_val and pd.notna(p_hr_val):
                st.metric("P(HR)", f"{float(p_hr_val) * 100:.1f}%")
            else:
                st.metric("P(HR)", "--")
    else:
        st.info("No HR feature data available for this player. Run the pipeline to generate HR features.")
else:
    st.info("No power profile data available for this player in the selected date range.")

# ── Full width: Prediction History ───────────────────────────────────────────

st.markdown("---")
st.subheader("Prediction History")

if not lb_history.empty:
    # Build combined chart: daily score line + actual hits overlay
    fig_pred = make_subplots(specs=[[{"secondary_y": True}]])

    # Daily score line
    fig_pred.add_trace(
        go.Scatter(
            x=lb_history["prediction_date"],
            y=lb_history["daily_score"],
            name="Daily Score",
            mode="lines+markers",
            line=dict(color="#4dabf7", width=2),
            marker=dict(size=6),
        ),
        secondary_y=False,
    )

    # Overlay actual hits from predictions if available
    if not predictions.empty:
        pred_with_actual = predictions[predictions["actual_hits"].notna()].copy()
        if not pred_with_actual.empty:
            fig_pred.add_trace(
                go.Bar(
                    x=pred_with_actual["prediction_date"],
                    y=pred_with_actual["actual_hits"],
                    name="Actual Hits",
                    marker_color=[
                        "#6bcb77" if c == 1 else "#ff6b6b"
                        for c in pred_with_actual["hit_1_correct"]
                    ],
                    opacity=0.6,
                ),
                secondary_y=True,
            )

    fig_pred.update_layout(
        title="Daily Score Over Time with Actual Performance",
        height=400,
        margin=dict(t=50, b=40),
        legend=dict(orientation="h", y=-0.15),
        hovermode="x unified",
    )
    fig_pred.update_yaxes(title_text="Daily Score (0-100)", secondary_y=False)
    fig_pred.update_yaxes(title_text="Actual Hits", secondary_y=True)
    fig_pred.update_xaxes(title_text="Date")

    st.plotly_chart(fig_pred, use_container_width=True)
    st.caption(
        "Green bars = correctly predicted 1+ hit. Red bars = prediction missed. "
        "Blue line = composite daily score."
    )
else:
    st.info("No leaderboard history found for this player in the selected range.")

# ── Signal History ───────────────────────────────────────────────────────────

st.markdown("---")

with st.expander("Signal History", expanded=False):
    if not signals_data.empty:
        sig_display = signals_data[
            ["signal_date", "signal_type", "confidence", "headline"]
        ].copy()
        sig_display.columns = ["Date", "Signal Type", "Confidence", "Headline"]
        sig_display["Confidence"] = sig_display["Confidence"].apply(
            lambda x: f"{x:.0%}" if pd.notna(x) else "—"
        )
        sig_display["Signal Type"] = sig_display["Signal Type"].str.replace(
            "_", " "
        ).str.title()
        st.dataframe(sig_display, use_container_width=True, hide_index=True)
    else:
        st.info("No signal history for this player in the selected range.")

# ── Accuracy Summary ─────────────────────────────────────────────────────────

# ── Game Log ────────────────────────────────────────────────────────────────

st.subheader("Game Log")

if not batter_data.empty:
    game_log = batter_data[
        [c for c in ["game_date", "pa", "ab", "hits", "hr", "bb", "so", "avg", "obp", "slg"]
         if c in batter_data.columns]
    ].copy()

    game_log = game_log.sort_values("game_date", ascending=False)

    col_rename = {
        "game_date": "Date",
        "pa": "PA",
        "ab": "AB",
        "hits": "H",
        "hr": "HR",
        "bb": "BB",
        "so": "K",
        "avg": "AVG",
        "obp": "OBP",
        "slg": "SLG",
    }
    game_log = game_log.rename(columns=col_rename)

    # Format decimal columns
    for col in ["AVG", "OBP", "SLG"]:
        if col in game_log.columns:
            game_log[col] = game_log[col].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "—"
            )

    st.dataframe(
        game_log,
        use_container_width=True,
        hide_index=True,
        height=min(len(game_log) * 38 + 40, 600),
    )
else:
    st.info("No game log data available for the selected date range.")

st.markdown("---")

# ── Accuracy Summary ─────────────────────────────────────────────────────────

with st.expander("Accuracy Summary", expanded=False):
    if not predictions.empty:
        pred_with_act = predictions[predictions["actual_hits"].notna()].copy()

        if not pred_with_act.empty:
            total_preds = len(pred_with_act)

            # Overall accuracy
            overall_acc = pred_with_act["hit_1_correct"].mean() * 100
            st.metric("Overall 1+ Hit Accuracy", f"{overall_acc:.1f}%", delta=f"n={total_preds}")

            # Accuracy by rank tier
            st.markdown("**Accuracy by Rank Tier**")
            tiers = [
                ("Top 10", pred_with_act[pred_with_act["daily_rank"] <= 10]),
                ("Top 25", pred_with_act[pred_with_act["daily_rank"] <= 25]),
                ("Top 50", pred_with_act[pred_with_act["daily_rank"] <= 50]),
            ]
            tier_data = []
            for label, subset in tiers:
                if len(subset) > 0:
                    tier_data.append(
                        {
                            "Tier": label,
                            "Predictions": len(subset),
                            "1+ Hit Accuracy": f"{subset['hit_1_correct'].mean() * 100:.1f}%",
                            "2+ Hit Accuracy": f"{subset['hit_2_correct'].mean() * 100:.1f}%",
                            "Avg Actual Hits": f"{subset['actual_hits'].mean():.2f}",
                        }
                    )
            if tier_data:
                st.dataframe(
                    pd.DataFrame(tier_data),
                    use_container_width=True,
                    hide_index=True,
                )

            # Prediction accuracy over time (if enough data)
            if len(pred_with_act) >= 5:
                pred_with_act = pred_with_act.sort_values("prediction_date")
                pred_with_act["rolling_acc"] = (
                    pred_with_act["hit_1_correct"]
                    .rolling(window=min(5, len(pred_with_act)), min_periods=1)
                    .mean()
                    * 100
                )
                fig_acc = px.line(
                    pred_with_act,
                    x="prediction_date",
                    y="rolling_acc",
                    title="Rolling 5-Game Prediction Accuracy",
                    labels={
                        "prediction_date": "Date",
                        "rolling_acc": "Accuracy %",
                    },
                    color_discrete_sequence=["#4dabf7"],
                )
                fig_acc.add_hline(
                    y=50,
                    line_dash="dash",
                    line_color="gray",
                    annotation_text="50% baseline",
                )
                fig_acc.update_layout(height=300, margin=dict(t=40, b=40))
                st.plotly_chart(fig_acc, use_container_width=True)
        else:
            st.info("No predictions with actual results available for this player.")
    else:
        st.info("No prediction history for this player in the selected range.")
