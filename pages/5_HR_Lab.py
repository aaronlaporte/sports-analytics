"""
MLB Player Insights Platform — HR Lab

Dedicated home run analytics: today's picks, power profiles,
model accuracy tracking, and statcast deep dives.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import numpy as np
from datetime import datetime, timedelta

st.set_page_config(page_title="MLB Insights — HR Lab", layout="wide", page_icon="\u26be")

DB_PATH = "data/mlb.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


# ── Cached queries ──────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def get_hr_dates():
    """All dates with HR feature data."""
    conn = get_connection()
    df = pd.read_sql(
        "SELECT DISTINCT feature_date FROM hr_features ORDER BY feature_date DESC",
        conn,
    )
    conn.close()
    return df["feature_date"].tolist()


@st.cache_data(ttl=300)
def get_hr_leaderboard(date_str: str):
    """HR leaderboard for a date (from dedicated table if available, else features)."""
    conn = get_connection()
    # Try dedicated leaderboard first
    df = pd.read_sql(
        """SELECT h.*, p.player_name AS pname
           FROM hr_leaderboard h
           LEFT JOIN player_lookup p ON h.player_id = p.mlbam_id
           WHERE h.prediction_date = ?
           ORDER BY h.hr_rank""",
        conn,
        params=(date_str,),
    )
    if not df.empty:
        # Use leaderboard player_name if present, else fallback to lookup
        df["player_name"] = df["player_name"].fillna(df["pname"])
        df.drop(columns=["pname"], inplace=True, errors="ignore")
        conn.close()
        return df

    # Fallback: build from hr_features
    df = pd.read_sql(
        """SELECT h.*, p.player_name, b.team
           FROM hr_features h
           LEFT JOIN player_lookup p ON h.batter_id = p.mlbam_id
           LEFT JOIN (
               SELECT DISTINCT batter_id, team FROM batter_stats WHERE game_date = ?
           ) b ON h.batter_id = b.batter_id
           WHERE h.feature_date = ?
           ORDER BY h.p_hr DESC""",
        conn,
        params=(date_str, date_str),
    )
    conn.close()
    if not df.empty:
        df.insert(0, "hr_rank", range(1, len(df) + 1))
        df["hr_score"] = None
        df["player_id"] = df["batter_id"]
    return df


@st.cache_data(ttl=300)
def get_hr_actuals(date_str: str):
    """Get actual HR data from prediction_tracking for a date."""
    conn = get_connection()
    df = pd.read_sql(
        """SELECT player_id, actual_hr, actual_hits, actual_pa
           FROM prediction_tracking
           WHERE prediction_date = ? AND actual_hr IS NOT NULL""",
        conn,
        params=(date_str,),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_hr_watch_accuracy():
    """Daily HR Watch top-10 accuracy over time."""
    conn = get_connection()
    df = pd.read_sql(
        """SELECT summary_date, value, sample_size
           FROM calibration_summary
           WHERE metric_type = 'hr_watch_top10_accuracy' AND window_days = 1
           ORDER BY summary_date""",
        conn,
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_statcast_hrs(start_date: str, end_date: str):
    """All HR events from statcast in a date range."""
    conn = get_connection()
    df = pd.read_sql(
        """SELECT s.game_date, s.batter, s.launch_speed, s.launch_angle,
                  s.hit_distance_sc, s.home_team, s.away_team,
                  p.player_name
           FROM statcast_staging s
           LEFT JOIN player_lookup p ON s.batter = p.mlbam_id
           WHERE s.events = 'home_run'
             AND s.game_date BETWEEN ? AND ?
           ORDER BY s.game_date DESC, s.launch_speed DESC""",
        conn,
        params=(start_date, end_date),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_daily_hr_counts(start_date: str):
    """Daily HR totals across MLB."""
    conn = get_connection()
    df = pd.read_sql(
        """SELECT game_date, COUNT(*) as hr_count
           FROM statcast_staging
           WHERE events = 'home_run' AND game_date >= ?
           GROUP BY game_date ORDER BY game_date""",
        conn,
        params=(start_date,),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_hr_leaders_season(year: int, top_n: int = 25):
    """Season HR leaders from batter_stats."""
    conn = get_connection()
    season_start = f"{year}-01-01"
    season_end = f"{year}-12-31"
    df = pd.read_sql(
        """SELECT b.batter_id, p.player_name, b.team,
                  SUM(b.hr) as season_hr,
                  SUM(b.pa) as season_pa,
                  SUM(b.hits) as season_h,
                  COUNT(DISTINCT b.game_date) as games,
                  ROUND(CAST(SUM(b.hr) AS REAL) / NULLIF(SUM(b.pa), 0), 4) as hr_rate
           FROM batter_stats b
           LEFT JOIN player_lookup p ON b.batter_id = p.mlbam_id
           WHERE b.game_date BETWEEN ? AND ?
             AND b.pa > 0
           GROUP BY b.batter_id
           HAVING SUM(b.pa) >= 30
           ORDER BY season_hr DESC
           LIMIT ?""",
        conn,
        params=(season_start, season_end, top_n),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_hr_features_scatter(date_str: str):
    """HR features for scatter plots."""
    conn = get_connection()
    df = pd.read_sql(
        """SELECT h.batter_id, h.barrel_rate, h.avg_exit_velo, h.p_hr,
                  h.park_factor, h.pitcher_hr_vuln,
                  p.player_name, b.team
           FROM hr_features h
           LEFT JOIN player_lookup p ON h.batter_id = p.mlbam_id
           LEFT JOIN (
               SELECT DISTINCT batter_id, team FROM batter_stats WHERE game_date = ?
           ) b ON h.batter_id = b.batter_id
           WHERE h.feature_date = ?""",
        conn,
        params=(date_str, date_str),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_signal_effectiveness():
    """HR signal accuracy from calibration_summary and main prediction tracking."""
    conn = get_connection()
    # Use daily_signals to check which HR signals fire and whether players actually HR'd
    df = pd.read_sql(
        """SELECT ds.signal_type, ds.signal_date,
                  ds.player_id, ds.confidence,
                  pt.actual_hr
           FROM daily_signals ds
           JOIN prediction_tracking pt
             ON ds.player_id = pt.player_id AND ds.signal_date = pt.prediction_date
           WHERE ds.signal_type LIKE 'hr_%'
             AND pt.actual_hr IS NOT NULL""",
        conn,
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_monster_hrs(start_date: str, top_n: int = 20):
    """Hardest-hit HRs in date range."""
    conn = get_connection()
    df = pd.read_sql(
        """SELECT s.game_date, s.batter, s.launch_speed, s.launch_angle,
                  s.hit_distance_sc, s.home_team, s.away_team,
                  p.player_name
           FROM statcast_staging s
           LEFT JOIN player_lookup p ON s.batter = p.mlbam_id
           WHERE s.events = 'home_run' AND s.game_date >= ?
             AND s.launch_speed IS NOT NULL
           ORDER BY s.launch_speed DESC
           LIMIT ?""",
        conn,
        params=(start_date, top_n),
    )
    conn.close()
    return df


# ── Color helpers ────────────────────────────────────────────────────────────


def phr_bg(val):
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ""
    if val >= 0.20:
        return "background-color: rgba(255, 60, 60, 0.40); font-weight: bold"
    elif val >= 0.12:
        return "background-color: rgba(255, 140, 0, 0.35)"
    elif val >= 0.08:
        return "background-color: rgba(255, 200, 0, 0.30)"
    else:
        return ""


def score_bg(val):
    if pd.isna(val) or not isinstance(val, (int, float)):
        return ""
    if val >= 85:
        return "background-color: rgba(255, 60, 60, 0.40); font-weight: bold"
    elif val >= 70:
        return "background-color: rgba(255, 140, 0, 0.35)"
    elif val >= 55:
        return "background-color: rgba(255, 200, 0, 0.30)"
    else:
        return ""


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("HR Lab Controls")

hr_dates = get_hr_dates()
if not hr_dates:
    st.error("No HR data available. Run the pipeline first.")
    st.stop()

selected_date = st.sidebar.selectbox("Date", hr_dates, index=0)
data_date = datetime.strptime(selected_date, "%Y-%m-%d").date()
prediction_target = data_date + timedelta(days=1)


# ── Header ───────────────────────────────────────────────────────────────────

st.markdown(
    """
    <div style="text-align: center; padding: 0.5rem 0 1rem 0;">
        <h1 style="margin-bottom: 0.2rem;">HR Lab</h1>
        <p style="color: #888; font-size: 1.1rem; margin-top: 0;">
            Deep dive into home run analytics, predictions, and power metrics
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(f"**Predictions for {prediction_target}** | Data through {selected_date}")
st.markdown("---")


# ── KPI Strip ────────────────────────────────────────────────────────────────

# Daily HR count
daily_counts = get_daily_hr_counts((data_date - timedelta(days=30)).isoformat())
today_hrs = daily_counts[daily_counts["game_date"] == selected_date]
today_hr_count = int(today_hrs["hr_count"].iloc[0]) if not today_hrs.empty else 0
avg_daily_hrs = daily_counts["hr_count"].mean() if not daily_counts.empty else 0

# Watch accuracy
acc_df = get_hr_watch_accuracy()
latest_acc = acc_df[acc_df["summary_date"] == selected_date]
latest_acc_val = latest_acc["value"].iloc[0] if not latest_acc.empty else None

# Season HR leader
year = data_date.year
season_leaders = get_hr_leaders_season(year, top_n=1)
top_hr_name = season_leaders["player_name"].iloc[0] if not season_leaders.empty else "N/A"
top_hr_count = int(season_leaders["season_hr"].iloc[0]) if not season_leaders.empty else 0

k1, k2, k3, k4 = st.columns(4)
k1.metric(
    "HRs Hit (Data Date)",
    today_hr_count,
    delta=f"{today_hr_count - avg_daily_hrs:+.0f} vs avg" if avg_daily_hrs > 0 else None,
)
k2.metric(
    "HR Watch Top-10 Accuracy",
    f"{latest_acc_val:.0%}" if latest_acc_val is not None else "N/A",
)
k3.metric("30d Avg HRs / Day", f"{avg_daily_hrs:.1f}")
k4.metric("Season HR Leader", f"{top_hr_name}: {top_hr_count}")

st.markdown("---")


# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_picks, tab_power, tab_accuracy, tab_statcast = st.tabs([
    "Today's HR Picks",
    "Power Profiles",
    "Model Accuracy",
    "Statcast Lab",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1: Today's HR Picks
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_picks:
    lb = get_hr_leaderboard(selected_date)
    actuals = get_hr_actuals(selected_date)
    has_actuals = not actuals.empty

    if lb.empty:
        st.info("No HR leaderboard data for this date.")
    else:
        # Merge actuals
        if has_actuals and "player_id" in lb.columns:
            lb = lb.merge(actuals, on="player_id", how="left")

        # Top 3 spotlight cards
        st.subheader("Top Picks")
        top3 = lb.head(3)
        cols = st.columns(3)
        for i, (_, row) in enumerate(top3.iterrows()):
            with cols[i]:
                name = row.get("player_name", f"ID {row.get('player_id', '?')}")
                team = row.get("team", "--") or "--"
                phr = row.get("p_hr", 0)
                score = row.get("hr_score", 0) or 0
                barrel = row.get("barrel_rate", 0) or 0
                signal = row.get("top_hr_signal", "--") or "--"

                # Card color by rank
                card_colors = ["#ff4444", "#ff7700", "#ffaa00"]
                medal = ["\U0001f947", "\U0001f948", "\U0001f949"][i]

                actual_badge = ""
                if has_actuals and "actual_hr" in row.index and pd.notna(row["actual_hr"]):
                    ahr = int(row["actual_hr"])
                    if ahr >= 1:
                        actual_badge = f'<div style="background: #00c853; color: white; padding: 4px 8px; border-radius: 4px; text-align: center; font-weight: bold; margin-top: 6px;">{ahr} HR</div>'
                    else:
                        actual_badge = f'<div style="background: #666; color: white; padding: 4px 8px; border-radius: 4px; text-align: center; margin-top: 6px;">0 HR</div>'

                st.markdown(
                    f"""
                    <div style="border: 2px solid {card_colors[i]}; border-radius: 12px; padding: 16px; text-align: center;">
                        <div style="font-size: 2rem;">{medal}</div>
                        <div style="font-size: 1.3rem; font-weight: bold;">{name}</div>
                        <div style="color: #888;">{team}</div>
                        <div style="font-size: 1.8rem; font-weight: bold; color: {card_colors[i]}; margin: 8px 0;">{phr:.1%}</div>
                        <div style="color: #aaa; font-size: 0.85rem;">P(HR)</div>
                        <div style="margin-top: 8px; font-size: 0.9rem;">
                            Score: <b>{score:.0f}</b> | Barrel: <b>{barrel:.1%}</b>
                        </div>
                        <div style="margin-top: 4px; font-size: 0.85rem; color: #aaa;">{signal.replace('_', ' ').title()}</div>
                        {actual_badge}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("")

        # Full leaderboard table
        st.subheader("Full HR Leaderboard")

        top_n_picks = st.select_slider(
            "Show top N", options=[10, 15, 25, 50], value=15, key="hr_picks_n"
        )
        table_df = lb.head(top_n_picks).copy()

        # Build display
        disp_cols = ["hr_rank", "player_name"]
        col_labels = ["#", "Player"]

        if "team" in table_df.columns:
            table_df["team"] = table_df["team"].fillna("--")
            disp_cols.append("team")
            col_labels.append("Team")

        disp_cols += ["p_hr", "barrel_rate", "fly_ball_rate", "park_factor",
                      "active_hr_signal_count", "top_hr_signal"]
        col_labels += ["P(HR)", "Barrel", "FB%", "Park", "Sigs", "Top Signal"]

        if "hr_score" in table_df.columns and table_df["hr_score"].notna().any():
            disp_cols.insert(3, "hr_score")
            col_labels.insert(3, "Score")

        if has_actuals and "actual_hr" in table_df.columns:
            disp_cols += ["actual_hr", "actual_hits"]
            col_labels += ["HR", "H"]

        # Only keep columns that exist
        disp_cols = [c for c in disp_cols if c in table_df.columns]
        col_labels = col_labels[:len(disp_cols)]

        show_df = table_df[disp_cols].copy()
        show_df.columns = col_labels

        # Format
        if "P(HR)" in show_df.columns:
            show_df["_phr_raw"] = show_df["P(HR)"]
            show_df["P(HR)"] = show_df["P(HR)"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "--"
            )
        if "Barrel" in show_df.columns:
            show_df["Barrel"] = show_df["Barrel"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "--"
            )
        if "FB%" in show_df.columns:
            show_df["FB%"] = show_df["FB%"].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "--"
            )
        if "Park" in show_df.columns:
            show_df["Park"] = show_df["Park"].apply(
                lambda x: f"{x:.2f}" if pd.notna(x) else "--"
            )
        if "Score" in show_df.columns:
            show_df["Score"] = show_df["Score"].apply(
                lambda x: f"{x:.0f}" if pd.notna(x) else "--"
            )
        if "Top Signal" in show_df.columns:
            show_df["Top Signal"] = show_df["Top Signal"].apply(
                lambda x: x.replace("_", " ").title() if pd.notna(x) and x else "--"
            )
        if "Sigs" in show_df.columns:
            show_df["Sigs"] = show_df["Sigs"].fillna(0).astype(int)
        for c in ["HR", "H"]:
            if c in show_df.columns:
                show_df[c] = show_df[c].apply(lambda x: int(x) if pd.notna(x) else "--")

        show_df.drop(columns=["_phr_raw"], errors="ignore", inplace=True)

        st.dataframe(
            show_df,
            use_container_width=True,
            hide_index=True,
            height=min(len(show_df) * 38 + 40, 700),
        )

        if has_actuals:
            st.caption("HR / H columns show actual results.")
        else:
            st.caption("Actuals appear after games complete and data refreshes.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2: Power Profiles
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_power:
    scatter_df = get_hr_features_scatter(selected_date)

    if scatter_df.empty:
        st.info("No HR feature data for this date.")
    else:
        st.subheader("Exit Velocity vs Barrel Rate")
        st.caption("Each dot is a batter. Size = P(HR). Hover for details.")

        scatter_df["player_name"] = scatter_df["player_name"].fillna("Unknown")
        scatter_df["p_hr_pct"] = scatter_df["p_hr"] * 100

        fig_scatter = px.scatter(
            scatter_df,
            x="avg_exit_velo",
            y="barrel_rate",
            size="p_hr_pct",
            color="p_hr_pct",
            color_continuous_scale=["#2196f3", "#ff9800", "#f44336"],
            hover_name="player_name",
            hover_data={
                "avg_exit_velo": ":.1f",
                "barrel_rate": ":.3f",
                "p_hr_pct": ":.1f",
                "team": True,
                "park_factor": ":.2f",
            },
            labels={
                "avg_exit_velo": "Avg Exit Velo (mph)",
                "barrel_rate": "Barrel Rate",
                "p_hr_pct": "P(HR) %",
            },
            size_max=25,
        )
        fig_scatter.update_layout(
            height=500,
            coloraxis_colorbar_title="P(HR)%",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        # Add quadrant lines at league medians
        med_ev = scatter_df["avg_exit_velo"].median()
        med_br = scatter_df["barrel_rate"].median()
        fig_scatter.add_hline(y=med_br, line_dash="dot", line_color="#555", opacity=0.5)
        fig_scatter.add_vline(x=med_ev, line_dash="dot", line_color="#555", opacity=0.5)
        fig_scatter.add_annotation(
            x=scatter_df["avg_exit_velo"].max(), y=scatter_df["barrel_rate"].max(),
            text="DANGER ZONE", showarrow=False,
            font=dict(size=12, color="rgba(255,60,60,0.4)"),
            xanchor="right",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        # P(HR) distribution
        col_a, col_b = st.columns(2)

        with col_a:
            st.subheader("P(HR) Distribution")
            fig_dist = px.histogram(
                scatter_df,
                x="p_hr",
                nbins=30,
                color_discrete_sequence=["#f44336"],
                labels={"p_hr": "P(HR)"},
            )
            fig_dist.update_layout(
                height=350,
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_tickformat=".0%",
                yaxis_title="Count",
                showlegend=False,
            )
            # Add median line
            fig_dist.add_vline(
                x=scatter_df["p_hr"].median(), line_dash="dash",
                line_color="#ffaa00", opacity=0.7,
                annotation_text=f"Median: {scatter_df['p_hr'].median():.1%}",
                annotation_font_color="#ffaa00",
            )
            st.plotly_chart(fig_dist, use_container_width=True)

        with col_b:
            st.subheader("Park Factor Impact")
            pf_df = scatter_df.dropna(subset=["park_factor", "p_hr"])
            if not pf_df.empty:
                fig_park = px.scatter(
                    pf_df,
                    x="park_factor",
                    y="p_hr",
                    hover_name="player_name",
                    color="park_factor",
                    color_continuous_scale=["#2196f3", "#4caf50", "#f44336"],
                    labels={
                        "park_factor": "Park HR Factor",
                        "p_hr": "P(HR)",
                    },
                )
                fig_park.update_layout(
                    height=350,
                    template="plotly_dark",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                    yaxis_tickformat=".0%",
                    showlegend=False,
                )
                fig_park.add_hline(y=pf_df["p_hr"].median(), line_dash="dot", line_color="#555")
                fig_park.add_vline(x=1.0, line_dash="dot", line_color="#555",
                                   annotation_text="Neutral", annotation_font_color="#888")
                st.plotly_chart(fig_park, use_container_width=True)

        # Season HR leaderboard
        st.markdown("---")
        st.subheader(f"{year} Season HR Leaders")
        season_df = get_hr_leaders_season(year, top_n=20)
        if not season_df.empty:
            season_display = season_df.copy()
            season_display.insert(0, "#", range(1, len(season_display) + 1))
            season_display["hr_rate"] = season_display["hr_rate"].apply(
                lambda x: f"{x:.3f}" if pd.notna(x) else "--"
            )
            season_display = season_display.rename(columns={
                "player_name": "Player", "team": "Team", "season_hr": "HR",
                "season_pa": "PA", "games": "G", "hr_rate": "HR/PA",
            })
            season_display.drop(columns=["batter_id", "season_h"], inplace=True, errors="ignore")

            st.dataframe(
                season_display,
                use_container_width=True,
                hide_index=True,
                height=min(len(season_display) * 38 + 40, 700),
            )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3: Model Accuracy
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_accuracy:
    acc_df = get_hr_watch_accuracy()

    if acc_df.empty:
        st.info("No HR Watch accuracy data yet. Accumulates as predictions are scored.")
    else:
        st.subheader("HR Watch Top-10 Accuracy Over Time")
        st.caption("Each point: what % of our top-10 HR picks actually homered that day.")

        acc_df["value_pct"] = acc_df["value"] * 100
        acc_df["summary_date"] = pd.to_datetime(acc_df["summary_date"])

        # Rolling average
        acc_df["rolling_7d"] = acc_df["value_pct"].rolling(7, min_periods=3).mean()

        fig_acc = go.Figure()
        fig_acc.add_trace(go.Bar(
            x=acc_df["summary_date"],
            y=acc_df["value_pct"],
            name="Daily",
            marker_color=acc_df["value_pct"].apply(
                lambda v: "#00c853" if v >= 20 else ("#ff9800" if v >= 10 else "#f44336")
            ),
            opacity=0.6,
        ))
        fig_acc.add_trace(go.Scatter(
            x=acc_df["summary_date"],
            y=acc_df["rolling_7d"],
            name="7-Day Rolling Avg",
            line=dict(color="#ffaa00", width=3),
            mode="lines",
        ))

        # MLB baseline: ~4.5% of PA result in HR
        fig_acc.add_hline(
            y=10, line_dash="dash", line_color="#666",
            annotation_text="10% baseline", annotation_font_color="#888",
        )

        fig_acc.update_layout(
            height=400,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Top-10 Hit Rate (%)",
            xaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            bargap=0.3,
        )
        st.plotly_chart(fig_acc, use_container_width=True)

        # Summary metrics
        c1, c2, c3 = st.columns(3)
        overall_acc = acc_df["value_pct"].mean()
        best_day = acc_df.loc[acc_df["value_pct"].idxmax()]
        hit_days = (acc_df["value_pct"] > 0).sum()
        total_days = len(acc_df)

        c1.metric("Overall Avg Accuracy", f"{overall_acc:.1f}%")
        c2.metric("Best Day", f"{best_day['value_pct']:.0f}% ({best_day['summary_date'].strftime('%m/%d')})")
        c3.metric("Days with 1+ Hit", f"{hit_days}/{total_days} ({hit_days/total_days*100:.0f}%)")

    # P(HR) vs actual outcomes — calibration check
    st.markdown("---")
    st.subheader("P(HR) Calibration Check")
    st.caption("Are our P(HR) estimates well-calibrated? Binned predicted vs actual HR rates.")

    conn = get_connection()
    cal_df = pd.read_sql(
        """SELECT p_hr, actual_hr
           FROM prediction_tracking
           WHERE actual_hr IS NOT NULL AND p_hr IS NOT NULL""",
        conn,
    )
    conn.close()

    if len(cal_df) > 100:
        cal_df["hr_binary"] = (cal_df["actual_hr"] >= 1).astype(int)
        cal_df["phr_bin"] = pd.cut(cal_df["p_hr"], bins=10)
        binned = cal_df.groupby("phr_bin", observed=True).agg(
            predicted=("p_hr", "mean"),
            actual=("hr_binary", "mean"),
            count=("hr_binary", "count"),
        ).reset_index()
        binned = binned[binned["count"] >= 5]

        if not binned.empty:
            fig_cal = go.Figure()
            # Perfect calibration line
            fig_cal.add_trace(go.Scatter(
                x=[0, binned["predicted"].max() * 1.1],
                y=[0, binned["predicted"].max() * 1.1],
                mode="lines",
                name="Perfect",
                line=dict(color="#555", dash="dash"),
            ))
            fig_cal.add_trace(go.Scatter(
                x=binned["predicted"],
                y=binned["actual"],
                mode="markers+lines",
                name="Model",
                marker=dict(
                    size=binned["count"].apply(lambda x: min(max(x / 10, 6), 25)),
                    color="#f44336",
                ),
                line=dict(color="#f44336", width=2),
                text=binned["count"].apply(lambda x: f"n={x}"),
                hovertemplate="Predicted: %{x:.1%}<br>Actual: %{y:.1%}<br>%{text}",
            ))
            fig_cal.update_layout(
                height=400,
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Predicted P(HR)",
                yaxis_title="Actual HR Rate",
                xaxis_tickformat=".0%",
                yaxis_tickformat=".0%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_cal, use_container_width=True)
    else:
        st.info("Need 100+ scored predictions for calibration analysis.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4: Statcast Lab
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

with tab_statcast:
    lookback = st.select_slider(
        "Lookback window", options=[7, 14, 30, 60, 90], value=30, key="sc_lookback"
    )
    start = (data_date - timedelta(days=lookback)).isoformat()
    end = selected_date

    # Daily HR volume chart
    st.subheader("MLB HR Volume")
    daily_df = get_daily_hr_counts(start)
    if not daily_df.empty:
        daily_df["game_date"] = pd.to_datetime(daily_df["game_date"])
        daily_df["rolling_7d"] = daily_df["hr_count"].rolling(7, min_periods=3).mean()

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Bar(
            x=daily_df["game_date"],
            y=daily_df["hr_count"],
            name="Daily HRs",
            marker_color="#f44336",
            opacity=0.5,
        ))
        fig_vol.add_trace(go.Scatter(
            x=daily_df["game_date"],
            y=daily_df["rolling_7d"],
            name="7-Day Avg",
            line=dict(color="#ffaa00", width=3),
        ))
        fig_vol.update_layout(
            height=350,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Home Runs",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            bargap=0.3,
        )
        st.plotly_chart(fig_vol, use_container_width=True)

    # Monster dingers
    st.markdown("---")
    st.subheader("Hardest-Hit Home Runs")
    st.caption(f"Top 20 by exit velocity in the last {lookback} days")

    monster_df = get_monster_hrs(start, top_n=20)
    if not monster_df.empty:
        monster_display = monster_df.copy()
        monster_display.insert(0, "#", range(1, len(monster_display) + 1))
        monster_display["player_name"] = monster_display["player_name"].fillna("Unknown")
        monster_display["launch_speed"] = monster_display["launch_speed"].apply(
            lambda x: f"{x:.1f}" if pd.notna(x) else "--"
        )
        monster_display["launch_angle"] = monster_display["launch_angle"].apply(
            lambda x: f"{x}" if pd.notna(x) else "--"
        )
        monster_display["hit_distance_sc"] = monster_display["hit_distance_sc"].apply(
            lambda x: f"{int(x)}" if pd.notna(x) else "--"
        )
        monster_display["matchup"] = monster_display["away_team"] + " @ " + monster_display["home_team"]

        show_monster = monster_display[["#", "game_date", "player_name", "launch_speed",
                                        "launch_angle", "hit_distance_sc", "matchup"]].copy()
        show_monster.columns = ["#", "Date", "Player", "Exit Velo", "Launch Angle", "Distance", "Game"]

        st.dataframe(show_monster, use_container_width=True, hide_index=True)

    # Launch speed vs distance scatter
    st.markdown("---")
    st.subheader("HR Launch Profiles")
    hr_sc = get_statcast_hrs(start, end)
    if not hr_sc.empty and "launch_speed" in hr_sc.columns and "launch_angle" in hr_sc.columns:
        hr_sc = hr_sc.dropna(subset=["launch_speed", "launch_angle"])
        hr_sc["player_name"] = hr_sc["player_name"].fillna("Unknown")

        col1, col2 = st.columns(2)

        with col1:
            fig_la = px.scatter(
                hr_sc,
                x="launch_angle",
                y="launch_speed",
                color="hit_distance_sc",
                color_continuous_scale=["#2196f3", "#4caf50", "#f44336"],
                hover_name="player_name",
                hover_data={
                    "game_date": True,
                    "launch_speed": ":.1f",
                    "launch_angle": True,
                    "hit_distance_sc": True,
                },
                labels={
                    "launch_angle": "Launch Angle",
                    "launch_speed": "Exit Velo (mph)",
                    "hit_distance_sc": "Distance (ft)",
                },
                title="Exit Velo vs Launch Angle",
            )
            fig_la.update_layout(
                height=400,
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_la, use_container_width=True)

        with col2:
            # Launch angle distribution
            fig_angle = px.histogram(
                hr_sc,
                x="launch_angle",
                nbins=25,
                color_discrete_sequence=["#f44336"],
                title="HR Launch Angle Distribution",
                labels={"launch_angle": "Launch Angle"},
            )
            fig_angle.update_layout(
                height=400,
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                yaxis_title="Count",
                showlegend=False,
            )
            # Sweet spot annotation
            fig_angle.add_vrect(
                x0=25, x1=35,
                fillcolor="#ff9800", opacity=0.15,
                annotation_text="Sweet Spot", annotation_position="top",
                annotation_font_color="#ff9800",
            )
            st.plotly_chart(fig_angle, use_container_width=True)

        # Exit velo distribution
        st.subheader("Exit Velocity Distribution (HRs Only)")
        fig_ev = px.histogram(
            hr_sc,
            x="launch_speed",
            nbins=30,
            color_discrete_sequence=["#2196f3"],
            labels={"launch_speed": "Exit Velo (mph)"},
        )
        fig_ev.update_layout(
            height=300,
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Count",
            showlegend=False,
        )
        mean_ev = hr_sc["launch_speed"].mean()
        fig_ev.add_vline(
            x=mean_ev, line_dash="dash", line_color="#ffaa00",
            annotation_text=f"Mean: {mean_ev:.1f} mph",
            annotation_font_color="#ffaa00",
        )
        st.plotly_chart(fig_ev, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "HR Lab | Data from Statcast + MLB Insights Pipeline | "
    "Predictions refresh daily at 3 AM PT"
)
