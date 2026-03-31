"""
MLB Player Insights Platform — Model Trends & Calibration
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sqlite3
from datetime import datetime, timedelta

st.set_page_config(page_title="MLB Insights — Trends", layout="wide")

DB_PATH = "data/mlb.db"

DARK_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=50, b=40, l=50, r=20),
    font=dict(size=12),
)


def get_connection():
    return sqlite3.connect(DB_PATH)


# ── Cached queries ───────────────────────────────────────────────────────────


@st.cache_data(ttl=300)
def get_date_range():
    conn = get_connection()
    df = pd.read_sql(
        "SELECT MIN(prediction_date) AS mn, MAX(prediction_date) AS mx "
        "FROM prediction_tracking WHERE actual_hits IS NOT NULL",
        conn,
    )
    conn.close()
    if df.empty or df["mn"].iloc[0] is None:
        return None, None
    return df["mn"].iloc[0], df["mx"].iloc[0]


@st.cache_data(ttl=300)
def get_daily_accuracy(date_start: str, date_end: str):
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT prediction_date, daily_rank,
               hit_1_correct, hit_2_correct, hr_correct,
               p_1hit, p_2hit, p_hr,
               actual_hits, actual_hr
        FROM prediction_tracking
        WHERE actual_hits IS NOT NULL
          AND prediction_date BETWEEN ? AND ?
        ORDER BY prediction_date
        """,
        conn,
        params=(date_start, date_end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_bss_trends(date_start: str, date_end: str, window_days: int):
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT summary_date, metric_type, value, sample_size
        FROM calibration_summary
        WHERE metric_type IN ('bss_1hit', 'bss_2hit', 'bss_hr',
                              'brier_1hit', 'brier_2hit', 'brier_hr')
          AND window_days = ?
          AND summary_date BETWEEN ? AND ?
        ORDER BY summary_date
        """,
        conn,
        params=(window_days, date_start, date_end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_signal_accuracy(date_start: str, date_end: str):
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT ds.signal_date, ds.signal_type, ds.player_id,
               pt.hit_1_correct, pt.hit_2_correct, pt.hr_correct
        FROM daily_signals ds
        JOIN prediction_tracking pt
            ON ds.player_id = pt.player_id
            AND ds.signal_date = pt.prediction_date
        WHERE pt.actual_hits IS NOT NULL
          AND ds.signal_date BETWEEN ? AND ?
        """,
        conn,
        params=(date_start, date_end),
    )
    conn.close()
    return df


@st.cache_data(ttl=300)
def get_hr_watch_trend(date_start: str, date_end: str, window_days: int):
    conn = get_connection()
    df = pd.read_sql(
        """
        SELECT summary_date, value, sample_size
        FROM calibration_summary
        WHERE metric_type = 'hr_watch_top10_accuracy'
          AND window_days = ?
          AND summary_date BETWEEN ? AND ?
        ORDER BY summary_date
        """,
        conn,
        params=(window_days, date_start, date_end),
    )
    conn.close()
    return df


# ── Helpers ──────────────────────────────────────────────────────────────────


def compute_weekly_accuracy(df: pd.DataFrame, max_rank: int) -> pd.DataFrame:
    """Aggregate daily accuracy by week for a rank tier."""
    filtered = df[df["daily_rank"] <= max_rank].copy()
    if filtered.empty:
        return pd.DataFrame()
    filtered["week"] = pd.to_datetime(filtered["prediction_date"]).dt.to_period("W").dt.start_time
    weekly = (
        filtered.groupby("week")
        .agg(
            hit_1_acc=("hit_1_correct", "mean"),
            hit_2_acc=("hit_2_correct", "mean"),
            hr_acc=("hr_correct", "mean"),
            sample_size=("hit_1_correct", "count"),
        )
        .reset_index()
    )
    weekly["hit_1_pct"] = (weekly["hit_1_acc"] * 100).round(1)
    weekly["hit_2_pct"] = (weekly["hit_2_acc"] * 100).round(1)
    weekly["hr_pct"] = (weekly["hr_acc"] * 100).round(1)
    return weekly


def compute_daily_accuracy(df: pd.DataFrame, max_rank: int) -> pd.DataFrame:
    """Aggregate accuracy by day for a rank tier."""
    filtered = df[df["daily_rank"] <= max_rank].copy()
    if filtered.empty:
        return pd.DataFrame()
    daily = (
        filtered.groupby("prediction_date")
        .agg(
            hit_1_acc=("hit_1_correct", "mean"),
            hit_2_acc=("hit_2_correct", "mean"),
            hr_acc=("hr_correct", "mean"),
            avg_p1hit=("p_1hit", "mean"),
            avg_actual=("hit_1_correct", "mean"),
            sample_size=("hit_1_correct", "count"),
        )
        .reset_index()
    )
    daily["hit_1_pct"] = (daily["hit_1_acc"] * 100).round(1)
    daily["hit_2_pct"] = (daily["hit_2_acc"] * 100).round(1)
    daily["date"] = pd.to_datetime(daily["prediction_date"])
    return daily


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Trend Controls")

min_date, max_date = get_date_range()
if min_date is None:
    st.error("No scored predictions available yet. Run the pipeline and check back.")
    st.stop()

min_dt = datetime.strptime(min_date, "%Y-%m-%d").date()
max_dt = datetime.strptime(max_date, "%Y-%m-%d").date()

date_range = st.sidebar.date_input(
    "Date Range",
    value=(min_dt, max_dt),
    min_value=min_dt,
    max_value=max_dt,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d_start = date_range[0].strftime("%Y-%m-%d")
    d_end = date_range[1].strftime("%Y-%m-%d")
else:
    d_start = min_date
    d_end = max_date

window_days = st.sidebar.selectbox("BSS Rolling Window", [7, 14, 30, 90], index=2)

rank_tier = st.sidebar.selectbox("Rank Tier", ["Top 10", "Top 25", "Top 50", "All"], index=0)
max_rank = {"Top 10": 10, "Top 25": 25, "Top 50": 50, "All": 9999}[rank_tier]

# ── Load data ────────────────────────────────────────────────────────────────

accuracy_df = get_daily_accuracy(d_start, d_end)
bss_df = get_bss_trends(d_start, d_end, window_days)
signal_df = get_signal_accuracy(d_start, d_end)
hr_watch_df = get_hr_watch_trend(d_start, d_end, window_days)

# ── Header ───────────────────────────────────────────────────────────────────

st.title("Model Trends & Calibration")
st.caption(f"Tracking prediction accuracy from {d_start} to {d_end}")
st.markdown("---")

# KPI row
daily_acc = compute_daily_accuracy(accuracy_df, max_rank)

if not daily_acc.empty:
    latest_acc = daily_acc.iloc[-1]
    prev_acc = daily_acc.iloc[-2] if len(daily_acc) > 1 else None

    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        f"{rank_tier} Hit Accuracy",
        f"{latest_acc['hit_1_pct']:.1f}%",
        delta=f"{latest_acc['hit_1_pct'] - prev_acc['hit_1_pct']:.1f}pp" if prev_acc is not None else None,
    )
    k2.metric(
        f"{rank_tier} Multi-Hit",
        f"{latest_acc['hit_2_pct']:.1f}%",
        delta=f"{latest_acc['hit_2_pct'] - prev_acc['hit_2_pct']:.1f}pp" if prev_acc is not None else None,
    )

    # Latest BSS
    bss_1hit = bss_df[bss_df["metric_type"] == "bss_1hit"]
    if not bss_1hit.empty:
        latest_bss = bss_1hit.iloc[-1]["value"]
        prev_bss = bss_1hit.iloc[-2]["value"] if len(bss_1hit) > 1 else None
        k3.metric(
            f"BSS 1-Hit ({window_days}d)",
            f"{latest_bss:.4f}",
            delta=f"{latest_bss - prev_bss:.4f}" if prev_bss is not None else None,
        )
    else:
        k3.metric(f"BSS 1-Hit ({window_days}d)", "N/A")

    k4.metric("Total Predictions Scored", f"{len(accuracy_df):,}")
else:
    st.info("Not enough scored predictions to display trends.")
    st.stop()

st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────────────────

tab_acc, tab_bss, tab_cal, tab_sig, tab_hr = st.tabs(
    ["Accuracy Trends", "BSS Trends", "Calibration Curve", "Signal Performance", "HR Watch"]
)

# ── Tab 1: Accuracy Trends ───────────────────────────────────────────────────

with tab_acc:
    st.subheader(f"{rank_tier} Accuracy Over Time")

    with st.popover("?"):
        st.markdown(
            "**Hit Accuracy** = % of players in the selected rank tier who recorded "
            "at least 1 hit. **Multi-Hit** = at least 2 hits.\n\n"
            "The dashed gray line shows the approximate MLB league average (~65% of "
            "qualified batters get 1+ hit per game). The model should consistently "
            "beat this baseline for the top-ranked players."
        )

    col_daily, col_weekly = st.columns(2)

    with col_daily:
        st.markdown("**Daily**")
        if not daily_acc.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily_acc["date"], y=daily_acc["hit_1_pct"],
                mode="lines+markers", name="1+ Hit %",
                line=dict(color="#4dabf7", width=2),
                marker=dict(size=6),
            ))
            fig.add_trace(go.Scatter(
                x=daily_acc["date"], y=daily_acc["hit_2_pct"],
                mode="lines+markers", name="2+ Hit %",
                line=dict(color="#6bcb77", width=2),
                marker=dict(size=6),
            ))
            fig.add_hline(y=65, line_dash="dash", line_color="gray",
                          annotation_text="League avg (~65%)", annotation_position="bottom right")
            fig.update_layout(
                **DARK_LAYOUT,
                height=400,
                yaxis_title="Accuracy %",
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data for the selected range.")

    with col_weekly:
        st.markdown("**Weekly Averages**")
        weekly_acc = compute_weekly_accuracy(accuracy_df, max_rank)
        if not weekly_acc.empty:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=weekly_acc["week"], y=weekly_acc["hit_1_pct"],
                name="1+ Hit %", marker_color="#4dabf7",
                text=weekly_acc["hit_1_pct"].apply(lambda x: f"{x:.0f}%"),
                textposition="outside",
            ))
            fig.add_trace(go.Bar(
                x=weekly_acc["week"], y=weekly_acc["hit_2_pct"],
                name="2+ Hit %", marker_color="#6bcb77",
            ))
            fig.add_hline(y=65, line_dash="dash", line_color="gray",
                          annotation_text="League avg")
            fig.update_layout(
                **DARK_LAYOUT,
                height=400,
                yaxis_title="Accuracy %",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough data for weekly view.")

    # Tier comparison
    st.markdown("---")
    st.subheader("Accuracy by Rank Tier")

    tier_data = []
    for tier_name, tier_max in [("Top 10", 10), ("Top 25", 25), ("Top 50", 50)]:
        tier_daily = compute_daily_accuracy(accuracy_df, tier_max)
        if not tier_daily.empty:
            tier_data.append({
                "Tier": tier_name,
                "1+ Hit %": round(tier_daily["hit_1_pct"].mean(), 1),
                "2+ Hit %": round(tier_daily["hit_2_pct"].mean(), 1),
                "HR %": round(tier_daily["hr_pct"].mean() if "hr_pct" in tier_daily.columns else 0, 1),
                "Predictions": len(accuracy_df[accuracy_df["daily_rank"] <= tier_max]),
            })

    if tier_data:
        tier_df = pd.DataFrame(tier_data)
        st.dataframe(tier_df, use_container_width=True, hide_index=True)
        st.caption(
            "Higher-ranked tiers should show higher accuracy — this validates "
            "the ranking model is ordering players correctly."
        )

# ── Tab 2: BSS Trends ───────────────────────────────────────────────────────

with tab_bss:
    st.subheader(f"Brier Skill Score — {window_days}-Day Rolling Window")

    with st.popover("?"):
        st.markdown(
            "**Brier Skill Score (BSS)** measures how much better the model is "
            "vs always predicting the league-average probability.\n\n"
            "- **BSS > 0**: Model adds value (beats naive baseline)\n"
            "- **BSS = 0**: No better than guessing the base rate\n"
            "- **BSS < 0**: Worse than baseline\n\n"
            "As the model calibrates with more data, BSS should trend upward. "
            "Even small positive values (0.01-0.05) indicate meaningful skill."
        )

    bss_only = bss_df[bss_df["metric_type"].str.startswith("bss_")]
    if not bss_only.empty:
        bss_pivot = bss_only.pivot_table(
            index="summary_date", columns="metric_type", values="value"
        ).reset_index()
        bss_pivot["date"] = pd.to_datetime(bss_pivot["summary_date"])

        fig = go.Figure()

        color_map = {"bss_1hit": "#4dabf7", "bss_2hit": "#6bcb77", "bss_hr": "#ff6b6b"}
        label_map = {"bss_1hit": "1+ Hit", "bss_2hit": "2+ Hit", "bss_hr": "HR"}

        for col in ["bss_1hit", "bss_2hit", "bss_hr"]:
            if col in bss_pivot.columns:
                fig.add_trace(go.Scatter(
                    x=bss_pivot["date"], y=bss_pivot[col],
                    mode="lines+markers", name=label_map[col],
                    line=dict(color=color_map[col], width=2),
                    marker=dict(size=5),
                ))

        fig.add_hline(y=0, line_dash="dash", line_color="gray",
                      annotation_text="No skill (baseline)",
                      annotation_position="bottom right")
        fig.update_layout(
            **DARK_LAYOUT,
            height=450,
            yaxis_title="Brier Skill Score",
            xaxis_title="Date",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No BSS data available for the selected range and window.")

    # Raw Brier scores in expander
    with st.expander("Raw Brier Scores (lower is better)"):
        brier_only = bss_df[bss_df["metric_type"].str.startswith("brier_")]
        if not brier_only.empty:
            brier_pivot = brier_only.pivot_table(
                index="summary_date", columns="metric_type", values="value"
            ).reset_index()
            brier_pivot["date"] = pd.to_datetime(brier_pivot["summary_date"])

            fig = go.Figure()
            brier_color = {"brier_1hit": "#4dabf7", "brier_2hit": "#6bcb77", "brier_hr": "#ff6b6b"}
            brier_label = {"brier_1hit": "1+ Hit", "brier_2hit": "2+ Hit", "brier_hr": "HR"}

            for col in ["brier_1hit", "brier_2hit", "brier_hr"]:
                if col in brier_pivot.columns:
                    fig.add_trace(go.Scatter(
                        x=brier_pivot["date"], y=brier_pivot[col],
                        mode="lines+markers", name=brier_label[col],
                        line=dict(color=brier_color[col], width=2),
                        marker=dict(size=5),
                    ))

            fig.update_layout(
                **DARK_LAYOUT,
                height=350,
                yaxis_title="Brier Score",
                xaxis_title="Date",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.caption("Brier Score = mean squared error of probability predictions. Lower = better.")
        else:
            st.info("No raw Brier data available.")

# ── Tab 3: Calibration Curve ────────────────────────────────────────────────

with tab_cal:
    st.subheader("Calibration Curve — Predicted vs Actual")

    with st.popover("?"):
        st.markdown(
            "A **calibration curve** shows whether predicted probabilities match "
            "actual outcomes. Points on the diagonal (dashed line) mean the model "
            "is perfectly calibrated.\n\n"
            "- Points **above** the diagonal: model underestimates (actual > predicted)\n"
            "- Points **below** the diagonal: model overestimates (actual < predicted)\n\n"
            "As the season progresses and the feedback loop tunes the model, "
            "the dots should converge toward the diagonal."
        )

    cal_col1, cal_col2 = st.columns(2)

    with cal_col1:
        st.markdown("**1+ Hit Calibration**")
        if not accuracy_df.empty:
            cal_data = accuracy_df[["p_1hit", "hit_1_correct"]].dropna().copy()
            if len(cal_data) > 20:
                cal_data["bin"] = pd.cut(cal_data["p_1hit"], bins=10, duplicates="drop")
                cal_agg = cal_data.groupby("bin", observed=True).agg(
                    pred_mean=("p_1hit", "mean"),
                    actual_rate=("hit_1_correct", "mean"),
                    count=("hit_1_correct", "count"),
                ).reset_index()
                cal_agg = cal_agg[cal_agg["count"] >= 5]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=cal_agg["pred_mean"] * 100,
                    y=cal_agg["actual_rate"] * 100,
                    mode="markers+text",
                    marker=dict(
                        size=cal_agg["count"].clip(upper=100) / 2 + 8,
                        color="#4dabf7",
                    ),
                    text=cal_agg["count"].apply(lambda x: f"n={x}"),
                    textposition="top center",
                    textfont=dict(size=10),
                    name="Bins",
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 100], y=[0, 100],
                    mode="lines", line=dict(dash="dash", color="gray"),
                    name="Perfect calibration",
                ))
                fig.update_layout(
                    **DARK_LAYOUT,
                    height=400,
                    xaxis_title="Predicted P(1+ Hit) %",
                    yaxis_title="Actual Hit Rate %",
                    showlegend=False,
                    xaxis=dict(range=[0, 100]),
                    yaxis=dict(range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need more data for calibration curve (min 20 scored predictions).")
        else:
            st.info("No data available.")

    with cal_col2:
        st.markdown("**HR Calibration**")
        if not accuracy_df.empty:
            hr_cal = accuracy_df[["p_hr", "hr_correct"]].dropna().copy()
            if len(hr_cal) > 20:
                hr_cal["bin"] = pd.cut(hr_cal["p_hr"], bins=8, duplicates="drop")
                hr_agg = hr_cal.groupby("bin", observed=True).agg(
                    pred_mean=("p_hr", "mean"),
                    actual_rate=("hr_correct", "mean"),
                    count=("hr_correct", "count"),
                ).reset_index()
                hr_agg = hr_agg[hr_agg["count"] >= 5]

                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hr_agg["pred_mean"] * 100,
                    y=hr_agg["actual_rate"] * 100,
                    mode="markers+text",
                    marker=dict(
                        size=hr_agg["count"].clip(upper=100) / 2 + 8,
                        color="#ff6b6b",
                    ),
                    text=hr_agg["count"].apply(lambda x: f"n={x}"),
                    textposition="top center",
                    textfont=dict(size=10),
                    name="Bins",
                ))
                fig.add_trace(go.Scatter(
                    x=[0, 50], y=[0, 50],
                    mode="lines", line=dict(dash="dash", color="gray"),
                    name="Perfect calibration",
                ))
                fig.update_layout(
                    **DARK_LAYOUT,
                    height=400,
                    xaxis_title="Predicted P(HR) %",
                    yaxis_title="Actual HR Rate %",
                    showlegend=False,
                    xaxis=dict(range=[0, 50]),
                    yaxis=dict(range=[0, 50]),
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Need more data for HR calibration curve.")
        else:
            st.info("No data available.")

    # Predicted vs Actual averages over time
    st.markdown("---")
    st.subheader("Predicted vs Actual — Daily Averages")

    if not daily_acc.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_acc["date"],
            y=(daily_acc["avg_p1hit"] if "avg_p1hit" in daily_acc.columns
               else daily_acc["hit_1_acc"]),
            mode="lines+markers", name="Avg Predicted P(1+ Hit)",
            line=dict(color="#4dabf7", width=2, dash="dot"),
            marker=dict(size=5),
        ))
        fig.add_trace(go.Scatter(
            x=daily_acc["date"],
            y=daily_acc["hit_1_acc"],
            mode="lines+markers", name="Actual Hit Rate",
            line=dict(color="#6bcb77", width=2),
            marker=dict(size=5),
        ))
        fig.update_layout(
            **DARK_LAYOUT,
            height=350,
            yaxis_title="Rate",
            yaxis_tickformat=".0%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "The predicted line (dotted) and actual line (solid) should converge "
            "over time as the model calibrates. Persistent gaps indicate systematic "
            "over- or under-confidence."
        )

# ── Tab 4: Signal Performance ───────────────────────────────────────────────

with tab_sig:
    st.subheader("Signal-Level Performance")

    with st.popover("?"):
        st.markdown(
            "Each signal detector fires when it identifies a statistical pattern. "
            "This tab shows which signals are most predictive.\n\n"
            "**Lift** = signal accuracy / league average accuracy. "
            "Lift > 1.0 means the signal adds value. "
            "Signals that consistently underperform should be downweighted or removed."
        )

    if not signal_df.empty:
        # Aggregate by signal type
        sig_summary = (
            signal_df.groupby("signal_type")
            .agg(
                times_fired=("player_id", "count"),
                hit_1_acc=("hit_1_correct", "mean"),
                hit_2_acc=("hit_2_correct", "mean"),
                hr_acc=("hr_correct", "mean"),
            )
            .reset_index()
        )
        sig_summary["1+ Hit %"] = (sig_summary["hit_1_acc"] * 100).round(1)
        sig_summary["2+ Hit %"] = (sig_summary["hit_2_acc"] * 100).round(1)

        # Compute lift vs overall baseline
        overall_1hit = accuracy_df["hit_1_correct"].mean() if not accuracy_df.empty else 0.65
        sig_summary["Lift vs Avg"] = (sig_summary["hit_1_acc"] / overall_1hit).round(2) if overall_1hit > 0 else 1.0

        display_sig = sig_summary[["signal_type", "times_fired", "1+ Hit %", "2+ Hit %", "Lift vs Avg"]].copy()
        display_sig.columns = ["Signal", "Times Fired", "1+ Hit %", "2+ Hit %", "Lift"]
        display_sig = display_sig.sort_values("Lift", ascending=False)

        st.dataframe(display_sig, use_container_width=True, hide_index=True)

        # Signal accuracy bar chart
        st.markdown("---")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=display_sig["Signal"],
            y=display_sig["1+ Hit %"],
            name="1+ Hit %",
            marker_color="#4dabf7",
            text=display_sig["1+ Hit %"].apply(lambda x: f"{x:.0f}%"),
            textposition="outside",
        ))
        fig.add_hline(
            y=overall_1hit * 100, line_dash="dash", line_color="gray",
            annotation_text=f"Overall avg ({overall_1hit*100:.0f}%)",
        )
        fig.update_layout(
            **DARK_LAYOUT,
            height=400,
            yaxis_title="Accuracy %",
            title="Signal Accuracy — When This Signal Fires, How Often Is the Prediction Correct?",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Signal accuracy over time (weekly)
        st.markdown("---")
        st.subheader("Signal Accuracy Over Time (Weekly)")

        signal_df_copy = signal_df.copy()
        signal_df_copy["week"] = pd.to_datetime(signal_df_copy["signal_date"]).dt.to_period("W").dt.start_time

        sig_weekly = (
            signal_df_copy.groupby(["week", "signal_type"])
            .agg(acc=("hit_1_correct", "mean"), count=("hit_1_correct", "count"))
            .reset_index()
        )
        sig_weekly["acc_pct"] = (sig_weekly["acc"] * 100).round(1)

        sig_types = sig_weekly["signal_type"].unique()
        colors = ["#4dabf7", "#6bcb77", "#ff6b6b", "#ffd93d", "#c084fc", "#fb923c"]

        fig = go.Figure()
        for i, sig_type in enumerate(sorted(sig_types)):
            sig_data = sig_weekly[sig_weekly["signal_type"] == sig_type]
            fig.add_trace(go.Scatter(
                x=sig_data["week"], y=sig_data["acc_pct"],
                mode="lines+markers", name=sig_type,
                line=dict(color=colors[i % len(colors)], width=2),
                marker=dict(size=5),
            ))

        fig.add_hline(y=overall_1hit * 100, line_dash="dash", line_color="gray")
        fig.update_layout(
            **DARK_LAYOUT,
            height=400,
            yaxis_title="1+ Hit Accuracy %",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No signal data with actuals available for the selected range.")

# ── Tab 5: HR Watch ──────────────────────────────────────────────────────────

with tab_hr:
    st.subheader("HR Watch Performance")

    with st.popover("?"):
        st.markdown(
            "**HR Watch** tracks how often the model's top HR candidates actually "
            "hit home runs.\n\n"
            "The league average HR rate is ~3.5% per PA or ~12% per game. "
            "The top-10 HR candidates should exceed this significantly. "
            "Tracking this over time shows whether the HR model is identifying "
            "true power opportunities."
        )

    # HR tracking from calibration_summary
    if not hr_watch_df.empty:
        hr_watch_df["date"] = pd.to_datetime(hr_watch_df["summary_date"])

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hr_watch_df["date"], y=hr_watch_df["value"] * 100,
            mode="lines+markers", name="Top-10 HR Accuracy",
            line=dict(color="#ff6b6b", width=2),
            marker=dict(size=6),
        ))
        fig.add_hline(y=12, line_dash="dash", line_color="gray",
                      annotation_text="League avg (~12% per game)")
        fig.update_layout(
            **DARK_LAYOUT,
            height=400,
            yaxis_title="% of Top-10 Who Hit HR",
            xaxis_title="Date",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No HR Watch tracking data available yet.")

    # HR prediction detail
    with st.expander("HR Prediction Detail"):
        if not accuracy_df.empty:
            hr_detail = accuracy_df[
                (accuracy_df["daily_rank"] <= 10)
                & (accuracy_df["p_hr"].notna())
            ][["prediction_date", "daily_rank", "p_hr", "actual_hr", "hr_correct"]].copy()

            if not hr_detail.empty:
                hr_detail = hr_detail.sort_values(
                    ["prediction_date", "p_hr"], ascending=[False, False]
                )
                hr_detail.columns = ["Date", "Rank", "P(HR)", "Actual HR", "Correct"]
                hr_detail["P(HR)"] = hr_detail["P(HR)"].apply(
                    lambda x: f"{x*100:.1f}%" if pd.notna(x) else "--"
                )
                hr_detail["Actual HR"] = hr_detail["Actual HR"].apply(
                    lambda x: int(x) if pd.notna(x) else "--"
                )
                hr_detail["Correct"] = hr_detail["Correct"].map({1: "Yes", 0: "No"})
                st.dataframe(hr_detail, use_container_width=True, hide_index=True, height=400)
            else:
                st.info("No HR predictions in top-10 with actuals.")
        else:
            st.info("No data available.")

# ── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Trends update as the pipeline scores predictions against actual outcomes. "
    "Weekly averages should improve as the model's calibration feedback loop accumulates data."
)
