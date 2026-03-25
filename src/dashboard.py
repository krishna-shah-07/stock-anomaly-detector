# src/dashboard.py
# Run: streamlit run src/dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — clean, minimal, no overlaps ─────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"]      { font-size: 0.88rem; font-weight: 500;
                                        padding: 0.4rem 0.9rem; }
    div[data-testid="stMetric"]       { background: #f8fafc; border-radius: 10px;
                                        padding: 0.7rem 1rem; }
    div[data-testid="stMetricValue"]  { font-size: 1.35rem; font-weight: 700; }
    div[data-testid="stMetricLabel"]  { font-size: 0.75rem; color: #64748b; }
    div[data-testid="stMetricDelta"]  { font-size: 0.75rem; }
    section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Stock Anomaly Detection Dashboard")
st.caption(
    "Dual-model system · Isolation Forest + Prophet · SPY (S&P 500 ETF)  "
    "— adjust the sidebar threshold to explore without retraining."
)

# ── Data loader ────────────────────────────────────────────────────────────────
@st.cache_data
def load_results() -> pd.DataFrame:
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results.csv"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results.csv"),
        "data/results.csv",
    ]
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_csv(p, index_col="Date", parse_dates=True)
            return df
    st.error("❌ data/results.csv not found. Run `python src/detect.py` first.")
    st.stop()


df = load_results()

# ── Pandas-version-safe resample ─────────────────────────────────────────────
# "ME" (Month End) added in pandas 2.2 — fall back to "M" for older installs
def safe_month_resample(frame):
    try:
        return frame.resample("ME")
    except ValueError:
        return frame.resample("M")


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    min_date = df.index.min().date()
    max_date = df.index.max().date()

    date_range = st.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(min_date, max_date),
    )

    st.divider()

    pct_2 = float(df["if_score"].quantile(0.02))
    threshold = st.slider(
        "IF Score threshold (live)",
        min_value=float(df["if_score"].min()),
        max_value=float(df["if_score"].max()),
        value=pct_2,
        step=0.005,
        help="Lower = stricter (fewer anomalies). Default = 2nd percentile.",
    )
    st.caption(f"Default (2nd pct): `{pct_2:.4f}`")

    st.divider()

    anomaly_filter = st.selectbox(
        "Table / data filter",
        ["All Data", "IF Anomalies", "Prophet Anomalies", "Consensus Only"],
    )

    st.divider()
    st.caption(f"**Ticker:** SPY")
    st.caption(f"**Range:** {min_date} → {max_date}")
    st.caption(f"**Rows:** {len(df):,}")

# ── Filter + recompute anomaly flags live ─────────────────────────────────────
df_date = df[
    (df.index.date >= date_range[0]) &
    (df.index.date <= date_range[1])
].copy()

df_date["if_anomaly"]        = (df_date["if_score"] <= threshold).astype(int)
df_date["consensus_anomaly"] = (
    (df_date["if_anomaly"] == 1) & (df_date["prophet_anomaly"] == 1)
).astype(int)


def apply_filter(frame: pd.DataFrame) -> pd.DataFrame:
    if anomaly_filter == "IF Anomalies":
        return frame[frame["if_anomaly"] == 1]
    elif anomaly_filter == "Prophet Anomalies":
        return frame[frame["prophet_anomaly"] == 1]
    elif anomaly_filter == "Consensus Only":
        return frame[frame["consensus_anomaly"] == 1]
    return frame


df_view = apply_filter(df_date)

# ── Derived metrics (computed once, reused everywhere) ────────────────────────
total_days      = len(df_date)
if_count        = int(df_date["if_anomaly"].sum())
prophet_count   = int(df_date["prophet_anomaly"].sum())
consensus_count = int(df_date["consensus_anomaly"].sum())
period_return   = (df_date["Close"].iloc[-1] / df_date["Close"].iloc[0] - 1) * 100
volatility      = (
    df_date["daily_return"].std() * 100
    if "daily_return" in df_date.columns
    else df_date["Close"].pct_change().std() * 100
)
denom           = max(if_count, prophet_count, 1)
agreement_pct   = 100 * consensus_count / denom

# ── KPI row ────────────────────────────────────────────────────────────────────
st.subheader("Summary")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Trading days",     f"{total_days:,}")
k2.metric("IF anomalies",     f"{if_count}",        f"{100*if_count/total_days:.1f}% of days")
k3.metric("Prophet anom.",    f"{prophet_count}",   f"{100*prophet_count/total_days:.1f}% of days")
k4.metric("Consensus",        f"{consensus_count}", f"{100*consensus_count/total_days:.2f}% of days")
k5.metric("Period return",    f"{period_return:+.2f}%")
k6.metric("Daily volatility", f"{volatility:.2f}%", "price std dev")

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈  Overview",
    "🎯  Anomalies",
    "📊  Analysis",
    "📋  Data",
    "📝  Findings",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("#### Price chart with live anomaly overlay")
    st.caption(
        "🔴 Red diamond = IF anomaly · 🟠 Orange star = Prophet anomaly · "
        "🟢 Green ring = Consensus (both models)"
    )

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        subplot_titles=("Close price + anomaly markers", "IF anomaly score", "Prophet residual"),
        vertical_spacing=0.08,
        row_heights=[0.55, 0.22, 0.23],
    )

    fig.add_trace(go.Scatter(
        x=df_date.index, y=df_date["Close"],
        name="Close", mode="lines",
        line=dict(color="#1e40af", width=1.8),
    ), row=1, col=1)

    covid_start = pd.Timestamp("2020-02-15")
    covid_end   = pd.Timestamp("2020-04-15")
    if df_date.index.min() <= covid_end and df_date.index.max() >= covid_start:
        fig.add_vrect(
            x0=covid_start, x1=covid_end,
            fillcolor="rgba(234,179,8,0.12)", line_width=0,
            annotation_text="COVID crash", annotation_position="top left",
        )

    if_anom = df_date[df_date["if_anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=if_anom.index, y=if_anom["Close"],
        name="IF anomaly", mode="markers",
        marker=dict(color="#dc2626", size=9, symbol="diamond",
                    line=dict(color="#991b1b", width=1)),
    ), row=1, col=1)

    p_anom = df_date[df_date["prophet_anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=p_anom.index, y=p_anom["Close"],
        name="Prophet anomaly", mode="markers",
        marker=dict(color="#f97316", size=7, symbol="star"),
    ), row=1, col=1)

    c_anom = df_date[df_date["consensus_anomaly"] == 1]
    fig.add_trace(go.Scatter(
        x=c_anom.index, y=c_anom["Close"],
        name="Consensus", mode="markers",
        marker=dict(color="rgba(0,0,0,0)", size=16, symbol="circle",
                    line=dict(color="#16a34a", width=2.5)),
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=df_date.index, y=df_date["if_score"],
        name="IF score", mode="lines",
        line=dict(color="#16a34a", width=1.5),
        showlegend=False,
    ), row=2, col=1)
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="#dc2626", line_width=1.5,
        annotation_text=f"Threshold: {threshold:.3f}",
        annotation_position="bottom right",
        row=2, col=1,
    )

    bar_colors = ["#dc2626" if v < 0 else "#2563eb" for v in df_date["residual"]]
    fig.add_trace(go.Bar(
        x=df_date.index, y=df_date["residual"],
        name="Residual", marker_color=bar_colors, opacity=0.65,
        showlegend=False,
    ), row=3, col=1)
    fig.add_hline(y=0, line_color="#94a3b8", line_width=0.8, row=3, col=1)

    fig.update_yaxes(title_text="Price ($)",  row=1, col=1, title_font_size=11)
    fig.update_yaxes(title_text="IF score",   row=2, col=1, title_font_size=11)
    fig.update_yaxes(title_text="Residual $", row=3, col=1, title_font_size=11)
    fig.update_xaxes(title_text="Date",       row=3, col=1, title_font_size=11)
    fig.update_layout(
        height=700,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.03, x=0, font_size=12),
        margin=dict(t=80, b=20, l=60, r=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9", gridwidth=0.5)
    fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9", gridwidth=0.5)
    st.plotly_chart(fig, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Anomalies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    left, right = st.columns([1.1, 1], gap="large")

    with left:
        st.markdown("#### Top consensus anomalies")
        df_cons = df_date[df_date["consensus_anomaly"] == 1].copy()
        df_cons["severity"] = df_cons["if_score"].abs()
        df_cons = df_cons.sort_values("severity", ascending=False)

        if df_cons.empty:
            st.info("No consensus anomalies in this range. Try moving the threshold slider right.")
        else:
            st.caption(f"{len(df_cons)} days flagged by both models simultaneously.")
            max_n = min(30, len(df_cons))
            default_n = min(10, len(df_cons))
            if max_n <= 1:
                top_n = 1
                st.markdown("*Only one consensus anomaly is available to show.*")
            else:
                top_n = st.slider("Show top N", 1, max_n, default_n, key="top_n")
            disp = [c for c in ["Close", "daily_return", "if_score", "residual", "volume_zscore"]
                    if c in df_cons.columns]
            fmt = {k: v for k, v in {
                "Close":         "${:.2f}",
                "daily_return":  "{:.2%}",
                "if_score":      "{:.4f}",
                "residual":      "{:.2f}",
                "volume_zscore": "{:.2f}",
            }.items() if k in disp}
            st.dataframe(
                df_cons[disp].head(top_n)
                .style.background_gradient(subset=["if_score"], cmap="RdYlGn_r")
                .format(fmt),
                use_container_width=True,
                height=360,
            )

    with right:
        st.markdown("#### Anomalies per month")
        agg = safe_month_resample(df_date).agg(
            IF=("if_anomaly", "sum"),
            Prophet=("prophet_anomaly", "sum"),
            Consensus=("consensus_anomaly", "sum"),
        )
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(x=agg.index, y=agg["IF"],        name="IF",        marker_color="#dc2626", opacity=0.8))
        fig_bar.add_trace(go.Bar(x=agg.index, y=agg["Prophet"],   name="Prophet",   marker_color="#f97316", opacity=0.8))
        fig_bar.add_trace(go.Bar(x=agg.index, y=agg["Consensus"], name="Consensus", marker_color="#16a34a", opacity=0.9))
        fig_bar.update_layout(
            barmode="group", height=380,
            legend=dict(orientation="h", y=1.08, font_size=11),
            margin=dict(t=50, b=30, l=40, r=20),
            xaxis_title="Month", yaxis_title="Count",
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig_bar.update_xaxes(showgrid=False)
        fig_bar.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig_bar, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Analysis
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("#### Score distributions")
    d1, d2 = st.columns(2, gap="medium")

    with d1:
        fig_if = go.Figure()
        fig_if.add_trace(go.Histogram(
            x=df_date["if_score"], nbinsx=60,
            marker_color="#16a34a", opacity=0.7,
        ))
        fig_if.add_vline(x=threshold, line_dash="dash", line_color="#dc2626", line_width=1.5,
                         annotation_text=f"Threshold {threshold:.3f}", annotation_position="top right")
        fig_if.update_layout(
            height=300, title="Isolation Forest scores",
            xaxis_title="Score", yaxis_title="Days",
            margin=dict(t=40, b=40, l=40, r=20),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_if, use_container_width=True)
        st.dataframe(df_date["if_score"].describe().rename("IF score").to_frame(),
                     use_container_width=True)

    with d2:
        fig_res = go.Figure()
        fig_res.add_trace(go.Histogram(
            x=df_date["residual"], nbinsx=60,
            marker_color="#7c3aed", opacity=0.7,
        ))
        fig_res.add_vline(x=0, line_color="#94a3b8", line_dash="dot", line_width=1)
        fig_res.update_layout(
            height=300, title="Prophet residuals",
            xaxis_title="Residual ($)", yaxis_title="Days",
            margin=dict(t=40, b=40, l=40, r=20),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        )
        st.plotly_chart(fig_res, use_container_width=True)
        st.dataframe(df_date["residual"].describe().rename("Residual ($)").to_frame(),
                     use_container_width=True)

    st.markdown("#### 30-day rolling anomaly rate")
    rolling = df_date[["consensus_anomaly"]].rolling(30).mean() * 100
    fig_roll = go.Figure(go.Scatter(
        x=rolling.index, y=rolling["consensus_anomaly"],
        mode="lines", fill="tozeroy",
        line=dict(color="#16a34a", width=1.5),
        fillcolor="rgba(22,163,74,0.08)",
    ))
    fig_roll.add_hline(y=2, line_dash="dot", line_color="#f97316", line_width=1,
                       annotation_text="2% baseline", annotation_position="right")
    fig_roll.update_layout(
        height=240, xaxis_title="Date", yaxis_title="Anomaly rate (%)",
        margin=dict(t=20, b=40, l=50, r=80),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
    )
    fig_roll.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig_roll.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig_roll, use_container_width=True)

    st.markdown("#### Feature correlation matrix")
    exclude  = ["if_anomaly", "prophet_anomaly", "consensus_anomaly",
                "yhat", "yhat_lower", "yhat_upper"]
    num_cols = [c for c in df_date.select_dtypes(include=np.number).columns if c not in exclude]
    corr     = df_date[num_cols].corr().round(2)
    fig_corr = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu", zmid=0,
        text=corr.values, texttemplate="%{text:.2f}",
        colorbar=dict(thickness=12, len=0.8),
    ))
    fig_corr.update_layout(
        height=500,
        margin=dict(t=20, b=60, l=120, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_corr, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown(f"#### Dataset — {anomaly_filter}")
    st.caption(f"{len(df_view):,} rows shown")

    disp = [c for c in ["Close", "daily_return", "if_score", "residual",
                         "if_anomaly", "prophet_anomaly", "consensus_anomaly"]
            if c in df_view.columns]
    fmt = {k: v for k, v in {
        "Close":        "${:.2f}",
        "daily_return": "{:.2%}",
        "if_score":     "{:.4f}",
        "residual":     "{:.2f}",
    }.items() if k in disp}

    st.dataframe(
        df_view[disp].style
        .background_gradient(subset=["if_score"], cmap="RdYlGn_r")
        .format(fmt),
        use_container_width=True,
        height=520,
    )
    st.download_button(
        "⬇️  Download filtered CSV",
        data=df_view[disp].to_csv().encode(),
        file_name=f"anomalies_{date_range[0]}_{date_range[1]}.csv",
        mime="text/csv",
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Findings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab5:
    st.markdown("#### Key findings & interpretation")

    top_cons = (
        df_date[df_date["consensus_anomaly"] == 1].copy()
        .assign(severity=lambda x: x["if_score"].abs())
        .sort_values("severity", ascending=False)
    )

    fl, fr = st.columns([1.6, 1], gap="large")

    with fl:
        st.markdown("**1. Anomaly detection activity**")
        st.markdown(f"""
Over **{total_days:,} trading days**, the system flagged:
- `{if_count}` unusual days via Isolation Forest ({100*if_count/total_days:.1f}%)
- `{prophet_count}` trend deviations via Prophet ({100*prophet_count/total_days:.1f}%)
- **{consensus_count} high-confidence events** confirmed by both models ({100*consensus_count/total_days:.2f}%)

Consensus events are the strongest signals — both models detected something independently.
        """)

        st.markdown("**2. Price performance**")
        start_p = df_date["Close"].iloc[0]
        end_p   = df_date["Close"].iloc[-1]
        st.markdown(f"""
| Metric | Value |
|:--|:--|
| Start price | ${start_p:.2f} |
| End price | ${end_p:.2f} |
| Period return | {period_return:+.2f}% |
| Daily volatility | {volatility:.2f}% |
| Price range | ${df_date['Close'].min():.2f} – ${df_date['Close'].max():.2f} |
        """)

        if not top_cons.empty:
            st.markdown(f"**3. Top {min(5, len(top_cons))} most extreme days**")
            for i, (date, row) in enumerate(top_cons.head(5).iterrows(), 1):
                icon = "📉" if row.get("residual", 0) < 0 else "📈"
                st.markdown(
                    f"**{i}. {date.strftime('%b %d, %Y')}** {icon}  "
                    f"Close `${row['Close']:.2f}` · IF score `{row['if_score']:.4f}` · "
                    f"Residual `{row['residual']:.2f}`"
                )

        st.markdown("**4. Model agreement**")
        st.markdown(f"""
- Agreement rate: **{agreement_pct:.1f}%** of anomalies confirmed by both models
- IF-only detections: `{if_count - consensus_count}` (multivariate outliers)
- Prophet-only detections: `{prophet_count - consensus_count}` (trend band violations)
        """)

    with fr:
        st.markdown("**Interpretation**")
        rate = 100 * consensus_count / total_days
        if rate > 10:
            st.warning(f"⚠️ High anomaly rate ({rate:.1f}%)\n\nConsider tightening the threshold.")
        elif rate > 3:
            st.info(f"ℹ️ Active period ({rate:.1f}%)\n\nHealthy range for trading signals.")
        else:
            st.success(f"✅ Stable period ({rate:.2f}%)\n\nFew extreme events — typical SPY behaviour.")

        st.divider()
        vol_icon  = "🔴" if volatility > 2.5 else "🟡" if volatility > 1.5 else "🟢"
        vol_label = "High" if volatility > 2.5 else "Moderate" if volatility > 1.5 else "Low"
        st.markdown(f"**Volatility: {vol_icon} {vol_label}**  \n`{volatility:.2f}%` daily std dev")

        st.divider()
        st.markdown("""
**How to use this dashboard**
1. Adjust **date range** to zoom into a period
2. Move **IF threshold** slider for live sensitivity
3. Check **Anomalies tab** for ranked event list
4. Use **Analysis tab** to validate distributions
5. **Export** filtered data from Data tab
        """)

    st.divider()
    m1, m2, m3 = st.columns(3)
    m1.metric("Model agreement",        f"{agreement_pct:.1f}%",                  "both models agree")
    m2.metric("Anomaly concentration",  f"{100*consensus_count/total_days:.2f}%", "consensus days")
    m3.metric("Daily volatility",       f"{volatility:.2f}%",                     "price std dev")

    st.info("""
**Reading the numbers**
- Model agreement >50% → high-confidence events
- Anomaly concentration 1–3% → normal for SPY. >5% → tighten threshold
- Daily volatility >2% → volatile regime. 1–2% → normal. <1% → very calm
    """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Tip: the sidebar IF threshold slider recomputes all anomaly flags live — "
    "no retraining needed. Lower = stricter. Default = 2nd percentile of score distribution."
)