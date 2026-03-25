# src/dashboard.py
# Run: streamlit run src/dashboard.py

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Anomaly Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"]      { font-size: 0.88rem; font-weight: 500; padding: 0.4rem 0.9rem; }
    div[data-testid="stMetric"]       { background: #f8fafc; border-radius: 10px; padding: 0.7rem 1rem; }
    div[data-testid="stMetricValue"]  { font-size: 1.3rem; font-weight: 700; }
    div[data-testid="stMetricLabel"]  { font-size: 0.75rem; color: #64748b; }
    div[data-testid="stMetricDelta"]  { font-size: 0.75rem; }
    .stDataFrame { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
TICKER_COLORS = {
    "SPY":     "#1e40af",
    "QQQ":     "#16a34a",
    "GLD":     "#d97706",
    "BTC-USD": "#dc2626",
    "AAPL":    "#7c3aed",
}
DEFAULT_COLOR = "#64748b"

TICKER_LABELS = {
    "SPY":     "SPY — S&P 500",
    "QQQ":     "QQQ — Nasdaq 100",
    "GLD":     "GLD — Gold ETF",
    "BTC-USD": "BTC — Bitcoin",
    "AAPL":    "AAPL — Apple",
}

# ── Helpers ────────────────────────────────────────────────────────────────────
def safe_month_resample(frame):
    try:
        return frame.resample("ME")
    except ValueError:
        return frame.resample("M")

def ticker_color(t: str) -> str:
    return TICKER_COLORS.get(t, DEFAULT_COLOR)

def ticker_label(t: str) -> str:
    return TICKER_LABELS.get(t, t)

# ── Data loaders ───────────────────────────────────────────────────────────────
def _results_root() -> str:
    # Works from src/ or project root
    candidates = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "results"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "results"),
        "data/results",
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return "data/results"

@st.cache_data(ttl=300)
def load_portfolio() -> pd.DataFrame:
    root = _results_root()
    path = os.path.join(root, "portfolio_results.csv")
    if not os.path.exists(path):
        st.error("❌ `data/results/portfolio_results.csv` not found. Run `python src/detect.py` first.")
        st.stop()
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df

@st.cache_data(ttl=300)
def load_ticker(ticker: str) -> pd.DataFrame:
    root = _results_root()
    path = os.path.join(root, f"{ticker}_results.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col="Date", parse_dates=True)

# ── Load data ──────────────────────────────────────────────────────────────────
portfolio_raw = load_portfolio()
all_tickers   = sorted(portfolio_raw["ticker"].unique().tolist())

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Controls")

    global_min = portfolio_raw.index.min().date()
    global_max = portfolio_raw.index.max().date()
    date_range = st.slider("Date range", min_value=global_min, max_value=global_max,
                            value=(global_min, global_max))

    st.divider()

    selected_tickers = st.multiselect(
        "Active tickers",
        all_tickers,
        default=all_tickers,
        help="Used in Portfolio and Comparison tabs"
    )
    if not selected_tickers:
        st.warning("Select at least one ticker.")
        st.stop()

    st.divider()

    focus_ticker = st.selectbox(
        "Single-ticker focus",
        selected_tickers,
        help="Used in the Deep Dive tab"
    )

    # Per-ticker threshold (live, no retraining)
    pct_2     = float(load_ticker(focus_ticker)["if_score"].quantile(0.02)) \
                if not load_ticker(focus_ticker).empty else -0.6
    threshold = st.slider(
        f"IF threshold ({focus_ticker})",
        min_value=-0.80, max_value=-0.30,
        value=round(pct_2, 3), step=0.005,
        help="Recomputes anomalies live without retraining"
    )

    st.divider()
    st.caption(f"**Tickers:** {', '.join(all_tickers)}")
    st.caption(f"**Range:** {global_min} → {global_max}")
    st.caption(f"**Total rows:** {len(portfolio_raw):,}")

# ── Filter portfolio by date + selected tickers ────────────────────────────────
port = portfolio_raw[
    (portfolio_raw.index.date >= date_range[0]) &
    (portfolio_raw.index.date <= date_range[1]) &
    (portfolio_raw["ticker"].isin(selected_tickers))
].copy()

# ── Header ─────────────────────────────────────────────────────────────────────
st.title("📈 Multi-Asset Anomaly Detection Dashboard")
st.caption(
    f"Isolation Forest + Prophet · {len(selected_tickers)} tickers active · "
    f"{date_range[0]} → {date_range[1]} · cache refreshes every 5 min"
)

# ── Portfolio-level KPIs ───────────────────────────────────────────────────────
total_consensus = int(port["consensus_anomaly"].sum())
total_if        = int(port["if_anomaly"].sum())
total_prophet   = int(port["prophet_anomaly"].sum())
total_days      = len(port)
most_flagged    = (port.groupby("ticker")["consensus_anomaly"].sum()
                   .idxmax() if total_consensus > 0 else "—")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Total trading rows",   f"{total_days:,}")
k2.metric("IF anomalies",         f"{total_if}",        f"{100*total_if/max(total_days,1):.1f}%")
k3.metric("Prophet anomalies",    f"{total_prophet}",   f"{100*total_prophet/max(total_days,1):.1f}%")
k4.metric("Consensus anomalies",  f"{total_consensus}", f"{100*total_consensus/max(total_days,1):.2f}%")
k5.metric("Most flagged ticker",  most_flagged)

st.divider()

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab_port, tab_compare, tab_deep, tab_heat, tab_data, tab_findings = st.tabs([
    "🗂️  Portfolio",
    "📊  Comparison",
    "🔬  Deep Dive",
    "🌡️  Heatmap",
    "📋  Data",
    "📝  Findings",
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — Portfolio Overview
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_port:
    st.markdown("#### Normalised price performance (base = 100 at start)")
    st.caption("Lets you compare SPY, QQQ, GLD, BTC, AAPL on the same scale regardless of price.")

    fig_norm = go.Figure()
    for ticker in selected_tickers:
        t_df = port[port["ticker"] == ticker].sort_index()
        if t_df.empty or t_df["Close"].iloc[0] == 0:
            continue
        norm = (t_df["Close"] / t_df["Close"].iloc[0]) * 100

        # Consensus anomaly markers
        c_anom = t_df[t_df["consensus_anomaly"] == 1]
        c_norm = (c_anom["Close"] / t_df["Close"].iloc[0]) * 100

        fig_norm.add_trace(go.Scatter(
            x=t_df.index, y=norm,
            name=ticker_label(ticker), mode="lines",
            line=dict(width=2, color=ticker_color(ticker)),
        ))
        if not c_anom.empty:
            fig_norm.add_trace(go.Scatter(
                x=c_anom.index, y=c_norm,
                name=f"{ticker} anomaly", mode="markers",
                marker=dict(color=ticker_color(ticker), size=8, symbol="diamond",
                            line=dict(color="white", width=1.5)),
                showlegend=False,
            ))

    fig_norm.update_layout(
        height=420, hovermode="x unified",
        legend=dict(orientation="h", y=1.07, font_size=12),
        margin=dict(t=60, b=30, l=55, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="Indexed price (base 100)",
    )
    fig_norm.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig_norm.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    fig_norm.add_hline(y=100, line_dash="dot", line_color="#94a3b8", line_width=1)
    st.plotly_chart(fig_norm, use_container_width=True)

    # ── Summary table ──────────────────────────────────────────────────────
    st.markdown("#### Portfolio summary")
    rows = []
    for ticker in selected_tickers:
        t_df = port[port["ticker"] == ticker].sort_index()
        if t_df.empty:
            continue
        ret     = (t_df["Close"].iloc[-1] / t_df["Close"].iloc[0] - 1) * 100
        vol     = t_df["Close"].pct_change().std() * 100
        n_cons  = int(t_df["consensus_anomaly"].sum())
        n_days  = len(t_df)
        rows.append({
            "Ticker":         ticker,
            "Period return":  f"{ret:+.1f}%",
            "Daily vol":      f"{vol:.2f}%",
            "IF anomalies":   int(t_df["if_anomaly"].sum()),
            "Prophet anom.":  int(t_df["prophet_anomaly"].sum()),
            "Consensus":      n_cons,
            "Cons. rate":     f"{100*n_cons/max(n_days,1):.2f}%",
        })

    summary_df = pd.DataFrame(rows).set_index("Ticker")
    st.dataframe(summary_df, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — Side-by-side Comparison
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_compare:
    st.markdown("#### IF anomaly scores — all tickers overlaid")
    st.caption("Lower score = more anomalous. Lets you see which assets were stressed simultaneously.")

    fig_scores = go.Figure()
    for ticker in selected_tickers:
        t_df = port[port["ticker"] == ticker].sort_index()
        if t_df.empty:
            continue
        fig_scores.add_trace(go.Scatter(
            x=t_df.index, y=t_df["if_score"],
            name=ticker, mode="lines",
            line=dict(width=1.5, color=ticker_color(ticker)),
            opacity=0.85,
        ))

    fig_scores.update_layout(
        height=340, hovermode="x unified",
        legend=dict(orientation="h", y=1.07, font_size=12),
        margin=dict(t=60, b=30, l=55, r=20),
        plot_bgcolor="white", paper_bgcolor="white",
        yaxis_title="IF anomaly score",
    )
    fig_scores.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig_scores.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig_scores, use_container_width=True)

    # ── Monthly anomaly bar chart ──────────────────────────────────────────
    st.markdown("#### Consensus anomalies per month — grouped by ticker")
    agg = (
        safe_month_resample(port.set_index(port.index))
        if False  # placeholder — use groupby below instead
        else port.assign(month=port.index.to_period("M").astype(str))
                 .groupby(["month", "ticker"])["consensus_anomaly"]
                 .sum()
                 .reset_index()
    )

    fig_bar = go.Figure()
    for ticker in selected_tickers:
        t_agg = agg[agg["ticker"] == ticker]
        fig_bar.add_trace(go.Bar(
            x=t_agg["month"], y=t_agg["consensus_anomaly"],
            name=ticker, marker_color=ticker_color(ticker), opacity=0.85,
        ))

    fig_bar.update_layout(
        barmode="group", height=320,
        legend=dict(orientation="h", y=1.07, font_size=12),
        margin=dict(t=60, b=60, l=40, r=20),
        xaxis=dict(tickangle=-45, tickfont_size=9),
        yaxis_title="Consensus anomalies",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig_bar.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Correlation of IF scores across tickers ────────────────────────────
    st.markdown("#### IF score correlation across assets")
    st.caption("High correlation = assets were stressed at the same time (market-wide event). Low = idiosyncratic.")

    score_pivot = (
        port[["ticker", "if_score"]]
        .reset_index()
        .pivot_table(index="Date", columns="ticker", values="if_score", aggfunc="mean")
        .dropna()
    )
    if score_pivot.shape[1] > 1:
        corr = score_pivot.corr().round(2)
        fig_corr = go.Figure(go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.index,
            colorscale="RdBu", zmid=0,
            text=corr.values, texttemplate="%{text:.2f}",
            colorbar=dict(thickness=12, len=0.8),
            zmin=-1, zmax=1,
        ))
        fig_corr.update_layout(
            height=320,
            margin=dict(t=20, b=60, l=80, r=20),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        st.caption(
            "SPY & QQQ typically correlate highly (both US equities). "
            "GLD often decorrelates during equity stress. "
            "BTC tends to be idiosyncratic."
        )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — Deep Dive (single ticker)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_deep:
    st.markdown(f"#### Deep dive — {ticker_label(focus_ticker)}")

    t_df = load_ticker(focus_ticker)
    if t_df.empty:
        st.warning(f"No results file found for {focus_ticker}.")
    else:
        # Apply date filter + live threshold
        t_df = t_df[
            (t_df.index.date >= date_range[0]) &
            (t_df.index.date <= date_range[1])
        ].copy()
        t_df["if_anomaly"]       = (t_df["if_score"] <= threshold).astype(int)
        t_df["consensus_anomaly"] = (
            (t_df["if_anomaly"] == 1) & (t_df["prophet_anomaly"] == 1)
        ).astype(int)

        # KPIs for this ticker
        n_if   = int(t_df["if_anomaly"].sum())
        n_prop = int(t_df["prophet_anomaly"].sum())
        n_cons = int(t_df["consensus_anomaly"].sum())
        n_days = len(t_df)
        ret    = (t_df["Close"].iloc[-1] / t_df["Close"].iloc[0] - 1) * 100

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Trading days",   f"{n_days:,}")
        d2.metric("IF anomalies",   f"{n_if}",   f"{100*n_if/max(n_days,1):.1f}%")
        d3.metric("Consensus",      f"{n_cons}",  f"{100*n_cons/max(n_days,1):.2f}%")
        d4.metric("Period return",  f"{ret:+.2f}%")

        st.markdown("")

        # 3-panel chart
        color = ticker_color(focus_ticker)
        fig = make_subplots(
            rows=3, cols=1, shared_xaxes=True,
            subplot_titles=("Price + anomalies", "IF score", "Prophet residual"),
            vertical_spacing=0.08, row_heights=[0.55, 0.22, 0.23],
        )

        fig.add_trace(go.Scatter(
            x=t_df.index, y=t_df["Close"], name="Close",
            mode="lines", line=dict(color=color, width=1.8),
        ), row=1, col=1)

        # COVID shading
        cs, ce = pd.Timestamp("2020-02-15"), pd.Timestamp("2020-04-15")
        if t_df.index.min() <= ce and t_df.index.max() >= cs:
            fig.add_vrect(x0=cs, x1=ce,
                          fillcolor="rgba(234,179,8,0.12)", line_width=0,
                          annotation_text="COVID crash", annotation_position="top left")

        if_anom = t_df[t_df["if_anomaly"] == 1]
        fig.add_trace(go.Scatter(
            x=if_anom.index, y=if_anom["Close"], name="IF anomaly", mode="markers",
            marker=dict(color="#dc2626", size=9, symbol="diamond",
                        line=dict(color="#991b1b", width=1)),
        ), row=1, col=1)

        p_anom = t_df[t_df["prophet_anomaly"] == 1]
        fig.add_trace(go.Scatter(
            x=p_anom.index, y=p_anom["Close"], name="Prophet anomaly", mode="markers",
            marker=dict(color="#f97316", size=7, symbol="star"),
        ), row=1, col=1)

        c_anom = t_df[t_df["consensus_anomaly"] == 1]
        fig.add_trace(go.Scatter(
            x=c_anom.index, y=c_anom["Close"], name="Consensus", mode="markers",
            marker=dict(color="rgba(0,0,0,0)", size=16, symbol="circle",
                        line=dict(color="#16a34a", width=2.5)),
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=t_df.index, y=t_df["if_score"], name="IF score",
            mode="lines", line=dict(color="#16a34a", width=1.5), showlegend=False,
        ), row=2, col=1)
        fig.add_hline(y=threshold, line_dash="dash", line_color="#dc2626", line_width=1.5,
                      annotation_text=f"Threshold {threshold:.3f}",
                      annotation_position="bottom right", row=2, col=1)

        bar_colors = ["#dc2626" if v < 0 else "#2563eb" for v in t_df["residual"]]
        fig.add_trace(go.Bar(
            x=t_df.index, y=t_df["residual"],
            name="Residual", marker_color=bar_colors, opacity=0.65, showlegend=False,
        ), row=3, col=1)
        fig.add_hline(y=0, line_color="#94a3b8", line_width=0.8, row=3, col=1)

        fig.update_yaxes(title_text="Price ($)",  row=1, col=1, title_font_size=11)
        fig.update_yaxes(title_text="IF score",   row=2, col=1, title_font_size=11)
        fig.update_yaxes(title_text="Residual",   row=3, col=1, title_font_size=11)
        fig.update_xaxes(title_text="Date",       row=3, col=1, title_font_size=11)
        fig.update_layout(
            height=680, hovermode="x unified",
            legend=dict(orientation="h", y=1.03, x=0, font_size=12),
            margin=dict(t=80, b=20, l=60, r=40),
            plot_bgcolor="white", paper_bgcolor="white",
        )
        fig.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
        fig.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
        st.plotly_chart(fig, use_container_width=True)

        # Top anomalies table
        st.markdown("#### Top consensus anomaly days")
        top = (t_df[t_df["consensus_anomaly"] == 1]
               .copy()
               .assign(severity=lambda x: x["if_score"].abs())
               .sort_values("severity", ascending=False))

        if top.empty:
            st.info("No consensus anomalies with current threshold. Try moving the slider right.")
        else:
            disp = [c for c in ["Close", "daily_return", "if_score", "residual", "volume_zscore"]
                    if c in top.columns]
            fmt  = {k: v for k, v in {
                "Close": "${:.2f}", "daily_return": "{:.2%}",
                "if_score": "{:.4f}", "residual": "{:.2f}", "volume_zscore": "{:.2f}",
            }.items() if k in disp}
            st.dataframe(
                top[disp].head(15).style
                .background_gradient(subset=["if_score"], cmap="RdYlGn_r")
                .format(fmt),
                use_container_width=True, height=380,
            )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — Anomaly Heatmap
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_heat:
    st.markdown("#### Consensus anomaly heatmap — ticker × month")
    st.caption(
        "Red = many anomalies that month. Green = calm. "
        "Columns where multiple tickers are red = market-wide stress event."
    )

    pivot = (
        port.assign(month=port.index.to_period("M").astype(str))
        .groupby(["ticker", "month"])["consensus_anomaly"]
        .sum()
        .reset_index()
        .pivot(index="ticker", columns="month", values="consensus_anomaly")
        .fillna(0)
        .reindex(index=selected_tickers)
    )

    # Sort columns chronologically
    pivot = pivot[sorted(pivot.columns)]

    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=[ticker_label(t) for t in pivot.index],
        colorscale=[[0, "#f0fdf4"], [0.4, "#fde68a"], [1, "#dc2626"]],
        colorbar=dict(title="Anomalies", thickness=14, len=0.8),
        hovertemplate="%{y}<br>%{x}: %{z:.0f} anomalies<extra></extra>",
    ))
    fig_heat.update_layout(
        height=max(250, len(selected_tickers) * 70 + 100),
        margin=dict(t=20, b=80, l=140, r=20),
        xaxis=dict(tickangle=-60, tickfont_size=9),
        plot_bgcolor="white", paper_bgcolor="white",
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Rolling 30-day anomaly rate per ticker ─────────────────────────────
    st.markdown("#### 30-day rolling consensus anomaly rate — by ticker")
    fig_roll = go.Figure()
    for ticker in selected_tickers:
        t_df = port[port["ticker"] == ticker].sort_index()
        if t_df.empty:
            continue
        rolling = t_df["consensus_anomaly"].rolling(30).mean() * 100
        fig_roll.add_trace(go.Scatter(
            x=rolling.index, y=rolling,
            name=ticker, mode="lines",
            line=dict(color=ticker_color(ticker), width=1.8),
        ))

    fig_roll.add_hline(y=2, line_dash="dot", line_color="#94a3b8", line_width=1,
                       annotation_text="2% baseline", annotation_position="right")
    fig_roll.update_layout(
        height=300, hovermode="x unified",
        legend=dict(orientation="h", y=1.07, font_size=12),
        margin=dict(t=60, b=30, l=55, r=80),
        yaxis_title="30d anomaly rate (%)",
        plot_bgcolor="white", paper_bgcolor="white",
    )
    fig_roll.update_xaxes(showgrid=True, gridcolor="#f1f5f9")
    fig_roll.update_yaxes(showgrid=True, gridcolor="#f1f5f9")
    st.plotly_chart(fig_roll, use_container_width=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 5 — Data
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_data:
    st.markdown("#### Full portfolio dataset")

    ticker_filter = st.selectbox("Filter by ticker", ["All"] + selected_tickers)
    show_only     = st.checkbox("Show only consensus anomalies", value=False)

    display = port.copy()
    if ticker_filter != "All":
        display = display[display["ticker"] == ticker_filter]
    if show_only:
        display = display[display["consensus_anomaly"] == 1]

    st.caption(f"{len(display):,} rows shown")

    disp_cols = [c for c in ["ticker", "Close", "daily_return", "if_score",
                              "residual", "if_anomaly", "prophet_anomaly", "consensus_anomaly"]
                 if c in display.columns]
    fmt = {k: v for k, v in {
        "Close": "${:.2f}", "daily_return": "{:.2%}",
        "if_score": "{:.4f}", "residual": "{:.2f}",
    }.items() if k in disp_cols}

    st.dataframe(
        display[disp_cols].reset_index().style
        .background_gradient(subset=["if_score"], cmap="RdYlGn_r")
        .format(fmt),
        use_container_width=True, height=500,
    )

    st.download_button(
        "⬇️  Download filtered CSV",
        data=display[disp_cols].to_csv().encode(),
        file_name=f"anomalies_{ticker_filter}_{date_range[0]}_{date_range[1]}.csv",
        mime="text/csv",
    )

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 6 — Findings
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab_findings:
    st.markdown("#### Portfolio-level findings & interpretation")

    fl, fr = st.columns([1.6, 1], gap="large")

    with fl:
        st.markdown("**1. Detection activity across all assets**")
        st.markdown(f"""
Over the selected period across **{len(selected_tickers)} tickers**:
- `{total_if}` IF anomaly days total ({100*total_if/max(total_days,1):.1f}% of all ticker-days)
- `{total_prophet}` Prophet anomaly days
- **{total_consensus} consensus events** confirmed by both models

Most active ticker: **{most_flagged}**
        """)

        st.markdown("**2. Per-ticker breakdown**")
        for ticker in selected_tickers:
            t_df  = port[port["ticker"] == ticker]
            if t_df.empty:
                continue
            n_c   = int(t_df["consensus_anomaly"].sum())
            n_d   = len(t_df)
            ret   = (t_df["Close"].iloc[-1] / t_df["Close"].iloc[0] - 1) * 100
            vol   = t_df["Close"].pct_change().std() * 100
            label = ticker_label(ticker)
            st.markdown(
                f"**{label}** — {n_c} consensus anomalies "
                f"({100*n_c/max(n_d,1):.2f}% of days) · "
                f"Return: {ret:+.1f}% · Vol: {vol:.2f}%"
            )

        st.markdown("**3. Cross-asset observations**")

        # Compute score correlation for insight
        score_pivot = (
            port[["ticker", "if_score"]]
            .reset_index()
            .pivot_table(index="Date", columns="ticker", values="if_score", aggfunc="mean")
            .dropna()
        )
        if "SPY" in score_pivot and "QQQ" in score_pivot:
            spy_qqq_corr = score_pivot["SPY"].corr(score_pivot["QQQ"])
            st.markdown(f"""
- SPY ↔ QQQ score correlation: **{spy_qqq_corr:.2f}** 
  ({'very high — move together' if spy_qqq_corr > 0.8 else 'moderate — some divergence'})
            """)
        if "GLD" in score_pivot and "SPY" in score_pivot:
            gld_spy_corr = score_pivot["GLD"].corr(score_pivot["SPY"])
            st.markdown(f"""
- GLD ↔ SPY score correlation: **{gld_spy_corr:.2f}**
  ({'positive — moved together' if gld_spy_corr > 0.3 else 'low/negative — gold provided diversification'})
            """)

        st.markdown("**4. Why AAPL and BTC show fewer consensus anomalies**")
        st.markdown("""
This is expected behaviour, not a model bug:
- **AAPL** — Prophet fits wide confidence bands to AAPL's smooth uptrend, 
  so prices rarely break out of the expected range even on volatile days.
  Fix: lower `changepoint_prior_scale` in Prophet for AAPL to make bands tighter.
- **BTC-USD** — the opposite problem: Prophet flags *too many* days (173) because 
  crypto trends change rapidly. The IF model's 2% threshold catches only ~51 days.
  Overlap is small. Fix: raise `contamination` to 0.03–0.04 for BTC.
  
Both are tunable per-asset — a natural next iteration for client engagements.
        """)

    with fr:
        st.markdown("**Risk summary**")
        for ticker in selected_tickers:
            t_df = port[port["ticker"] == ticker]
            if t_df.empty:
                continue
            n_c  = int(t_df["consensus_anomaly"].sum())
            rate = 100 * n_c / max(len(t_df), 1)
            vol  = t_df["Close"].pct_change().std() * 100
            if rate > 5 or vol > 2.5:
                st.warning(f"⚠️ **{ticker}** — elevated anomaly rate ({rate:.1f}%) or high vol ({vol:.2f}%)")
            elif rate > 2:
                st.info(f"ℹ️ **{ticker}** — moderate activity ({rate:.1f}%)")
            else:
                st.success(f"✅ **{ticker}** — stable ({rate:.2f}% anomaly rate)")

        st.divider()
        st.markdown("""
**Reading the heatmap**

Months where 3+ tickers show red simultaneously = macro-level stress event. 
Check those months in the Deep Dive tab for individual price behaviour.

**Model agreement guide**
- >60% agreement → genuine stress event
- 20–60% → model sees different types of risk  
- <20% → may need threshold tuning
        """)

        st.divider()
        st.markdown("""
**Next tuning steps**
1. Lower Prophet `changepoint_prior_scale` for AAPL
2. Raise `contamination` to 0.03 for BTC-USD
3. Add sector ETF benchmarks (XLF, XLK, XLE)
4. Connect email alerts to Airflow DAG
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data refreshes every 5 min from cached CSV. "
    "Use sidebar IF threshold to tune the Deep Dive tab live. "
    "Run `python src/detect.py` to refresh all model scores."
)