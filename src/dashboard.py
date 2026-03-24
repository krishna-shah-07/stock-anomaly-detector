import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

# Set page config
st.set_page_config(page_title="Stock Anomaly Detector", layout="wide", initial_sidebar_state="expanded")
st.title("📈 Stock Anomaly Detection Dashboard")

# Load results with caching
@st.cache_data
def load_results():
    """Load detection results from CSV."""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(project_root, "data", "results.csv")
    if not os.path.exists(path):
        st.error(f"Results file not found at {path}. Run `python detect.py` first.")
        st.stop()
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df

df = load_results()

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Date range filter
    date_range = st.slider(
        "Select date range:",
        min_value=df.index.min().date(),
        max_value=df.index.max().date(),
        value=(df.index.min().date(), df.index.max().date()),
        help="Filter data by date range"
    )
    
    df_filtered_date = df[(df.index.date >= date_range[0]) & (df.index.date <= date_range[1])]
    
    # Anomaly type filter
    anomaly_type = st.selectbox(
        "View anomalies by:",
        ["All Data", "Isolation Forest Only", "Prophet Only", "Consensus (Both)"],
        help="Filter to specific anomaly detection methods"
    )
    
    # Dynamic threshold adjustment
    st.subheader("🎯 Threshold Control")
    default_threshold = df["if_score"].quantile(0.02)
    threshold = st.slider(
        "Isolation Forest Score Threshold:",
        min_value=float(df["if_score"].min()),
        max_value=float(df["if_score"].max()),
        value=default_threshold,
        step=0.01,
        help="Lower scores = more anomalous. Adjust to find your sweet spot."
    )

# Apply filters
def apply_filters(df, anomaly_type, date_filter=True):
    df_view = df_filtered_date if date_filter else df
    
    if anomaly_type == "Isolation Forest Only":
        return df_view[df_view["if_anomaly"] == 1]
    elif anomaly_type == "Prophet Only":
        return df_view[df_view["prophet_anomaly"] == 1]
    elif anomaly_type == "Consensus (Both)":
        return df_view[df_view["consensus_anomaly"] == 1]
    else:
        return df_view

filtered_df = apply_filters(df, anomaly_type)

# Key metrics row
st.subheader("📊 Quick Stats")
col1, col2, col3, col4, col5 = st.columns(5)
with col1:
    st.metric("📅 Trading Days", len(df_filtered_date), help="Total trading days in selected range")
with col2:
    if_count = int(df_filtered_date["if_anomaly"].sum())
    st.metric("🔴 IF Anomalies", if_count, f"{100*if_count/len(df_filtered_date):.1f}%")
with col3:
    prophet_count = int(df_filtered_date["prophet_anomaly"].sum())
    st.metric("🟠 Prophet Anomalies", prophet_count, f"{100*prophet_count/len(df_filtered_date):.1f}%")
with col4:
    consensus_count = int(df_filtered_date["consensus_anomaly"].sum())
    st.metric("✅ Consensus", consensus_count, f"{100*consensus_count/len(df_filtered_date):.1f}%")
with col5:
    price_change = ((df_filtered_date["Close"].iloc[-1] - df_filtered_date["Close"].iloc[0]) / df_filtered_date["Close"].iloc[0]) * 100
    st.metric("💹 Price Change", f"{price_change:.2f}%", help="Period return")

# Tabs for different views
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📈 Overview", "🎯 Anomalies", "📊 Analysis", "📋 Details", "📝 Findings"])

with tab1:
    st.subheader("Price & Anomalies with Dynamic Threshold")
    
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        subplot_titles=("Close Price with Anomalies", "Isolation Forest Score", "Prophet Residual"),
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25]
    )
    
    # Row 1: Price
    fig.add_trace(
        go.Scatter(x=df_filtered_date.index, y=df_filtered_date["Close"], name="Close Price", 
                   mode="lines", line=dict(color="darkblue", width=2)),
        row=1, col=1
    )
    
    # IF Anomalies (top plot)
    if_anom = df_filtered_date[df_filtered_date["if_anomaly"] == 1]
    fig.add_trace(
        go.Scatter(x=if_anom.index, y=if_anom["Close"], mode="markers", name="IF Anomaly",
                   marker=dict(color="red", size=10, symbol="diamond")),
        row=1, col=1
    )
    
    # Prophet Anomalies (top plot)
    prophet_anom = df_filtered_date[df_filtered_date["prophet_anomaly"] == 1]
    fig.add_trace(
        go.Scatter(x=prophet_anom.index, y=prophet_anom["Close"], mode="markers", name="Prophet Anomaly",
                   marker=dict(color="orange", size=7, symbol="star")),
        row=1, col=1
    )
    
    # Row 2: IF Score
    fig.add_trace(
        go.Scatter(x=df_filtered_date.index, y=df_filtered_date["if_score"], name="IF Score",
                   mode="lines", line=dict(color="green", width=2)),
        row=2, col=1
    )
    
    # Dynamic threshold line
    fig.add_hline(y=threshold, line_dash="dash", line_color="red", line_width=2,
                  annotation_text=f"Threshold: {threshold:.3f}", 
                  annotation_position="right", row=2, col=1)
    
    # Row 3: Prophet Residual
    fig.add_trace(
        go.Scatter(x=df_filtered_date.index, y=df_filtered_date["residual"], name="Prophet Residual",
                   mode="lines", line=dict(color="purple", width=2), fill="tozeroy"),
        row=3, col=1
    )
    
    fig.add_hline(y=0, line_color="gray", line_width=1, row=3, col=1)
    
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="IF Score", row=2, col=1)
    fig.update_yaxes(title_text="Residual", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_layout(height=700, hovermode="x unified", legend=dict(x=0.01, y=0.99))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("🎯 Top Anomalies")
    
    # Calculate anomaly severity (how extreme the scores are)
    df_anomalies = df_filtered_date[df_filtered_date["consensus_anomaly"] == 1].copy()
    df_anomalies["severity"] = df_anomalies["if_score"].abs()
    df_anomalies = df_anomalies.sort_values("severity", ascending=False)
    
    if len(df_anomalies) == 0:
        st.info("No consensus anomalies found in the selected range and threshold.")
    else:
        st.write(f"**Found {len(df_anomalies)} consensus anomalies** (flagged by both models)")
        
        # Top anomalies
        top_n = st.slider("Show top N anomalies:", 1, min(20, len(df_anomalies)), 10)
        
        display_cols = ["Close", "if_score", "residual", "if_anomaly", "prophet_anomaly"]
        st.dataframe(
            df_anomalies[display_cols].head(top_n).style.background_gradient(subset=["if_score"], cmap="RdYlGn_r"),
            use_container_width=True,
            height=400
        )
    
    # Anomaly timeline
    st.subheader("Anomaly Frequency Over Time")
    anomaly_freq = df_filtered_date.resample("M").agg({
        "if_anomaly": "sum",
        "prophet_anomaly": "sum",
        "consensus_anomaly": "sum"
    })
    
    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Bar(x=anomaly_freq.index, y=anomaly_freq["if_anomaly"], name="IF", marker_color="red"))
    fig_timeline.add_trace(go.Bar(x=anomaly_freq.index, y=anomaly_freq["prophet_anomaly"], name="Prophet", marker_color="orange"))
    fig_timeline.add_trace(go.Bar(x=anomaly_freq.index, y=anomaly_freq["consensus_anomaly"], name="Consensus", marker_color="green"))
    
    fig_timeline.update_layout(barmode="group", height=400, title="Anomalies per Month")
    fig_timeline.update_xaxes(title="Month")
    fig_timeline.update_yaxes(title="Count")
    st.plotly_chart(fig_timeline, use_container_width=True)

with tab3:
    st.subheader("📊 Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**IF Score Distribution**")
        fig_if = go.Figure(data=[go.Histogram(x=df_filtered_date["if_score"], nbinsx=50, name="IF Score")])
        fig_if.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text="Current Threshold")
        fig_if.update_layout(height=400, xaxis_title="IF Score", yaxis_title="Frequency")
        st.plotly_chart(fig_if, use_container_width=True)
        
        st.write(df_filtered_date["if_score"].describe().to_frame().T)
    
    with col2:
        st.write("**Prophet Residual Distribution**")
        fig_residual = go.Figure(data=[go.Histogram(x=df_filtered_date["residual"], nbinsx=50, name="Residual", marker_color="purple")])
        fig_residual.update_layout(height=400, xaxis_title="Residual", yaxis_title="Frequency")
        st.plotly_chart(fig_residual, use_container_width=True)
        
        st.write(df_filtered_date["residual"].describe().to_frame().T)
    
    # Correlation
    st.subheader("Feature Correlation")
    numeric_cols = df_filtered_date.select_dtypes(include=[np.number]).columns
    corr_matrix = df_filtered_date[numeric_cols].corr()
    
    fig_corr = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns, colorscale="RdBu"))
    fig_corr.update_layout(height=500, title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

with tab4:
    st.subheader("📋 Full Dataset")
    
    # Filtered view based on anomaly type
    display_cols = ["Close", "if_score", "residual", "if_anomaly", "prophet_anomaly", "consensus_anomaly"]
    st.dataframe(
        filtered_df[display_cols].style.background_gradient(subset=["if_score"], cmap="RdYlGn_r"),
        use_container_width=True,
        height=600
    )
    
    # Export options
    st.subheader("📥 Export")
    csv = filtered_df[display_cols].to_csv()
    st.download_button(
        label="Download as CSV",
        data=csv,
        file_name="anomalies.csv",
        mime="text/csv"
    )

with tab5:
    st.subheader("📝 Key Findings & Insights")
    
    # Calculate key metrics
    total_days = len(df_filtered_date)
    total_anomalies_if = int(df_filtered_date["if_anomaly"].sum())
    total_anomalies_prophet = int(df_filtered_date["prophet_anomaly"].sum())
    consensus = int(df_filtered_date["consensus_anomaly"].sum())
    
    # Price metrics
    start_price = df_filtered_date["Close"].iloc[0]
    end_price = df_filtered_date["Close"].iloc[-1]
    price_change_pct = ((end_price - start_price) / start_price) * 100
    max_price = df_filtered_date["Close"].max()
    min_price = df_filtered_date["Close"].min()
    volatility = df_filtered_date["daily_return"].std() * 100 if "daily_return" in df_filtered_date.columns else df_filtered_date["Close"].pct_change().std() * 100
    
    # Top anomalies
    top_anomalies = df_filtered_date[df_filtered_date["consensus_anomaly"] == 1].copy()
    top_anomalies["severity"] = top_anomalies["if_score"].abs()
    top_anomalies = top_anomalies.sort_values("severity", ascending=False)
    
    # Most volatile month
    monthly_vol = df_filtered_date.groupby(df_filtered_date.index.to_period("M"))["daily_return"].std().max() * 100 if "daily_return" in df_filtered_date.columns else df_filtered_date.groupby(df_filtered_date.index.to_period("M"))["Close"].pct_change().std().max() * 100
    
    st.write("""
    ### Summary Overview
    This section provides a plain-language explanation of the anomaly detection results 
    and what they mean for your stock analysis.
    """)
    
    # Create two columns for findings
    col1, col2 = st.columns([1.5, 1])
    
    with col1:
        st.write("### 📌 Main Findings")
        
        # Finding 1: Overall Activity
        st.write(f"""
        **1. Anomaly Detection Activity**
        
        Over the selected period of **{total_days} trading days**, our dual-model detection system flagged:
        - **{total_anomalies_if}** unusual price movements** (Isolation Forest)
        - **{total_anomalies_prophet}** trend deviations** (Prophet forecasting)
        - **{consensus} high-confidence anomalies** detected by both models
        
        This means about **{100*consensus/total_days:.1f}% of trading days** showed unusual behavior that both models agreed on.
        The consensus anomalies are the most reliable — these are the events you should pay attention to.
        """)
        
        # Finding 2: Price Performance
        st.write(f"""
        **2. Price Performance & Volatility**
        
        During this period:
        - **Price change**: {price_change_pct:+.2f}% (from ${start_price:.2f} to ${end_price:.2f})
        - **Range**: ${min_price:.2f} to ${max_price:.2f}
        - **Volatility**: {volatility:.2f}% daily standard deviation
        
        {"**High volatility** environment detected — more price swings means more anomalies." if volatility > 2.5 else "**Moderate volatility** — relatively stable trading conditions." if volatility > 1.5 else "**Low volatility** — calm market with few sudden moves."}
        """)
        
        # Finding 3: Top Events
        if len(top_anomalies) > 0:
            st.write(f"""
            **3. Top {min(5, len(top_anomalies))} Most Extreme Events**
            
            These are the trading days where BOTH models detected anomalies:
            """)
            
            for idx, (date, row) in enumerate(top_anomalies.head(5).iterrows(), 1):
                price_dir = "📈" if row.get("residual", 0) > 0 else "📉"
                st.write(f"   **{idx}. {date.strftime('%B %d, %Y')}** {price_dir}")
                st.write(f"      Price: ${row['Close']:.2f} | IF Score: {row['if_score']:.3f} | Residual: {row['residual']:.2f}")
        else:
            st.write("""
            **3. No Extreme Events Detected**
            
            No dates were flagged by both models simultaneously. The anomalies are scattered across models.
            """)
        
        # Finding 4: Model Agreement
        st.write(f"""
        **4. Model Agreement Analysis**
        
        - **Consensus rate**: {100*consensus/max(total_anomalies_if, total_anomalies_prophet):.1f}% (anomalies both models agree on)
        - **IF unique detections**: {total_anomalies_if - consensus} (extremes in feature distribution)
        - **Prophet unique detections**: {total_anomalies_prophet - consensus} (trend deviations)
        
        When models disagree, it means one detected a statistical outlier while the other saw normal trend behavior.
        Both high and low agreement rates can be informative!
        """)
    
    with col2:
        st.write("### 💡 Recommendations")
        
        if consensus > 0.10 * total_days:
            st.warning(f"""
            ⚠️ **High Anomaly Rate**
            
            {100*consensus/total_days:.1f}% consensus anomalies is elevated.
            Consider if:
            - Market is genuinely volatile
            - Threshold needs adjustment
            - Different assets needed
            """)
        elif consensus > 0.05 * total_days:
            st.info(f"""
            ℹ️ **Moderate Activity**
            
            {100*consensus/total_days:.1f}% is a healthy anomaly rate
            for active trading signals.
            """)
        else:
            st.success(f"""
            ✅ **Stable Period**
            
            {100*consensus/total_days:.1f}% - Few anomalies suggest
            stable trading with limited extreme events.
            """)
        
        st.divider()
        
        if volatility > 2.5:
            st.write("""
            🔴 **High Volatility Alert**
            
            With {:.1f}% daily volatility, expect frequent
            anomaly detections. This is normal during
            market stress periods.
            """.format(volatility))
        
        st.divider()
        
        st.write("""
        **🎯 Next Steps**
        
        1. Review the **📈 Overview** tab to visualize events
        2. Check **🎯 Anomalies** for specific dates
        3. Use sidebar to **adjust threshold** for your needs
        4. **Export data** for further analysis
        """)
    
    st.divider()
    
    # Key insights box
    st.write("### 🔍 Technical Interpretation")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            "Model Agreement",
            f"{100*consensus/max(total_anomalies_if, total_anomalies_prophet):.1f}%",
            "How often models agree"
        )
    
    with col2:
        st.metric(
            "Anomaly Concentration",
            f"{100*consensus/total_days:.2f}%",
            "% of days flagged (both)"
        )
    
    with col3:
        st.metric(
            "Avg Daily Volatility",
            f"{volatility:.2f}%",
            "Price movement std dev"
        )
    
    st.info("""
    💭 **What These Numbers Mean**
    
    - **Model Agreement**: High = consensus is reliable. Low = models have different perspectives.
    - **Anomaly Concentration**: Typical range is 1-5% for consensus. >10% suggests high volatility.
    - **Daily Volatility**: >2% = volatile market. 1-2% = normal. <1% = very stable.
    """)

# Footer
st.divider()
st.caption("💡 **Tip:** Adjust the threshold slider to tune sensitivity. Lower threshold = fewer anomalies (higher precision). Higher threshold = more anomalies (higher recall).")
