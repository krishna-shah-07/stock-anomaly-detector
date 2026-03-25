# src/live.py
# Run: python src/live.py
# Pulls latest price every N minutes and scores against saved models.
# Does NOT retrain — pure inference only.

import os
import sys
import time
import joblib
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

# ── Config ─────────────────────────────────────────────────────────────────────
TICKERS       = [t.strip() for t in os.getenv("TICKERS",       "SPY").split(",")]
MODEL_PATH    = os.getenv("MODEL_PATH",    "models")
RESULTS_PATH  = os.getenv("RESULTS_PATH", "data/results")
REFRESH_SECS  = int(os.getenv("LIVE_REFRESH_SECONDS", "300"))   # default 5 min
LOOKBACK_DAYS = 90   # needs 90 days — rolling windows go up to 50 days, dropna() removes the rest

# Add src/ to path so imports work from any working directory
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

from features import build_features, get_model_features


# ── Model loader ───────────────────────────────────────────────────────────────
def load_model(ticker: str):
    """
    Load saved Isolation Forest + scaler for a given ticker.
    Returns (model, scaler) or (None, None) if not found.
    """
    ticker_path = os.path.join(MODEL_PATH, ticker)
    if_path     = os.path.join(ticker_path, "isolation_forest.pkl")
    sc_path     = os.path.join(ticker_path, "scaler.pkl")

    if not os.path.exists(if_path) or not os.path.exists(sc_path):
        return None, None

    model  = joblib.load(if_path)
    scaler = joblib.load(sc_path)
    return model, scaler


# ── Per-ticker threshold loader ────────────────────────────────────────────────
def load_threshold(ticker: str) -> float:
    """
    Read the 2nd percentile IF score from saved results as the anomaly threshold.
    Falls back to -0.60 if results file not found.
    """
    path = os.path.join(RESULTS_PATH, f"{ticker}_results.csv")
    if not os.path.exists(path):
        return -0.60
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return float(df["if_score"].quantile(0.02))


# ── Data fetcher ───────────────────────────────────────────────────────────────
def fetch_recent(ticker: str, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """
    Pull recent daily OHLCV data for a ticker.
    We need enough history for rolling feature windows (20-50 days).
    """
    df = yf.download(
        ticker,
        period=f"{days}d",
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    return df


# ── Single ticker scorer ───────────────────────────────────────────────────────
def score_ticker(ticker: str, model, scaler, threshold: float) -> dict | None:
    """
    Score the most recent trading day for a ticker.
    Returns a result dict or None if data/model unavailable.
    """
    df = fetch_recent(ticker)
    if df.empty or len(df) < 60:   # need at least 25 rows for rolling features
        return None

    try:
        feat = build_features(df)
        if feat.empty:
            print(f"  ✗ {ticker}: features empty after dropna — fetched {len(df)} rows but rolling windows need ~90 days.")
            return None
        X    = get_model_features(feat)
        X_sc = scaler.transform(X)
        scores = model.score_samples(X_sc)

        latest_score = float(scores[-1])
        latest_row   = feat.iloc[-1]
        latest_date  = feat.index[-1]
        is_anomaly   = latest_score <= threshold

        return {
            "ticker":    ticker,
            "date":      latest_date,
            "close":     float(latest_row["Close"]),
            "if_score":  latest_score,
            "threshold": threshold,
            "anomaly":   is_anomaly,
            "daily_ret": float(latest_row.get("daily_return", 0)),
            "vol_z":     float(latest_row.get("volume_zscore", 0)),
        }
    except Exception as e:
        print(f"  ✗ {ticker}: scoring error — {e}")
        return None


# ── Pretty printer ─────────────────────────────────────────────────────────────
def print_results(results: list[dict]) -> None:
    """Print a clean table of live scores."""
    print(f"\n{'─'*72}")
    print(f"  {'Ticker':<10} {'Date':<12} {'Close':>9}  {'IF Score':>9}  "
          f"{'Threshold':>10}  {'Daily Ret':>10}  {'Status'}")
    print(f"{'─'*72}")

    anomalies = []
    for r in results:
        if r is None:
            continue
        status = "🚨 ANOMALY" if r["anomaly"] else "✓  normal"
        date_s = r["date"].strftime("%Y-%m-%d") if hasattr(r["date"], "strftime") else str(r["date"])
        print(
            f"  {r['ticker']:<10} {date_s:<12} "
            f"${r['close']:>8.2f}  "
            f"{r['if_score']:>9.4f}  "
            f"{r['threshold']:>10.4f}  "
            f"{r['daily_ret']:>+9.2%}  "
            f"{status}"
        )
        if r["anomaly"]:
            anomalies.append(r)

    print(f"{'─'*72}")

    if anomalies:
        print(f"\n  ⚠️  {len(anomalies)} anomaly detected — "
              f"tickers: {[a['ticker'] for a in anomalies]}")
        print(f"  Run `python src/alerts.py` to send an email alert.\n")
    else:
        print(f"\n  All tickers normal.\n")


# ── Model preloader ────────────────────────────────────────────────────────────
def preload_models() -> tuple[dict, dict]:
    """
    Load all models and thresholds once at startup.
    Returns (models_dict, thresholds_dict).
    """
    models     = {}
    thresholds = {}
    missing    = []

    print(f"\nLoading models for: {TICKERS}")
    for ticker in TICKERS:
        model, scaler = load_model(ticker)
        if model is None:
            print(f"  ✗ {ticker}: no saved model — run `python src/detect.py` first")
            missing.append(ticker)
        else:
            models[ticker]     = (model, scaler)
            thresholds[ticker] = load_threshold(ticker)
            print(f"  ✓ {ticker}: model loaded  (threshold={thresholds[ticker]:.4f})")

    if missing:
        print(f"\n  Skipping {missing} — no models found.")

    return models, thresholds


# ── Main loop ──────────────────────────────────────────────────────────────────
def run_live_loop(refresh_secs: int = REFRESH_SECS) -> None:
    """
    Main monitoring loop.
    Scores all tickers every `refresh_secs` seconds.
    Press Ctrl+C to stop.
    """
    models, thresholds = preload_models()

    if not models:
        print("\nNo models loaded. Exiting.")
        return

    active_tickers = list(models.keys())
    print(f"\nLive monitoring: {active_tickers}")
    print(f"Refresh interval: {refresh_secs}s  ({refresh_secs//60}m {refresh_secs%60}s)")
    print("Press Ctrl+C to stop.\n")

    cycle = 0
    while True:
        cycle += 1
        now = datetime.now().strftime("%H:%M:%S")
        print(f"[Cycle {cycle}] Scoring at {now} ...")

        results = []
        for ticker in active_tickers:
            model, scaler = models[ticker]
            threshold     = thresholds[ticker]
            result        = score_ticker(ticker, model, scaler, threshold)
            results.append(result)

        print_results([r for r in results if r is not None])

        print(f"Next refresh in {refresh_secs}s — Ctrl+C to stop.")
        try:
            time.sleep(refresh_secs)
        except KeyboardInterrupt:
            print("\nStopped by user.")
            break


# ── Quick single-shot mode ─────────────────────────────────────────────────────
def run_once() -> None:
    """Score all tickers once and exit. Useful for testing."""
    models, thresholds = preload_models()
    if not models:
        return

    print(f"\nScoring {list(models.keys())} once ...\n")
    results = []
    for ticker, (model, scaler) in models.items():
        result = score_ticker(ticker, model, scaler, thresholds[ticker])
        results.append(result)

    print_results([r for r in results if r is not None])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Live anomaly monitor")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Score once and exit (default: continuous loop)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=REFRESH_SECS,
        help=f"Refresh interval in seconds (default: {REFRESH_SECS})"
    )
    args = parser.parse_args()

    if args.once:
        run_once()
    else:
        run_live_loop(refresh_secs=args.interval)