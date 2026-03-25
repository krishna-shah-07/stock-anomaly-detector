# src/detect.py

import pandas as pd
import numpy as np
import os
import joblib
from dotenv import load_dotenv

from ingestion import load_data, TICKERS
from features  import build_features, get_model_features
from model     import (train_isolation_forest, load_isolation_forest,
                       score_isolation_forest, train_prophet, score_prophet)

load_dotenv()

ANOMALY_PERCENTILE = float(os.getenv("ANOMALY_PERCENTILE", "0.02"))
MODEL_PATH         = os.getenv("MODEL_PATH",   "models")
RESULTS_PATH       = os.getenv("RESULTS_PATH", "data/results")


def run_single(ticker: str, retrain: bool = False) -> pd.DataFrame:
    """
    Full pipeline for one ticker.
    Models are saved per-ticker so SPY and QQQ don't share weights.
    """
    ticker_model_path = os.path.join(MODEL_PATH, ticker)
    os.makedirs(ticker_model_path, exist_ok=True)
    os.makedirs(RESULTS_PATH, exist_ok=True)

    # 1. Load + feature engineer
    df     = load_data(ticker)
    feat   = build_features(df)
    X      = get_model_features(feat)

    # 2. Isolation Forest (per-ticker model)
    if_path = os.path.join(ticker_model_path, "isolation_forest.pkl")
    sc_path = os.path.join(ticker_model_path, "scaler.pkl")
    if retrain or not os.path.exists(if_path):
        model_if, scaler = train_isolation_forest(X, model_path=ticker_model_path)
    else:
        model_if = joblib.load(if_path)
        scaler   = joblib.load(sc_path)

    if_scores = score_isolation_forest(X, model_if, scaler)

    # 3. Dynamic threshold — 2nd percentile of THIS ticker's scores
    threshold = float(if_scores.quantile(ANOMALY_PERCENTILE))

    # 4. Prophet (per-ticker model)
    pr_path = os.path.join(ticker_model_path, "prophet.pkl")
    if retrain or not os.path.exists(pr_path):
        model_p = train_prophet(df, model_path=ticker_model_path)
    else:
        model_p = joblib.load(pr_path)

    prophet_res = score_prophet(df, model_p)

    # 5. Combine
    results = feat.copy()
    results["ticker"]          = ticker
    results["if_score"]        = if_scores
    results["if_anomaly"]      = (if_scores <= threshold).astype(int)
    results = results.join(prophet_res, how="left")
    results["consensus_anomaly"] = (
        (results["if_anomaly"] == 1) & (results["prophet_anomaly"] == 1)
    ).astype(int)

    # 6. Save per-ticker results
    out = os.path.join(RESULTS_PATH, f"{ticker}_results.csv")
    results.to_csv(out)

    n_if   = int(results["if_anomaly"].sum())
    n_prop = int(results["prophet_anomaly"].sum())
    n_cons = int(results["consensus_anomaly"].sum())
    print(f"  {ticker}: IF={n_if}  Prophet={n_prop}  Consensus={n_cons}  threshold={threshold:.4f}")
    return results


def run_all(tickers: list = TICKERS, retrain: bool = False) -> pd.DataFrame:
    """
    Run detection on all tickers and save a combined portfolio CSV.
    Returns the combined DataFrame.
    """
    print(f"\nRunning detection on: {tickers}")
    all_results = []
    failed      = []

    for ticker in tickers:
        try:
            df = run_single(ticker, retrain=retrain)
            all_results.append(df)
        except Exception as e:
            print(f"  ✗ {ticker} failed: {e}")
            failed.append(ticker)

    if not all_results:
        raise RuntimeError("All tickers failed. Check your data files.")

    combined = pd.concat(all_results)
    combined.to_csv(os.path.join(RESULTS_PATH, "portfolio_results.csv"))
    print(f"\nDone. Portfolio saved. Failed: {failed if failed else 'none'}")
    return combined


if __name__ == "__main__":
    # Fetch fresh data then run detection on all tickers
    from ingestion import fetch_all
    fetch_all()
    run_all(retrain=True)