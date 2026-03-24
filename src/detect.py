# src/detect.py

import pandas as pd
import os
from dotenv import load_dotenv

from ingestion import load_data
from features  import build_features, get_model_features
from model     import (train_isolation_forest, load_isolation_forest,
                       score_isolation_forest, train_prophet, score_prophet)

load_dotenv()
THRESHOLD  = float(os.getenv("ANOMALY_THRESHOLD", "-0.15"))
MODEL_PATH = os.getenv("MODEL_PATH", "models")


def run_detection(retrain: bool = False) -> pd.DataFrame:
    """
    Full pipeline: load data → features → score → combine → return results.
    retrain=True: retrain both models from scratch.
    retrain=False: load saved models (faster, for daily runs).
    """
    # 1. Load data
    df = load_data()

    # 2. Feature engineering
    feat_df    = build_features(df)
    X          = get_model_features(feat_df)

    # 3. Isolation Forest
    if retrain or not os.path.exists(os.path.join(MODEL_PATH, "isolation_forest.pkl")):
        model_if, scaler = train_isolation_forest(X)
    else:
        model_if, scaler = load_isolation_forest()

    if_scores = score_isolation_forest(X, model_if, scaler)

    # 4. Prophet
    if retrain or not os.path.exists(os.path.join(MODEL_PATH, "prophet.pkl")):
        model_p = train_prophet(df)
    else:
        import joblib
        model_p = joblib.load(os.path.join(MODEL_PATH, "prophet.pkl"))

    prophet_results = score_prophet(df, model_p)

    # 5. Combine results
    results = feat_df.copy()
    results["if_score"]        = if_scores
    threshold_2pct = if_scores.quantile(0.02)
    results["if_anomaly"] = (if_scores <= threshold_2pct).astype(int)
    print(f"IF dynamic threshold (2nd percentile): {threshold_2pct:.4f}")
    results = results.join(prophet_results, how="left")

    # 6. Consensus anomaly — flagged by BOTH models
    results["consensus_anomaly"] = (
        (results["if_anomaly"] == 1) & (results["prophet_anomaly"] == 1)
    ).astype(int)

    # Save results
    out_path = "data/results.csv"
    results.to_csv(out_path)
    print(f"Detection complete. Results saved to {out_path}")
    print(f"Total anomalies (IF): {results['if_anomaly'].sum()}")
    print(f"Total anomalies (Prophet): {results['prophet_anomaly'].sum()}")
    print(f"Consensus anomalies: {results['consensus_anomaly'].sum()}")

    return results


if __name__ == "__main__":
    results = run_detection(retrain=True)
    print(results[results["consensus_anomaly"] == 1][["Close", "if_score", "residual"]].tail(20))