# src/model.py

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from prophet import Prophet
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "models")


# ── Isolation Forest ─────────────────────────────────────────────────────────

def train_isolation_forest(X: pd.DataFrame,
                            model_path: str = None,
                            contamination: float = 0.01) -> tuple:  # 1% not 5%
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "models")
    os.makedirs(model_path, exist_ok=True)

    model = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    joblib.dump(model, os.path.join(model_path, "isolation_forest.pkl"))
    joblib.dump(scaler, os.path.join(model_path, "scaler.pkl"))
    print("Isolation Forest trained and saved.")
    return model, scaler


def load_isolation_forest(model_path: str = None) -> tuple:
    """Load saved model and scaler from disk."""
    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "models")
    model = joblib.load(os.path.join(model_path, "isolation_forest.pkl"))
    scaler = joblib.load(os.path.join(model_path, "scaler.pkl"))
    return model, scaler


def score_isolation_forest(X: pd.DataFrame,
                            model, scaler) -> pd.Series:
    """
    Returns anomaly scores. More negative = more anomalous.
    Scores below -0.1 are typically strong signals.
    """
    X_scaled = scaler.transform(X)
    scores = model.score_samples(X_scaled)   # raw scores (negative)
    return pd.Series(scores, index=X.index, name="if_score")


# ── Prophet ──────────────────────────────────────────────────────────────────

def train_prophet(df: pd.DataFrame, model_path: str = None) -> Prophet:
    """
    Train Prophet on Close price to learn trend + weekly + yearly seasonality.
    Anomalies = dates where actual price deviates significantly from forecast.
    """
    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]

    if model_path is None:
        model_path = os.getenv("MODEL_PATH", "models")
    os.makedirs(model_path, exist_ok=True)

    # flatten timezone if present
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,  # lower = less flexible trend (prevents overfitting)
        interval_width=0.95
    )
    model.fit(prophet_df)

    joblib.dump(model, os.path.join(model_path, "prophet.pkl"))
    print("Prophet trained and saved.")
    return model


def score_prophet(df: pd.DataFrame, model: Prophet) -> pd.DataFrame:
    """
    Returns forecast vs actual with residual-based anomaly scores.
    Large residuals (outside confidence band) = anomalies.
    """
    prophet_df = df[["Close"]].reset_index()
    prophet_df.columns = ["ds", "y"]
    prophet_df["ds"] = pd.to_datetime(prophet_df["ds"]).dt.tz_localize(None)

    forecast = model.predict(prophet_df[["ds"]])

    result = prophet_df.copy()
    result["yhat"]       = forecast["yhat"].values
    result["yhat_lower"] = forecast["yhat_lower"].values
    result["yhat_upper"] = forecast["yhat_upper"].values
    result["residual"]   = result["y"] - result["yhat"]
    result["prophet_anomaly"] = (
        (result["y"] < result["yhat_lower"]) |
        (result["y"] > result["yhat_upper"])
    ).astype(int)

    result.set_index("ds", inplace=True)
    result.index.name = "Date"
    return result[["yhat", "yhat_lower", "yhat_upper", "residual", "prophet_anomaly"]]