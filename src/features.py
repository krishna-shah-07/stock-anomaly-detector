# src/features.py

import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features from raw OHLCV data.
    These features are what the Isolation Forest model sees.
    """
    feat = df.copy()

    # --- Price-based features ---
    feat["daily_return"]    = feat["Close"].pct_change()
    feat["log_return"]      = np.log(feat["Close"] / feat["Close"].shift(1))
    feat["price_range"]     = (feat["High"] - feat["Low"]) / feat["Close"]  # intraday volatility

    # --- Rolling volatility (20-day window) ---
    feat["volatility_20d"]  = feat["daily_return"].rolling(20).std()

    # --- Volume features ---
    feat["volume_change"]   = feat["Volume"].pct_change()
    feat["volume_zscore"]   = (
        (feat["Volume"] - feat["Volume"].rolling(20).mean())
        / feat["Volume"].rolling(20).std()
    )

    # --- Moving average deviation ---
    feat["ma_20"]           = feat["Close"].rolling(20).mean()
    feat["ma_50"]           = feat["Close"].rolling(50).mean()
    feat["price_vs_ma20"]   = (feat["Close"] - feat["ma_20"]) / feat["ma_20"]
    feat["price_vs_ma50"]   = (feat["Close"] - feat["ma_50"]) / feat["ma_50"]

    # --- Momentum ---
    feat["momentum_5d"]     = feat["Close"].pct_change(5)
    feat["momentum_20d"]    = feat["Close"].pct_change(20)

    # Drop rows with NaN (result of rolling windows)
    feat.dropna(inplace=True)

    print(f"Features built: {feat.shape[0]} rows, {feat.shape[1]} columns")
    return feat


def get_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return only the columns the Isolation Forest model trains on.
    Keeping this separate makes it easy to add/remove features later.
    """
    feature_cols = [
        "daily_return",
        "log_return",
        "price_range",
        "volatility_20d",
        "volume_change",
        "volume_zscore",
        "price_vs_ma20",
        "price_vs_ma50",
        "momentum_5d",
        "momentum_20d",
    ]
    return df[feature_cols]