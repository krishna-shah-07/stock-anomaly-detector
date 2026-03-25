# src/ingestion.py

import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

TICKERS    = [t.strip() for t in os.getenv("TICKERS", "SPY").split(",")]
START_DATE = os.getenv("START_DATE", "2018-01-01")
END_DATE   = os.getenv("END_DATE",   "2025-01-01")
DATA_PATH  = os.getenv("DATA_PATH",  "data/raw")


def fetch_data(ticker: str,
               start: str = START_DATE,
               end: str = END_DATE) -> pd.DataFrame:
    print(f"  Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}.")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"
    return df


def save_data(df: pd.DataFrame, ticker: str) -> str:
    os.makedirs(DATA_PATH, exist_ok=True)
    path = os.path.join(DATA_PATH, f"{ticker}_raw.csv")
    df.to_csv(path)
    return path


def load_data(ticker: str) -> pd.DataFrame:
    path = os.path.join(DATA_PATH, f"{ticker}_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved data for {ticker}. Run fetch_data() first."
        )
    return pd.read_csv(path, index_col="Date", parse_dates=True)


def fetch_all(tickers: list = TICKERS) -> dict:
    """Fetch and save data for all tickers. Returns dict of {ticker: df}."""
    print(f"\nFetching {len(tickers)} tickers: {tickers}")
    results = {}
    failed  = []
    for ticker in tickers:
        try:
            df = fetch_data(ticker)
            save_data(df, ticker)
            results[ticker] = df
            print(f"  ✓ {ticker}: {len(df)} rows")
        except Exception as e:
            print(f"  ✗ {ticker}: {e}")
            failed.append(ticker)
    if failed:
        print(f"\nFailed tickers: {failed}")
    return results


if __name__ == "__main__":
    fetch_all()