# src/ingestion.py

import yfinance as yf
import pandas as pd
import os
from dotenv import load_dotenv

load_dotenv()

TICKER     = os.getenv("TICKER", "SPY")
START_DATE = os.getenv("START_DATE", "2018-01-01")
END_DATE   = os.getenv("END_DATE", "2025-01-01")
# Get path relative to project root, not current working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.getenv("DATA_PATH", os.path.join(PROJECT_ROOT, "data", "raw"))


def fetch_data(ticker: str = TICKER,
               start: str = START_DATE,
               end: str = END_DATE) -> pd.DataFrame:
    """
    Download OHLCV data from Yahoo Finance.
    Returns a clean DataFrame indexed by date.
    """
    print(f"Fetching {ticker} from {start} to {end}...")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)

    if df.empty:
        raise ValueError(f"No data returned for {ticker}. Check ticker and date range.")

    # Flatten MultiIndex columns if present (yfinance sometimes returns them)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index = pd.to_datetime(df.index)
    df.index.name = "Date"

    print(f"Downloaded {len(df)} rows. Columns: {list(df.columns)}")
    return df


def save_data(df: pd.DataFrame, ticker: str = TICKER) -> str:
    """Save raw data to CSV, return file path."""
    os.makedirs(DATA_PATH, exist_ok=True)
    path = os.path.join(DATA_PATH, f"{ticker}_raw.csv")
    df.to_csv(path)
    print(f"Saved to {path}")
    return path


def load_data(ticker: str = TICKER) -> pd.DataFrame:
    """Load previously saved data from CSV."""
    path = os.path.join(DATA_PATH, f"{ticker}_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No saved data found at {path}. Run fetch_data() first.")
    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    return df


if __name__ == "__main__":
    df = fetch_data()
    save_data(df)
    print(df.tail())