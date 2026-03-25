# Stock Anomaly Detector

## Overview

Stock Anomaly Detector is an end-to-end pipeline for detecting unusual market behavior in historical price data. It uses a hybrid ensemble of:

- **Isolation Forest** for multivariate feature outlier detection.
- **Prophet** for time-series prediction residual anomaly detection.

It also includes an interactive Streamlit dashboard for exploration.

## Repository Structure

- `data/raw/` - raw downloaded OHLCV data from Yahoo Finance.
- `data/results.csv` - output dataset with anomaly flags and scores.
- `models/` - saved trained models (`isolation_forest.pkl`, `prophet.pkl`).
- `src/` - pipeline source code:
  - `ingestion.py` - fetches/saves/loads data.
  - `features.py` - feature engineering and model input selector.
  - `model.py` - training and scoring functions for both models.
  - `detect.py` - orchestrates the pipeline and saves results.
  - `dashboard.py` - Streamlit app with tabs: Overview, Anomalies, Analysis, Details, Findings.
- `.env` - optional configuration (TICKER, START_DATE, END_DATE, etc.)

## Dependencies

\`\`\`bash
pip install -r requirements.txt
\`\`\`

Required packages include:
- `yfinance`
- `pandas`, `numpy`
- `scikit-learn`
- `prophet`
- `plotly`, `streamlit`
- `joblib`, `python-dotenv`

## Config

Optionally set `.env`:

\`\`\`dotenv
TICKER=SPY
START_DATE=2018-01-01
END_DATE=2025-01-01
ANOMALY_THRESHOLD=-0.15
DATA_PATH=data/raw
MODEL_PATH=models
\`\`\`

## Usage

### 1) Ingest raw data

\`\`\`bash
python src/ingestion.py
\`\`\`

### 2) Run detection

\`\`\`bash
python src/detect.py
\`\`\`

### 3) Run dashboard

\`\`\`bash
streamlit run src/dashboard.py
\`\`\`

Open browser at `http://localhost:8501`.

## Workflow

1. `ingestion.fetch_data()` downloads and stores raw stock data.
2. `ingestion.load_data()` reads local CSV for downstream runs.
3. `features.build_features()` creates model-ready inputs.
4. `model.train_isolation_forest()` and `model.train_prophet()` fit models.
5. `detect.run_detection()` computes scores and consensus flags.
6. dashboard renders charts, tables, and plain-language insights.

## Anomaly definitions

- `if_score`: Isolation Forest output (lower = more anomalous).
- `if_anomaly`: binary flag from IF score threshold.
- `residual`: actual - Prophet forecast.
- `prophet_anomaly`: binary flag for large residuals.
- `consensus_anomaly`: both IF and Prophet flagged.

## Airflow-friendly setup

- Task 1: fetch and save raw data (once per interval)
- Task 2: detect anomalies with local CSV input
- Task 3: dashboard reads results file, no repeated API fetch

## Notes

- The dashboard includes a dynamic threshold slider and a fallback for `ME`/`M` period resampling.
- The `Findings` tab provides plain-language insights and recommended actions.

## Troubleshooting

- If there is `FileNotFoundError` for `data/results.csv`, run `python src/detect.py` first.
- If the Streamlit bar chart error `Invalid frequency: ME` appears, ensure your pandas supports `ME` or restart.

## Contact

For changes or issues, use the repository issue tracker.
'@
