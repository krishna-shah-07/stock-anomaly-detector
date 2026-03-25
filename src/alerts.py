# src/alerts.py
# Two alert modes:
#   1. check_saved_results()  — reads consensus_anomaly from saved CSVs (for Airflow DAG)
#   2. check_live_scores()    — scores tickers live and alerts on IF anomaly alone
#                               (used when you haven't re-run detect.py today)
#   run_alerts() automatically picks the right mode.

import smtplib
import os
import sys
import joblib
import pandas as pd
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from datetime import date, timedelta

load_dotenv()

SMTP_HOST    = os.getenv("SMTP_HOST",  "smtp.gmail.com")
SMTP_PORT    = int(os.getenv("SMTP_PORT", 587))
FROM_EMAIL   = os.getenv("ALERT_EMAIL_FROM", "")
TO_EMAIL     = os.getenv("ALERT_EMAIL_TO",   "")
PASSWORD     = os.getenv("ALERT_EMAIL_PASSWORD", "")
RESULTS_PATH = os.getenv("RESULTS_PATH", "data/results")
TICKERS      = [t.strip() for t in os.getenv("TICKERS", "SPY").split(",")]

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))


# ── Mode 1: check saved CSV results (Airflow / post-detect.py) ─────────────────
def check_saved_results(lookback_days: int = 1) -> dict:
    """
    Read consensus_anomaly from saved results CSVs.
    Use this after running detect.py — both models must agree.
    Returns {ticker: DataFrame of flagged rows}.
    """
    cutoff  = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    flagged = {}

    for ticker in TICKERS:
        path = os.path.join(RESULTS_PATH, f"{ticker}_results.csv")
        if not os.path.exists(path):
            continue
        df     = pd.read_csv(path, index_col="Date", parse_dates=True)
        recent = df[df.index >= cutoff]
        hits   = recent[recent["consensus_anomaly"] == 1]
        if not hits.empty:
            flagged[ticker] = hits[["Close", "if_score", "residual"]].copy()

    return flagged


# ── Mode 2: check live scores (IF only, no Prophet refit needed) ───────────────
def check_live_scores() -> dict:
    """
    Fetch latest prices and score with saved IF models right now.
    Alerts on IF anomaly alone — no need for consensus since Prophet
    can't be run live without retraining.
    Returns {ticker: DataFrame of flagged rows}.
    """
    from features import build_features, get_model_features
    import yfinance as yf

    MODEL_PATH = os.getenv("MODEL_PATH", "models")
    flagged    = {}

    for ticker in TICKERS:
        try:
            # Load model
            ticker_path = os.path.join(MODEL_PATH, ticker)
            if_path     = os.path.join(ticker_path, "isolation_forest.pkl")
            sc_path     = os.path.join(ticker_path, "scaler.pkl")
            if not os.path.exists(if_path):
                continue

            model  = joblib.load(if_path)
            scaler = joblib.load(sc_path)

            # Load threshold from saved results
            res_path = os.path.join(RESULTS_PATH, f"{ticker}_results.csv")
            threshold = -0.60
            if os.path.exists(res_path):
                saved = pd.read_csv(res_path, index_col="Date", parse_dates=True)
                threshold = float(saved["if_score"].quantile(0.02))

            # Fetch live data
            df = yf.download(ticker, period="90d", interval="1d",
                             auto_adjust=True, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df.index.name = "Date"

            feat = build_features(df)
            if feat.empty:
                continue

            X      = get_model_features(feat)
            X_sc   = scaler.transform(X)
            scores = model.score_samples(X_sc)

            feat["if_score"] = scores
            latest = feat.iloc[[-1]]  # just today

            if float(latest["if_score"].iloc[0]) <= threshold:
                flagged[ticker] = latest[["Close", "if_score"]].copy()
                flagged[ticker]["residual"] = 0.0   # Prophet not run live
                flagged[ticker]["source"]   = "live_IF_only"

        except Exception as e:
            print(f"  Live score failed for {ticker}: {e}")

    return flagged


# ── Auto-mode selector ──────────────────────────────────────────────────────────
def check_anomalies(lookback_days: int = 1, force_live: bool = False) -> dict:
    """
    Smart selector:
    - If today's results exist in CSVs → use saved consensus (both models)
    - Otherwise → fall back to live IF scoring
    - force_live=True → always use live IF scoring
    """
    if force_live:
        print("Mode: live IF scoring")
        return check_live_scores()

    # Check if saved results are fresh (written today or yesterday)
    cutoff      = pd.Timestamp(date.today() - timedelta(days=lookback_days))
    port_path   = os.path.join(RESULTS_PATH, "portfolio_results.csv")
    results_fresh = False

    if os.path.exists(port_path):
        df   = pd.read_csv(port_path, index_col="Date", parse_dates=True)
        last = df.index.max()
        if last >= cutoff:
            results_fresh = True

    if results_fresh:
        print("Mode: saved consensus results (both models)")
        return check_saved_results(lookback_days)
    else:
        print("Mode: live IF scoring (saved results are stale — run detect.py to update)")
        return check_live_scores()


# ── Email builder ───────────────────────────────────────────────────────────────
def build_email(flagged: dict) -> str:
    rows = ""
    for ticker, df in flagged.items():
        source = df.get("source", pd.Series(["consensus"])).iloc[0] \
                 if "source" in df.columns else "consensus"
        badge  = (
            '<span style="background:#fef3c7;color:#92400e;padding:2px 6px;'
            'border-radius:4px;font-size:11px">IF only</span>'
            if source == "live_IF_only" else
            '<span style="background:#dcfce7;color:#166534;padding:2px 6px;'
            'border-radius:4px;font-size:11px">consensus</span>'
        )
        for date_idx, row in df.iterrows():
            direction = "📉 Below trend" if row.get("residual", 0) < 0 else "📈 Above trend"
            rows += f"""
            <tr>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;
                           font-weight:600;color:#1e40af">{ticker}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
                    {date_idx.strftime('%b %d, %Y') if hasattr(date_idx,'strftime') else date_idx}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
                    ${row['Close']:.2f}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb;
                           color:#dc2626">{row['if_score']:.4f}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">
                    {direction}</td>
                <td style="padding:8px 12px;border-bottom:1px solid #e5e7eb">{badge}</td>
            </tr>"""

    return f"""
    <html><body style="font-family:Arial,sans-serif;color:#1f2937;max-width:660px;margin:auto">
      <div style="background:#1e40af;padding:20px 24px;border-radius:8px 8px 0 0">
        <h2 style="color:white;margin:0">📈 Anomaly Alert — Stock Monitor</h2>
        <p style="color:#bfdbfe;margin:4px 0 0">{len(flagged)} ticker(s) flagged</p>
      </div>
      <div style="background:#f8fafc;padding:20px 24px;border:1px solid #e5e7eb;
                  border-top:none;border-radius:0 0 8px 8px">
        <table style="width:100%;border-collapse:collapse;font-size:14px">
          <thead>
            <tr style="background:#f1f5f9">
              <th style="padding:8px 12px;text-align:left;color:#64748b">Ticker</th>
              <th style="padding:8px 12px;text-align:left;color:#64748b">Date</th>
              <th style="padding:8px 12px;text-align:left;color:#64748b">Close</th>
              <th style="padding:8px 12px;text-align:left;color:#64748b">IF Score</th>
              <th style="padding:8px 12px;text-align:left;color:#64748b">Direction</th>
              <th style="padding:8px 12px;text-align:left;color:#64748b">Source</th>
            </tr>
          </thead>
          <tbody>{rows}</tbody>
        </table>
        <p style="margin-top:16px;font-size:12px;color:#94a3b8">
          <b>consensus</b> = Isolation Forest + Prophet both flagged this day (high confidence)<br>
          <b>IF only</b> = Live score, Prophet not refit — run detect.py for full consensus
        </p>
      </div>
    </body></html>"""


# ── Email sender ────────────────────────────────────────────────────────────────
def send_alert(flagged: dict) -> bool:
    if not all([FROM_EMAIL, TO_EMAIL, PASSWORD]):
        print("Email config missing in .env — skipping.")
        return False

    n_consensus = sum(
        1 for df in flagged.values()
        if "source" not in df.columns or df["source"].iloc[0] != "live_IF_only"
    )
    n_live = len(flagged) - n_consensus
    subject_tag = "consensus" if n_consensus > 0 and n_live == 0 \
                  else "IF only" if n_consensus == 0 \
                  else "mixed"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"🚨 Anomaly Alert ({subject_tag}) — {len(flagged)} ticker(s)"
    msg["From"]    = FROM_EMAIL
    msg["To"]      = TO_EMAIL
    msg.attach(MIMEText(build_email(flagged), "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.login(FROM_EMAIL, PASSWORD)
            server.sendmail(FROM_EMAIL, TO_EMAIL, msg.as_string())
        print(f"✓ Alert sent to {TO_EMAIL}  [{subject_tag}]")
        return True
    except Exception as e:
        print(f"✗ Alert failed: {e}")
        return False


# ── Main entry point ────────────────────────────────────────────────────────────
def run_alerts(lookback_days: int = 1, force_live: bool = False) -> None:
    print("Checking for anomalies...")
    flagged = check_anomalies(lookback_days=lookback_days, force_live=force_live)

    if flagged:
        print(f"Found anomalies in: {list(flagged.keys())}")
        send_alert(flagged)
    else:
        print("No anomalies detected. No alert sent.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Anomaly alert system")
    parser.add_argument("--live",     action="store_true",
                        help="Force live IF scoring instead of saved results")
    parser.add_argument("--lookback", type=int, default=1,
                        help="Days to look back in saved results (default: 1)")
    args = parser.parse_args()

    run_alerts(lookback_days=args.lookback, force_live=args.live)