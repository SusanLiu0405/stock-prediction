"""
infra/upload_sample_data.py
---------------------------
Phase 2 helper: fetches real yfinance prices for 5 tickers,
sets predicted_close to null, uploads to S3.

Usage:
  python infra/upload_sample_data.py
"""

import json
import boto3
import yfinance as yf
from datetime import date

BUCKET = "stock-predictor-863084987436"
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL"]

s3 = boto3.client("s3", region_name="us-east-1")
today = date.today().isoformat()

for sym in TICKERS:
    print(f"Fetching {sym}...")
    ticker = yf.Ticker(sym)
    hist = ticker.history(period="2d")

    if len(hist) < 2:
        print(f"  Not enough history for {sym}, skipping")
        continue

    open_price = round(float(hist["Open"].iloc[-1]), 2)
    prev_close = round(float(hist["Close"].iloc[-2]), 2)

    # Grab up to 3 news headlines
    raw_news = ticker.news or []
    news = []
    for item in raw_news[:3]:
        content = item.get("content", {})
        title = content.get("title", item.get("title", ""))
        url = content.get("canonicalUrl", {}).get("url", item.get("link", ""))
        if title:
            news.append({"title": title, "sentiment": None, "url": url})

    payload = {
        "ticker": sym,
        "date": today,
        "open": open_price,
        "previous_close": prev_close,
        "predicted_close": None,
        "news": news,
    }

    key = f"predictions/{sym}/latest.json"
    s3.put_object(
        Bucket=BUCKET,
        Key=key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    print(f"  Uploaded s3://{BUCKET}/{key}")

print("\nDone.")
