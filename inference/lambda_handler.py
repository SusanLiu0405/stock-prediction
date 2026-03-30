"""
inference/lambda_handler.py
---------------------------
Daily Lambda entry point. Triggered by EventBridge at 6:30 PM ET (after market close).

Steps:
  1. Rank all S&P 500 tickers by market cap via yfinance, select Top 25
  2. Fetch last 500 trading days of close prices for each ticker
  3. Fetch news (yfinance) + sentiment (Alpha Vantage) for each ticker
  4. Call SageMaker Chronos-2 endpoint to get next-day predicted close
  5. Write prediction JSON to S3

Environment variables (set by setup.py):
  S3_BUCKET   - destination bucket name
  SM_ENDPOINT - SageMaker endpoint name
  REGION      - AWS region
"""

import os
import json
import time
import logging
import datetime
import boto3
import yfinance as yf
import requests
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REGION      = os.environ.get("REGION", "us-east-1")
S3_BUCKET   = os.environ.get("S3_BUCKET", "stock-predictor-863084987436")
SM_ENDPOINT = os.environ.get("SM_ENDPOINT", "stock-predictor-chronos-endpoint")
SECRET_NAME = "stock-predictor/alpha-vantage-key"
TOP_N       = 25
PRICE_DAYS  = 500   # trading days of history sent to Chronos

# Broad S&P 500 universe — yfinance will rank by market cap at runtime
SP500_UNIVERSE = [
    "AAPL","MSFT","NVDA","AMZN","GOOGL","META","BRK-B","TSLA","LLY","V",
    "JPM","UNH","XOM","WMT","MA","JNJ","PG","COST","HD","AVGO",
    "MRK","CVX","ABBV","KO","NFLX","PEP","ADBE","CRM","ACN","TMO",
    "MCD","CSCO","ABT","BAC","DHR","ORCL","INTC","NOW","GE","AMD",
    "NEE","TXN","PM","QCOM","HON","IBM","AMGN","CAT","INTU","RTX",
]

COMPANY_NAMES = {
    "AAPL": "Apple Inc.", "MSFT": "Microsoft Corporation",
    "NVDA": "NVIDIA Corporation", "AMZN": "Amazon.com Inc.",
    "GOOGL": "Alphabet Inc.", "META": "Meta Platforms Inc.",
    "BRK-B": "Berkshire Hathaway", "TSLA": "Tesla Inc.",
    "LLY": "Eli Lilly", "V": "Visa Inc.",
    "JPM": "JPMorgan Chase", "UNH": "UnitedHealth Group",
    "XOM": "Exxon Mobil", "WMT": "Walmart Inc.",
    "MA": "Mastercard", "JNJ": "Johnson & Johnson",
    "PG": "Procter & Gamble", "COST": "Costco",
    "HD": "Home Depot", "AVGO": "Broadcom Inc.",
    "MRK": "Merck & Co.", "CVX": "Chevron Corporation",
    "ABBV": "AbbVie Inc.", "KO": "Coca-Cola",
    "NFLX": "Netflix Inc.",
}

# ---------------------------------------------------------------------------
# Secrets
# ---------------------------------------------------------------------------

def get_secret():
    """Retrieve Alpha Vantage API key from AWS Secrets Manager."""
    session = boto3.session.Session()
    client = session.client(service_name="secretsmanager", region_name=REGION)
    response = client.get_secret_value(SecretId=SECRET_NAME)
    secret = json.loads(response["SecretString"])
    # Secret stored as {"ALPHA_VANTAGE_API_KEY": "YOUR_KEY"}
    return secret["ALPHA_VANTAGE_API_KEY"]

# ---------------------------------------------------------------------------
# Step 1: Rank tickers by market cap
# ---------------------------------------------------------------------------

def get_top_n_tickers(n=TOP_N):
    """
    Fetch market cap for each ticker in SP500_UNIVERSE via yfinance,
    return the top-n sorted descending by market cap.
    """
    caps = {}
    for sym in SP500_UNIVERSE:
        try:
            info = yf.Ticker(sym).info
            cap = info.get("marketCap") or 0
            if cap > 0:
                caps[sym] = cap
        except Exception as e:
            logger.warning(f"market cap fetch failed for {sym}: {e}")

    ranked = sorted(caps, key=caps.get, reverse=True)[:n]
    logger.info(f"Top {n} tickers by market cap: {ranked}")
    return ranked

# ---------------------------------------------------------------------------
# Step 2: Price history
# ---------------------------------------------------------------------------

def get_close_prices(ticker: str, days: int = PRICE_DAYS):
    """
    Return the last `days` trading-day close prices for a ticker as a list of floats,
    chronological order (oldest first).
    """
    # Fetch extra calendar days to ensure we get enough trading days
    hist = yf.Ticker(ticker).history(period="3y")
    closes = hist["Close"].dropna().tolist()
    return closes[-days:]

# ---------------------------------------------------------------------------
# Step 3: News + sentiment
# ---------------------------------------------------------------------------

def get_news(ticker: str):
    """Return up to 3 recent news headlines from yfinance."""
    raw = yf.Ticker(ticker).news or []
    items = []
    for item in raw[:3]:
        content = item.get("content", {})
        title = content.get("title") or item.get("title", "")
        url   = (content.get("canonicalUrl") or {}).get("url") or item.get("link", "")
        if title:
            items.append({"title": title, "url": url, "sentiment": None})
    return items

def get_sentiment(ticker: str, api_key: str):
    """
    Fetch the latest sentiment score from Alpha Vantage News Sentiment API.
    Returns a float in [-1, 1] or None on failure.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=NEWS_SENTIMENT&tickers={ticker}&limit=5&apikey={api_key}"
    )
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        feed = data.get("feed", [])
        if not feed:
            return None
        # Average the ticker-specific relevance-weighted sentiment scores
        scores = []
        for article in feed:
            for ts in article.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    try:
                        scores.append(float(ts["ticker_sentiment_score"]))
                    except (KeyError, ValueError):
                        pass
        return round(sum(scores) / len(scores), 4) if scores else None
    except Exception as e:
        logger.warning(f"sentiment fetch failed for {ticker}: {e}")
        return None

# ---------------------------------------------------------------------------
# Step 4: Chronos inference
# ---------------------------------------------------------------------------

def call_chronos(prices: list, sm_client) -> float:
    """
    Send a time series to the SageMaker Chronos-2 endpoint and return
    the predicted next-day close price.

    The custom inference.py on the endpoint returns:
      {"predictions": [<median_float>, ...]}  — one value per input series.
    """
    payload = {
        "inputs": [prices],          # one series wrapped in a list
        "parameters": {
            "prediction_length": 1,
        }
    }
    response = sm_client.invoke_endpoint(
        EndpointName=SM_ENDPOINT,
        ContentType="application/json",
        Body=json.dumps(payload),
    )
    result = json.loads(response["Body"].read())
    # predictions[0] is the median forecast for the single input series
    return round(float(result["predictions"][0]), 2)

# ---------------------------------------------------------------------------
# Step 5: Write to S3
# ---------------------------------------------------------------------------

def write_to_s3(payload: dict, s3_client):
    """Upload prediction JSON to s3://{bucket}/predictions/{ticker}/latest.json"""
    ticker = payload["ticker"]
    key = f"predictions/{ticker}/latest.json"
    s3_client.put_object(
        Bucket=S3_BUCKET,
        Key=key,
        Body=json.dumps(payload, indent=2),
        ContentType="application/json",
    )
    logger.info(f"Written to s3://{S3_BUCKET}/{key}")

# ---------------------------------------------------------------------------
# Main handler
# ---------------------------------------------------------------------------

def handler(event, context):
    """Lambda entry point."""
    logger.info("Stock predictor Lambda started")
    today = datetime.date.today().isoformat()

    # Initialise AWS clients once and reuse across tickers
    s3_client = boto3.client("s3", region_name=REGION)
    sm_client = boto3.client("sagemaker-runtime", region_name=REGION)

    # Fetch Alpha Vantage key once
    try:
        av_api_key = get_secret()
    except Exception as e:
        logger.error(f"Could not retrieve secret: {e}")
        raise

    # Step 1: determine Top 25 tickers
    tickers = get_top_n_tickers(TOP_N)

    results = []
    for sym in tickers:
        logger.info(f"Processing {sym}…")
        try:
            # Step 2: price history
            prices = get_close_prices(sym)
            if len(prices) < 10:
                logger.warning(f"Not enough price history for {sym}, skipping")
                continue

            # Derive open and previous_close from the last two trading days
            hist = yf.Ticker(sym).history(period="2d")
            open_price  = round(float(hist["Open"].iloc[-1]), 2)  if len(hist) >= 1 else None
            prev_close  = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else None

            # Step 3: news + sentiment
            news        = get_news(sym)
            sentiment   = get_sentiment(sym, av_api_key)

            # Attach sentiment score to each news item (shared score for now)
            for item in news:
                item["sentiment"] = sentiment

            # Step 4: Chronos prediction
            predicted_close = call_chronos(prices, sm_client)

            # Step 5: build payload and write to S3
            payload = {
                "ticker":          sym,
                "name":            COMPANY_NAMES.get(sym, sym),
                "date":            today,
                "open":            open_price,
                "previous_close":  prev_close,
                "predicted_close": predicted_close,
                "news":            news,
            }
            write_to_s3(payload, s3_client)
            results.append(sym)

        except Exception as e:
            logger.error(f"Failed to process {sym}: {e}", exc_info=True)
            # Continue with remaining tickers rather than aborting the whole run

        # Respect Alpha Vantage free-tier rate limit (5 req/min)
        time.sleep(12)

    logger.info(f"Done. Processed {len(results)}/{len(tickers)} tickers: {results}")
    return {"statusCode": 200, "processed": results}


# ---------------------------------------------------------------------------
# Local test entrypoint — run a single ticker without EventBridge
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    # Override TOP_N to test a single ticker quickly
    test_ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"
    logger.addHandler(logging.StreamHandler())

    s3 = boto3.client("s3", region_name=REGION)
    sm = boto3.client("sagemaker-runtime", region_name=REGION)

    try:
        av_key = get_secret()
    except Exception as e:
        print(f"Secret fetch failed: {e}")
        sys.exit(1)

    prices = get_close_prices(test_ticker)
    print(f"{test_ticker}: {len(prices)} days of price history loaded")

    hist = yf.Ticker(test_ticker).history(period="2d")
    open_price = round(float(hist["Open"].iloc[-1]), 2)
    prev_close = round(float(hist["Close"].iloc[-2]), 2)

    news      = get_news(test_ticker)
    sentiment = get_sentiment(test_ticker, av_key)
    for item in news:
        item["sentiment"] = sentiment

    predicted_close = call_chronos(prices, sm)

    payload = {
        "ticker":          test_ticker,
        "name":            COMPANY_NAMES.get(test_ticker, test_ticker),
        "date":            datetime.date.today().isoformat(),
        "open":            open_price,
        "previous_close":  prev_close,
        "predicted_close": predicted_close,
        "news":            news,
    }

    print(json.dumps(payload, indent=2))
    write_to_s3(payload, s3)
    print(f"Written to S3.")
