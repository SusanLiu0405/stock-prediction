# Stock Predictor

## Project Overview
A fully automated next-day close price prediction system for the top 25 S&P 500 tickers by market cap.
- Top 25 tickers determined daily by market cap via yfinance `ticker.info["marketCap"]`
- Next-day close price prediction using **Amazon Chronos-2** (zero-shot, no per-ticker training)
- News sentiment via yfinance + Alpha Vantage (25 req/day free tier — matches top 25 exactly)
- Results displayed on a read-only static frontend
- No user-facing compute — frontend only reads pre-generated JSON from S3

## Tech Stack
- **Forecasting**: Amazon Chronos-2 via SageMaker Serverless Inference (zero-shot, no training needed)
- **Data**: yfinance (prices + news + market cap), Alpha Vantage (sentiment)
- **Cloud**: AWS — Lambda, SageMaker, S3, EventBridge
- **Frontend**: React.js, reads JSON directly from S3

## Directory Structure
```
stock-predictor/
├── CLAUDE.md
├── inference/
│   └── lambda_handler.py     # Triggered daily — ranks top 25 by market cap, fetches prices + news, calls Chronos endpoint, writes JSON to S3
├── frontend/
│   └── index.html            # Static page — reads prediction JSON from S3, no backend
└── infra/
    └── setup.py              # Creates AWS resources (S3 bucket, SageMaker endpoint, Lambda, EventBridge)
```

## Why Chronos-2 (not custom LSTM)
- **Zero-shot**: one model predicts all 25 tickers — no per-ticker training, no retraining pipeline
- **No EC2 needed**: removed entirely, SageMaker handles inference
- **Better accuracy**: outperforms task-specific deep learning models on benchmarks
- **SageMaker Serverless**: scales to zero when idle, no charges between daily runs

## S3 Path Conventions
- Predictions: `s3://stock-predictor-bucket/predictions/{ticker}/latest.json`
- News:        `s3://stock-predictor-bucket/news/{ticker}/{YYYY-MM-DD}.json`

## AWS Resource Naming
All resources prefixed with `stock-predictor-`
- S3 bucket:        `stock-predictor-bucket`
- Lambda:           `stock-predictor-daily-inference`
- SageMaker:        `stock-predictor-chronos-endpoint` (Serverless, model: `amazon/chronos-2`)
- EventBridge:      `stock-predictor-daily-trigger`

## Scheduling
- **Daily** (6:30 PM ET, after market close): EventBridge → Lambda
  1. Fetch market cap for all S&P 500 tickers via yfinance, rank and select Top 25
  2. Fetch last 500 trading days of close prices for the Top 25 tickers via yfinance
  3. Fetch today's news + Alpha Vantage sentiment for the Top 25 tickers
  4. Call SageMaker Chronos endpoint → get next-day predicted close for each of the Top 25
  5. Write prediction JSON to S3 per ticker

## News & Sentiment

Data sources (fetched daily in Lambda, for Top 25 tickers only):
- yfinance `ticker.news` → recent news headlines
- Alpha Vantage sentiment API → sentiment score (free tier: 25 req/day — matches top 25 exactly)

API keys stored in AWS Secrets Manager. You can use get_secret in setup.py to fetch it.
Lambda fetches both, stores raw news to S3,
includes top 3 news items + sentiment score in prediction JSON


## Inference Details

Input per ticker:
- Last 500 trading days of close prices (fetched via yfinance)
- Formatted as a time series array, chronological order

Chronos call:
- Model: amazon/chronos-2 via SageMaker Serverless endpoint
- prediction_length: 1 (next trading day only)
- Output: predicted close price for next trading day

Expected output per ticker:
- Single float value → predicted_close in JSON


## Prediction JSON Schema(example)
```json
{
  "ticker": "AAPL",
  "date": "2026-01-28",
  "open": 272.26,
  "previous_close": 271.01,
  "predicted_close": 258.34,
  "news": [
    {
      "title": "Apple reports record quarterly revenue",
      "sentiment": 0.82,
      "url": "https://..."
    }
  ]
}
```

## Frontend

Grid layout showing the Top 25 S&P 500 tickers by market cap. Each row displays:
- **Ticker** + company name
- **Open**: today's open price
- **Previous close**: last trading day's close
- **Predicted close**: next-day close from Chronos
  - If today is a trading day → show predicted close for today
  - If today is not a trading day (weekend/holiday) → show predicted close for next trading day

Search bar with fuzzy search (use [Fuse.js](https://fusejs.io/)):
- Default: show all 25 tickers
- On input: fuzzy match against ticker symbol and company name, show a dropdown list of matched items, highlight the hovered item
- No results state: show "No tickers found"
- If any price field is null, display "--"

## Key Constraints
- Frontend is fully static — no API gateway, no Lambda invocation from browser
- SageMaker endpoint uses Serverless Inference — no idle charges between daily runs
- Lambda only does inference + data fetching — no model training anywhere in this system
- No OpenSearch / vector DB — news stored per-ticker in S3, no semantic search needed
- No EC2 — removed entirely, Chronos requires no retraining
- Lambda reserved concurrency = 1 — prevents duplicate runs if EventBridge fires twice
- EventBridge rule must have a dead-letter queue (DLQ) configured — failed invocations do not retry infinitely
- S3 bucket must have CORS enabled for GET requests from https://你的用户名.github.io
- Alpha Vantage free tier (25 req/day) is the binding constraint — Top 25 scope is chosen to match this exactly

## Development Phases

Build in order. Do not proceed to the next phase until the current one is verified working.

### Phase 1 — Static Frontend (no AWS)
**Scope**: `frontend/index.html` only  
**Goal**: UI is fully functional with hardcoded fake data  
- Hardcode 3–5 fake ticker rows directly in the HTML (no S3 reads)
- Grid layout, all columns displayed correctly
- Fuse.js fuzzy search working (ticker symbol + company name)
- Null fields display "--"
- Verify on localhost only

### Phase 2 — S3 Integration (no Chronos)
**Scope**: `infra/setup.py` + `frontend/index.html`  
**Goal**: Frontend reads real JSON from S3  
- Run `infra/setup.py` to create S3 bucket with correct CORS config
- Manually write 3–5 real prediction JSON files (use real yfinance prices, set `predicted_close` to null)
- Upload to S3 under correct paths
- Update frontend to fetch from S3 instead of hardcoded data
- Verify CORS is working from localhost and from https://SusanLiu0405.github.io

### Phase 3 — Lambda + Chronos (single ticker)
**Scope**: `inference/lambda_handler.py` + SageMaker endpoint  
**Goal**: Full inference pipeline works end-to-end for one ticker  
- Run `infra/setup.py` to create SageMaker endpoint, Lambda, Secrets Manager entry
- Run `lambda_handler.py` locally for a single ticker (e.g. AAPL)
- Verify: yfinance prices fetched → Chronos returns predicted_close → JSON written to S3
- Verify frontend displays the real predicted value

### Phase 4 — Full Pipeline (Top 25 + EventBridge)
**Scope**: All components  
**Goal**: Fully automated daily runs for all Top 25 tickers  
- Extend Lambda to rank Top 25 by market cap daily
- Add Alpha Vantage sentiment calls (25 req/day)
- Run `infra/setup.py` to create EventBridge rule + DLQ
- Trigger manually once and verify all 25 JSON files land in S3 correctly
- Deploy frontend to GitHub Pages at https://SusanLiu0405.github.io/stock-predictor/