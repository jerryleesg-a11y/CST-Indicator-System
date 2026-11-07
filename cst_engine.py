import os
from datetime import datetime
import numpy as np
import pandas as pd
import pytz
import yfinance as yf

# File paths
PORTFOLIO_FILE = "Intrinsic-Value-Portfolio-Nov2026.txt"
OUTPUT_FILE = "CST_Portfolio_Analysis_Results.csv"


def load_portfolio(filepath):
  tickers = []
  intrinsic_values = {}

  if not os.path.exists(filepath):
    print(f"Portfolio file {filepath} not found. Using default mock values.")
    defaults = {
        "AAPL": 180.0,
        "MSFT": 400.0,
        "GOOGL": 150.0,
        "AMZN": 170.0,
        "NVDA": 120.0,
    }
    return list(defaults.keys()), defaults

  with open(filepath, "r") as f:
    for line in f:
      line = line.strip()
      if not line or line.startswith("#"):
        continue
      parts = line.split(",")
      if len(parts) >= 2:
        ticker = parts[0].strip().upper()
        try:
          iv = float(parts[1].strip())
          tickers.append(ticker)
          intrinsic_values[ticker] = iv
        except ValueError:
          continue
  return tickers, intrinsic_values


def fetch_ticker_data(ticker):
  try:
    df = yf.download(
        ticker, period="1y", interval="1d", auto_adjust=True, progress=False
    )
    if isinstance(df.columns, pd.MultiIndex):
      df.columns = df.columns.get_level_values(0)
    return df
  except Exception as e:
    print(f"Error downloading {ticker}: {e}")
    return pd.DataFrame()


def calculate_indicators(ticker, df_ticker, iv):
  if df_ticker.empty or len(df_ticker) < 30:
    return None

  close = df_ticker["Close"].dropna()
  high = df_ticker["High"].dropna()
  low = df_ticker["Low"].dropna()
  volume = df_ticker["Volume"].dropna()

  if len(close) < 30:
    return None

  current_price = float(close.iloc[-1])

  # Moving Averages
  ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
  ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
  sma200 = (
      float(close.rolling(window=200).mean().iloc[-1])
      if len(close) >= 200
      else float(close.mean())
  )

  # RSI (14)
  delta = close.diff()
  gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
  loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
  rs = gain / loss
  rsi = float(
      (100 - (100 / (1 + rs))).iloc[-1] if not loss.iloc[-1] == 0 else 100
  )

  # ATR (14)
  high_low = high - low
  high_close = np.abs(high - close.shift())
  low_close = np.abs(low - close.shift())
  tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
  atr14 = float(tr.rolling(window=14).mean().iloc[-1])
  atr_pct = (atr14 / current_price) * 100

  # Distance from EMA20 in ATR terms
  dist_ema20_atr = (current_price - ema20) / atr14 if atr14 > 0 else 0.0

  # MACD
  exp1 = close.ewm(span=12, adjust=False).mean()
  exp2 = close.ewm(span=26, adjust=False).mean()
  macd_line = exp1 - exp2
  signal_line = macd_line.ewm(span=9, adjust=False).mean()
  macd_val = float(macd_line.iloc[-1])
  macd_sig = float(signal_line.iloc[-1])

  # Volume Ratio (Current Vol vs 20D MA Vol)
  vol_ma20 = volume.rolling(window=20).mean().iloc[-1]
  vol_ratio = (
      float(volume.iloc[-1] / vol_ma20)
      if vol_ma20 and not np.isnan(vol_ma20) and vol_ma20 > 0
      else 1.0
  )

  # Support & Resistance (20-day Low / High)
  support_20d = float(low.rolling(window=20).min().iloc[-1])
  resistance_20d = float(high.rolling(window=20).max().iloc[-1])
  high_52w = float(high.rolling(window=252).max().iloc[-1])

  # Valuation Ratio P/IV
  p_iv_ratio = current_price / iv if iv > 0 else 1.0

  # Action Signal Logic
  if p_iv_ratio > 1.25 or rsi > 75:
    signal = "🔴 STRONG SELL/TRIM"
    priority = 1
  elif p_iv_ratio > 1.10 or (rsi > 65 and current_price > resistance_20d * 0.98):
    signal = "🟡 TRIM SATELLITE"
    priority = 2
  elif p_iv_ratio < 0.80 and rsi < 40:
    signal = "🚀 STRONG BUY"
    priority = 4
  elif p_iv_ratio < 0.90 or current_price <= support_20d * 1.02:
    signal = "⚡ RE-BUY DIP"
    priority = 3
  else:
    signal = "⚪ NEUTRAL"
    priority = 5

  target_trim = round(iv * 1.20, 2)
  target_rebuy = round(iv * 0.85, 2)

  return {
      "Priority": priority,
      "Ticker": ticker,
      "Current Price": round(current_price, 2),
      "Intrinsic Value": round(iv, 2),
      "P/IV Ratio": round(p_iv_ratio, 2),
      "Valuation Status": (
          "Overvalued"
          if p_iv_ratio > 1.1
          else ("Undervalued" if p_iv_ratio < 0.9 else "Fair Value")
      ),
      "EMA20": round(ema20, 2),
      "EMA50": round(ema50, 2),
      "200 SMA": round(sma200, 2),
      "RSI (14)": round(rsi, 1),
      "ATR (14)": round(atr14, 2),
      "ATR %": round(atr_pct, 2),
      "Dist EMA20 (ATR)": round(dist_ema20_atr, 2),
      "MACD": round(macd_val, 2),
      "MACD Signal": round(macd_sig, 2),
      "Vol Ratio": round(vol_ratio, 2),
      "20D Support": round(support_20d, 2),
      "20D Resistance": round(resistance_20d, 2),
      "52W High": round(high_52w, 2),
      "CST Signal": signal,
      "Target Trim Price": target_trim,
      "Target Re-Buy Price": target_rebuy,
  }


def main():
  tickers, intrinsic_values = load_portfolio(PORTFOLIO_FILE)
  if not tickers:
    print("No tickers found to process.")
    return

  results = []
  print(
      f"Fetching and calculating market data for {len(tickers)} tickers..."
  )

  for ticker in tickers:
    df_ticker = fetch_ticker_data(ticker)
    if not df_ticker.empty:
      res = calculate_indicators(ticker, df_ticker, intrinsic_values[ticker])
      if res:
        results.append(res)

  if results:
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values(
        by=["Priority", "P/IV Ratio"], ascending=[True, False]
    )

    # Timestamp metadata
    sgt_time = datetime.now(pytz.timezone("Asia/Singapore")).strftime(
        "%Y-%m-%d %H:%M SGT"
    )
    df_results["Last Updated"] = sgt_time

    df_export = df_results.drop(columns=["Priority"])
    df_export.to_csv(OUTPUT_FILE, index=False)
    print(
        f"Analysis complete successfully at {sgt_time}. Results saved to"
        f" {OUTPUT_FILE}"
    )
  else:
    print("Error: No results generated.")


if __name__ == "__main__":
  main()
