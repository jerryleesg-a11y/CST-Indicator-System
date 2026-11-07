"""
Covered-Swing-Trade (CST) Technical Indicator & Valuation Engine (v2.1 Sorted)
================================================================================
Calculates: P/IV Ratio, 200 SMA, EMA20, EMA50, RSI(14), ATR(14), Dist EMA20 (in ATRs),
MACD (12,26,9), Vol Ratio (20D), 20D Support/Resistance, 52W High, Flash Icons,
Salient Remarks, Action Signals, and sorts output by highest action conviction.
"""

import os
import sys
import json
import urllib.request
import urllib.parse
import pandas as pd
import numpy as np

INTRINSIC_FILE = "Intrinsic-Value-Portfolio-Nov2026.txt"
OUTPUT_CSV = "CST_Portfolio_Analysis_Results.csv"
LEGEND_CSV = "CST_Indicator_Legend_Guide.csv"

TICKER_MAP = {
    '9988': '9988.HK', 'AMZN': 'AMZN', 'ASML': 'ASML', 'AVGO': 'AVGO',
    'BXSL': 'BXSL', 'CPRT': 'CPRT', 'CRM': 'CRM', 'CSPX': 'CSPX.L',
    'D05.SI': 'D05.SI', 'ES3.SI': 'ES3.SI', 'GOOGL': 'GOOGL', 'IBM': 'IBM',
    'JEPG.LSE': 'JEPG.L', 'JEPQ.LSE': 'JEPQ.L', 'LMT': 'LMT', 'META': 'META',
    'PANW': 'PANW', 'MSFT': 'MSFT', 'MU': 'MU', 'NVDA': 'NVDA',
    'NVO': 'NVO', 'PEP': 'PEP', 'S63': 'S63.SI', 'SPGI': 'SPGI',
    'TSM': 'TSM', 'U11.SI': 'U11.SI', 'UNH': 'UNH', 'V': 'V',
    'VWRA.LSE': 'VWRA.L', 'WDEF.LSE': 'WDEF.L'
}

HEADERS = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

def load_intrinsic_values(filepath):
    iv_dict = {}
    if not os.path.exists(filepath):
        return iv_dict
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines[1:]:
            parts = line.strip().split(',')
            if len(parts) >= 1:
                ticker = parts[0].strip()
                val = parts[1].strip() if len(parts) > 1 and parts[1].strip() != '' else None
                iv_dict[ticker] = float(val) if val else None
    return iv_dict

def fetch_yahoo_ohlcv(yf_symbol):
    encoded_symbol = urllib.parse.quote(yf_symbol)
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{encoded_symbol}?range=1y&interval=1d"
    req = urllib.request.Request(url, headers=HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=10) as response:
            res = json.loads(response.read().decode('utf-8'))
            chart = res['chart']['result'][0]
            timestamps = chart['timestamp']
            quote = chart['indicators']['quote'][0]
            df = pd.DataFrame({
                'Date': pd.to_datetime(timestamps, unit='s'),
                'Close': quote['close'],
                'High': quote['high'],
                'Low': quote['low'],
                'Volume': quote['volume']
            }).dropna().reset_index(drop=True)
            return df
    except Exception as e:
        print(f"Error fetching {yf_symbol}: {e}")
        return None

def compute_indicators(df):
    close = df['Close']
    high = df['High']
    low = df['Low']
    volume = df['Volume']

    df['EMA20'] = close.ewm(span=20, adjust=False).mean()
    df['EMA50'] = close.ewm(span=50, adjust=False).mean()
    df['SMA200'] = close.rolling(window=min(200, len(df))).mean()

    delta = close.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI14'] = 100 - (100 / (1 + rs))

    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df['ATR14'] = tr.rolling(window=14).mean()

    df['ATR_Pct'] = (df['ATR14'] / close) * 100
    df['Dist_EMA20_ATR'] = (close - df['EMA20']) / df['ATR14']

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    vol20 = volume.rolling(window=20).mean()
    df['Vol_Ratio'] = volume / vol20.replace(0, np.nan)

    df['Support_20D'] = low.rolling(window=20).min()
    df['Resistance_20D'] = high.rolling(window=20).max()
    df['High_52Wk'] = high.rolling(window=min(252, len(df))).max()

    return df

def generate_signals_and_remarks(price, iv, ema20, ema50, sma200, rsi, dist_ema20, vol_ratio, support, resistance, high_52wk):
    p_iv = round(price / iv, 4) if iv else 1.0
    val_status = "Overvalued" if p_iv > 1.0 else "Undervalued"
    val_icon = "🔴" if p_iv > 1.0 else "🟢"

    trend_icon = "🚀" if (sma200 != "N/A" and price > sma200 and ema20 > ema50) else "📉"
    rsi_icon = "🔥" if rsi >= 70 else ("🧊" if rsi <= 40 else "⚖️")
    ext_icon = "⚡" if dist_ema20 >= 2.0 else ("🧲" if dist_ema20 <= -1.2 else "🔹")

    remarks = []
    if price > ema20:
        remarks.append("Above EMA20 (Short Bull)")
    else:
        remarks.append("Below EMA20 (Short Bear)")

    if price > ema50:
        remarks.append("Above EMA50 (Med Bull)")
    else:
        remarks.append("Below EMA50 (Med Dip)")

    if sma200 != "N/A":
        if price > sma200:
            remarks.append("Above 200 SMA (Macro Uptrend 🟢)")
        else:
            remarks.append("Below 200 SMA (Macro Caution 🔴)")

    if price >= high_52wk * 0.98:
        remarks.append("Near 52W High (Breakout Watch 🌟)")
    elif price <= support * 1.02:
        remarks.append("At 20D Support (Re-buy Zone 🛒)")

    if rsi >= 70:
        remarks.append("RSI Overbought (>70)")
    elif rsi <= 35:
        remarks.append("RSI Oversold (<35)")

    remark_str = " | ".join(remarks)

    if dist_ema20 >= 2.2 and rsi >= 65 and p_iv > 1.0:
        cst_sig = "STRONG SELL / TRIM"
        priority = 1
    elif dist_ema20 >= 1.8 or rsi >= 70 or (p_iv > 1.05 and dist_ema20 > 1.2):
        cst_sig = "TRIM SATELLITE"
        priority = 2
    elif p_iv <= 1.0 and (dist_ema20 <= -1.2 or rsi <= 40 or price <= support * 1.02):
        cst_sig = "RE-BUY DIP"
        priority = 3
    elif p_iv < 0.90 and dist_ema20 < 0.0:
        cst_sig = "STRONG BUY / RE-BUY"
        priority = 4
    else:
        cst_sig = "NEUTRAL / HOLD"
        priority = 5

    return val_status, val_icon, trend_icon, rsi_icon, ext_icon, remark_str, cst_sig, priority

def main():
    print("Executing CST Technical & Valuation Analysis Engine v2.1 (Sorted)...")
    iv_dict = load_intrinsic_values(INTRINSIC_FILE)
    results = []

    for raw_ticker, yf_symbol in TICKER_MAP.items():
        iv = iv_dict.get(raw_ticker, None)
        df = fetch_yahoo_ohlcv(yf_symbol)
        if df is None or len(df) == 0:
            continue
        df = compute_indicators(df)
        latest = df.iloc[-1]

        price = round(float(latest['Close']), 2)
        p_iv = round(price / iv, 4) if iv else None
        ema20 = round(float(latest['EMA20']), 2)
        ema50 = round(float(latest['EMA50']), 2)
        sma200 = round(float(latest['SMA200']), 2) if not pd.isna(latest['SMA200']) else "N/A"
        rsi14 = round(float(latest['RSI14']), 2) if not pd.isna(latest['RSI14']) else 50.0
        atr14 = round(float(latest['ATR14']), 2) if not pd.isna(latest['ATR14']) else 0.0
        atr_pct = round(float(latest['ATR_Pct']), 2) if not pd.isna(latest['ATR_Pct']) else 0.0
        dist_ema20 = round(float(latest['Dist_EMA20_ATR']), 2) if not pd.isna(latest['Dist_EMA20_ATR']) else 0.0
        macd = round(float(latest['MACD']), 2) if not pd.isna(latest['MACD']) else 0.0
        vol_ratio = round(float(latest['Vol_Ratio']), 2) if not pd.isna(latest['Vol_Ratio']) else 1.0
        support = round(float(latest['Support_20D']), 2)
        resistance = round(float(latest['Resistance_20D']), 2)
        high_52wk = round(float(latest['High_52Wk']), 2)

        val_status, val_icon, trend_icon, rsi_icon, ext_icon, remarks, cst_sig, priority = generate_signals_and_remarks(
            price, iv, ema20, ema50, sma200, rsi14, dist_ema20, vol_ratio, support, resistance, high_52wk
        )

        target_trim = round(max(resistance, ema20 + (2.0 * atr14)), 2)
        target_rebuy = round(min(support, ema50 - (0.5 * atr14)), 2)

        results.append({
            "Priority": priority,
            "Ticker": raw_ticker,
            "Current Price": price,
            "Intrinsic Value": iv if iv else "N/A",
            "P/IV Ratio": p_iv if p_iv else "N/A",
            "Valuation Status": f"{val_icon} {val_status}",
            "Macro Trend (200 SMA)": f"{trend_icon} {'Bullish' if trend_icon == '🚀' else 'Bearish'}",
            "EMA20": ema20,
            "EMA50": ema50,
            "200 SMA": sma200,
            "RSI (14)": f"{rsi_icon} {rsi14}",
            "ATR (14)": atr14,
            "ATR %": f"{atr_pct}%",
            "Dist EMA20 (ATR)": f"{ext_icon} {dist_ema20}x",
            "MACD": macd,
            "Vol Ratio": f"{vol_ratio}x",
            "20D Support": support,
            "20D Resistance": resistance,
            "52W High": high_52wk,
            "CST Action Signal": cst_sig,
            "Target Trim Price": target_trim,
            "Target Re-Buy Price": target_rebuy,
            "Salient Indicator Remarks": remarks,
            "_sort_val": abs(dist_ema20)
        })

    results = sorted(results, key=lambda x: (x['Priority'], -x['_sort_val']))

    for r in results:
        del r['Priority']
        del r['_sort_val']

    out_df = pd.DataFrame(results)
    out_df.to_csv(OUTPUT_CSV, index=False)
    
    legend_data = [
        {"Indicator / Metric": "P/IV Ratio", "Good Threshold (🟢)": "< 1.0 (Undervalued)", "Caution Threshold (🔴)": "> 1.0 (Overvalued)", "Definition & Usage": "Price to Intrinsic Value ratio. Values >1.0 indicate trading at a premium over your target valuation."},
        {"Indicator / Metric": "200 SMA", "Good Threshold (🟢)": "Price > 200 SMA (Bullish)", "Caution Threshold (🔴)": "Price < 200 SMA (Bearish)", "Definition & Usage": "200-day Simple Moving Average. Baseline macro trend filter."},
        {"Indicator / Metric": "EMA20 / EMA50", "Good Threshold (🟢)": "Price > EMA20 > EMA50", "Caution Threshold (🔴)": "Price < EMA20 / EMA50", "Definition & Usage": "Exponential Moving Averages for short (20D) and medium (50D) swing trend tracking."},
        {"Indicator / Metric": "RSI (14)", "Good Threshold (🟢)": "35 - 55 (Healthy Dip)", "Caution Threshold (🔴)": "> 70 (Overbought 🔥) / < 30 (Oversold 🧊)", "Definition & Usage": "Relative Strength Index measuring momentum strength and extreme overextension."},
        {"Indicator / Metric": "Dist EMA20 (ATR)", "Good Threshold (🟢)": "-0.5x to +1.5x ATR", "Caution Threshold (🔴)": "> 2.0x ATR (Stretched ⚡)", "Definition & Usage": "Distance above/below EMA20 expressed in ATR multiples. Highlights overextension."},
        {"Indicator / Metric": "Vol Ratio", "Good Threshold (🟢)": "> 1.2x Volume", "Caution Threshold (🔴)": "< 0.8x Volume", "Definition & Usage": "Ratio of current daily volume to 20-day average volume."},
        {"Indicator / Metric": "20D Support / Resistance", "Good Threshold (🟢)": "Support = 20-Day Low", "Caution Threshold (🔴)": "Resistance = 20-Day High", "Definition & Usage": "Key price boundaries over the past 20 trading sessions."},
        {"Indicator / Metric": "CST Signal: TRIM SATELLITE", "Good Threshold (🟢)": "Lock in profits", "Caution Threshold (🔴)": "N/A", "Definition & Usage": "Sell a portion of satellite position when overextended/overvalued while holding core."},
        {"Indicator / Metric": "CST Signal: RE-BUY DIP", "Good Threshold (🟢)": "Re-acquire at support", "Caution Threshold (🔴)": "N/A", "Definition & Usage": "Buy back trimmed satellite shares at 20D support or EMA50 pullback."}
    ]
    pd.DataFrame(legend_data).to_csv(LEGEND_CSV, index=False)
    print("Sorted analysis results successfully exported!")

if __name__ == "__main__":
    main()
