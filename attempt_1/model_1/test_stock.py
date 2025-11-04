#!/usr/bin/env python3
"""
get_tsla_data.py
Fetch full TSLA historical daily data with yfinance.
"""

import pandas as pd, yfinance as yf
df = yf.download("AAPL", period="6mo")
print(df.head())
df.to_parquet("test.parquet", engine="fastparquet")
print("Saved parquet, size:", os.path.getsize("test.parquet"))
