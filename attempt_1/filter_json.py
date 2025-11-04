#!/usr/bin/env python3
"""
filter_tickers.py

Reads all_tickers.json (list of tickers),
keeps only those with usable Yahoo Finance data,
and writes usable_tickers.json + bad_tickers.json.
"""


import time
import json
import yfinance as yf
import argparse
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.simplefilter("ignore", category=FutureWarning)


def looks_like_junk(ticker: str) -> bool:
    """Heuristic: skip warrants, rights, units, prefs."""
    bad_suffixes = ("W", "WS", "U", "R", "P", "-WT", "-WS")
    return ticker.endswith(bad_suffixes)

def is_usable(ticker: str) -> bool:
    try:
        df = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
        # Require non-empty and at least ~120 trading days (~6 months)
        if df is not None and len(df) >= 120:
            return True
        return False
    except Exception as e:
        msg = str(e)
        if "Invalid Crumb" in msg or "Unauthorized" in msg:
            time.sleep(2)
            try:
                df = yf.download(ticker, period="6mo", interval="1d", progress=False, threads=False)
                return df is not None and len(df) >= 120
            except Exception:
                return False
        return False


def main(infile: str, outfile: str):
    with open(infile, "r") as f:
        tickers = json.load(f)

    tickers = [t.strip().upper() for t in tickers if isinstance(t, str) and t.strip()]

    usable, bad = [], []

    # Use thread pool for speed
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(is_usable, t): t for t in tickers if not looks_like_junk(t)}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                if fut.result():
                    usable.append(t)
                else:
                    bad.append(t)
            except Exception:
                bad.append(t)

    with open(outfile, "w") as f:
        json.dump(sorted(usable), f)

    with open("bad_tickers.json", "w") as f:
        json.dump(sorted(bad), f)

    print(f"✅ Saved {len(usable)} usable tickers to {outfile}")
    print(f"❌ Skipped {len(bad)} unusable tickers -> bad_tickers.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", type=str, default="all_tickers.json",
                        help="Path to input JSON with all tickers")
    parser.add_argument("--outfile", type=str, default="usable_tickers.json",
                        help="Path to output JSON with filtered tickers")
    args = parser.parse_args()
    main(args.infile, args.outfile)
