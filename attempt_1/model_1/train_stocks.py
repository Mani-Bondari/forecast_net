#!/usr/bin/env python3
"""
Training pipeline for ForecastNet.

- Dynamically fetches OHLCV from Yahoo Finance via yfinance
- Randomly samples (ticker, anchor_date) each step
- 90 trading days -> inputs; next 7 trading days -> normalized target returns
- Checkpointing (model + optimizer + scaler + scheduler + RNG)
- TensorBoard logging
"""

import os, json, math, time, random, argparse, signal, io
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

# --- 3rd party fetch ---
import yfinance as yf

# --- your model ---
from forecast_net import ForecastNet
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")



# ---------------------------
# Utilities & feature engineer
# ---------------------------

def trading_days_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    """Assumes df indexed by DatetimeIndex with market days only."""
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must be indexed by DatetimeIndex.")
    return df.index

def pct_return(series: pd.Series) -> pd.Series:
    return series.pct_change().fillna(0.0)

def log_return(series: pd.Series) -> pd.Series:
    with np.errstate(divide='ignore', invalid='ignore'):
        lr = np.log(series / series.shift(1))
    return lr.replace([np.inf, -np.inf], 0.0).fillna(0.0)

def rolling_std(series: pd.Series, win: int) -> pd.Series:
    return series.pct_change().rolling(win).std().fillna(0.0)

def sma(series: pd.Series, win: int) -> pd.Series:
    return series.rolling(win).mean().fillna(method="bfill").fillna(0.0)

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean().fillna(0.0)

def rsi(series: pd.Series, win: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0).rolling(win).mean()
    down = (-delta.clip(upper=0)).rolling(win).mean()
    rs = (up / (down + 1e-12)).fillna(0.0)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50.0)

def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series]:
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line.fillna(0.0), signal_line.fillna(0.0)

def compute_beta(ticker_rets: pd.Series, market_rets: pd.Series, lookback: int = 252) -> float:
    """OLS beta = Cov(R_i, R_m) / Var(R_m)."""
    a = ticker_rets.dropna().tail(lookback).align(market_rets.dropna().tail(lookback), join="inner")
    if a[0].empty or a[1].var() == 0:
        return 1.0
    cov = np.cov(a[0], a[1])[0, 1]
    var = np.var(a[1])
    if var == 0:
        return 1.0
    return float(cov / var)

def safe_float(x, default=0.0):
    try:
        if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
            return default
        return float(x)
    except Exception:
        return default

# ---------------------------
# Data cache
# ---------------------------

import time
import random

class YahooCache:
    """
    Simple per-run in-memory & optional on-disk parquet cache for OHLCV.
    Falls back from 'max' -> '1y' -> '6mo' if necessary.
    Retries with backoff on transient errors (timeouts, rate limits).
    """
    def __init__(self, cache_dir: Optional[str] = None, session: Optional[yf.Ticker] = None):
        self.cache: Dict[str, pd.DataFrame] = {}
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

    def _cache_path(self, ticker: str) -> str:
        return os.path.join(self.cache_dir, f"{ticker.replace('^','SPX_')}.parquet")

    def _download_with_fallback(self, ticker: str, interval: str = "1d") -> pd.DataFrame:
        periods = ["max", "1y", "6mo"]
        for period in periods:
            for attempt in range(5):  # up to 5 retries per period
                try:
                    logging.info(f"Downloading {ticker} ({period}, attempt {attempt+1})")
                    df = yf.download(
                        ticker, period=period, interval=interval,
                        auto_adjust=False, progress=False, threads=False
                    )
                    if df is not None and not df.empty:
                        logging.info(f"✅ {ticker} succeeded with period={period}, rows={len(df)}")
                        return df
                    break
                except Exception as e:
                    msg = str(e)
                    if any(x in msg for x in ["timed out", "429", "Too Many Requests", "Connection", "Invalid Crumb", "Unauthorized"]):
                        wait = (2 ** attempt) + random.random()
                        logging.warning(f"[WARN] {ticker}: transient error ({msg}), retrying in {wait:.1f}s...")
                        time.sleep(wait)
                        continue
                    else:
                        logging.error(f"[ERROR] {ticker}: permanent error ({msg}), skipping period={period}")
                        break
        logging.error(f"❌ {ticker}: all periods failed")
        return pd.DataFrame()
    
    def get_history(self, ticker: str, period: str = "max", interval: str = "1d") -> pd.DataFrame:
        if ticker in self.cache:
            return self.cache[ticker]

        df = None
        if self.cache_dir:
            path = self._cache_path(ticker)
            if os.path.exists(path):
                try:
                    df = pd.read_parquet(path)
                    logging.info(f"📦 Loaded {ticker} from cache ({len(df)} rows)")
                except Exception as e:
                    logging.warning(f"[WARN] Failed to read cache for {ticker}: {e}")
                    df = None

        if df is None or df.empty:
            data = self._download_with_fallback(ticker, interval=interval)

            if isinstance(data, pd.DataFrame) and not data.empty:
                # Flatten MultiIndex if needed
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = ["_".join([str(c) for c in col if c]) for col in data.columns]

                # Normalize column names (case-insensitive)
                rename_map = {}
                for c in data.columns:
                    c_low = c.lower().strip()
                    if c_low == "open":
                        rename_map[c] = "open"
                    elif c_low == "high":
                        rename_map[c] = "high"
                    elif c_low == "low":
                        rename_map[c] = "low"
                    elif c_low == "close":
                        rename_map[c] = "close"
                    elif c_low in ("adj close", "adj_close", "adjclose"):
                        rename_map[c] = "adj_close"
                    elif c_low == "volume":
                        rename_map[c] = "volume"

                data = data.rename(columns=rename_map)

                # Ensure all expected columns exist
                for col in ["open", "high", "low", "close", "adj_close", "volume"]:
                    if col not in data.columns:
                        data[col] = np.nan

                # Fallbacks
                if data["adj_close"].isna().all() and not data["close"].isna().all():
                    data["adj_close"] = data["close"]
                if data["close"].isna().all() and not data["adj_close"].isna().all():
                    data["close"] = data["adj_close"]
                if data["volume"].isna().all():
                    data["volume"] = 0.0

                # Clean index
                try:
                    data.index = pd.to_datetime(data.index)
                except Exception:
                    pass
                data = data[~data.index.duplicated(keep="last")].sort_index()

                # Drop rows only if OHLC totally missing
                df = data.dropna(subset=["open", "high", "low", "close"], how="any")

                if df.empty:
                    logging.warning(f"⚠️ {ticker}: all rows invalid after cleaning (raw rows={len(data)})")
                else:
                    if self.cache_dir:
                        path = self._cache_path(ticker)
                        try:
                            df.to_parquet(path, engine="fastparquet")
                            logging.info(f"💾 Saved {ticker} to cache -> {path} ({len(df)} rows)")
                        except Exception as e:
                            logging.error(f"❌ Failed to save {ticker} to {path}: {e}")
            else:
                df = pd.DataFrame(columns=["open", "high", "low", "close", "adj_close", "volume"])

        self.cache[ticker] = df
        return df







# ---------------------------
# Dynamic sampler dataset
# ---------------------------

@dataclass
class SampleConfig:
    lookback: int = 90
    horizon: int = 7
    min_history: int = 400   # need at least this many rows to compute beta etc.
    mask_prob: float = 0.1   # random mask probability per timestep per channel
    noise_std: float = 0.01  # Gaussian noise std on normalized features

class DynamicYahooDataset(Dataset):
    """
    Each __getitem__ creates one random sample:
      - Choose random ticker from list (train or val set)
      - Choose random anchor date from its history
      - Build 90-day window features & 7-day normalized future-return target
    """
    def __init__(
        self,
        tickers: List[str],
        cache: YahooCache,
        market_ticker: str = "^GSPC",
        cfg: SampleConfig = SampleConfig(),
        eval_mode: bool = False,
        seed: int = 1234,
    ):
        self.tickers = tickers
        self.cache = cache
        self.market_ticker = market_ticker
        self.cfg = cfg
        self.eval_mode = eval_mode
        self.rng = random.Random(seed)
        # prefetch market history
        self.market_df = self.cache.get_history(market_ticker)
        self.market_rets = pct_return(self.market_df["adj_close"]) if not self.market_df.empty else pd.Series(dtype=float)

    def __len__(self):
        # It's conceptually infinite; pick a big number. DataLoader's num_steps controls epochs.
        return 10**12

    def _pick_ticker_and_anchor(self) -> Tuple[str, pd.Timestamp]:
        for _ in range(100):  # try until success
            tkr = self.rng.choice(self.tickers)
            df = self.cache.get_history(tkr)
            if df is None or df.empty or len(df) < (self.cfg.min_history + self.cfg.lookback + self.cfg.horizon + 5):
                continue
            idx = trading_days_index(df)
            # anchor ranges from (lookback) .. (len - horizon - 1)
            lo = self.cfg.lookback
            hi = len(idx) - self.cfg.horizon - 1
            if hi <= lo:
                continue
            anchor_pos = self.rng.randint(lo, hi)
            return tkr, idx[anchor_pos]
        # fallback (should be rare if all tickers usable)
        tkr = self.rng.choice(self.tickers)
        df = self.cache.get_history(tkr)
        idx = trading_days_index(df)
        anchor_pos = min(len(idx) - self.cfg.horizon - 1, self.cfg.lookback + 1)
        anchor_pos = max(anchor_pos, self.cfg.lookback)
        return tkr, idx[anchor_pos]

    def _build_features_and_target(self, tkr: str, anchor: pd.Timestamp) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        df = self.cache.get_history(tkr)
        if df is None or df.empty:
            raise RuntimeError(f"No data for {tkr}")

        # find index of anchor
        idx = df.index.get_indexer([anchor])
        if idx.size == 0 or idx[0] == -1:
            raise RuntimeError("Anchor not in index (sanity)")

        anchor_i = idx[0]
        start_i = anchor_i - self.cfg.lookback
        end_i = anchor_i  # inclusive anchor day in history window
        fut_start = anchor_i + 1
        fut_end = anchor_i + self.cfg.horizon

        hist = df.iloc[start_i:end_i]

        # --- NEW: pad if shorter than lookback ---
        if len(hist) < self.cfg.lookback:
            pad_len = self.cfg.lookback - len(hist)
            pad_df = pd.DataFrame(
                {c: [0.0]*pad_len for c in df.columns}, 
                index=pd.date_range(end=hist.index[0], periods=pad_len, freq="B")
            )
            hist = pd.concat([pad_df, hist])

        fut = df.iloc[fut_start:fut_end]  # length = horizon

        # basic series
        close = hist["close"].astype(float)
        open_ = hist["open"].astype(float)
        adj_close = hist["adj_close"].astype(float)
        volume = hist["volume"].astype(float)

        # engineered indicators (computed on adj_close unless otherwise specified)
        ret = pct_return(adj_close)
        logret = log_return(adj_close)
        ma5 = sma(adj_close, 5)
        ma10 = sma(adj_close, 10)
        ma20 = sma(adj_close, 20)
        ma50 = sma(adj_close, 50)
        vol20 = rolling_std(adj_close, 20)
        rsi14 = rsi(adj_close, 14)
        macd_line, macd_sig = macd(adj_close, 12, 26, 9)
        momentum = adj_close.diff().fillna(0.0)

        # Construct channels expected by your model:
        # 1) val_ts: return-like signal (close - open) normalized
        val_ts = (close - open_).to_numpy().reshape(-1, 1)

        # 2) vol_ts: volume (log-scaled then standardized)
        vol_ts_raw = volume.replace(0, np.nan).fillna(method="bfill").fillna(1.0)
        vol_ts = np.log(vol_ts_raw).to_numpy().reshape(-1, 1)

        # 3) mval_ts: adj_close (price-level proxy, log-scaled)
        mval_ts = np.log(adj_close.replace(0, np.nan).fillna(method="bfill")).to_numpy().reshape(-1, 1)

        # 4) beta: compute over longer lookback vs market
        ticker_rets_full = pct_return(self.cache.get_history(tkr)["adj_close"])
        beta_val = compute_beta(ticker_rets_full, self.market_rets, lookback=252)
        beta_ts = np.array([[beta_val]], dtype=np.float32)  # shape (1, 1) later broadcast to (B,1,1)

        # 5) pe_ts: Yahoo does not provide a reliable daily PE series via API.
        # Fallback: try trailing PE snapshot and broadcast across window; else 0.
        trailing_pe = None
        try:
            info = yf.Ticker(tkr).fast_info  # faster than .info, may not have PE
            trailing_pe = getattr(info, "trailing_pe", None)
        except Exception:
            trailing_pe = None
        if trailing_pe is None:
            # Try legacy .info as a fallback (slower; may be deprecated)
            try:
                trailing_pe = yf.Ticker(tkr).info.get("trailingPE", None)
            except Exception:
                trailing_pe = None

        pe_scalar = safe_float(trailing_pe, default=0.0)
        pe_ts = (np.ones((len(hist), 1), dtype=np.float32) * pe_scalar)

        def pad_to_len(arr, target_len):
            if len(arr) < target_len:
                pad_width = target_len - len(arr)
                arr = np.pad(arr, ((pad_width,0),(0,0)), mode="constant", constant_values=0.0)
            elif len(arr) > target_len:
                arr = arr[-target_len:]
            return arr

        # Normalize the three sequences (val_ts, vol_ts, mval_ts) robustly
        def robust_norm(x: np.ndarray) -> np.ndarray:
            mu = np.nanmean(x)
            sd = np.nanstd(x)
            if not np.isfinite(sd) or sd < 1e-12:
                sd = 1.0
            return ((x - mu) / sd).astype(np.float32)

        val_ts = robust_norm(val_ts)
        vol_ts = robust_norm(vol_ts)
        mval_ts = robust_norm(mval_ts)
        pe_ts = robust_norm(pe_ts)
        
        val_ts = pad_to_len(val_ts, self.cfg.lookback)
        vol_ts = pad_to_len(vol_ts, self.cfg.lookback)
        mval_ts = pad_to_len(mval_ts, self.cfg.lookback)
        pe_ts  = pad_to_len(pe_ts,  self.cfg.lookback)


        # Optional random masking & noise (train only)
        if not self.eval_mode and self.cfg.mask_prob > 0:
            for arr in (val_ts, vol_ts, mval_ts, pe_ts):
                mask = (np.random.rand(*arr.shape) < self.cfg.mask_prob).astype(np.float32)
                arr[mask == 1] = 0.0
        if not self.eval_mode and self.cfg.noise_std > 0:
            for arr in (val_ts, vol_ts, mval_ts, pe_ts):
                arr += np.random.normal(0.0, self.cfg.noise_std, size=arr.shape).astype(np.float32)

        # Target: normalized 7-day future returns on adj_close
        fut_adj = fut["adj_close"].astype(float)
        fut_lr = log_return(fut_adj)
        # normalize the future returns by window stats for stationarity
        base_lr = logret  # for normalization stats
        mu = np.nanmean(base_lr.values)
        sd = np.nanstd(base_lr.values)
        if not np.isfinite(sd) or sd < 1e-12:
            sd = 1.0
        fut_norm = ((fut_lr.values).reshape(-1, 1) - mu) / sd
        # handle any length mismatches (should be horizon)
        if fut_norm.shape[0] != self.cfg.horizon:
            pad = self.cfg.horizon - fut_norm.shape[0]
            fut_norm = np.pad(fut_norm, ((0, pad), (0, 0)), mode="constant", constant_values=0.0)
        target = fut_norm.astype(np.float32)  # (7, 1)

        features = {
            "val_ts": val_ts.astype(np.float32),   # (90, 1)
            "vol_ts": vol_ts.astype(np.float32),   # (90, 1)
            "mval_ts": mval_ts.astype(np.float32), # (90, 1)
            "beta_ts": beta_ts.astype(np.float32), # (1, 1)
            "pe_ts": pe_ts.astype(np.float32),     # (90, 1)
        }
        return features, target

    def __getitem__(self, idx):
        for _ in range(10):
            try:
                tkr, anchor = self._pick_ticker_and_anchor()
                feats, target = self._build_features_and_target(tkr, anchor)
                return feats, target, tkr, anchor
            except Exception:
                continue
        # last resort: random dummy (should not happen frequently)
        dummy = np.zeros((self.cfg.lookback, 1), dtype=np.float32)
        features = {"val_ts": dummy, "vol_ts": dummy, "mval_ts": dummy, "beta_ts": np.zeros((1,1), np.float32), "pe_ts": dummy}
        target = np.zeros((self.cfg.horizon, 1), dtype=np.float32)
        return features, target, "DUMMY", pd.Timestamp.utcnow()

# ---------------------------
# Collate fn
# ---------------------------

def collate_fn(batch):
    feats_list, target_list, tickers, anchors = zip(*batch)
    def stack(key, pad_to=None):
        arrs = [torch.from_numpy(f[key]) for f in feats_list]
        return torch.stack(arrs, dim=0)  # (B, L, 1) or (B, 1, 1)
    val_ts = stack("val_ts")
    vol_ts = stack("vol_ts")
    mval_ts = stack("mval_ts")
    beta = stack("beta_ts")  # (B,1,1)
    pe_ts = stack("pe_ts")

    target = torch.stack([torch.from_numpy(x) for x in target_list], dim=0)  # (B, 7, 1)
    return (val_ts, vol_ts, mval_ts, beta, pe_ts), target, tickers, anchors

# ---------------------------
# Training, Eval, Checkpoint
# ---------------------------

def save_checkpoint(path: str, model, optimizer, scaler, scheduler, global_step, epoch, rng_state, args):
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "global_step": global_step,
        "epoch": epoch,
        "rng_state": {
            "python": rng_state["python"],
            "numpy": rng_state["numpy"],
            "torch": rng_state["torch"],
            "torch_cuda": rng_state.get("torch_cuda", None)
        },
        "args": vars(args),
    }
    torch.save(ckpt, path)

def load_checkpoint(path: str, model, optimizer=None, scaler=None, scheduler=None, map_location="cpu"):
    ckpt = torch.load(path, map_location=map_location)
    model.load_state_dict(ckpt["model"], strict=True)
    if optimizer and "optimizer" in ckpt and ckpt["optimizer"]:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt and ckpt["scaler"]:
        scaler.load_state_dict(ckpt["scaler"])
    if scheduler and "scheduler" in ckpt and ckpt["scheduler"]:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt

@torch.no_grad()
def evaluate(model, loader, device, max_batches=100):
    model.eval()
    mse = nn.MSELoss()
    l_sum = 0.0
    n = 0
    # simple direction accuracy: sign match on sum of 7-day returns
    dir_right = 0
    dir_total = 0
    for i, (feats, target, tickers, anchors) in enumerate(loader):
        if i >= max_batches:
            break
        val_ts, vol_ts, mval_ts, beta, pe_ts = [x.to(device) for x in feats]
        target = target.to(device)  # (B,7,1)
        preds = model(val_ts, vol_ts, mval_ts, beta, pe_ts)  # (B,7,1)
        loss = mse(preds, target)
        l_sum += float(loss.item()) * target.size(0)
        n += target.size(0)

        pred_sum = preds.sum(dim=1)  # (B,1)
        tgt_sum = target.sum(dim=1)
        dir_right += ((pred_sum.sign() == tgt_sum.sign()).float().sum().item())
        dir_total += target.size(0)
    return (l_sum / max(n, 1)), (dir_right / max(dir_total, 1))

# ---------------------------
# Main train
# ---------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers_json", type=str, default="all_tickers.json", help="JSON list of US tickers")
    p.add_argument("--cache_dir", type=str, default="./yf_cache")
    p.add_argument("--log_dir", type=str, default="./runs")
    p.add_argument("--save_dir", type=str, default="./checkpoints")
    p.add_argument("--epochs", type=int, default=1000000, help="Large number; loop is step-based")
    p.add_argument("--steps_per_epoch", type=int, default=1000)
    p.add_argument("--val_steps", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.999))
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--accum_steps", type=int, default=2, help="gradient accumulation steps")
    p.add_argument("--cosine_t0", type=int, default=10000, help="T_max for CosineAnnealingLR in steps")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--dropout", type=float, default=0.1, help="(If your model exposes it)")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to resume")
    # model sizes
    p.add_argument("--d_model", type=int, default=1024)
    p.add_argument("--nhead_enc", type=int, default=8)
    p.add_argument("--nhead_dec", type=int, default=8)
    p.add_argument("--layers_enc", type=int, default=16)
    p.add_argument("--layers_dec", type=int, default=16)
    p.add_argument("--query_len", type=int, default=7)
    p.add_argument("--lookback", type=int, default=90)
    p.add_argument("--horizon", type=int, default=7)
    # regularization / data
    p.add_argument("--mask_prob", type=float, default=0.1)
    p.add_argument("--noise_std", type=float, default=0.01)
    p.add_argument("--pe_as_sequence", action="store_true", help="Use full 90-day PE history as sequence tokens instead of scalar conditioning")

    args = p.parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    # Seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Load tickers
    with open(args.tickers_json, "r", encoding="utf-8") as f:
        all_tickers = json.load(f)
    # Basic cleaning (filter non-empty strings)
    all_tickers = [t.strip().upper() for t in all_tickers if isinstance(t, str) and t.strip()]

    # Split tickers into train/val (ticker-level generalization)
    rng = random.Random(args.seed)
    rng.shuffle(all_tickers)
    val_frac = 0.1
    n_val = max(1, int(len(all_tickers) * val_frac))
    val_tickers = all_tickers[:n_val]
    train_tickers = all_tickers[n_val:]

    cache = YahooCache(cache_dir=args.cache_dir)

    train_ds = DynamicYahooDataset(
        tickers=train_tickers,
        cache=cache,
        cfg=SampleConfig(lookback=args.lookback, horizon=args.horizon, mask_prob=args.mask_prob, noise_std=args.noise_std),
        eval_mode=False,
        seed=args.seed,
    )
    val_ds = DynamicYahooDataset(
        tickers=val_tickers,
        cache=cache,
        cfg=SampleConfig(lookback=args.lookback, horizon=args.horizon, mask_prob=0.0, noise_std=0.0),
        eval_mode=True,
        seed=args.seed + 1,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True, drop_last=True,
                            collate_fn=collate_fn)

    # Model
    model = ForecastNet(
        in_dim=1,
        d_model=args.d_model,
        n_head_enc=args.nhead_enc,
        n_head_dec=args.nhead_dec,
        num_layers_enc=args.layers_enc,
        num_layers_dec=args.layers_dec,
        query_len=args.query_len,
        pe_as_sequence=args.pe_as_sequence,
    ).to(args.device)

    # Losses (MSE for regression)
    criterion = nn.MSELoss()

    # Optimizer / Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=tuple(args.betas))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosine_t0, eta_min=args.lr * 0.1)

    scaler = GradScaler(enabled=args.amp)

    global_step = 0
    start_epoch = 0

    # Resume
    if args.resume and os.path.isfile(args.resume):
        ckpt = load_checkpoint(args.resume, model, optimizer, scaler, scheduler, map_location=args.device)
        global_step = ckpt.get("global_step", 0)
        start_epoch = ckpt.get("epoch", 0)
        # restore RNG
        rs = ckpt.get("rng_state", {})
        try:
            random.setstate(rs["python"])
            np.random.set_state(rs["numpy"])
            torch.set_rng_state(rs["torch"])
            if torch.cuda.is_available() and rs.get("torch_cuda") is not None:
                torch.cuda.set_rng_state(rs["torch_cuda"])
        except Exception:
            pass
        print(f"Resumed from {args.resume} @ epoch {start_epoch}, step {global_step}")

    # Graceful interrupt to save checkpoint
    stop_requested = False
    def handle_sigint(signum, frame):
        nonlocal stop_requested
        stop_requested = True
        print("\nStop requested. Will save checkpoint at end of this step/epoch.")
    signal.signal(signal.SIGINT, handle_sigint)

    model.train()
    for epoch in range(start_epoch, args.epochs):
        running = 0.0
        train_iter = iter(train_loader)
        for step in range(args.steps_per_epoch):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            (val_ts, vol_ts, mval_ts, beta, pe_ts), target, tickers, anchors = batch


            val_ts = val_ts.to(args.device, non_blocking=True)
            vol_ts = vol_ts.to(args.device, non_blocking=True)
            mval_ts = mval_ts.to(args.device, non_blocking=True)
            beta = beta.to(args.device, non_blocking=True)
            pe_ts = pe_ts.to(args.device, non_blocking=True)
            target = target.to(args.device, non_blocking=True)

            with autocast(enabled=args.amp):
                preds = model(val_ts, vol_ts, mval_ts, beta, pe_ts)  # (B,7,1)
                loss = criterion(preds, target) / args.accum_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % args.accum_steps == 0:
                # gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            running += loss.item() * args.accum_steps
            if global_step % 50 == 0:
                writer.add_scalar("train/loss", running / max(1, (step + 1)), global_step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            global_step += 1

            if stop_requested:
                break

        # Eval
        val_mse, val_dir = evaluate(model, val_loader, args.device, max_batches=args.val_steps)
        writer.add_scalar("val/mse", val_mse, global_step)
        writer.add_scalar("val/direction_acc", val_dir, global_step)

        # Save checkpoint per epoch
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
        }
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch:06d}_step_{global_step}.pt")
        save_checkpoint(ckpt_path, model, optimizer, scaler, scheduler, global_step, epoch, rng_state, args)
        print(f"[epoch {epoch}] train_loss={running/args.steps_per_epoch:.6f}  val_mse={val_mse:.6f}  val_dir={val_dir:.3f}  -> {ckpt_path}")

        if stop_requested:
            print("Exiting after graceful stop.")
            break

    writer.close()

if __name__ == "__main__":
    main()
