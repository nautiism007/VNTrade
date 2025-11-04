#!/usr/bin/env python3
"""
sma_backtest.py

Simple backtesting prototype for 1-minute BTC data.
Implements:
  - 1-minute returns
  - N-period simple moving average (SMA)
  - Buy when price crosses above SMA, Sell when price crosses below SMA
  - Full long (position=+1) or full short (position=-1); switch only on opposite signal
  - No leverage, no partial positions
  - Metrics: total return (%), number of trades, max drawdown, Sharpe ratio (annualized)
  - Equity curve plot
  - Optional transaction cost per trade
  - Optional run over multiple SMA periods and compare results

Input data: CSV with columns (timestamp, mid_price, volume)
 - timestamp: unix timestamp (seconds)
 - mid_price: price at end of minute
 - volume: optional, unused by default
"""

import os
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Utilities / computations
# -------------------------
def compute_returns(df: pd.DataFrame) -> pd.Series:
    """Compute 1-minute relative returns: (p_t - p_{t-1}) / p_{t-1}"""
    return df["mid_price"].pct_change()


def compute_sma(df: pd.DataFrame, period: int) -> pd.Series:
    """Simple moving average of mid_price"""
    return df["mid_price"].rolling(period).mean()


def assign_volume_clusters(df: pd.DataFrame, n_clusters: int = 3, window: int = 1000) -> pd.Series:
    """
    Assign discrete volume regimes based on rolling quantiles (to avoid look-ahead bias).
    0 = low, 1 = medium, 2 = high
    """
    if "volume" not in df.columns:
        return pd.Series(1, index=df.index)  # neutral if no volume

    vol_clusters = pd.Series(index=df.index, dtype=float)
    vol = df["volume"]

    for i in range(window, len(df)):
        past_vol = vol.iloc[i - window:i].dropna()
        if len(past_vol) < 10:
            vol_clusters.iloc[i] = 1  # default medium if not enough history
            continue
        try:
            cluster = pd.qcut(past_vol, q=n_clusters, labels=False, duplicates="drop")
            vol_clusters.iloc[i] = cluster.iloc[-1]
        except Exception:
            vol_clusters.iloc[i] = 1  # fallback medium

    vol_clusters = vol_clusters.fillna(1)
    return vol_clusters



def detect_cross_signals(price: pd.Series, sma: pd.Series) -> pd.Series:
    """Detect SMA crossover signals"""
    prev_price = price.shift(1)
    prev_sma = sma.shift(1)
    cross_above = (prev_price <= prev_sma) & (price > sma)
    cross_below = (prev_price >= prev_sma) & (price < sma)
    signals = pd.Series(0, index=price.index)
    signals[cross_above.fillna(False)] = 1
    signals[cross_below.fillna(False)] = -1
    return signals


def generate_position_from_signals(signals: pd.Series) -> pd.Series:
    """Hold position until opposite signal appears"""
    pos = pd.Series(index=signals.index, dtype=float)
    current = 0
    for idx, s in signals.items():
        if s != 0:
            current = s
        pos.at[idx] = current
    return pos


def backtest_strategy(df: pd.DataFrame, sma_period: int,
                      initial_capital: float = 100000.0,
                      tx_cost: float = 0.0) -> Dict:
    """Run SMA crossover backtest"""
    df = df.copy().sort_index()
    returns = compute_returns(df)
    sma = compute_sma(df, sma_period)
    # Volume clustering
    df["vol_cluster"] = assign_volume_clusters(df)
    signals = detect_cross_signals(df["mid_price"], sma)
    # Filter signals: only act in medium/high-volume regimes
    signals[df["vol_cluster"] < 1] = 0
    raw_positions = generate_position_from_signals(signals)

    # Note: signals are executed on the next bar (no lookahead).
    position = raw_positions.shift(1).fillna(0).astype(int)
    strat_ret = position * returns
    strat_ret = strat_ret.fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret.cumsum())

    # Transaction costs — charged at execution (next bar) proportional to trade notional
    trades_mask = raw_positions != raw_positions.shift(1)
    trade_indices = trades_mask[trades_mask].index

    if tx_cost > 0 and len(trade_indices) > 0:
        cost_series = pd.Series(0.0, index=equity.index)
        # shift trade times forward by one bar to align with execution
        trade_exec_indices = []
        idx_list = equity.index.to_list()
        for t in trade_indices:
            try:
                next_idx = idx_list[idx_list.index(t) + 1]
                trade_exec_indices.append(next_idx)
            except (ValueError, IndexError):
                continue  # skip last signal with no next bar

        for exec_t in trade_exec_indices:
            cost_at_exec = equity.loc[exec_t] * tx_cost
            cost_series.loc[exec_t:] -= cost_at_exec
        equity = equity + cost_series

    total_return = (equity.iloc[-1] / initial_capital - 1) * 100

    # Count every executed order (every change in raw_positions)
    num_trades = int((raw_positions != raw_positions.shift(1)).sum())

    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min() * 100

    # Annualization assumes 1-min data 24/7 crypto
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(365*24*60) if strat_ret.std() != 0 else np.nan

    return {
        "sma_period": sma_period,
        "equity": equity,
        "strategy_returns": strat_ret,
        "position": position,
        "signals": signals,
        "total_return_pct": total_return,
        "num_trades": num_trades,
        "max_drawdown_pct": max_drawdown,
        "sharpe": sharpe,
        "tx_cost": tx_cost
    }


# -------------------------
# Plot & summary
# -------------------------
def plot_equity(equity: pd.Series, title: str):
    plt.figure(figsize=(12, 5))
    plt.plot(equity.index, equity.values)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Equity (USD)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def summarise_result(res: Dict) -> None:
    print(f"SMA period: {res['sma_period']}")
    print(f"Initial capital: 100,000 USD")
    print(f"Transaction cost: {res['tx_cost']:.6f}")
    print(f"Total return: {res['total_return_pct']:.2f}%")
    print(f"Number of trades: {res['num_trades']}")
    print(f"Max drawdown: {res['max_drawdown_pct']:.2f}%")
    print(f"Sharpe ratio: {res['sharpe']:.3f}")
    if "vol_cluster" in df.columns:
        print("Volume-aware filter active: low-volume signals suppressed.")



# -------------------------
# Load data
# -------------------------
def load_data(path: str) -> pd.DataFrame:
    """
    Load CSV or TSV or Excel file into a DataFrame.
    Handles timestamps in seconds, milliseconds, or ISO strings.
    Expected columns: timestamp, mid_price, (volume optional)
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Input file not found: {path}")

    df = pd.read_csv(path)
    df.columns = [c.lower() for c in df.columns]
    if "timestamp" not in df or "mid_price" not in df:
        raise ValueError("CSV must have 'timestamp' and 'mid_price' columns")

    # Clean timestamp values (strip possible trailing T or Z)
    df["timestamp"] = df["timestamp"].astype(str).str.replace("T", "", regex=False).str.replace("Z", "", regex=False)

    # Robust timestamp parsing
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    numeric_mask = ts.notna()

    if numeric_mask.mean() > 0.9:
        # mostly numeric
        unit = "ms" if ts.dropna().astype(str).str.len().median() > 11 else "s"
        df["timestamp"] = pd.to_datetime(ts, unit=unit, utc=True)
    else:
        # parse as datetime strings
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")

    # Drop rows that couldn't be parsed
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()
    return df


# -------------------------
# Execution (no CLI)
# -------------------------
input_file = "BTCUSDT_price_data_2024-01-24.csv"
sma_periods = [10, 20, 50]   # test multiple SMA windows
tx_cost = 0.0002             # example 0.02%
initial_capital = 100000.0

print(f"Running backtest on {input_file}...\n")
df = load_data(input_file)

results = {}
for sma in sma_periods:
    res = backtest_strategy(df, sma, initial_capital, tx_cost)
    results[sma] = res
    print("\n============================")
    summarise_result(res)
    plot_equity(res["equity"], f"Equity Curve (SMA {sma})")

# Comparison plot
plt.figure(figsize=(12, 6))
for sma, res in results.items():
    plt.plot(res["equity"].index, res["equity"].values, label=f"SMA {sma}")
plt.legend()
plt.title("Equity Curves Comparison")
plt.xlabel("Timestamp")
plt.ylabel("Equity (USD)")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
