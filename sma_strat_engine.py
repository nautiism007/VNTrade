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
    signals = detect_cross_signals(df["mid_price"], sma)
    raw_positions = generate_position_from_signals(signals)
    position = raw_positions.shift(1).fillna(0).astype(int)
    strat_ret = position * returns
    strat_ret = strat_ret.fillna(0.0)
    equity = initial_capital * (1.0 + strat_ret.cumsum())

    # Transaction costs
    trades_mask = raw_positions != raw_positions.shift(1)
    trades_mask = trades_mask & (raw_positions != 0)
    trade_indices = trades_mask[trades_mask].index
    if tx_cost > 0 and len(trade_indices) > 0:
        cost_series = pd.Series(0.0, index=equity.index)
        cost_per_trade = initial_capital * tx_cost
        for t in trade_indices:
            cost_series.loc[t:] -= cost_per_trade
        equity += cost_series

    total_return = (equity.iloc[-1] / initial_capital - 1) * 100
    num_trades = len(trade_indices)
    drawdown = equity / equity.cummax() - 1
    max_drawdown = drawdown.min() * 100
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

    # Detect milliseconds vs seconds
    try:
        # try parsing as integers
        ts = pd.to_numeric(df["timestamp"], errors="coerce")
        if ts.dropna().astype(int).astype(str).str.len().median() > 11:
            # likely milliseconds
            df["timestamp"] = pd.to_datetime(ts, unit="ms", utc=True)
        else:
            df["timestamp"] = pd.to_datetime(ts, unit="s", utc=True)
    except Exception:
        # fallback: generic parser for string datetimes
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
