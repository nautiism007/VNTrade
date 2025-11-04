# SMA Backtest

Simple Python script for backtesting an SMA crossover strategy on 1-minute BTC data.  
Buys when price crosses above SMA, sells when below. Tracks total return, trades, drawdown, and Sharpe ratio.  
Supports multiple SMA periods and optional transaction cost.  

Input CSV must have: timestamp, mid_price (volume optional).  
Edit file paths and SMA settings in the script, then run:  
    python3 sma_backtest.py  

Requires: pandas, numpy, matplotlib.
