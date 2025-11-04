# SMA Backtest

Simple Python script for backtesting a Simple Moving Average (SMA) crossover strategy on 1-minute BTC data.  
Buys when price crosses above the SMA and sells when it crosses below.  
Now includes **volume clustering heuristics** — trades are only executed during medium or high-volume regimes to reduce false signals in low-liquidity periods.  

Tracks key metrics: total return, number of trades, max drawdown, and Sharpe ratio.  
Supports multiple SMA periods and optional transaction costs per trade.  

### Input Data
Input CSV must include:  
- `timestamp`: Unix timestamp (seconds or milliseconds)  
- `mid_price`: mid price at each minute  
- `volume` *(optional)*: used for volume clustering, if available  

### Usage
Edit file paths and SMA settings in the script, then run:
