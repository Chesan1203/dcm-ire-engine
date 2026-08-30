from typing import Tuple
import pandas as pd
import yfinance as yf
import numpy as np


class MarketDataFetcher:
    """Fetches equity pricing, market cap, and trailing volatility via yfinance[cite: 1, 2]."""

    def __init__(self) -> None:
        pass

    def get_equity_metrics(self, ticker: str, window_days: int = 252) -> Tuple[float, float, float]:
        """Fetches history and runs validation."""
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1y")

        # Defensive validation
        if hist.empty or len(hist) < 30:
            raise ValueError(
                f"Insufficient pricing history available for ticker: {ticker}"
            )

        # Extract price and compute Market Equity Value (E)
        current_price = float(hist["Close"].iloc[-1])
        shares_outstanding = stock.info.get("sharesOutstanding")

        if not shares_outstanding:
            shares_outstanding = stock.fast_info.get("shares", 0)

        if shares_outstanding == 0:
            market_cap = stock.fast_info.get(
                "market_cap", current_price * 1_000_000
            )
        else:
            market_cap = float(current_price * shares_outstanding)

        # Compute daily log returns and annualized sample standard deviation
        log_returns = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
        trailing_window = log_returns.tail(window_days)
        sigma_E = float(trailing_window.std() * np.sqrt(252))

        return market_cap, sigma_E, current_price
        pass

    def get_historical_close(
        self, ticker: str, period: str = "1y"
    ) -> pd.DataFrame:
        # TODO: Fetch time-series historical pricing data[cite: 1]
        pass