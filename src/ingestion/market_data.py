from typing import Tuple
import pandas as pd
import yfinance as yf


class MarketDataFetcher:
    """Fetches equity pricing, market cap, and trailing volatility via yfinance[cite: 1, 2]."""

    def __init__(self) -> None:
        pass

    def get_equity_metrics(
        self, ticker: str, window_days: int = 252
    ) -> Tuple[float, float, float]:
        # TODO: Fetch daily equity prices and shares outstanding[cite: 1]
        # TODO: Compute equity value (E) and trailing equity volatility (sigma_E)[cite: 1]
        pass

    def get_historical_close(
        self, ticker: str, period: str = "1y"
    ) -> pd.DataFrame:
        # TODO: Fetch time-series historical pricing data[cite: 1]
        pass