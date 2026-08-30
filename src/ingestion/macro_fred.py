import pandas as pd
import yfinance as yf


class MacroFREDFetcher:
    """Pipeline for FRED macroeconomic data: Yield curve and VIX[cite: 1, 2]."""

    def __init__(self, api_key: str = "") -> None:
        """Stores API credentials if needed for direct FRED pipelines."""
        self.api_key = api_key

    def get_risk_free_rate(self) -> float:
        """Ingests current baseline continuous risk-free rate r (10Y Treasury proxy)."""
        try:
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")
            if hist.empty:
                return 0.042
            latest_yield = float(hist["Close"].iloc[-1])
            return latest_yield / 100.0
        except Exception:
            return 0.042
        
    def get_yield_curve_spread(self) -> float:
        """Retrieves or approximates current 10Y-2Y Treasury Yield Spread."""
        try:
            # ^TNX = 10-Year Treasury Note Yield, ^IRX = 13-Week Treasury Bill Yield
            tnx_data = yf.Ticker("^TNX").history(period="5d")
            irx_data = yf.Ticker("^IRX").history(period="5d")

            if tnx_data.empty or irx_data.empty:
                return 0.15

            # Yield index values are scaled by 10 (e.g., 42.5 means 4.25%)
            tnx_yield = float(tnx_data["Close"].iloc[-1]) / 10.0
            irx_yield = float(irx_data["Close"].iloc[-1]) / 10.0

            return float(tnx_yield - irx_yield)
        except Exception:
            return 0.15  # Fallback neutral baseline spread

    def get_sector_vix(self) -> float:
        """Retrieves the current market-wide CBOE Volatility Index (VIX) level."""
        try:
            vix_ticker = yf.Ticker("^VIX")
            hist = vix_ticker.history(period="5d")
            if hist.empty:
                return 18.0  # Fallback neutral baseline level
            return float(hist["Close"].iloc[-1])
        except Exception:
            return 18.0  # Fallback neutral baseline level