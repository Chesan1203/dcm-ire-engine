import os
import numpy as np
import pandas as pd
import yfinance as yf

from src.math_core.merton_solver import MertonSolver


def fetch_historical_macro(start_date: str = "2007-01-01") -> pd.DataFrame:
    """Fetches real daily historical 10Y yield, 3M T-bill proxy, and VIX."""
    tnx = yf.Ticker("^TNX").history(start=start_date)["Close"] / 100.0
    irx = yf.Ticker("^IRX").history(start=start_date)["Close"] / 100.0
    vix = yf.Ticker("^VIX").history(start=start_date)["Close"]

    macro_df = pd.DataFrame({
        "risk_free_rate": tnx,
        "short_rate": irx,
        "sector_vix": vix,
    }).dropna()

    macro_df["yield_spread"] = macro_df["risk_free_rate"] - macro_df["short_rate"]
    macro_df.index = macro_df.index.tz_localize(None).strftime("%Y-%m-%d")
    return macro_df


def fetch_ticker_panel_data(
    ticker: str,
    macro_df: pd.DataFrame,
    solver: MertonSolver,
    start_date: str = "2007-01-01"
) -> pd.DataFrame:
    """
    Fetches multi-decade daily prices and balance sheet debt for a ticker,
    joins with daily historical macro data, and solves Merton DD per day.
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(start=start_date, timeout=15)
    if hist.empty or len(hist) < 120:
        print(f"[Skip] Insufficient history for {ticker}")
        return pd.DataFrame()

    hist.index = hist.index.tz_localize(None).strftime("%Y-%m-%d")

    # Extract shares outstanding and balance sheet debt barrier
    shares = stock.info.get("sharesOutstanding") or stock.fast_info.get("shares", 0)
    if not shares or shares <= 0:
        latest_p = float(hist["Close"].iloc[-1])
        mcap = stock.fast_info.get("market_cap", latest_p * 100_000_000)
        shares = mcap / latest_p

    # Pull total debt from balance sheet (fallback to fast_info/estimate if unavailable)
    debt_barrier = 10_000_000_000.0
    try:
        bs = stock.balance_sheet
        if bs is not None and not bs.empty:
            total_debt = bs.loc["Total Debt"].dropna().iloc[0] if "Total Debt" in bs.index else None
            if total_debt is not None and total_debt > 0:
                # Approximate structural barrier: Short-Term + 0.5 * Long-Term (~0.65 * Total Debt)
                debt_barrier = float(total_debt) * 0.65
    except Exception:
        pass

    # Compute 252-day trailing annualized equity volatility
    log_returns = np.log(hist["Close"] / hist["Close"].shift(1))
    rolling_vol = (log_returns.rolling(window=252, min_periods=30).std() * np.sqrt(252)).fillna(0.30)

    joined = hist[["Close"]].join(macro_df, how="inner")
    joined["equity_vol"] = rolling_vol

    records = []
    for date_str, row in joined.iterrows():
        price = float(row["Close"])
        E = price * shares
        sigma_E = max(float(row["equity_vol"]), 0.05)
        r = max(float(row["risk_free_rate"]), 0.005)

        try:
            V, sigma_V = solver.solve_asset_parameters(
                E=E, sigma_E=sigma_E, D=debt_barrier, r=r, T=4.0
            )
            dd = solver.compute_distance_to_default(
                V=V, sigma_V=sigma_V, D=debt_barrier, mu_V=r, T=4.0
            )
        except Exception:
            continue

        records.append({
            "date": date_str,
            "ticker": ticker,
            "distance_to_default": float(dd),
            "asset_volatility": float(sigma_V),
            "yield_spread": float(row["yield_spread"]),
            "sector_vix": float(row["sector_vix"]),
            "close_price": price,
            "market_cap": E,
        })

    return pd.DataFrame(records)


def build_real_historical_panel(output_path: str = "data/processed/historical_panel.csv") -> None:
    # Universe mixing blue-chip firms with cyclical/distressed firms across 2008 & 2020 crises
    tickers = [
        "AAPL", "MSFT", "F", "GE", "BAC", "C",
        "AAL", "CCL", "RCL", "M", "X", "AMC"
    ]

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    solver = MertonSolver()

    print("Fetching daily historical macro series (2007–Present)...")
    macro_df = fetch_historical_macro(start_date="2007-01-01")

    panel_dfs = []
    for ticker in tickers:
        print(f"Building multi-crisis Merton history for {ticker}...")
        df_t = fetch_ticker_panel_data(ticker, macro_df, solver, start_date="2007-01-01")
        if not df_t.empty:
            print(f"  -> {ticker}: {len(df_t)} rows | DD range: [{df_t['distance_to_default'].min():.2f}, {df_t['distance_to_default'].max():.2f}]")
            panel_dfs.append(df_t)

    full_panel = pd.concat(panel_dfs, ignore_index=True)
    full_panel.to_csv(output_path, index=False)
    print(f"\nSaved {len(full_panel)} historical rows to {output_path}")


if __name__ == "__main__":
    build_real_historical_panel()