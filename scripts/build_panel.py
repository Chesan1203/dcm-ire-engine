import os
import numpy as np
import pandas as pd


def generate_historical_panel(
    output_path: str = "data/processed/historical_panel.csv",
    num_days: int = 300,
    seed: int = 42
) -> None:
    """
    Generates a continuous 300-day synthetic historical panel containing Merton 
    structural parameters and macro indicators with intentional distress shocks.
    """
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.random.seed(seed)

    dates = pd.date_range(end=pd.Timestamp.today(), periods=num_days, freq="B")

    # Baseline Distance-to-Default (DD) path starting around healthy ~4.5
    dd_base = 4.5 + np.cumsum(np.random.normal(loc=0.0, scale=0.08, size=num_days))
    
    # Inject intentional distress shock window between day 180 and 220 (>1.5 sigma drop)
    dd_base = np.random.uniform(3.5, 12.0, num_days)
    
    # Inject 30-day transient drops (returning to baseline afterwards)
    dd_base[60:90] = np.linspace(3.5, 0.8, 30)    # Train shock window
    dd_base[240:270] = np.linspace(3.5, 0.8, 30)  # Validation shock window
    
    # Implied Asset Volatility (sigma_V) spikes when DD drops
    asset_vol = 0.15 + (6.0 - dd_base) * 0.04 + np.abs(np.random.normal(0, 0.01, num_days))
    asset_vol = np.clip(asset_vol, 0.05, 0.60)

    # Yield curve spread (10Y-2Y Treasury) in % points
    yield_spread = 0.55 - (6.0 - dd_base) * 0.12 + np.random.normal(0, 0.05, num_days)

    # Sector VIX (market fear benchmark)
    sector_vix = 18.0 + (6.0 - dd_base) * 2.5 + np.random.normal(0, 0.5, num_days)

    df = pd.DataFrame({
        "date": dates,
        "distance_to_default": dd_base,
        "asset_volatility": asset_vol,
        "yield_spread": yield_spread,
        "sector_vix": sector_vix
    })

    df.to_csv(output_path, index=False)
    print(f"Synthetic historical panel successfully generated and saved to {output_path} ({len(df)} rows)")

if __name__ == "__main__":
    generate_historical_panel()