import numpy as np
import pandas as pd
import joblib

from src.math_core.merton_solver import MertonSolver
from src.ml_engine.classifier import DistressClassifier


def run_bad_data_stress_test():
    print("=" * 65)
    print("      DCM-IRE ENGINE: CATASTROPHIC STRESS & BAD DATA TEST      ")
    print("=" * 65)

    # 1. Initialize solver & load trained XGBoost model
    solver = MertonSolver()
    try:
        classifier = joblib.load("models/distress_xgboost.pkl")
        print("[Setup] Successfully loaded trained XGBoost model from models/distress_xgboost.pkl")
    except FileNotFoundError:
        print("[Error] Trained model artifact not found. Please run 'python -m scripts.train_model' first.")
        return

    # 2. Define CATASTROPHIC "Bad Data" Parameters
    # Simulating a severely distressed firm: Tiny equity, massive debt, extreme volatility
    E_bad = 50_000_000.0      # Tiny market cap ($50M)
    D_bad = 2_000_000_000.0   # Huge debt barrier ($2B)
    sigma_E_bad = 0.85        # 85% extreme equity volatility
    r = 0.045                 # 4.5% risk-free rate
    T = 4.0                   # 4 years maturity

    print(f"\n[Input Parameters - Distressed Scenario]")
    print(f"  - Market Equity (E)       : ${E_bad:,.2f}")
    print(f"  - Debt Barrier (D)        : ${D_bad:,.2f}")
    print(f"  - Equity Volatility (σ_E) : {sigma_E_bad:.2%}")

    # 3. Solve Merton structural parameters for bad data
    V, sigma_V = solver.solve_asset_parameters(
        E=E_bad, sigma_E=sigma_E_bad, D=D_bad, r=r, T=T
    )
    dd = solver.compute_distance_to_default(
        V=V, sigma_V=sigma_V, D=D_bad, mu_V=r, T=T
    )
    pd_merton = solver.compute_probability_of_default(dd)

    print(f"\n[Merton Structural Solver Output]")
    print(f"  - Implied Asset Value (V)       : ${V:,.2f}")
    print(f"  - Implied Asset Volatility (σ_V) : {sigma_V:.2%}")
    print(f"  - Distance-to-Default (DD)       : {dd:.4f} σ")
    print(f"  - Merton Theoretical PD          : {pd_merton:.2%}")

    # 4. Construct ML feature row simulating acute stress and negative momentum
    feature_row = pd.DataFrame([{
        "distance_to_default": dd,
        "asset_volatility": sigma_V,
        "yield_spread": 0.25,        # Widened yield spread (stress indicator)
        "sector_vix": 38.0,          # High market fear index
        "dd_lag_30": dd + 0.9,       # Simulated recent sharp drop from a higher DD
        "asset_vol_lag_30": sigma_V - 0.25,
        "yield_spread_lag_30": 0.15,
        "vix_lag_30": 18.0,
        "delta_dd_30d": -0.9,        # Sharp negative momentum
        "delta_asset_vol_30d": 0.25
    }])

    # 5. Run ML Classifier Inference & Threshold Check
    prob = float(classifier.model.predict_proba(feature_row)[:, 1][0])
    threshold = getattr(classifier, "optimal_threshold_", 0.5)
    alert = "ALERT" if prob >= threshold else "NORMAL"

    print(f"\n[Machine Learning Inference Output]")
    print(f"  - ML Distress Probability : {prob:.2%}")
    print(f"  - Calibrated F1 Threshold : {threshold:.4f}")
    print(f"  - Final Alert Status      : {alert}")

    print("\n" + "=" * 65)
    if alert == "ALERT":
        print("          SUCCESS: ENGINE CORRECTLY FLAGGED DISTRESS ALERT         ")
    else:
        print("          NOTE: EVALUATED AS NORMAL UNDER CURRENT THRESHOLD        ")
    print("=" * 65)


if __name__ == "__main__":
    run_bad_data_stress_test()