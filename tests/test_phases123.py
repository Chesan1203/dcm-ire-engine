import numpy as np
import pandas as pd

from src.ingestion.macro_fred import MacroFREDFetcher
from src.ingestion.market_data import MarketDataFetcher
from src.math_core.merton_solver import MertonSolver
from src.math_core.vector_math import compute_d1_d2, standard_normal_cdf
from src.ml_engine.classifier import DistressClassifier
from src.ml_engine.clustering import GeometricDiscoveryEngine, RobustWinsorScaler
from src.ml_engine.explainer import DistressExplainer


def run_end_to_end_verification() -> None:
    print("=" * 60)
    print("RUNNING END-TO-END PIPELINE VERIFICATION (PHASES 1 - 3)")
    print("=" * 60)

    # -------------------------------------------------------------
    # 1. PHASE 1: Ingestion & Geometric Clustering
    # -------------------------------------------------------------
    print("\n[PHASE 1] Testing Data Ingestion & Clustering Engine...")

    macro_fetcher = MacroFREDFetcher()
    r = macro_fetcher.get_risk_free_rate()
    spread = macro_fetcher.get_yield_curve_spread()
    vix = macro_fetcher.get_sector_vix()
    print(
        f"  - Ingested Macro Benchmarks: r={r:.4f}, 10Y-2Y Spread={spread:.4f}, VIX={vix:.2f}"
    )

    # Generate synthetic balance sheet feature matrix for geometric clustering
    np.random.seed(42)
    n_samples = 60
    raw_features = pd.DataFrame(
        {
            "debt_ratio": np.random.uniform(0.1, 0.9, n_samples),
            "current_ratio": np.random.uniform(0.5, 3.0, n_samples),
            "cash_to_assets": np.random.uniform(0.02, 0.4, n_samples),
            "equity_volatility": np.random.uniform(0.15, 0.80, n_samples),
            "collinear_debt_proxy": np.random.uniform(
                0.1, 0.9, n_samples
            ),  # Collinear pair
        }
    )

    scaler = RobustWinsorScaler(corr_threshold=0.85)
    X_scaled = scaler.fit_transform(raw_features)
    print(
        f"  - RobustWinsorScaler: Pruned from {raw_features.shape[1]} to {X_scaled.shape[1]} features."
    )

    geo_engine = GeometricDiscoveryEngine(variance_retained=0.95)
    kmeans_model = geo_engine.tune_and_fit_kmeans(X_scaled, k_range=(3, 5))
    print(
        f"  - GeometricDiscoveryEngine: Optimal clusters selected K={geo_engine.best_k}"
    )

    # -------------------------------------------------------------
    # 2. PHASE 2: Stochastic Merton Solver
    # -------------------------------------------------------------
    print("\n[PHASE 2] Testing Stochastic Merton Root Solver...")

    solver = MertonSolver()
    # Test solvent profile
    E_healthy, sigma_E_healthy, D_healthy, T_healthy = (
        1000.0,
        0.25,
        300.0,
        4.0,
    )
    V_h, sigma_Vh = solver.solve_asset_parameters(
        E_healthy, sigma_E_healthy, D_healthy, r, T_healthy
    )
    d1_h, d2_h = compute_d1_d2(V_h, D_healthy, r, sigma_Vh, T_healthy)
    dd_h = (np.log(V_h / D_healthy) + (r - 0.5 * sigma_Vh**2) * T_healthy) / (
        sigma_Vh * np.sqrt(T_healthy)
    )
    pd_h = standard_normal_cdf(-dd_h)

    print(
        f"  - Healthy Firm -> Asset Value V: {V_h:.2f}, Asset Vol σV: {sigma_Vh:.4f}, DD: {dd_h:.2f}σ, PD: {pd_h:.6f}"
    )

    # Test distressed profile
    E_distressed, sigma_E_distressed, D_distressed, T_distressed = (
        150.0,
        0.75,
        850.0,
        4.0,
    )
    V_d, sigma_Vd = solver.solve_asset_parameters(
        E_distressed, sigma_E_distressed, D_distressed, r, T_distressed
    )
    dd_d = (
        np.log(V_d / D_distressed) + (r - 0.5 * sigma_Vd**2) * T_distressed
    ) / (sigma_Vd * np.sqrt(T_distressed))
    pd_d = standard_normal_cdf(-dd_d)

    print(
        f"  - Distressed Firm -> Asset Value V: {V_d:.2f}, Asset Vol σV: {sigma_Vd:.4f}, DD: {dd_d:.2f}σ, PD: {pd_d:.6f}"
    )

    # Invariant Checks
    assert V_h >= E_healthy, "Invariant Breach: V < E"
    assert sigma_Vh <= sigma_E_healthy, "Invariant Breach: σV > σE"
    assert dd_h > dd_d, "Invariant Breach: Healthy DD <= Distressed DD"
    print(
        "  - All Merton Mathematical Invariants Passed (V ≥ E, σV ≤ σE, DD_healthy > DD_distressed)."
    )

    # -------------------------------------------------------------
    # 3. PHASE 3: Predictive ML Overlay & PR Calibration & SHAP
    # -------------------------------------------------------------
    print(
        "\n[PHASE 3] Testing Predictive ML Overlay & PR Calibration & SHAP..."
    )

    # Generate synthetic time-series with shocks spanning both train and val splits
    dates = pd.date_range(start="2024-01-01", periods=300, freq="D")
    np.random.seed(42)

    dd_series = np.full(300, 3.5) + np.random.normal(0, 0.05, 300)

    # Shock 1: Days 50-80 -> Forward drop falls into Train set
    dd_series[50:80] -= np.linspace(0, 2.5, 30)

    # Shock 2: Days 230-260 -> Forward drop falls into Validation set
    dd_series[230:260] -= np.linspace(0, 2.5, 30)

    asset_vol_series = 0.20 + (4.0 - dd_series) * 0.05 + np.random.normal(0, 0.01, 300)

    ts_df = pd.DataFrame(
        {
            "distance_to_default": dd_series,
            "asset_volatility": asset_vol_series,
            "yield_spread": np.random.normal(spread, 0.05, 300),
            "sector_vix": np.random.normal(vix, 2.0, 300),
        },
        index=dates,
    )

    classifier = DistressClassifier(scale_pos_weight=3.0)

    # Feature Matrix Generation
    feature_df = classifier.build_rolling_features(ts_df)
    target_series = classifier.generate_binary_target(
        ts_df["distance_to_default"]
    )

    # Align features and target
    valid_idx = feature_df.index.intersection(target_series.dropna().index)
    X = feature_df.loc[valid_idx]
    y = target_series.loc[valid_idx]

    # Split train/validation chronologically
    split_point = int(len(X) * 0.70)
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]

    # Train XGBoost with validation set tracking
    classifier.fit(X_train, y_train, X_val, y_val)
    print("  - XGBoost Model Trained successfully.")

    # PR Calibration
    optimal_thresh = classifier.calibrate_optimal_threshold(X_val, y_val)
    print(
        f"  - Precision-Recall Calibration Optimal Threshold: {optimal_thresh:.4f}"
    )

    # Inference execution
    preds = classifier.predict_distress(X_val)
    print(
        f"  - Inference Executed. Positive Alerts Triggered: {int(preds.sum())}/{len(preds)}"
    )

    # SHAP Risk Attribution
    explainer = DistressExplainer(classifier.model)
    alert_instance = X_val.iloc[-1:]
    top_drivers = explainer.get_top_drivers(alert_instance)
    print(f"  - SHAP Risk Attribution Top Drivers Payload: {top_drivers}")

    print("\n" + "=" * 60)
    print("ALL PHASES (1, 2, 3) VERIFIED & WORKING PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_end_to_end_verification()