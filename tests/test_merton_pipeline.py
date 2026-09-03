import numpy as np
from src.math_core.merton_solver import MertonSolver


def run_pipeline_verification():
    solver = MertonSolver()

    # Shared macroeconomic parameters
    r = 0.042   # 4.2% Risk-Free Rate
    T = 4.0     # 4.0 Years Dynamic Remaining Maturity (per specification)

    print("=" * 75)
    print("       STAGE 2.5: MERTON SOLVER STOCHASTIC VERIFICATION        ")
    print("-" * 75)

    # -------------------------------------------------------------
    # Case 1: Healthy Firm Profile (High Equity, Low Leverage)

    E_healthy = 1_000_000.0   # $1,000,000 Equity Market Cap
    D_healthy = 200_000.0     # $200,000 Debt Barrier
    sigma_E_healthy = 0.25    # 25% Equity Volatility

    V_h, sigma_V_h = solver.solve_asset_parameters(
        E=E_healthy, sigma_E=sigma_E_healthy, D=D_healthy, r=r, T=T
    )
    dd_h = solver.compute_distance_to_default(
        V=V_h, sigma_V=sigma_V_h, D=D_healthy, mu_V=r, T=T
    )
    pd_h = solver.compute_probability_of_default(dd_h)

    print("\n[Case 1: Healthy Firm]")
    print(f"  Inputs  : E={E_healthy:,.0f}, D={D_healthy:,.0f}, sigma_E={sigma_E_healthy:.2%}, T={T}y")
    print(f"  Solved  : Implied Asset Value (V)       = {V_h:,.2f}")
    print(f"            Implied Asset Volatility (σ_V) = {sigma_V_h:.2%}")
    print(f"  Outputs : Distance-to-Default (DD)       = {dd_h:.4f} σ")
    print(f"            Default Probability (PD)       = {pd_h:.6%}")

    # Theoretical sanity assertions
    assert V_h >= E_healthy, "Asset value V must be greater than or equal to equity value E."
    assert sigma_V_h <= sigma_E_healthy, "Asset volatility must be less than or equal to equity volatility."
    assert dd_h >= 3.0, "Healthy profile should have Distance-to-Default >= 3.0."

    # -------------------------------------------------------------
    # Case 2: Distressed Firm Profile (High Leverage, Compressed Equity)

    E_distressed = 150_000.0  # $150,000 Equity Market Cap
    D_distressed = 850_000.0  # $850,000 Debt Barrier
    sigma_E_distressed = 0.75 # 75% Equity Volatility

    V_d, sigma_V_d = solver.solve_asset_parameters(
        E=E_distressed, sigma_E=sigma_E_distressed, D=D_distressed, r=r, T=T
    )
    dd_d = solver.compute_distance_to_default(
        V=V_d, sigma_V=sigma_V_d, D=D_distressed, mu_V=r, T=T
    )
    pd_d = solver.compute_probability_of_default(dd_d)

    print("\n[Case 2: Distressed Firm]")
    print(f"  Inputs  : E={E_distressed:,.0f}, D={D_distressed:,.0f}, sigma_E={sigma_E_distressed:.2%}, T={T}y")
    print(f"  Solved  : Implied Asset Value (V)       = {V_d:,.2f}")
    print(f"            Implied Asset Volatility (σ_V) = {sigma_V_d:.2%}")
    print(f"  Outputs : Distance-to-Default (DD)       = {dd_d:.4f} σ")
    print(f"            Default Probability (PD)       = {pd_d:.6%}")

    # Comparative assertions
    assert dd_h > dd_d, "Healthy Distance-to-Default must exceed distressed Distance-to-Default."
    assert pd_d > pd_h, "Distressed default probability must exceed healthy default probability."

    print("\n" + "-" * 65)
    print("           ALL MATHEMATICAL VERIFICATION CHECKS PASSED         ")
    print("-" * 65)


if __name__ == "__main__":
    run_pipeline_verification()