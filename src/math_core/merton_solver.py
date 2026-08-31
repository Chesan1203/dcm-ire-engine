from typing import Tuple
import numpy as np
from scipy.optimize import root
from scipy.stats import norm
from src.math_core.vector_math import compute_d1_d2, standard_normal_cdf


class MertonSolver:
    """Solves non-linear Black-Scholes-Merton system for unobservable assets[cite: 1, 2]."""

    def __init__(self) -> None:
        pass

    def _merton_objective(
        self,
        vars: np.ndarray,
        E: float,
        sigma_E: float,
        D: float,
        r: float,
        T: float,
    ) -> np.ndarray:
        """
        Computes residuals for equity value E and equity volatility sigma_E.
        Target is [0.0, 0.0] when V and sigma_V are correct.
        """
        V_candidate, sigma_V_candidate = float(vars[0]), float(vars[1])

        # 1. Compute d1 and d2 via defensive vector helper
        d1, d2 = compute_d1_d2(
            V=V_candidate,
            D=D,
            r=r,
            sigma_V=sigma_V_candidate,
            T=T
        )

        # 2. Evaluate normal CDF probabilities
        Nd1 = standard_normal_cdf(d1)
        Nd2 = standard_normal_cdf(d2)

        # 3. Compute residual differences
        discount_factor = np.exp(-r * T)
        equity_residual = (V_candidate * Nd1) - (discount_factor * D * Nd2) - E
        volatility_residual = ((V_candidate / max(E, 1e-8)) * Nd1 * sigma_V_candidate) - sigma_E

        return np.array([equity_residual, volatility_residual], dtype=float)

    def solve_asset_parameters(
        self, E: float, sigma_E: float, D: float, r: float, T: float
    ) -> Tuple[float, float]:
        # 1. Initial Guesses
        V_0 = float(E + D)
        sigma_V0 = float(sigma_E * (E / max(E + D, 1e-8)))
        x0 = np.array([V_0, sigma_V0], dtype=float)

        # 2. Execute Levenberg-Marquardt root solving
        sol = root(
            self._merton_objective,
            x0=x0,
            args=(E, sigma_E, D, r, T),
            method="lm",
        )

        # 3. Defensive validation and fallback
        if sol.success:
            V_sol = float(sol.x[0])
            sigma_V_sol = float(sol.x[1])
            return max(V_sol, 1e-8), max(sigma_V_sol, 1e-8) # To ensure non-negativity
        else:
            # Fallback to initial guess approximations if solver diverges
            return V_0, sigma_V0


    def compute_distance_to_default(
        self, V: float, sigma_V: float, D: float, mu_V: float, T: float, eps: float = 1e-8
    ) -> float:
        """
        Calculates the structural Distance-to-Default (DD) metric:
        DD = [ln(V / D) + (mu_V - 0.5 * sigma_V^2) * T] / (sigma_V * sqrt(T))
        """
        # 1. Defensive bounds
        V_safe = max(float(V), eps)
        D_safe = max(float(D), eps)
        sigma_safe = max(float(sigma_V), eps)
        T_safe = max(float(T), eps)

        # 2. Evaluate analytical Distance-to-Default
        numerator = np.log(V_safe / D_safe) + (mu_V - 0.5 * (sigma_safe ** 2)) * T_safe
        denominator = sigma_safe * np.sqrt(T_safe)

        return float(numerator / denominator)

    def compute_probability_of_default(self, distance_to_default: float) -> float:
        """
        Computes the theoretical market-implied probability of default: PD = N(-DD)
        """
        return standard_normal_cdf(-float(distance_to_default))