import os
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.main import app
from src.api.schemas import RiskReport
from src.ingestion.market_data import MarketDataFetcher
from src.math_core.merton_solver import MertonSolver
from src.math_core.vector_math import compute_d1_d2, standard_normal_cdf


# =====================================================================
# 1. PYDANTIC SCHEMA VALIDATION TESTS


class TestRiskReportSchema:
    def test_valid_risk_report_serialization(self):
        payload = {
            "ticker": "AAPL",
            "distance_to_default": 8.45,
            "merton_analytical_pd": 1.45e-17,
            "probability_of_distress": 0.012,
            "geometric_distance_to_distress": 4.12,
            "geometric_drift_velocity": -0.05,
            "geometric_drift_alert": False,
            "alert_status": "NORMAL",
            "shap_top_drivers": {
                "distance_to_default": -1.25,
                "asset_volatility": -0.84,
                "delta_dd_30d": -0.12,
            },
        }
        report = RiskReport(**payload)
        assert report.ticker == "AAPL"
        assert report.alert_status == "NORMAL"
        assert report.geometric_drift_alert is False
        assert len(report.shap_top_drivers) == 3

    def test_risk_report_optional_geometric_defaults(self):
        report = RiskReport(
            ticker="MSFT",
            distance_to_default=9.1,
            merton_analytical_pd=0.0,
            probability_of_distress=0.005,
            alert_status="NORMAL",
            shap_top_drivers={"distance_to_default": -1.5},
        )
        assert report.geometric_distance_to_distress is None
        assert report.geometric_drift_velocity is None
        assert report.geometric_drift_alert is False

    def test_risk_report_rejects_invalid_types(self):
        with pytest.raises(ValidationError):
            RiskReport(
                ticker="AAPL",
                distance_to_default="not-a-float",
                merton_analytical_pd=0.01,
                probability_of_distress=0.02,
                alert_status="NORMAL",
                shap_top_drivers={},
            )


# =====================================================================
# 2. MERTON SOLVER & VECTOR MATH EDGE-CASE TESTS


class TestMertonSolverEdgeCases:
    def setup_method(self):
        self.solver = MertonSolver()

    def test_healthy_vs_distressed_invariants(self):
        r, T = 0.045, 4.0

        V_h, sigma_h = self.solver.solve_asset_parameters(
            E=1_000_000.0, sigma_E=0.22, D=200_000.0, r=r, T=T
        )
        dd_h = self.solver.compute_distance_to_default(V_h, sigma_h, 200_000.0, r, T)
        pd_h = self.solver.compute_probability_of_default(dd_h)

        V_d, sigma_d = self.solver.solve_asset_parameters(
            E=50_000.0, sigma_E=0.85, D=2_000_000.0, r=r, T=T
        )
        dd_d = self.solver.compute_distance_to_default(V_d, sigma_d, 2_000_000.0, r, T)
        pd_d = self.solver.compute_probability_of_default(dd_d)

        assert V_h >= 1_000_000.0
        assert sigma_h <= 0.22
        assert dd_h > dd_d
        assert pd_d > pd_h
        assert np.isclose(pd_d, standard_normal_cdf(-dd_d))

    def test_near_zero_equity_extreme_leverage_stability(self):
        """Ensures solver does not produce NaN or Inf when E -> 0 and D is massive."""
        V, sigma_V = self.solver.solve_asset_parameters(
            E=1.0, sigma_E=1.50, D=10_000_000.0, r=0.05, T=4.0
        )
        dd = self.solver.compute_distance_to_default(V, sigma_V, 10_000_000.0, 0.05, 4.0)
        pd_val = self.solver.compute_probability_of_default(dd)

        assert np.isfinite(V) and V > 0
        assert np.isfinite(sigma_V) and sigma_V > 0
        assert np.isfinite(dd) and dd < 0
        assert 0.5 < pd_val <= 1.0

    def test_defensive_clamping_on_non_positive_inputs(self):
        """Validates compute_d1_d2 prevents division-by-zero or log(<=0) errors."""
        d1, d2 = compute_d1_d2(V=0.0, D=0.0, r=0.04, sigma_V=0.0, T=0.0)
        assert np.isfinite(d1)
        assert np.isfinite(d2)


# =====================================================================
# 3. LOCAL CACHE FALLBACK TESTS (RESILIENCY AGAINST API TIMEOUTS)
# =====================================================================


class TestLocalCacheFallback:
    def test_market_fetcher_falls_back_to_csv_cache_on_timeout(self, tmp_path):
        cache_file = tmp_path / "historical_panel.csv"
        cached_df = pd.DataFrame(
            {
                "date": ["2026-11-10", "2026-11-11"],
                "ticker": ["AAPL", "AAPL"],
                "distance_to_default": [9.8, 10.1],
                "asset_volatility": [0.21, 0.20],
                "yield_spread": [0.012, 0.013],
                "sector_vix": [14.5, 14.2],
                "close_price": [225.0, 227.5],
                "market_cap": [3_400_000_000_000.0, 3_450_000_000_000.0],
            }
        )
        cached_df.to_csv(cache_file, index=False)

        fetcher = MarketDataFetcher(cache_path=str(cache_file))

        with patch("src.ingestion.market_data.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = RuntimeError("Yahoo Finance API timeout")
            equity_val, equity_vol, price = fetcher.get_equity_metrics("AAPL")

        assert equity_val > 0
        assert equity_vol > 0
        assert price > 0

    def test_market_fetcher_uses_structural_baseline_when_cache_unavailable(self, tmp_path):
        missing_cache = tmp_path / "nonexistent_panel.csv"
        fetcher = MarketDataFetcher(cache_path=str(missing_cache))

        with patch("src.ingestion.market_data.yf.Ticker") as mock_ticker:
            mock_ticker.side_effect = RuntimeError("Network unreachable")
            equity_val, equity_vol, price = fetcher.get_equity_metrics("UNKNOWN_TICKER")

        assert equity_val > 0
        assert equity_vol > 0
        assert price > 0


# =====================================================================
# 4. FASTAPI ENDPOINT INTEGRATION TESTS

class TestFastAPIEndpoints:
    def setup_method(self):
        self.client = TestClient(app)

    def test_health_check_endpoint(self):
        response = self.client.get("/")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "online"

    @patch("src.api.main.macro_fetcher")
    @patch("src.api.main.market_fetcher")
    @patch("src.api.main.sec_ingestor")
    def test_predict_endpoint_healthy_ticker_returns_normal(
        self, mock_sec, mock_market, mock_macro
    ):
        mock_sec.fetch_company_facts = AsyncMock(return_value={"mock": "facts"})
        mock_sec.extract_liabilities = MagicMock(
            return_value={"total_debt_D": 50_000_000_000.0}
        )
        mock_market.get_equity_metrics = MagicMock(
            return_value=(3_000_000_000_000.0, 0.22, 200.0)
        )
        mock_macro.get_risk_free_rate = MagicMock(return_value=0.042)
        mock_macro.get_yield_curve_spread = MagicMock(return_value=0.015)
        mock_macro.get_sector_vix = MagicMock(return_value=15.0)

        response = self.client.get("/predict/aapl")
        assert response.status_code == 200

        data = response.json()
        validated = RiskReport(**data)
        assert validated.ticker == "AAPL"
        assert validated.distance_to_default > 4.0
        assert validated.merton_analytical_pd < 0.01
        assert validated.alert_status == "NORMAL"
        assert isinstance(validated.shap_top_drivers, dict)
        assert len(validated.shap_top_drivers) > 0

    @patch("src.api.main.macro_fetcher")
    @patch("src.api.main.market_fetcher")
    @patch("src.api.main.sec_ingestor")
    def test_predict_endpoint_distressed_inputs_triggers_alert(
        self, mock_sec, mock_market, mock_macro
    ):
        mock_sec.fetch_company_facts = AsyncMock(return_value={"mock": "facts"})
        mock_sec.extract_liabilities = MagicMock(
            return_value={"total_debt_D": 5_000_000_000.0}
        )
        # Tiny equity ($40M) against $5B debt with 90% equity volatility
        mock_market.get_equity_metrics = MagicMock(
            return_value=(40_000_000.0, 0.90, 1.20)
        )
        mock_macro.get_risk_free_rate = MagicMock(return_value=0.045)
        mock_macro.get_yield_curve_spread = MagicMock(return_value=-0.02)
        mock_macro.get_sector_vix = MagicMock(return_value=38.0)

        response = self.client.get("/predict/DISTRESSED_TEST")
        assert response.status_code == 200

        data = response.json()
        validated = RiskReport(**data)
        assert validated.ticker == "DISTRESSED_TEST"
        assert validated.distance_to_default < 0.0
        assert validated.merton_analytical_pd > 0.50
        assert validated.alert_status == "ALERT"

    @patch("src.api.main.sec_ingestor")
    def test_predict_endpoint_handles_upstream_failure_with_500(self, mock_sec):
        mock_sec.fetch_company_facts = AsyncMock(
            side_effect=ValueError("Ticker not found in SEC CIK registry")
        )
        response = self.client.get("/predict/INVALID_XYZ")
        assert response.status_code == 500
        assert "Ticker not found" in response.json()["detail"]