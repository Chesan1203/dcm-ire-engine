import os
import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Any
import pandas as pd
import numpy as np

# Import validation schemas
from src.api.schemas import RiskReport

# Import quantitative pipeline modules
from src.ingestion.sec_edgar import SECEdgarIngestor
from src.ingestion.market_data import MarketDataFetcher
from src.ingestion.macro_fred import MacroFREDFetcher
from src.math_core.merton_solver import MertonSolver
from src.ml_engine.classifier import DistressClassifier
from src.ml_engine.explainer import DistressExplainer

# --- Global Service & Ingestion Instances ---
sec_ingestor = SECEdgarIngestor()
market_fetcher = MarketDataFetcher()
macro_fetcher = MacroFREDFetcher()
merton_solver = MertonSolver()

# --- Load Trained Model Checkpoint ---
MODEL_PATH = "models/distress_xgboost.pkl"
if os.path.exists(MODEL_PATH):
    classifier = joblib.load(MODEL_PATH)
else:
    classifier = DistressClassifier(scale_pos_weight=3.0)
    classifier.optimal_threshold_ = 0.5

# Instantiate SHAP explainer on the trained classifier model
explainer = DistressExplainer(classifier.model)

app = FastAPI(
    title="DCM-IRE Credit Risk Engine API",
    description="Asynchronous microservice for real-time Merton structural solving, XGBoost default forecasting, and SHAP risk attributions.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root() -> Dict[str, str]:
    """Health check endpoint to verify API container status."""
    return {
        "status": "online",
        "system": "Dynamic Credit Migration & Market-Implied Risk Engine (DCM-IRE)"
    }

@app.get("/predict/{ticker}", response_model=RiskReport)
async def get_risk(ticker: str) -> RiskReport:
    """
    Asynchronous risk prediction endpoint for a given equity ticker:
    1. Ingests live SEC EDGAR, YFinance, and FRED data.
    2. Runs SciPy Levenberg-Marquardt Merton solver for unobservable asset metrics.
    3. Executes trained XGBoost inference using PR-calibrated optimal threshold.
    4. Computes SHAP risk attributions and returns structured JSON payload.
    """
    try:
        clean_ticker = ticker.upper()

        # Step 1: Fetch live data across sources using ingestion class methods
        facts_json = await sec_ingestor.fetch_company_facts(clean_ticker)
        liabilities = sec_ingestor.extract_liabilities(facts_json)

        equity_val, equity_vol, current_price = market_fetcher.get_equity_metrics(clean_ticker)

        risk_free_rate = macro_fetcher.get_risk_free_rate()
        yield_spread = macro_fetcher.get_yield_curve_spread()
        sector_vix = macro_fetcher.get_sector_vix()

        # Step 2: Extract core variables for Merton solver
        debt_barrier = liabilities["total_debt_D"]

        # Step 3: Run SciPy LM Merton Root Solver
        V, implied_asset_vol = merton_solver.solve_asset_parameters(
            E=equity_val,
            sigma_E=equity_vol,
            D=debt_barrier,
            r=risk_free_rate,
            T=4.0
        )

        distance_to_default = merton_solver.compute_distance_to_default(
            V=V,
            sigma_V=implied_asset_vol,
            D=debt_barrier,
            mu_V=risk_free_rate,  # Uses risk-free rate as baseline drift mu_V
            T=4.0
        )

        # Step 4: Construct feature row matching full 10-feature training set
        feature_row = pd.DataFrame([{
            "distance_to_default": distance_to_default,
            "asset_volatility": implied_asset_vol,
            "yield_spread": yield_spread,
            "sector_vix": sector_vix,
            "dd_lag_30": distance_to_default,
            "asset_vol_lag_30": implied_asset_vol,
            "yield_spread_lag_30": yield_spread,
            "vix_lag_30": sector_vix,
            "delta_dd_30d": 0.0,
            "delta_asset_vol_30d": 0.0
        }])

        # Step 5: Execute trained model inference & SHAP feature attributions
        probability_of_distress = float(classifier.model.predict_proba(feature_row)[:, 1][0])

        threshold = getattr(classifier, "optimal_threshold_", 0.5)
        alert_status = "ALERT" if probability_of_distress >= threshold else "NORMAL"

        shap_drivers = explainer.get_top_drivers(feature_row)

        return RiskReport(
            ticker=clean_ticker,
            distance_to_default=float(distance_to_default),
            probability_of_distress=probability_of_distress,
            alert_status=alert_status,
            shap_top_drivers=shap_drivers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))