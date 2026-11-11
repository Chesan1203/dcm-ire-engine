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

sec_ingestor = SECEdgarIngestor()
market_fetcher = MarketDataFetcher()
macro_fetcher = MacroFREDFetcher()
merton_solver = MertonSolver()

classifier = DistressClassifier(scale_pos_weight=3.0)
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
    # TODO: Fetch live data asynchronously (SEC EDGAR / YFinance)
    # TODO: Run SciPy Merton Solver for unobservable assets
    # TODO: Execute XGBoost Inference & SHAP Extraction
    # TODO: Return structured JSON payload to the client
    try:

        clean_ticker = ticker.upper()

        # Step 1: Fetch live data asynchronously across sources
        cik = await sec_ingestor.get_cik(clean_ticker)
        financials = await sec_ingestor.get_company_facts(cik)
        market_data = market_fetcher.get_historical_data(clean_ticker)
        macro_data = macro_fetcher.get_latest_macro()

        # Step 2: Extract core variables for Merton solver
        equity_val = market_data['market_cap']
        equity_vol = market_data['equity_volatility']
        debt_barrier = financials['total_debt']
        risk_free_rate = macro_data['risk_free_rate']

        # Step 3: Run SciPy LM Merton Root Solver
        merton_result = merton_solver.solve(
            E=equity_val,
            sigma_E=equity_vol,
            D=debt_barrier,
            r=risk_free_rate,
            T=4.0  # Default structural maturity
        )
        
        distance_to_default = merton_result['DD']
        implied_asset_vol = merton_result['sigma_V']

        # Step 4: Construct feature vector for ML inference & SHAP extraction
        feature_row = pd.DataFrame([{
            "distance_to_default": distance_to_default,
            "asset_volatility": implied_asset_vol,
            "yield_curve_spread": macro_data['yield_spread'],
            "sector_vix": macro_data['vix']
        }])

        if not hasattr(classifier, "optimal_threshold_"):
            classifier.optimal_threshold_ = 0.5  # Default fallback threshold

        probability_of_distress = float(classifier.model.predict_proba(feature_row)[:, 1][0])
        alert_status = "NORMAL" if probability_of_distress < classifier.optimal_threshold_ else "ALERT"
        
        shap_drivers = explainer.get_top_drivers(feature_row)

        # Mock structural payload for microservice wiring verification
        return RiskReport(
            ticker=clean_ticker,
            distance_to_default=float(distance_to_default),
            probability_of_distress=float(probability_of_distress),
            alert_status=alert_status,
            shap_top_drivers=shap_drivers
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))