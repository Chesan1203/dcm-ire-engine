# Dynamic Credit Migration & Market-Implied Risk Engine (DCM-IRE)

DCM-IRE is an academic quantitative risk engine designed to detect corporate credit deterioration early. Standard credit rating agencies often lag behind fast-moving market conditions. This project addresses that delay by combining structural option pricing (the Black-Scholes-Merton model) with unsupervised geometric clustering and gradient-boosted default forecasting.

---

## 🏗️ Architecture & Development Status

The repository is organized around four sequential engineering stages:

1. **Stage 1: Unsupervised Geometric Discovery (The Radar)** — *Completed*
2. **Stage 2: The Stochastic Math Core (The Merton Model)** — *Completed*
3. **Stage 3: Predictive ML Overlay & PR Calibration** — *In Progress*
4. **Stage 4: API Microservice & Dashboard Interface** — *Planned*

---

## 📁 Repository Layout

```text
dcm-ire-engine/
├── data/
│   ├── raw/                  # Ingested balance sheets and market prices
│   └── processed/            # Cleaned, winsorized, and scaled matrices
├── src/
│   ├── ingestion/            # Phase 1: Multi-source data ingestion pipelines
│   │   ├── sec_edgar.py      # Async SEC 10-Q parser and liability extractor
│   │   ├── market_data.py    # Equity market cap and trailing volatility
│   │   └── macro_fred.py     # Yield curve spreads (10Y-2Y), rates, and VIX
│   ├── math_core/            # Phase 2: Option pricing and root optimization
│   │   ├── vector_math.py    # Intermediate d1/d2 terms, CDF, and vector distance
│   │   └── merton_solver.py  # Levenberg-Marquardt solver and Distance-to-Default
│   ├── ml_engine/            # Phase 1 clustering & Phase 3 predictive models
│   │   ├── clustering.py     # Robust scaler, PCA, and dynamic K-Means
│   │   ├── classifier.py     # XGBoost distress model
│   │   └── explainer.py      # SHAP tree-attribution module
│   └── api/                  # Phase 4: FastAPI service schemas and endpoints
├── dashboard/                # Phase 4: Streamlit web dashboard
├── tests/
│   └── test_merton_pipeline.py # Stochastic engine verification suite
├── Dockerfile
├── requirements.txt
└── README.md
