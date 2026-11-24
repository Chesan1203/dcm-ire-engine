# Dynamic Credit Migration & Market-Implied Risk Engine (DCM-IRE)

DCM-IRE is a quantitative credit risk engine designed to detect corporate credit deterioration early. Standard credit rating agencies often lag behind fast-moving market conditions by 45–90 days due to quarterly accounting cycles. This project addresses that delay by combining structural option pricing (the Black-Scholes-Merton model) with unsupervised geometric clustering, Precision-Recall calibrated XGBoost default forecasting, and TreeSHAP explainability—served through an asynchronous FastAPI microservice and an interactive Streamlit analytics dashboard.

---

## Architecture & Development Status

The repository is organized around four sequential engineering stages:

1. **Stage 1: Ingestion & Unsupervised Geometric Discovery (The Radar)** — *Completed*
2. **Stage 2: The Stochastic Math Core (The Merton Model)** — *Completed*
3. **Stage 3: Multi-Crisis Panel & Predictive ML Overlay** — *Completed*
4. **Stage 4: FastAPI Microservice & Streamlit Dashboard** — *Completed*

---

## Repository Layout

```text
dcm-ire-engine/
├── dashboard/
│   └── app.py                    # Interactive 3-tab Streamlit analytics dashboard
├── data/
│   ├── raw/                      # Cached SEC XBRL filings and raw market series
│   └── processed/                # 2007–Present historical panel (historical_panel.csv)
├── models/
│   └── distress_xgboost.pkl      # PR-F1 calibrated XGBoost model (cutoff: 0.9249)
├── scripts/
│   ├── build_panel.py            # Multi-crisis historical panel generator (2007–Present)
│   └── train_model.py            # Chronological training & F1 threshold calibration
├── src/
│   ├── ingestion/                # Phase 1: Multi-source data ingestion pipelines
│   │   ├── sec_edgar.py          # Async SEC XBRL parser and debt barrier extractor
│   │   ├── market_data.py        # Equity market cap (E) and 252-day volatility (σ_E)
│   │   └── macro_fred.py         # 10Y Treasury rate (r), 10Y–2Y spread, and VIX
│   ├── math_core/                # Phase 2: Option pricing and root optimization
│   │   ├── vector_math.py        # Clamped d1/d2 terms, Normal CDF, and vector drift
│   │   └── merton_solver.py      # Levenberg-Marquardt solver, DD, and analytical PD
│   ├── ml_engine/                # Phase 1 clustering & Phase 3 predictive models
│   │   ├── clustering.py         # RobustWinsorScaler, PCA, and dynamic K-Means
│   │   ├── classifier.py         # 10-feature DistressClassifier & PR-F1 calibration
│   │   └── explainer.py          # TreeSHAP top-3 risk driver attribution module
│   └── api/                      # Phase 4: FastAPI service schemas and endpoints
│       ├── main.py               # /predict/{ticker} pipeline, startup radar & SEC cache
│       └── schemas.py            # Typed Pydantic RiskReport response contract
├── tests/
│   ├── test_merton_pipeline.py   # Stochastic solver boundary invariant checks
│   ├── test_bad_data.py          # Synthetic catastrophic insolvency stress test
│   └── test_api.py               # End-to-end API, schema & fallback test suite
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## Completed Work

### Phase 1: Ingestion & Geometric Anomaly Detection

- **Multi-Source Data Ingestion (`src/ingestion/`):**
  - `sec_edgar.py`: Asynchronously retrieves quarterly financial disclosures from the SEC EDGAR XBRL API, maps tickers to 10-digit zero-padded CIK numbers via an in-memory cache, parses short-term and long-term liabilities across prioritized US-GAAP fallback tags, and calculates the structural default barrier ($D = \text{Short-Term Debt} + 0.5 \times \text{Long-Term Debt}$) along with dynamic debt maturity ($T$).
  - `market_data.py`: Queries historical stock market data using `yfinance` to determine current equity market capitalization ($E$) and computes annualized trailing equity volatility ($\sigma_E$) from daily log returns over a 252-trading-day window, with local CSV cache fallback resilience.
  - `macro_fred.py`: Captures macroeconomic benchmark indicators, including the continuous 10-year Treasury risk-free rate ($r$), the 10Y–2Y Treasury yield curve spread to monitor economic cycle contractions, and market-wide volatility via the CBOE VIX.

- **Feature Preprocessing & Clustering (`src/ml_engine/clustering.py`):**
  - `RobustWinsorScaler`: A Scikit-Learn-compliant transformer that limits fat-tailed financial anomalies by applying 1st and 99th percentile winsorization, scales variables using Median and Interquartile Range (IQR) with a zero-division safeguard, and prunes collinear feature pairs with Pearson correlation $|r| > 0.85$.
  - `GeometricDiscoveryEngine`: Reduces feature dimensions using Principal Component Analysis (PCA) while retaining 95% cumulative variance, evaluates optimal cluster counts ($K \in [3, 6]$) via Silhouette Score sweeps, and isolates the Distress Centroid ($\mu_{\text{risk}}$) characterized by minimum liquidity and maximum leverage coordinates.
  - **Vector Drift Tracking**: Computes the Euclidean distance $d_e(t)$ of corporate entities to the distress centroid in PCA space and flags accelerated migration toward insolvency if drift velocity drops below the threshold ($\Delta d_e < -0.5$).

---

### Phase 2: The Stochastic Math Core

- **Vector Mathematics (`src/math_core/vector_math.py`):**
  - Implements intermediate Black-Scholes terms ($d_1, d_2$) and standard normal cumulative distribution transforms ($N(x) = \Phi(x)$) with defensive numerical clamping ($\epsilon = 10^{-8}$) to prevent zero-division or invalid logarithmic calculations.
  - Provides vector helper routines for Euclidean distance calculations, rolling drift velocity estimation, and automated distress alert triggers.

- **Merton Structural Solver (`src/math_core/merton_solver.py`):**
  - Formulates a simultaneous non-linear equation system treating corporate equity as a European call option on underlying firm assets to extract unobservable Asset Value ($V$) and Asset Volatility ($\sigma_V$).
  - Computes deterministic initial parameter guesses ($V_0 = E + D$, $\sigma_{V0} = \sigma_E \cdot \frac{E}{E+D}$) grounded in balance sheet identities to initialize numerical optimization reliably.
  - Executes SciPy's Levenberg-Marquardt root-finding algorithm (`method='lm'`) across non-linear equity and volatility residuals, incorporating dynamic debt maturity discounting ($e^{-rT}$) and fallback protections against numerical divergence.
  - Evaluates closed-form structural **Distance-to-Default ($DD$)** and calculates theoretical market-implied **Default Probability ($PD = N(-DD)$)**.

---

### Phase 3: Multi-Crisis Panel & Predictive ML Overlay

- **Multi-Crisis Historical Panel (`scripts/build_panel.py`):**
  - Constructs a 53,193-row historical panel spanning **2007–Present** across 11 US equities (`AAPL, MSFT, F, GE, BAC, C, AAL, CCL, RCL, M, AMC`), capturing the 2008 Global Financial Crisis, 2015–2016 Energy/Credit Crunch, and 2020 COVID-19 Crash alongside daily Treasury yields (`^TNX`, `^IRX`) and `^VIX`.

- **10-Feature Momentum Classifier & Calibration (`src/ml_engine/classifier.py`, `scripts/train_model.py`):**
  - Builds a 10-feature rolling time-series matrix combining current levels ($DD_t$, $\sigma_{V,t}$, yield spread, VIX), 30-trading-day lags, and 30-day momentum deltas ($\Delta DD_{30d}$, $\Delta \sigma_{V,30d}$) to predict forward 30-day structural distress shocks ($y = 1$ if $\Delta DD_{t+30} < -1.5\sigma_{DD}$ or $DD_{t+30} < 1.0$).
  - Trains `XGBClassifier(scale_pos_weight=3.0)` on a strict chronological 70%/30% split (`37,004` training / `15,859` validation rows) and sweeps the Precision-Recall curve to calibrate the $F_1$-optimal decision threshold (`0.9249`), persisted to `models/distress_xgboost.pkl`.

- **Explainable AI Attributions (`src/ml_engine/explainer.py`):**
  - Integrates `shap.TreeExplainer` around the calibrated XGBoost classifier to decompose every distress score into local feature contributions and extract the top three financial risk drivers ranked by absolute SHAP impact.

---

### Phase 4: FastAPI Microservice, Dashboard & Verification

- **Asynchronous FastAPI Backend (`src/api/main.py`, `src/api/schemas.py`):**
  - Exposes `GET /predict/{ticker}` executing a six-stage live inference pipeline validated by Pydantic `RiskReport` contracts.
  - Accelerates startup initialization to $<1$ second using stride-based Silhouette clustering and hardens SEC EDGAR ingestion with local disk caching and baseline fallback protection against transient connection resets.

- **Interactive Streamlit Dashboard (`dashboard/app.py`):**
  - Connects to the live FastAPI backend and historical crisis panel with three interactive views:
    - **Micro Entity Risk Profile:** Solvency status banner (`NORMAL` vs. `ALERT`), color-coded Distance-to-Default gauge (`< 1.0σ`, `1.0σ–2.5σ`, `> 2.5σ`), analytical $PD$, XGBoost distress probability, geometric drift velocity, and horizontal TreeSHAP risk-driver chart.
    - **Macro Geometric Radar:** Zoomable 2D PCA regime map plotting the Distress Centroid ($\mu_{\text{risk}}$) and 30-day migration vectors across Crisis, Transitional, and Solvent regimes.
    - **Historical Solvency Trajectory:** Full 2007–Present daily Merton $DD$ timeline benchmarked against the $1.0\sigma$ insolvency threshold.

- **Verification & Stress Testing (`tests/`):**
  - `test_merton_pipeline.py`: Verifies Merton root-finding convergence and boundary invariants ($V \ge E$, $\sigma_V \le \sigma_E$, $DD_{\text{healthy}} \gg DD_{\text{distressed}}$).
  - `test_bad_data.py`: Validates end-to-end distress triggering (`ALERT`, $100.00\%$ ML $PD$, $74.68\%$ Merton $PD$) under a synthetic catastrophic insolvency scenario.
  - `test_api.py`: Automated 12-test `pytest` suite covering Pydantic schemas, near-zero equity solver stability, CSV cache fallbacks, and FastAPI endpoints.

---

## Getting Started

### 1. Environment Setup

```bash
# Clone repository
git clone [https://github.com/Chesan1203/dcm-ire-engine.git](https://github.com/Chesan1203/dcm-ire-engine.git)
cd dcm-ire-engine

# Initialize and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Build Panel & Train Model (Optional — Pretrained Artifact Included)

```bash
# Generate 2007–Present multi-crisis historical panel
python -m scripts.build_panel

# Train and calibrate the XGBoost classifier
python -m scripts.train_model
```

### 3. Launch Backend API & Frontend Dashboard

Start the FastAPI microservice in one terminal:

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Start the Streamlit dashboard in a second terminal:

```bash
streamlit run dashboard/app.py
```

### 4. Run Verification & Stress Tests

```bash
python -m pytest tests/ -v
```
