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
```

## ⚙️ Completed Work

### Phase 1: Ingestion & Geometric Anomaly Detection

* **Multi-Source Data Ingestion (`src/ingestion/`):**
  * `sec_edgar.py`: Asynchronously retrieves quarterly financial disclosures from the SEC EDGAR API, maps tickers to 10-digit zero-padded CIK numbers via an in-memory cache, parses short-term and long-term liabilities across prioritized US-GAAP fallback tags, and calculates the structural default barrier ($D = \text{Short-Term Debt} + 0.5 \times \text{Long-Term Debt}$) along with dynamic debt maturity ($T$).
  * `market_data.py`: Queries historical stock market data using `yfinance` to determine current equity market capitalization ($E$) and computes annualized trailing equity volatility ($\sigma_E$) from daily log returns over a 252-trading-day window.
  * `macro_fred.py`: Captures macroeconomic benchmark indicators, including the continuous 10-year Treasury risk-free rate ($r$), the 10Y–2Y Treasury yield curve spread to monitor economic cycle contractions, and market-wide volatility via the CBOE VIX.

* **Feature Preprocessing & Clustering (`src/ml_engine/clustering.py`):**
  * `RobustWinsorScaler`: A Scikit-Learn-compliant transformer that limits fat-tailed financial anomalies by applying 1st and 99th percentile winsorization, scales variables using Median and Interquartile Range (IQR) with a zero-division safeguard, and prunes collinear feature pairs with Pearson correlation $|r| > 0.85$.
  * `GeometricDiscoveryEngine`: Reduces feature dimensions using Principal Component Analysis (PCA) while retaining 95% cumulative variance, evaluates optimal cluster counts ($K \in [3, 6]$) via Silhouette Score sweeps, and isolates the Distress Centroid ($\mu_{\text{risk}}$) characterized by minimum liquidity and maximum leverage coordinates.
  * **Vector Drift Tracking**: Computes the Euclidean distance $d_e(t)$ of corporate entities to the distress centroid in PCA space and flags accelerated migration toward insolvency if drift velocity drops below the threshold ($\Delta d_e < -0.5$).

---

### Phase 2: The Stochastic Math Core

* **Vector Mathematics (`src/math_core/vector_math.py`):**
  * Implements intermediate Black-Scholes terms ($d_1, d_2$) and standard normal cumulative distribution transforms ($N(x) = \Phi(x)$) with defensive numerical clamping ($\epsilon = 10^{-8}$) to prevent zero-division or invalid logarithmic calculations.
  * Provides vector helper routines for Euclidean distance calculations, rolling drift velocity estimation, and automated distress alert triggers.

* **Merton Structural Solver (`src/math_core/merton_solver.py`):**
  * Formulates a simultaneous non-linear equation system treating corporate equity as a European call option on underlying firm assets to extract unobservable Asset Value ($V$) and Asset Volatility ($\sigma_V$).
  * Computes deterministic initial parameter guesses ($V_0 = E + D$, $\sigma_{V0} = \sigma_E \cdot \frac{E}{E+D}$) grounded in balance sheet identities to initialize numerical optimization reliably.
  * Executes SciPy's Levenberg-Marquardt root-finding algorithm (`method='lm'`) across non-linear equity and volatility residuals, incorporating dynamic debt maturity discounting ($e^{-rT}$) and fallback protections against numerical divergence.
  * Evaluates closed-form structural **Distance-to-Default ($DD$)** and calculates theoretical market-implied **Default Probability ($PD = N(-DD)$)**.

* **Pipeline Verification (`tests/test_merton_pipeline.py`):**
  * Tests root-finding convergence and parameter sensitivity across synthetic profiles for solvent and distressed firms.
  * Validates theoretical boundary invariants across all solved scenarios: $V \ge E$, $\sigma_V \le \sigma_E$, and $DD_{\text{healthy}} \gg DD_{\text{distressed}}$.
