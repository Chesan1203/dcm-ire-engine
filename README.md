# Dynamic Credit Migration & Market-Implied Risk Engine (DCM-IRE)

DCM-IRE is an academic quantitative credit risk engine that bridges structural asset pricing (Black-Scholes-Merton) with unsupervised geometric anomaly detection and gradient-boosted default classification.

---

## 🏗️ System Architecture

1. **Unsupervised Geometric Discovery (The Radar)**: Custom Winsorized scaling, PCA feature reduction, dynamic Silhouette-tuned K-Means, and vector drift velocity tracking ($\Delta d_e$).
2. **Stochastic Math Core (The Merton Model)**: Non-linear Levenberg-Marquardt root-finding (`scipy.optimize.root`) to estimate unobservable asset values ($V$) and asset volatilities ($\sigma_V$) with dynamic time-to-maturity ($T$) tracking.
3. **Predictive ML Overlay**: Precision-Recall calibrated XGBoost classifier forecasting $\gt 1.5\sigma$ Distance-to-Default drops over 30-day windows, explained via TreeSHAP payloads.
4. **Serving & Analytics Dashboard**: Asynchronous FastAPI microservice serving real-time inferences to an interactive dual-pane Streamlit dashboard.

---

## 📁 Repository Layout

```text
dcm-ire-engine/
|-- data/
|   |-- raw/                  # Ingested raw SEC/YFinance JSON/CSVs
|   |-- processed/            # Winsorized, robust-scaled matrices
|-- src/
|   |-- ingestion/            # SEC EDGAR, yfinance, and FRED pipelines
|   |-- math_core/            # Merton solver and vector distance calculators
|   |-- ml_engine/            # K-Means, XGBoost classifier, SHAP explainer
|   |-- api/                  # FastAPI endpoints and Pydantic schemas
|-- dashboard/                # Streamlit user interface
|-- Dockerfile
|-- requirements.txt
|-- README.md