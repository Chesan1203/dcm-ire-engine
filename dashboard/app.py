import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# Ensure project root is on sys.path so `src.*` imports resolve via `streamlit run`
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

from src.ml_engine.clustering import GeometricDiscoveryEngine, RobustWinsorScaler

DEFAULT_API_URL = "http://127.0.0.1:8000"
PANEL_PATH = "data/processed/historical_panel.csv"
CLUSTER_COLS = ["distance_to_default", "asset_volatility", "yield_spread", "sector_vix"]
UNIVERSE_TICKERS = ["AAPL", "MSFT", "GE", "BAC", "C", "F", "AAL", "CCL", "RCL", "M", "AMC"]

FEATURE_LABELS = {
    "distance_to_default": "Distance-to-Default (DD)",
    "asset_volatility": "Implied Asset Volatility (σ_V)",
    "yield_spread": "Treasury Yield Spread (10Y–3M)",
    "sector_vix": "CBOE VIX Index",
    "dd_lag_30": "30d Lagged DD",
    "asset_vol_lag_30": "30d Lagged Asset Vol",
    "yield_spread_lag_30": "30d Lagged Yield Spread",
    "vix_lag_30": "30d Lagged VIX",
    "delta_dd_30d": "30d DD Momentum (ΔDD)",
    "delta_asset_vol_30d": "30d Volatility Change (Δσ_V)",
}


@st.cache_data(show_spinner=False)
def load_macro_pca_data(
    panel_path: str = PANEL_PATH,
) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Loads the historical panel, fits the Geometric Radar, and returns 2D PCA coordinates."""
    if not os.path.exists(panel_path):
        return None, None

    df = pd.read_csv(panel_path)
    if df.empty or not all(col in df.columns for col in CLUSTER_COLS):
        return None, None

    clean_df = df.dropna(subset=CLUSTER_COLS).copy().reset_index(drop=True)
    if clean_df.empty:
        return None, None

    scaler = RobustWinsorScaler(corr_threshold=0.85)
    engine = GeometricDiscoveryEngine(variance_retained=0.95)

    X_scaled = scaler.fit_transform(clean_df[CLUSTER_COLS])
    kmeans_model = engine.tune_and_fit_kmeans(X_scaled[::10], k_range=(3, 6))
    distress_centroid = engine.identify_distress_centroid(
        kmeans_model, scaler.selected_features_
    )

    pca_coords = engine.fitted_pca.transform(X_scaled)
    cluster_labels = kmeans_model.predict(pca_coords)

    clean_df["PC1"] = pca_coords[:, 0]
    clean_df["PC2"] = pca_coords[:, 1] if pca_coords.shape[1] > 1 else 0.0
    clean_df["_cluster_id"] = cluster_labels

    # Rank clusters by distance to distress_centroid so regime names are intuitive
    centers = kmeans_model.cluster_centers_
    dists_to_risk = {
        cid: float(np.linalg.norm(centers[cid] - distress_centroid))
        for cid in range(len(centers))
    }
    ordered_cids = sorted(dists_to_risk.keys(), key=lambda c: dists_to_risk[c])

    regime_names: Dict[int, str] = {}
    for rank, cid in enumerate(ordered_cids):
        if rank == 0:
            regime_names[cid] = "Distress / Crisis Regime"
        elif rank == len(ordered_cids) - 1:
            regime_names[cid] = "Solvent / Low-Vol Regime"
        else:
            regime_names[cid] = f"Transitional Regime ({rank})"

    clean_df["Regime Cluster"] = clean_df["_cluster_id"].map(regime_names)

    centroid_2d = np.array([
        float(distress_centroid[0]),
        float(distress_centroid[1]) if len(distress_centroid) > 1 else 0.0,
    ])
    return clean_df, centroid_2d


def check_api_health(api_url: str) -> bool:
    """Pings the FastAPI root health-check endpoint."""
    try:
        resp = requests.get(f"{api_url.rstrip('/')}/", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def fetch_risk_report(ticker: str, api_url: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Fetches the RiskReport JSON payload from FastAPI /predict/{ticker}."""
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/predict/{ticker}", timeout=20
        )
        if response.status_code != 200:
            detail = response.json().get("detail", response.text)
            return None, f"API Error ({response.status_code}): {detail}"
        return response.json(), None
    except requests.RequestException as exc:
        return None, (
            f"Cannot reach FastAPI backend at `{api_url}`. "
            f"Start it with `uvicorn src.api.main:app --reload`.\n\nDetails: {exc}"
        )


def _render_dd_gauge(dd_value: float) -> go.Figure:
    """Builds a clean Plotly gauge meter for Merton Distance-to-Default (DD)."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=dd_value,
            number={"suffix": " σ", "valueformat": ".2f", "font": {"size": 38}},
            title={"text": "Merton Distance-to-Default (DD)", "font": {"size": 16}},
            gauge={
                "axis": {"range": [-2.5, 10.0], "tickwidth": 1},
                "bar": {"color": "#2563EB"},
                "steps": [
                    {"range": [-2.5, 1.0], "color": "rgba(220, 38, 38, 0.25)"},
                    {"range": [1.0, 2.5], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [2.5, 10.0], "color": "rgba(16, 185, 129, 0.25)"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.85,
                    "value": 1.0,
                },
            },
        )
    )
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=55, b=15),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _render_shap_bar_chart(shap_drivers: Dict[str, float]) -> go.Figure:
    """Builds a horizontal Plotly bar chart for Top-3 TreeSHAP attributions."""
    if not shap_drivers:
        return go.Figure()

    items = list(shap_drivers.items())[::-1]
    labels = [FEATURE_LABELS.get(k, k.replace("_", " ").title()) for k, _ in items]
    values = [float(v) for _, v in items]
    colors = ["#EF4444" if v > 0 else "#10B981" for v in values]

    max_abs = max(abs(v) for v in values) if values else 1.0
    pad = max_abs * 0.28

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            text=[f"{v:+.3f}" for v in values],
            textposition="outside",
            cliponaxis=False,
        )
    )
    fig.add_vline(x=0, line_width=1, line_color="#6B7280")
    fig.update_layout(
        title=dict(text="Top-3 TreeSHAP Risk Drivers (Log-Odds Impact)", font=dict(size=16)),
        height=280,
        margin=dict(l=10, r=45, t=55, b=25),
        xaxis=dict(
            title="SHAP Value (← Reduces Risk | Increases Risk →)",
            range=[min(values) - pad, max(values) + pad],
            zeroline=False,
        ),
        yaxis=dict(title="", automargin=True),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def render_micro_pane(ticker: str, report: Dict[str, Any]) -> None:
    """Renders the Micro Entity Risk Profile with clear spacing and zero label truncation."""
    clean_ticker = ticker.upper().strip()
    alert_status = report.get("alert_status", "NORMAL")
    geo_alert = bool(report.get("geometric_drift_alert", False))

    if alert_status == "ALERT":
        st.error(
            f"**DISTRESS ALERT ({clean_ticker})** — Calibrated 30-day forward distress "
            "probability breached the PR-F1 threshold (`0.9249`)."
        )
    else:
        st.success(
            f"**STATUS: NORMAL ({clean_ticker})** — Entity remains within solvent structural bounds."
        )

    if geo_alert:
        st.warning(
            "**GEOMETRIC DRIFT WARNING** — Rapid migration toward the Distress Centroid "
            "detected (`Δd_e < -0.5`)."
        )

    dd_val = float(report["distance_to_default"])
    merton_pd = float(report["merton_analytical_pd"]) * 100.0
    ml_prob = float(report["probability_of_distress"]) * 100.0
    geo_dist = report.get("geometric_distance_to_distress")
    geo_vel = report.get("geometric_drift_velocity")

    row1_col1, row1_col2 = st.columns(2, gap="medium")
    with row1_col1:
        with st.container(border=True):
            st.metric(
                label="Merton Distance-to-Default (DD)",
                value=f"{dd_val:.2f} σ",
                delta="Solvent Buffer (> 1.0σ)" if dd_val >= 1.0 else "Breached (< 1.0σ)",
                delta_color="normal" if dd_val >= 1.0 else "inverse",
            )
    with row1_col2:
        with st.container(border=True):
            st.metric(
                label="Merton Analytical Default Prob (PD)",
                value=f"{merton_pd:.2f}%",
                delta="Closed-Form N(-DD)",
                delta_color="off",
            )

    row2_col1, row2_col2 = st.columns(2, gap="medium")
    with row2_col1:
        with st.container(border=True):
            st.metric(
                label="XGBoost 30-Day Distress Probability",
                value=f"{ml_prob:.2f}%",
                delta="Cutoff: 92.49% (PR-F1)",
                delta_color="inverse" if alert_status == "ALERT" else "off",
            )
    with row2_col2:
        with st.container(border=True):
            vel_str = f"{geo_vel:+.3f}" if geo_vel is not None else "N/A"
            dist_caption = (
                f"Dist to μ_risk: {geo_dist:.2f}" if geo_dist is not None else "Out-of-panel ticker"
            )
            st.metric(
                label="Geometric Drift Velocity (30d Δd_e)",
                value=vel_str,
                delta="Rapid Drift" if geo_alert else dist_caption,
                delta_color="inverse" if geo_alert else "off",
            )

    viz_left, viz_right = st.columns([1.0, 1.25], gap="medium")
    with viz_left:
        with st.container(border=True):
            st.plotly_chart(_render_dd_gauge(dd_val), use_container_width=True)
    with viz_right:
        with st.container(border=True):
            shap_drivers = report.get("shap_top_drivers", {})
            st.plotly_chart(_render_shap_bar_chart(shap_drivers), use_container_width=True)


def render_macro_pane(selected_ticker: str, report: Optional[Dict[str, Any]] = None) -> None:
    """Renders the 2D PCA Geometric Radar with high-visibility markers and clean side legend."""
    pca_df, centroid_2d = load_macro_pca_data()
    if pca_df is None or centroid_2d is None:
        st.warning(
            f"Historical panel not found at `{PANEL_PATH}`. "
            "Run `python -m scripts.build_panel` to populate the Macro PCA Radar."
        )
        return

    clean_ticker = selected_ticker.upper().strip()
    ticker_hist = (
        pca_df[pca_df["ticker"] == clean_ticker]
        if "ticker" in pca_df.columns
        else pd.DataFrame()
    )
    has_trajectory = len(ticker_hist) >= 30

    # Top Telemetry & View Controls
    ctrl_col1, ctrl_col2, ctrl_col3, ctrl_col4 = st.columns([1.2, 1.2, 1.2, 1.1])
    geo_dist = report.get("geometric_distance_to_distress") if report else None
    geo_vel = report.get("geometric_drift_velocity") if report else None
    geo_alert = bool(report.get("geometric_drift_alert", False)) if report else False

    with ctrl_col1:
        st.metric(
            "Distance to Distress (d_e)",
            f"{geo_dist:.2f}" if geo_dist is not None else "N/A",
        )
    with ctrl_col2:
        st.metric(
            "30d Drift Velocity (Δd_e)",
            f"{geo_vel:+.3f}" if geo_vel is not None else "N/A",
            delta="Alert (< -0.5)" if geo_alert else "Normal (> -0.5)",
            delta_color="inverse" if geo_alert else "off",
        )
    with ctrl_col3:
        current_regime = (
            str(ticker_hist.iloc[-1]["Regime Cluster"])
            if not ticker_hist.empty
            else "Out-of-Panel Entity"
        )
        st.metric("Current PCA Regime", current_regime.split(" (")[0])
    with ctrl_col4:
        st.write("")
        zoom_trajectory = st.toggle(
            "Zoom on Trajectory",
            value=False,
            disabled=not has_trajectory,
            help="Zooms the PCA camera closely onto the selected firm's 30-day migration vector.",
        )

    # Background regime cloud (subtle so foreground markers sit clearly on top)
    plot_df = pca_df.iloc[::10].copy()
    regime_color_map = {
        "Solvent / Low-Vol Regime": "#38BDF8",
        "Transitional Regime (1)": "#A78BFA",
        "Transitional Regime (2)": "#FBBF24",
        "Distress / Crisis Regime": "#F87171",
    }

    fig = px.scatter(
        plot_df,
        x="PC1",
        y="PC2",
        color="Regime Cluster",
        color_discrete_map=regime_color_map,
        hover_data={
            "ticker": True,
            "distance_to_default": ":.2f",
            "asset_volatility": ":.3f",
            "sector_vix": ":.1f",
            "PC1": ":.2f",
            "PC2": ":.2f",
        },
        opacity=0.16,
    )
    fig.update_traces(marker=dict(size=4))

    # Foreground Layer 1: Distress Centroid (mu_risk)
    fig.add_trace(
        go.Scatter(
            x=[centroid_2d[0]],
            y=[centroid_2d[1]],
            mode="markers",
            name="Distress Centroid (μ_risk)",
            marker=dict(
                symbol="diamond",
                size=14,
                color="#FF2E2E",
                opacity=1.0,
                line=dict(width=1.8, color="#FFFFFF"),
            ),
            hovertemplate=(
                "<b>Distress Centroid (μ_risk)</b><br>"
                "PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>"
            ),
        )
    )

    # Foreground Layer 2: 30-Day Trajectory Line, Start Circle & Latest Star
    if has_trajectory:
        prev_pt = ticker_hist.iloc[-30]
        curr_pt = ticker_hist.iloc[-1]
        x0, y0 = float(prev_pt["PC1"]), float(prev_pt["PC2"])
        x1, y1 = float(curr_pt["PC1"]), float(curr_pt["PC2"])

        # Connecting trajectory line
        fig.add_trace(
            go.Scatter(
                x=[x0, x1],
                y=[y0, y1],
                mode="lines",
                name=f"{clean_ticker} 30d Path",
                line=dict(color="#FACC15", width=2.5),
                hoverinfo="skip",
            )
        )

        # Start Circle (t - 30d)
        fig.add_trace(
            go.Scatter(
                x=[x0],
                y=[y0],
                mode="markers",
                name=f"{clean_ticker} (t - 30d)",
                marker=dict(
                    symbol="circle",
                    size=10,
                    color="#F59E0B",
                    opacity=1.0,
                    line=dict(width=1.8, color="#FFFFFF"),
                ),
                hovertemplate=(
                    f"<b>{clean_ticker} (30 Days Ago)</b><br>"
                    f"DD: {prev_pt['distance_to_default']:.2f}σ<br>"
                    "PC1: %{x:.2f}, PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )

        # Latest Star (Current)
        fig.add_trace(
            go.Scatter(
                x=[x1],
                y=[y1],
                mode="markers",
                name=f"{clean_ticker} (Latest)",
                marker=dict(
                    symbol="star",
                    size=15,
                    color="#00F5D4",
                    opacity=1.0,
                    line=dict(width=1.8, color="#FFFFFF"),
                ),
                hovertemplate=(
                    f"<b>{clean_ticker} (Latest)</b><br>"
                    f"DD: {curr_pt['distance_to_default']:.2f}σ<br>"
                    "PC1: %{x:.2f}, PC2: %{y:.2f}<extra></extra>"
                ),
            )
        )

        if zoom_trajectory:
            pad_x = max(abs(x1 - x0) * 1.8, 0.65)
            pad_y = max(abs(y1 - y0) * 1.8, 0.65)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            fig.update_xaxes(range=[cx - pad_x, cx + pad_x])
            fig.update_yaxes(range=[cy - pad_y, cy + pad_y])
    else:
        st.info(
            f"`{clean_ticker}` is an out-of-panel custom equity. The background map displays "
            "the 11-ticker historical crisis universe and Distress Centroid (`μ_risk`)."
        )

    fig.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=25, b=20),
        xaxis_title="Principal Component 1 (Structural Solvency & Leverage)",
        yaxis_title="Principal Component 2 (Macro Volatility & Curve Regime)",
        legend=dict(
            orientation="v",
            yanchor="top",
            y=0.98,
            xanchor="right",
            x=0.98,
            bgcolor="rgba(15, 23, 42, 0.88)",
            bordercolor="rgba(255, 255, 255, 0.2)",
            borderwidth=1,
            font=dict(size=12),
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "PCA projection fitted via RobustWinsorScaler (1st/99th percentile winsorization, "
        "Median-IQR scaling) and Silhouette-tuned K-Means across 2007–Present crisis regimes."
    )


def render_historical_pane(selected_ticker: str) -> None:
    """Renders historical Distance-to-Default trajectory when viewing a panel ticker."""
    pca_df, _ = load_macro_pca_data()
    clean_ticker = selected_ticker.upper().strip()
    if pca_df is None or "ticker" not in pca_df.columns:
        st.info("Historical panel data unavailable.")
        return

    ticker_hist = pca_df[pca_df["ticker"] == clean_ticker].copy()
    if ticker_hist.empty:
        st.info(
            f"`{clean_ticker}` is an out-of-panel custom ticker. "
            "Historical 2007–Present time series is available for the 11 panel equities."
        )
        return

    x_vals = ticker_hist["date"] if "date" in ticker_hist.columns else list(range(len(ticker_hist)))
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_vals,
            y=ticker_hist["distance_to_default"],
            mode="lines",
            name=f"{clean_ticker} Distance-to-Default",
            line=dict(color="#38BDF8", width=2),
        )
    )
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text="Structural Distress Threshold (DD = 1.0σ)",
        annotation_position="top left",
    )
    fig.update_layout(
        title=f"{clean_ticker} Historical Merton Distance-to-Default (2007–Present)",
        height=440,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis_title="Date" if "date" in ticker_hist.columns else "Trading Day Index",
        yaxis_title="Distance-to-Default (σ)",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True)


def main() -> None:
    st.set_page_config(
        layout="wide",
        page_title="DCM-IRE Credit Risk Dashboard",
        page_icon="📉",
    )

    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 3.25rem;
            padding-bottom: 2rem;
            max-width: 1440px;
        }
        [data-testid="stMetricValue"] {
            font-size: 1.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.title("Dynamic Credit Migration & Market-Implied Risk Engine")
    st.caption(
        "Real-Time Merton Structural Option Pricing • Unsupervised Geometric Radar • "
        "Calibrated XGBoost & TreeSHAP Early-Warning System"
    )

    with st.sidebar:
        st.header("Engine Controls")
        api_url = st.text_input("FastAPI Endpoint URL", value=DEFAULT_API_URL)
        is_online = check_api_health(api_url)
        if is_online:
            st.success("API Status: Online")
        else:
            st.error("API Status: Offline")

        st.divider()
        preset_ticker = st.selectbox(
            "Select Panel Ticker",
            options=UNIVERSE_TICKERS,
            index=1,  # MSFT
        )
        custom_ticker = st.text_input(
            "Or Enter Any US Listed Ticker",
            placeholder="e.g., TSLA, NVDA, BA",
        )
        active_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else preset_ticker

        st.button("Refresh Live Risk Assessment", type="primary", use_container_width=True)

    with st.spinner(f"Running Merton solver, XGBoost & TreeSHAP for {active_ticker}..."):
        report, error_msg = fetch_risk_report(active_ticker, api_url)

    tab_micro, tab_macro, tab_hist = st.tabs([
        f"Micro Entity Risk Profile ({active_ticker})",
        "Macro Geometric Radar (2D PCA)",
        f"Historical Solvency Trajectory ({active_ticker})",
    ])

    with tab_micro:
        if error_msg:
            st.error(error_msg)
        elif report:
            render_micro_pane(active_ticker, report)

    with tab_macro:
        render_macro_pane(active_ticker, report=report)

    with tab_hist:
        render_historical_pane(active_ticker)


if __name__ == "__main__":
    main()