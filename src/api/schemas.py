from typing import Dict, Optional
from pydantic import BaseModel, Field


class RiskReport(BaseModel):
    """Structured credit risk assessment payload returned by /predict/{ticker}."""

    ticker: str = Field(..., description="Normalized equity ticker symbol")
    distance_to_default: float = Field(
        ..., description="Merton structural Distance-to-Default (in standard deviations)"
    )
    merton_analytical_pd: float = Field(
        ..., description="Theoretical Black-Scholes-Merton default probability N(-DD)"
    )
    probability_of_distress: float = Field(
        ..., description="PR-calibrated XGBoost 30-day distress probability"
    )
    geometric_distance_to_distress: Optional[float] = Field(
        None, description="Euclidean L2 distance d_e(t) to the PCA distress centroid"
    )
    geometric_drift_velocity: Optional[float] = Field(
        None, description="30-day change in distance to distress centroid (delta d_e)"
    )
    geometric_drift_alert: bool = Field(
        False, description="True if drift velocity < -0.5 indicating rapid migration toward distress"
    )
    alert_status: str = Field(
        ..., description="Overall classification status ('ALERT' or 'NORMAL')"
    )
    shap_top_drivers: Dict[str, float] = Field(
        ..., description="Top 3 SHAP feature attributions driving the ML distress score"
    )