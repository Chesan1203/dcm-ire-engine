from typing import Dict
from pydantic import BaseModel, Field


class RiskReport(BaseModel):
    """Pydantic schema definition for credit risk inference response payload."""

    ticker: str = Field(..., description="Target equity ticker symbol")
    distance_to_default: float = Field(
        ..., description="Structural Merton Distance-to-Default metric"
    )
    probability_of_distress: float = Field(
        ..., description="Calibrated distress probability from XGBoost"
    )
    alert_status: str = Field(
        ..., description="Operational risk flag (e.g., ALERT, NORMAL)"
    )
    shap_top_drivers: Dict[str, float] = Field(
        ..., description="Key driving features and their relative SHAP attributions"
    )