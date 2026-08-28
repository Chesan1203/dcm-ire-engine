from typing import Dict
from pydantic import BaseModel


class RiskReport(BaseModel):
    ticker: str
    distance_to_default: float
    probability_of_distress: float
    alert_status: str
    shap_top_drivers: Dict[str, float]