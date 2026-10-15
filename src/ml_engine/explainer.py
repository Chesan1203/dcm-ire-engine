from typing import Dict
import pandas as pd
import shap
from xgboost import XGBClassifier


class DistressExplainer:
    """Generates SHAP risk attribution payloads for interpretability."""

    def __init__(self, trained_model: XGBClassifier) -> None:
        """Initializes the SHAP TreeExplainer on the fitted XGBoost model."""
        self.model = trained_model
        # TreeExplainer is optimized specifically for gradient-boosted trees
        self.explainer = shap.TreeExplainer(self.model)

    def get_top_drivers(self, x_alert: pd.DataFrame) -> Dict[str, float]:
        """
        Extracts the top 3 features driving the distress prediction.
        Returns a transparent JSON-ready dictionary of feature contributions.
        """
        # Ensure input is a single-row DataFrame
        if isinstance(x_alert, pd.Series):
            x_alert = x_alert.to_frame().T

        # Compute SHAP values for the specific instance
        shap_values = self.explainer.shap_values(x_alert)

        # Handle binary classification output formats (extract 1D array for the row)
        if isinstance(shap_values, list):
            instance_shap = shap_values[1][0]  # Positive distress class
        elif shap_values.ndim == 2:
            instance_shap = shap_values[0]
        else:
            instance_shap = shap_values

        # Pair feature names with their corresponding SHAP values
        feature_names = x_alert.columns.tolist()
        attributions = list(zip(feature_names, instance_shap))

        # Sort by absolute impact magnitude to find the most significant drivers
        attributions.sort(key=lambda x: abs(x[1]), reverse=True)

        # Extract the top 3 driving factors
        top_3 = attributions[:3]

        # Format as a dictionary (which FastAPI will automatically convert to JSON)
        payload = {
            feature: float(shap_val) 
            for feature, shap_val in top_3
        }

        return payload