from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve
from xgboost import XGBClassifier


class DistressClassifier:
    """XGBoost classifier with Precision-Recall F1 calibration."""

    def __init__(self, scale_pos_weight: float = 1.0) -> None:
        # Initialize XGBClassifier with penalization for false negatives
        self.model = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric="logloss"
        )

    def build_rolling_features(self, time_series_df: pd.DataFrame) -> pd.DataFrame:
        """Constructs 30-day lag matrix of Merton DD, Yield Spread, VIX, and sigma_V."""
        df = time_series_df.copy()
        
        # 30-day lags for structural and macro features
        df["dd_lag_30"] = df["distance_to_default"].shift(30)
        df["asset_vol_lag_30"] = df["asset_volatility"].shift(30)
        df["yield_spread_lag_30"] = df["yield_spread"].shift(30)
        df["vix_lag_30"] = df["sector_vix"].shift(30)

        # 30-day rolling changes (Momentum)
        df["delta_dd_30d"] = df["distance_to_default"] - df["dd_lag_30"]
        df["delta_asset_vol_30d"] = df["asset_volatility"] - df["asset_vol_lag_30"]
        
        # Drop NaN values created by the backward shift warmup period
        return df.dropna()

    def generate_binary_target(
        self, dd_series: pd.Series, threshold_sigma: float = 1.5
    ) -> pd.Series:
        """Defines y=1 if firm exhibits >1.5 sigma DD drop in next 30 days."""
        # Calculate the sample standard deviation of Distance-to-Default
        dd_std = float(dd_series.std())
        
        # Look forward 30 days to compute future DD change
        forward_dd = dd_series.shift(-30)
        forward_delta = forward_dd - dd_series
        
        # y = 1 if the drop is greater than the 1.5 sigma threshold
        shock_cutoff = -threshold_sigma * dd_std
        target_y = (forward_delta < shock_cutoff).astype(int)
        
        target_y.name = "target_distress"
        return target_y

    def calibrate_optimal_threshold(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> float:
        # TODO: Extract predicted probabilities y_probs
        # TODO: Compute Precision-Recall curve and find threshold maximizing F1-score
        pass