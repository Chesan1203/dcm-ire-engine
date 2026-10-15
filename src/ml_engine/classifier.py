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

    def fit(
        self, 
        X_train: pd.DataFrame, 
        y_train: pd.Series, 
        X_val: pd.DataFrame = None, 
        y_val: pd.Series = None
    ) -> None:
        """
        Trains the XGBoost classifier on the rolling feature matrix.
        Uses scale_pos_weight to heavily penalize misclassification of the minority distress class.
        Optionally tracks validation logloss to monitor for overfitting.
        """
        if X_val is not None and y_val is not None:
            # Fit with validation tracking
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_train, y_train), (X_val, y_val)],
                verbose=False
            )
        else:
            # Standard fit if no validation set is provided
            self.model.fit(X_train, y_train)

    def calibrate_optimal_threshold(
        self, X_val: pd.DataFrame, y_val: pd.Series
    ) -> float:
        """
        Extracts validation probabilities and computes the PR curve to find
        the exact classification threshold that maximizes the F1-score.
        """
        # Extract raw probabilities for the positive class (distress, y=1)
        y_probs = self.model.predict_proba(X_val)[:, 1]
        
        # Compute Precision-Recall curve
        precisions, recalls, thresholds = precision_recall_curve(y_val, y_probs)
        
        # Calculate F1-score for each threshold safely to avoid division by zero
        numerator = 2 * precisions * recalls
        denominator = precisions + recalls
        f1_scores = np.divide(
            numerator, denominator, out=np.zeros_like(numerator), where=(denominator != 0)
        )
        
        # Locate the index of the highest F1-score
        optimal_idx = int(np.argmax(f1_scores))
        
        # The thresholds array has one less element than precisions and recalls
        if optimal_idx < len(thresholds):
            optimal_threshold = float(thresholds[optimal_idx])
        else:
            optimal_threshold = 1.0  # Fallback if the maximum is at the absolute end
            
        # Store for future inference calls
        self.optimal_threshold_ = optimal_threshold
        
        return optimal_threshold

    def predict_distress(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Executes inference using the PR-calibrated custom threshold.
        """
        if not hasattr(self, 'optimal_threshold_'):
            raise ValueError("Model must be calibrated before calling predict_distress.")
            
        y_probs = self.model.predict_proba(X_new)[:, 1]
        return (y_probs >= self.optimal_threshold_).astype(int)