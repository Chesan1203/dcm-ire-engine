import os
import joblib
import pandas as pd
import numpy as np

from src.ml_engine.classifier import DistressClassifier


def run_training_pipeline(data_path: str = "data/processed/historical_panel.csv"):
    """
    Ingests real multi-ticker historical panel data, engineers rolling features per ticker,
    trains XGBoost, calibrates PR threshold, and saves model artifact to disk.
    """
    os.makedirs("models", exist_ok=True)

    if not os.path.exists(data_path):
        print(f"Data file {data_path} not found. Run python -m scripts.build_panel first.")
        return

    raw_df = pd.read_csv(data_path)
    classifier = DistressClassifier(scale_pos_weight=3.0)

    X_list = []
    y_list = []

    if "ticker" in raw_df.columns:
        grouped = raw_df.groupby("ticker", group_keys=False)
    else:
        grouped = [("single_ticker", raw_df)]

    for _, group in grouped:
        group_df = group.copy().reset_index(drop=True)

        # Build features and targets for this ticker's time-series
        feat_df = classifier.build_rolling_features(group_df)
        target_s = classifier.generate_binary_target(group_df["distance_to_default"])

        # Align valid row indices within this ticker
        valid_idx = feat_df.index.intersection(target_s.dropna().index)

        if len(valid_idx) == 0:
            continue

        X_list.append(feat_df.loc[valid_idx].reset_index(drop=True))
        y_list.append(target_s.loc[valid_idx].reset_index(drop=True))

    if not X_list:
        raise RuntimeError("No valid feature/target samples generated across tickers.")

    # Combine all ticker datasets with fresh 0..N integer indexing
    X = pd.concat(X_list, ignore_index=True).reset_index(drop=True)
    y = pd.concat(y_list, ignore_index=True).reset_index(drop=True)

    # Drop non-feature metadata columns to keep X strictly numerical
    metadata_cols = ["date", "ticker", "close_price", "market_cap"]
    X = X.drop(columns=[col for col in metadata_cols if col in X.columns])

    # Chronological 70/30 Train/Validation Split
    split_idx = int(len(X) * 0.70)
    X_train = X.iloc[:split_idx].reset_index(drop=True)
    X_val = X.iloc[split_idx:].reset_index(drop=True)
    y_train = y.iloc[:split_idx].reset_index(drop=True)
    y_val = y.iloc[split_idx:].reset_index(drop=True)

    print(f"Training set: {len(X_train)} samples | Validation set: {len(X_val)} samples")

    # Model Fitting & PR Threshold Calibration
    print("Fitting XGBoost model on real historical panel data...")
    classifier.fit(X_train, y_train, X_val, y_val)

    print("Calibrating Precision-Recall threshold...")
    optimal_threshold = classifier.calibrate_optimal_threshold(X_val, y_val)
    print(f"Optimal F1 Threshold Calibrated: {optimal_threshold:.4f}")

    # Save Model Checkpoint
    model_path = "models/distress_xgboost.pkl"
    joblib.dump(classifier, model_path)
    print(f"Model successfully saved to {model_path}")


if __name__ == "__main__":
    run_training_pipeline()