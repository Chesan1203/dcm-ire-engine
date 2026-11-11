import os
import joblib
import pandas as pd
import numpy as np

from src.ml_engine.classifier import DistressClassifier

def run_training_pipeline(data_path: str = "data/processed/historical_panel.csv"):
    """
    Ingests historical panel data, engineers rolling features, trains XGBoost,
    calibrates PR threshold, and saves model artifact to disk.
    """
    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # 1. Load historical panel dataset & set 'date' as index to strip non-numeric columns
    if os.path.exists(data_path):
        raw_df = pd.read_csv(data_path)
        if "date" in raw_df.columns:
            raw_df = raw_df.set_index("date")
    else:
        print(f"Data file {data_path} not found. Ensure historical panel is present.")
        return

    # 2. Instantiate classifier & build features/target
    classifier = DistressClassifier(scale_pos_weight=3.0)
    
    feature_df = classifier.build_rolling_features(raw_df)
    target_series = classifier.generate_binary_target(raw_df["distance_to_default"])
    
    common_idx = feature_df.index.intersection(target_series.dropna().index)
    X = feature_df.loc[common_idx]
    y = target_series.loc[common_idx]

    # 3. Chronological 70/30 Train/Validation Split (prevents data leakage)
    split_idx = int(len(X) * 0.70)
    X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_val = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Training set: {len(X_train)} samples | Validation set: {len(X_val)} samples")

    # 4. Fit XGBoost model and calibrate optimal F1 threshold
    print("Fitting XGBoost model with class imbalance penalization...")
    classifier.fit(X_train, y_train, X_val, y_val)

    print("Calibrating Precision-Recall threshold...")
    optimal_threshold = classifier.calibrate_optimal_threshold(X_val, y_val)
    print(f"Optimal F1 Threshold Calibrated: {optimal_threshold:.4f}")

    # 5. Save model artifact to disk
    model_path = "models/distress_xgboost.pkl"
    joblib.dump(classifier, model_path)
    print(f"Model successfully saved to {model_path}")

if __name__ == "__main__":
    run_training_pipeline()