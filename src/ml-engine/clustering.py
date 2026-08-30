from typing import Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from scipy.stats.mstats import winsorize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

class RobustWinsorScaler(BaseEstimator, TransformerMixin):
    """Custom scikit-learn transformer for winsorization and robust scaling[cite: 1, 2]."""

    def __init__(self, limits: Tuple[float, float] = (0.01, 0.01), corr_threshold: float = 0.85,) -> None:
        self.limits = limits
        self.corr_threshold = corr_threshold
        self.medians_: pd.Series = pd.Series(dtype=float)
        self.iqrs_: pd.Series = pd.Series(dtype=float)
        self.selected_features_: List[str] = []

    def fit(self, X: pd.DataFrame, y: Any = None) -> "RobustWinsorScaler":
        """Identifies non-collinear features and stores Median and IQR statistics."""
        df = X.copy()

        # Absolute correlation matrix
        corr_matrix = df.corr().abs()

        # Extract upper triangle of correlation matrix
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Drop features with |r| > corr_threshold
        to_drop = [
            col
            for col in upper.columns
            if any(upper[col] > self.corr_threshold)
        ]
        self.selected_features_ = [
            col for col in df.columns if col not in to_drop
        ]

        # Median and IQR on remaining features
        pruned_df = df[self.selected_features_]
        self.medians_ = pruned_df.median()
        q75 = pruned_df.quantile(0.75)
        q25 = pruned_df.quantile(0.25)
        iqr = q75 - q25

        # Prevent zero-division
        self.iqrs_ = iqr.replace(0.0, 1.0)

        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Applies 1st/99th percentile winsorization and Median-IQR scaling."""
        df = X[self.selected_features_].copy()

        # Apply scipy 1st/99th percentile winsorization per column
        for col in df.columns:
            df[col] = winsorize(df[col], limits=self.limits)

        # Standardize using stored Median and IQR
        scaled_df = (df - self.medians_) / self.iqrs_

        return scaled_df.to_numpy()


class GeometricDiscoveryEngine:
    """PCA compression and dynamic Silhouette-tuned K-Means clustering[cite: 1, 2]."""

    def __init__(self, variance_retained: float = 0.95) -> None:
        """Initializes PCA to dynamically retain a specified proportion of feature variance."""
        self.variance_retained = variance_retained
        self.pca = PCA(n_components=self.variance_retained)
        self.fitted_pca: Optional[PCA] = None
        self.best_k: int = 3
        self.kmeans_model: Optional[KMeans] = None

    def tune_and_fit_kmeans(self, X_scaled: np.ndarray, k_range: Tuple[int, int] = (3, 6)) -> KMeans:
        """Compresses feature space via PCA and selects optimal K using Silhouette Score."""
        # Fit PCA to retain variance and transform scaled features
        self.fitted_pca = self.pca.fit(X_scaled)
        X_pca = self.fitted_pca.transform(X_scaled)

        best_score = -1.0
        best_k = k_range[0]
        best_model = None

        # Dynamic sweep across K in [k_min, k_max]
        for k in range(k_range[0], k_range[1] + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X_pca)
            score = float(silhouette_score(X_pca, labels))

            if score > best_score:
                best_score = score
                best_k = k
                best_model = kmeans

        self.best_k = best_k
        self.kmeans_model = best_model
        return self.kmeans_model

    
    def identify_distress_centroid(self, kmeans_model: KMeans, feature_names: list) -> np.ndarray:
        """Isolates the cluster centroid with minimum liquidity and maximum leverage."""
        centroids = kmeans_model.cluster_centers_

        # The cluster center with the lowest overall mean coordinates corresponds to the lowest liquidity / highest distress profile
        distress_idx = int(np.argmin(centroids.mean(axis=1)))
        return centroids[distress_idx]

    def compute_distance_to_distress(self, x_entity_pca: np.ndarray, mu_risk: np.ndarray) -> float:
        """Calculates Euclidean distance d_e(t) to the distress centroid."""
        return float(np.linalg.norm(x_entity_pca - mu_risk))

    def evaluate_drift_velocity(self, current_distance: float, previous_distance: float) -> Tuple[float, bool]:
        """Calculates drift velocity delta d_e and flags if migration velocity < -0.5."""
        delta_d = current_distance - previous_distance
        is_alert = bool(delta_d < -0.5)
        return delta_d, is_alert