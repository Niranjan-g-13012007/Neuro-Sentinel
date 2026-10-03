"""K-Means behavioural clustering, cluster evaluation and PCA projection."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

K_MIN, K_MAX = 2, 10
N_INIT = 10
# Silhouette differences smaller than this are treated as a tie (see recommend_k).
SILHOUETTE_TOLERANCE = 0.02


@dataclass
class ClusteringResult:
    k: int
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    silhouette: float
    davies_bouldin: float


def evaluate_k_range(X: np.ndarray, k_min: int = K_MIN, k_max: int = K_MAX, seed: int = 42) -> pd.DataFrame:
    """Fit K-Means for each candidate K; record inertia (elbow) and silhouette."""
    rows = []
    for k in range(k_min, k_max + 1):
        model = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)
        rows.append({
            "k": k,
            "inertia": float(model.inertia_),
            "silhouette": float(silhouette_score(X, model.labels_)),
        })
    return pd.DataFrame(rows)


def recommend_k(k_evaluation: pd.DataFrame, tolerance: float = SILHOUETTE_TOLERANCE) -> int:
    """Select K from the silhouette scores.

    Rule: find the best silhouette score, then choose the *largest* K whose
    score is within ``tolerance`` of it. A plain argmax often returns K=2, which
    only separates one group from "everything else". Near-ties are therefore
    resolved towards the richer segmentation. The user can always override K.
    """
    best_score = k_evaluation["silhouette"].max()
    candidates = k_evaluation[k_evaluation["silhouette"] >= best_score - tolerance]
    return int(candidates["k"].max())


def fit_kmeans(X: np.ndarray, k: int, seed: int = 42) -> ClusteringResult:
    """Train the final K-Means model for the chosen K."""
    model = KMeans(n_clusters=k, n_init=N_INIT, random_state=seed).fit(X)
    return ClusteringResult(
        k=k,
        labels=model.labels_,
        centers=model.cluster_centers_,
        inertia=float(model.inertia_),
        silhouette=float(silhouette_score(X, model.labels_)),
        davies_bouldin=float(davies_bouldin_score(X, model.labels_)),
    )


def cluster_label(cluster_id: int) -> str:
    """Neutral display name; clusters are never called 'malicious'."""
    return f"Behavioral Cluster {cluster_id}"


def cluster_summary(frame: pd.DataFrame, feature_cols: list[str], labels: np.ndarray) -> pd.DataFrame:
    """Size, share and mean feature behaviour of every cluster."""
    data = frame[feature_cols].copy()
    data["cluster"] = labels
    summary = data.groupby("cluster")[feature_cols].mean()
    sizes = data["cluster"].value_counts().sort_index()
    summary.insert(0, "activities", sizes)
    summary.insert(1, "percent_of_dataset", (sizes / len(data) * 100).round(2))
    summary.insert(0, "name", [cluster_label(c) for c in summary.index])
    return summary.reset_index()


def describe_cluster(center: np.ndarray, feature_cols: list[str], top_n: int = 3, min_abs: float = 0.5) -> str:
    """Plain-language description of what makes a cluster different.

    ``center`` is in z-score units (the data is standardized), so values far
    from 0 mean 'far from the dataset average'.
    """
    order = np.argsort(-np.abs(center))[:top_n]
    parts = [f"{feature_cols[i]} ({center[i]:+.2f})" for i in order if abs(center[i]) >= min_abs]
    return ", ".join(parts) if parts else "close to the dataset average on all features"


def compute_pca(X: np.ndarray, seed: int = 42, n_components: int = 2):
    """Project the features onto the first principal components."""
    pca = PCA(n_components=n_components, random_state=seed)
    coords = pca.fit_transform(X)
    return coords, pca.explained_variance_ratio_, pca.components_


def centroid_distances(X: np.ndarray, labels: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Euclidean distance from each activity to its own cluster centroid."""
    return np.linalg.norm(X - centers[labels], axis=1)
