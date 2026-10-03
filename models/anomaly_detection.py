"""Thresholding and the combined K-Means + Autoencoder anomaly analysis.

Combined score (documented in the app and README)
-------------------------------------------------
For activity *i* in cluster *c*:

    e_i   = mean squared reconstruction error of the autoencoder
    tau_e = p-th percentile of e on the held-out validation set
    d_i   = Euclidean distance from x_i to the centroid of cluster c
    tau_c = p-th percentile of d over the members of cluster c

    r_AE  = sqrt(e_i / tau_e)      (RMSE relative to its threshold)
    r_CL  = d_i / tau_c            (distance relative to its cluster's threshold)
    S_i   = (r_AE + r_CL) / 2

Both ratios are expressed in the same linear z-score units (RMSE and Euclidean
distance), so an equal-weight average is a neutral choice: without labels there
is no evidence that one signal deserves more weight. ``S_i > 1`` means the
activity is, on average, beyond the thresholds of both detectors.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from models.clustering import cluster_label

STATUS_NORMAL = "Normal"
STATUS_ANOMALY = "Potential Anomaly"
FINAL_NORMAL = "Normal Behavior"
FINAL_UNUSUAL = "Potentially Unusual Behavior"

# Clusters with fewer members than this fall back to the global distance threshold
MIN_CLUSTER_SIZE_FOR_OWN_THRESHOLD = 20


def autoencoder_threshold(validation_errors: np.ndarray, percentile: float) -> float:
    """Threshold = chosen percentile of reconstruction errors on held-out data."""
    return float(np.percentile(validation_errors, percentile))


def cluster_distance_thresholds(distances: np.ndarray, labels: np.ndarray, percentile: float) -> dict[int, float]:
    """Per-cluster percentile of centroid distances (global fallback for tiny clusters)."""
    global_threshold = float(np.percentile(distances, percentile))
    thresholds = {}
    for cluster in np.unique(labels):
        members = distances[labels == cluster]
        if len(members) >= MIN_CLUSTER_SIZE_FOR_OWN_THRESHOLD:
            thresholds[int(cluster)] = float(np.percentile(members, percentile))
        else:
            thresholds[int(cluster)] = global_threshold
    return thresholds


def _evidence_label(ae_flag: bool, cluster_flag: bool) -> str:
    if ae_flag and cluster_flag:
        return "Both signals"
    if ae_flag:
        return "Autoencoder only"
    if cluster_flag:
        return "Cluster distance only"
    return "Neither"


def build_results(activity_ids: np.ndarray, labels: np.ndarray, errors: np.ndarray,
                  distances: np.ndarray, ae_threshold: float,
                  cluster_thresholds: dict[int, float]) -> pd.DataFrame:
    """Assemble one row per activity with every signal and the final status."""
    cluster_tau = np.array([cluster_thresholds[int(c)] for c in labels])
    ae_ratio = np.sqrt(errors / ae_threshold)
    cluster_ratio = distances / cluster_tau
    combined = (ae_ratio + cluster_ratio) / 2.0

    ae_flag = errors > ae_threshold
    cluster_flag = distances > cluster_tau
    results = pd.DataFrame({
        "activity_id": activity_ids,
        "cluster": labels.astype(int),
        "cluster_name": [cluster_label(int(c)) for c in labels],
        "reconstruction_error": errors,
        "anomaly_status": np.where(ae_flag, STATUS_ANOMALY, STATUS_NORMAL),
        "centroid_distance": distances,
        "autoencoder_ratio": ae_ratio,
        "cluster_ratio": cluster_ratio,
        "combined_score": combined,
        "evidence": [_evidence_label(a, c) for a, c in zip(ae_flag, cluster_flag)],
        "final_status": np.where(combined > 1.0, FINAL_UNUSUAL, FINAL_NORMAL),
    })
    return results


def explain_activity(row_index: int, X: np.ndarray, centers: np.ndarray, labels: np.ndarray,
                     squared_error: np.ndarray, feature_cols: list[str]) -> pd.DataFrame:
    """Per-feature view of one activity.

    * ``z_value``            : the activity's (standardized) value
    * ``cluster_mean``       : the centroid value of its cluster
    * ``deviation``          : z_value - cluster_mean
    * ``error_share_percent``: share of this activity's reconstruction error
                               caused by each feature
    """
    z_values = X[row_index]
    cluster_mean = centers[labels[row_index]]
    feature_error = squared_error[row_index]
    total = feature_error.sum()
    share = feature_error / total * 100 if total > 0 else np.zeros_like(feature_error)
    return pd.DataFrame({
        "feature": feature_cols,
        "z_value": z_values,
        "cluster_mean": cluster_mean,
        "deviation": z_values - cluster_mean,
        "error_share_percent": share,
    })
