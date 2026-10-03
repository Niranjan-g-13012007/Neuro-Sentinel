"""Orchestrates the NeuroSentinel ML pipeline with Streamlit caching.

Cache strategy
--------------
* ``st.cache_data``     : deterministic data transformations (loading, K evaluation, PCA,
                          reconstruction results).
* ``st.cache_resource`` : trained model objects (K-Means, Keras autoencoder).

Every cached function receives the parameters that influence its output, so a change in
the sidebar controls invalidates only the affected steps. Changing the anomaly percentile,
for example, retrains nothing - only the cheap thresholding step is recomputed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from models import anomaly_detection as ad
from models import autoencoder as ae_lib
from models import clustering as cl
from utils import preprocessing as prep_lib

DATA_PATH = Path(__file__).parent / "data" / "neurosentinel_final_cleaned.csv"
SEED = 42


@dataclass(frozen=True)
class Settings:
    """User-selected model controls (``k=None`` means automatic selection)."""
    k: int | None = None
    epochs: int = 100
    batch_size: int = 32
    percentile: float = 95.0


@dataclass
class Context:
    """All pipeline outputs consumed by the dashboard pages."""
    settings: Settings
    raw: pd.DataFrame
    report: prep_lib.ValidationReport
    prep: prep_lib.PreparedData
    k_evaluation: pd.DataFrame
    recommended_k: int
    k_was_overridden: bool
    clustering: cl.ClusteringResult
    cluster_summary: pd.DataFrame
    pca_coords: np.ndarray
    pca_variance: np.ndarray
    autoencoder: ae_lib.AutoencoderResult
    errors: np.ndarray
    squared_errors: np.ndarray
    latent: np.ndarray
    ae_threshold: float
    cluster_thresholds: dict[int, float]
    distances: np.ndarray
    results: pd.DataFrame
    table: pd.DataFrame  # results + feature values + protocol + PCA, for the UI


# --------------------------------------------------------------------------
# Cached building blocks
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_and_prepare(path: str, modified_time: float):
    """Load + validate + prepare. ``modified_time`` invalidates the cache if the file changes."""
    raw = prep_lib.load_dataset(path)
    report = prep_lib.validate_dataset(raw)
    prepared = prep_lib.prepare_dataset(raw)
    return raw, report, prepared


@st.cache_data(show_spinner=False)
def cached_k_evaluation(X: np.ndarray, seed: int) -> pd.DataFrame:
    return cl.evaluate_k_range(X, seed=seed)


@st.cache_resource(show_spinner=False)
def cached_kmeans(X: np.ndarray, k: int, seed: int) -> cl.ClusteringResult:
    return cl.fit_kmeans(X, k, seed=seed)


@st.cache_data(show_spinner=False)
def cached_pca(X: np.ndarray, seed: int):
    return cl.compute_pca(X, seed=seed)


@st.cache_resource(show_spinner=False)
def cached_autoencoder(X: np.ndarray, epochs: int, batch_size: int, seed: int) -> ae_lib.AutoencoderResult:
    return ae_lib.train_autoencoder(X, epochs=epochs, batch_size=batch_size, seed=seed)


@st.cache_data(show_spinner=False)
def cached_reconstruction(_autoencoder: ae_lib.AutoencoderResult, X: np.ndarray,
                          epochs: int, batch_size: int, seed: int):
    """Reconstruction results; the underscore keeps the model out of the hash,
    while epochs/batch_size/seed identify which trained model produced them."""
    errors, squared_errors, _ = ae_lib.reconstruction_errors(_autoencoder.model, X)
    latent = ae_lib.encode(_autoencoder.encoder, X)
    return errors, squared_errors, latent


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------
def build_context(settings: Settings) -> Context:
    modified = DATA_PATH.stat().st_mtime if DATA_PATH.exists() else 0.0
    raw, report, prepared = load_and_prepare(str(DATA_PATH), modified)
    X = prepared.X

    # --- Unsupervised learning: K-Means -----------------------------------
    k_evaluation = cached_k_evaluation(X, SEED)
    recommended_k = cl.recommend_k(k_evaluation)
    k = settings.k or recommended_k
    clustering = cached_kmeans(X, k, SEED)
    summary = cl.cluster_summary(prepared.frame, prepared.feature_cols, clustering.labels)
    pca_coords, pca_variance, _ = cached_pca(X, SEED)

    # --- Neural network: autoencoder --------------------------------------
    autoencoder = cached_autoencoder(X, settings.epochs, settings.batch_size, SEED)
    errors, squared_errors, latent = cached_reconstruction(
        autoencoder, X, settings.epochs, settings.batch_size, SEED)

    # --- Thresholds + combined analysis (cheap; recomputed on every change) --
    ae_threshold = ad.autoencoder_threshold(errors[autoencoder.val_idx], settings.percentile)
    distances = cl.centroid_distances(X, clustering.labels, clustering.centers)
    cluster_thresholds = ad.cluster_distance_thresholds(distances, clustering.labels, settings.percentile)
    results = ad.build_results(prepared.frame[prep_lib.ID_COLUMN].to_numpy(), clustering.labels,
                               errors, distances, ae_threshold, cluster_thresholds)

    table = pd.concat([results, prepared.frame[prepared.feature_cols]], axis=1)
    table["protocol"] = prepared.protocol.to_numpy()
    table["pc1"], table["pc2"] = pca_coords[:, 0], pca_coords[:, 1]
    table["latent_1"], table["latent_2"] = latent[:, 0], latent[:, 1]
    table["split"] = "Training"
    table.loc[autoencoder.val_idx, "split"] = "Validation"

    return Context(
        settings=settings, raw=raw, report=report, prep=prepared,
        k_evaluation=k_evaluation, recommended_k=recommended_k,
        k_was_overridden=settings.k is not None and settings.k != recommended_k,
        clustering=clustering, cluster_summary=summary,
        pca_coords=pca_coords, pca_variance=pca_variance,
        autoencoder=autoencoder, errors=errors, squared_errors=squared_errors, latent=latent,
        ae_threshold=ae_threshold, cluster_thresholds=cluster_thresholds,
        distances=distances, results=results, table=table,
    )
