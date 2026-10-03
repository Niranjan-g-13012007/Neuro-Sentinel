"""Autoencoder training, thresholds and combined NeuroSentinel analysis."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from models.anomaly_detection import (FINAL_UNUSUAL, MIN_CLUSTER_SIZE_FOR_OWN_THRESHOLD, STATUS_ANOMALY)
from utils import ui, visualization as viz


def _architecture_tab(ctx) -> None:
    result = ctx.autoencoder
    history = result.history
    st.markdown("**Network architecture** (number of units per layer)")
    ui.architecture_diagram(result.layer_sizes)
    st.caption("Hidden layers use ReLU. The output layer is linear because standardized data contains "
               "negative values. Optimizer: Adam. Loss: mean squared error.")

    ui.kpi_row([
        ("Epochs run", f"{result.epochs_run}", f"maximum {ctx.settings.epochs}; early stopping on", "primary"),
        ("Best epoch", f"{result.best_epoch}", "weights restored to this epoch", "primary"),
        ("Final training loss", f"{history['loss'][result.best_epoch - 1]:.4f}",
         f"{len(result.train_idx):,} training activities", "primary"),
        ("Final validation loss", f"{history['val_loss'][result.best_epoch - 1]:.4f}",
         f"{len(result.val_idx):,} held-out activities", "primary"),
    ])
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ui.show_plotly(viz.loss_curve(history, result.best_epoch), key="loss_curve")
    ui.note("What the loss means: the network squeezes each activity through a very narrow 2-unit bottleneck "
            "and then tries to rebuild the original 13 feature values. The loss is the average squared "
            "difference between the original and the rebuilt values. Falling training and validation losses "
            "mean it is learning the common patterns. If validation loss stayed far above training loss, "
            "the network would be memorizing instead of learning general patterns.")


def _threshold_tab(ctx) -> None:
    table, percentile = ctx.table, ctx.settings.percentile
    flagged = table["anomaly_status"] == STATUS_ANOMALY
    val = table["split"] == "Validation"

    ui.kpi_row([
        ("Reconstruction threshold", f"{ctx.ae_threshold:.4f}", f"{percentile:g}th percentile of validation error", "primary"),
        ("Flagged (all activities)", f"{int(flagged.sum()):,}", f"{flagged.mean() * 100:.1f}% exceed the threshold", "warn"),
        ("Flagged (validation set)", f"{int((flagged & val).sum()):,}", f"{flagged[val].mean() * 100:.1f}% of held-out data", "primary"),
        ("Flagged (training set)", f"{int((flagged & ~val).sum()):,}", f"{flagged[~val].mean() * 100:.1f}% of training data", "primary"),
    ])
    st.markdown("&nbsp;", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        ui.show_plotly(viz.error_histogram(table, ctx.ae_threshold, percentile), key="err_hist")
    with right:
        ui.show_plotly(viz.latent_scatter(table), key="latent")
    ui.note(
        f"How the threshold is derived: the autoencoder never trains on the validation activities, so their "
        f"reconstruction errors show how well it reconstructs data it has not memorized. The threshold is the "
        f"{percentile:g}th percentile of those errors. Activities above it are labelled 'Potential Anomaly'. "
        f"Because a percentile is used, a roughly fixed share of data is expected to exceed it by construction: "
        f"the threshold is a way to prioritize activities for review, not a measured detection rate. "
        f"Training activities tend to have slightly lower error than validation activities because the "
        f"network was fitted to them.")


def _combined_tab(ctx) -> None:
    table, percentile = ctx.table, ctx.settings.percentile
    st.markdown("**How the two models are combined**")
    st.markdown("Each activity receives two independent signals. Both are turned into a ratio against a "
                "percentile threshold, so that 1.0 means 'at the threshold':")
    st.latex(r"r_{AE}=\sqrt{\frac{e_i}{\tau_{AE}}}\qquad r_{CL}=\frac{d_i}{\tau_{c}}\qquad "
             r"S_i=\frac{r_{AE}+r_{CL}}{2}")
    st.markdown(
        f"- **e_i**: autoencoder reconstruction error (mean squared error) of activity *i*; "
        f"**tau_AE**: its {percentile:g}th percentile on the validation set.\n"
        f"- **d_i**: Euclidean distance from activity *i* to the centroid of its K-Means cluster; "
        f"**tau_c**: the {percentile:g}th percentile of distances inside that cluster, so each cluster is judged "
        f"against its *own* spread (clusters with fewer than {MIN_CLUSTER_SIZE_FOR_OWN_THRESHOLD} members "
        f"use the global distance threshold).\n"
        f"- The square root puts the autoencoder ratio on the same linear z-score scale as the distance.\n"
        f"- Equal weights are used because there are no labels that would justify trusting one signal more.\n"
        f"- **Potentially Unusual Behavior** when S > 1; otherwise **Normal Behavior**.")

    results = ctx.results
    unusual = results["final_status"] == FINAL_UNUSUAL
    ui.kpi_row([
        ("Potentially unusual", f"{int(unusual.sum()):,}", f"{unusual.mean() * 100:.1f}% of activities", "warn"),
        ("Both signals exceed", f"{int((results['evidence'] == 'Both signals').sum()):,}", "highest agreement", "primary"),
        ("Autoencoder only", f"{int((results['evidence'] == 'Autoencoder only').sum()):,}", "error above threshold", "primary"),
        ("Cluster distance only", f"{int((results['evidence'] == 'Cluster distance only').sum()):,}",
         "far from own centroid", "primary"),
    ])
    st.markdown("&nbsp;", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        ui.show_plotly(viz.signal_scatter(table), key="signals")
    with right:
        ui.show_plotly(viz.status_by_cluster(table), key="status_cluster")

    st.markdown("**Per-cluster thresholds**")
    rows = []
    for cluster, tau in sorted(ctx.cluster_thresholds.items()):
        members = int((ctx.clustering.labels == cluster).sum())
        flagged = int((unusual & (results["cluster"] == cluster)).sum())
        rows.append({"Cluster": cluster, "Members": members, f"Distance threshold (P{percentile:g})": tau,
                     "Potentially unusual": flagged,
                     "Share of cluster (%)": flagged / members * 100})
    ui.show_dataframe(pd.DataFrame(rows), hide_index=True, column_config={
        f"Distance threshold (P{percentile:g})": st.column_config.NumberColumn(format="%.3f"),
        "Share of cluster (%)": st.column_config.NumberColumn(format="%.1f")})
    ui.note("Interpretation: 'Both signals' means the activity is unusual relative to the neural network's "
            "learned patterns and also far from its own behavioral group. Activities flagged by a single signal "
            "are still worth a look but carry less agreement. None of these labels confirm an attack.")


def render(ctx) -> None:
    ui.hero("Neural Anomaly Detection", "Learn and detect - autoencoder reconstruction error",
            "The autoencoder learns to rebuild typical activity. Activities it cannot rebuild well deviate from "
            "the patterns it learned and are flagged as potential anomalies. The result is combined with "
            "K-Means cluster context.")
    tab_ae, tab_threshold, tab_combined = st.tabs(
        ["Autoencoder training", "Reconstruction error and threshold", "Combined NeuroSentinel analysis"])
    with tab_ae:
        _architecture_tab(ctx)
    with tab_threshold:
        _threshold_tab(ctx)
    with tab_combined:
        _combined_tab(ctx)
