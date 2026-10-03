"""Dashboard homepage."""
from __future__ import annotations

import streamlit as st

from models.anomaly_detection import FINAL_NORMAL, FINAL_UNUSUAL
from utils import ui, visualization as viz


def render(ctx) -> None:
    ui.hero(
        "NeuroSentinel",
        "Behavioral Intelligence Platform",
        "Unsupervised behavioral pattern discovery and neural anomaly detection for network activity. "
        "No labels are used: K-Means discovers behavioral groups and an autoencoder learns what typical "
        "activity looks like, so unusual activity can be surfaced for investigation.")

    ui.workflow_steps([
        ("Observe", "NeuroSentinel receives network activity data."),
        ("Discover", "K-Means identifies groups of similar behavioral patterns."),
        ("Learn", "The autoencoder learns how typical activity can be represented and reconstructed."),
        ("Detect", "Activities with unusually high reconstruction error are flagged for investigation."),
        ("Investigate", "Analysts can inspect individual activities and see why they were flagged."),
    ])

    results = ctx.results
    total = len(results)
    unusual = int((results["final_status"] == FINAL_UNUSUAL).sum())
    normal = total - unusual
    ui.kpi_row([
        ("Total activities", f"{total:,}", f"{len(ctx.prep.feature_cols)} model features", "primary"),
        ("Behavioral clusters", f"{ctx.clustering.k}", f"Silhouette score {ctx.clustering.silhouette:.3f}", "primary"),
        ("Normal behavior", f"{normal:,}", f"{normal / total * 100:.1f}% of activities", "primary"),
        ("Potentially unusual", f"{unusual:,}", f"{unusual / total * 100:.1f}% - requires investigation", "warn"),
    ])

    # Compact data-quality strip ------------------------------------------------
    report = ctx.report
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ui.pills([
        (f"{report.total_missing} missing values", report.total_missing == 0),
        (f"{report.duplicate_rows} duplicate rows", report.duplicate_rows == 0),
        (f"{report.infinite_values} infinite values", report.infinite_values == 0),
        ("Features already standardized - not re-scaled" if ctx.prep.was_already_standardized
         else "Features scaled with StandardScaler", True),
    ])
    if not report.is_clean:
        ui.note("Data-quality findings: " + " ".join(report.issues), caution=True)

    ui.section("Behavioral Pattern Map",
               "Each point is one network activity projected onto two PCA components. "
               "Red rings mark activities flagged as potentially unusual.")
    ui.show_plotly(viz.pca_scatter(ctx.table, ctx.pca_variance), key="overview_pca")

    ui.section("Anomaly Distribution")
    left, right = st.columns(2)
    with left:
        ui.show_plotly(viz.error_histogram(ctx.table, ctx.ae_threshold, ctx.settings.percentile),
                       key="overview_hist")
    with right:
        ui.show_plotly(viz.status_by_cluster(ctx.table), key="overview_status")

    ui.section("Highest-Scoring Activities",
               "Ranked by the combined NeuroSentinel score (see 'Neural Anomaly Detection' for the formula).")
    top = ctx.table.sort_values("combined_score", ascending=False).head(10)
    display = top[["activity_id", "cluster_name", "reconstruction_error", "combined_score", "evidence",
                   "final_status"]].rename(columns={
        "activity_id": "Activity ID", "cluster_name": "Cluster", "reconstruction_error": "Reconstruction error",
        "combined_score": "Combined score", "evidence": "Signals", "final_status": "Status"})
    ui.show_dataframe(display, hide_index=True, column_config={
        "Reconstruction error": st.column_config.NumberColumn(format="%.4f"),
        "Combined score": st.column_config.NumberColumn(format="%.3f"),
    })
    ui.note("A flagged activity is an unusual behavioral pattern that requires investigation. "
            "It is not a confirmed attack: this system has no ground-truth labels and does not classify attacks.")
