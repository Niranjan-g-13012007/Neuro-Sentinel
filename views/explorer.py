"""Interactive anomaly explorer with a per-activity investigation panel."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from models.anomaly_detection import FINAL_UNUSUAL, explain_activity
from utils import ui, visualization as viz

SORT_OPTIONS = {
    "Combined score": "combined_score",
    "Reconstruction error": "reconstruction_error",
    "Centroid distance": "centroid_distance",
    "Activity ID": "activity_id",
}
KEY_FEATURES = ["duration_sec", "src_bytes", "dst_bytes", "packets", "connections",
                "request_frequency", "upload_ratio", "unique_destinations"]


def _filter_table(ctx) -> pd.DataFrame:
    table = ctx.table
    col_search, col_cluster, col_status, col_evidence = st.columns([1.2, 1.6, 1.4, 1.4])
    search = col_search.text_input("Search activity ID", placeholder="e.g. 1619")
    clusters = col_cluster.multiselect("Cluster", sorted(table["cluster"].unique()),
                                       format_func=lambda c: f"Behavioral Cluster {c}")
    status = col_status.selectbox("Behavior status", ["All", "Potentially Unusual Behavior", "Normal Behavior"])
    evidence = col_evidence.selectbox("Signals", ["All", "Both signals", "Autoencoder only",
                                                  "Cluster distance only", "Neither"])
    col_sort, col_order, _ = st.columns([1.5, 1.2, 2.7])
    sort_label = col_sort.selectbox("Sort by", list(SORT_OPTIONS))
    descending = col_order.radio("Order", ["Descending", "Ascending"], horizontal=True) == "Descending"

    filtered = table
    if search.strip():
        filtered = filtered[filtered["activity_id"].astype(str).str.contains(search.strip(), regex=False)]
    if clusters:
        filtered = filtered[filtered["cluster"].isin(clusters)]
    if status != "All":
        filtered = filtered[filtered["final_status"] == status]
    if evidence != "All":
        filtered = filtered[filtered["evidence"] == evidence]
    return filtered.sort_values(SORT_OPTIONS[sort_label], ascending=not descending).reset_index(drop=True)


def _explanation_text(ctx, row: pd.Series, explanation: pd.DataFrame) -> str:
    """Plain-language reasoning built only from this activity's computed numbers."""
    parts = []
    if row["anomaly_status"] == "Potential Anomaly":
        parts.append(f"The autoencoder's reconstruction error ({row['reconstruction_error']:.4f}) is above the "
                     f"threshold ({ctx.ae_threshold:.4f}), so the learned patterns do not describe this activity well.")
    else:
        parts.append(f"The autoencoder reconstructs this activity within the threshold "
                     f"({row['reconstruction_error']:.4f} vs {ctx.ae_threshold:.4f}).")
    if row["cluster_ratio"] > 1:
        parts.append(f"It lies {row['cluster_ratio']:.2f}x the typical outer distance of {row['cluster_name']}, "
                     f"i.e. it is far from its group's center.")
    else:
        parts.append(f"It lies within the typical spread of {row['cluster_name']}.")
    top_error = explanation.nlargest(3, "error_share_percent")
    parts.append("Features hardest to reconstruct: " + ", ".join(
        f"{r.feature} ({r.error_share_percent:.0f}%)" for r in top_error.itertuples()) + ".")
    top_dev = explanation.reindex(explanation["deviation"].abs().nlargest(3).index)
    parts.append("Largest deviation from the cluster average: " + ", ".join(
        f"{r.feature} ({r.deviation:+.2f})" for r in top_dev.itertuples()) + ".")
    return " ".join(parts)


def _detail_panel(ctx, row: pd.Series) -> None:
    index = int(ctx.table.index[ctx.table["activity_id"] == row["activity_id"]][0])
    explanation = explain_activity(index, ctx.prep.X, ctx.clustering.centers, ctx.clustering.labels,
                                   ctx.squared_errors, ctx.prep.feature_cols)
    unusual = row["final_status"] == FINAL_UNUSUAL

    st.markdown(f"### Activity {int(row['activity_id'])}")
    ui.pills([(row["final_status"], not unusual), (row["cluster_name"], True),
              (f"Protocol: {row['protocol']}", True), (f"Signals: {row['evidence']}", True)])
    ui.kpi_row([
        ("Reconstruction error", f"{row['reconstruction_error']:.4f}", f"threshold {ctx.ae_threshold:.4f}",
         "warn" if row["anomaly_status"] == "Potential Anomaly" else "primary"),
        ("Autoencoder status", row["anomaly_status"], f"ratio {row['autoencoder_ratio']:.2f}", "primary"),
        ("Centroid distance", f"{row['centroid_distance']:.2f}", f"cluster ratio {row['cluster_ratio']:.2f}", "primary"),
        ("Combined score", f"{row['combined_score']:.2f}", "unusual if above 1.0", "warn" if unusual else "primary"),
    ])
    st.markdown("&nbsp;", unsafe_allow_html=True)
    ui.note(_explanation_text(ctx, row, explanation), caution=unusual)

    left, right = st.columns(2)
    with left:
        ui.show_plotly(viz.feature_comparison(explanation), key="detail_compare")
    with right:
        ui.show_plotly(viz.error_share_bar(explanation), key="detail_error")

    key_cols = [c for c in KEY_FEATURES if c in ctx.prep.feature_cols]
    st.markdown("**Key feature values (z-scores)**")
    ui.show_dataframe(pd.DataFrame([row[key_cols].astype(float)]), hide_index=True,
                      column_config={c: st.column_config.NumberColumn(format="%.2f") for c in key_cols})
    st.caption("Positive z-scores are above the dataset average, negative are below. "
               "A flag means 'worth investigating', not 'confirmed malicious'.")


def render(ctx) -> None:
    ui.hero("Anomaly Explorer", "Investigate - inspect individual activities",
            "Filter, search and sort all activities, then select one to see why NeuroSentinel considers it "
            "typical or unusual.")
    filtered = _filter_table(ctx)

    total_unusual = int((filtered["final_status"] == FINAL_UNUSUAL).sum())
    st.markdown(f"**{len(filtered):,}** activities match ({total_unusual:,} potentially unusual).")
    if filtered.empty:
        st.info("No activities match the current filters. Adjust or clear a filter to continue.")
        return

    display_cols = ["activity_id", "cluster_name", "protocol", "reconstruction_error", "anomaly_status",
                    "centroid_distance", "combined_score", "evidence", "final_status"]
    labels = {"activity_id": "Activity ID", "cluster_name": "Cluster", "protocol": "Protocol",
              "reconstruction_error": "Reconstruction error", "anomaly_status": "Autoencoder status",
              "centroid_distance": "Centroid distance", "combined_score": "Combined score",
              "evidence": "Signals", "final_status": "NeuroSentinel status"}
    view = filtered[display_cols].rename(columns=labels)
    event = ui.show_dataframe(view, hide_index=True, on_select="rerun", selection_mode="single-row",
                              key="explorer_table", height=330, column_config={
                                  "Reconstruction error": st.column_config.NumberColumn(format="%.4f"),
                                  "Centroid distance": st.column_config.NumberColumn(format="%.2f"),
                                  "Combined score": st.column_config.NumberColumn(format="%.3f")})
    st.download_button("Download filtered results (CSV)", view.to_csv(index=False).encode("utf-8"),
                       file_name="neurosentinel_filtered_activities.csv", mime="text/csv")

    selected_rows = event.selection.rows if event is not None and hasattr(event, "selection") else []
    if selected_rows and selected_rows[0] < len(filtered):
        selected = filtered.iloc[selected_rows[0]]
    else:
        st.caption("Click a row in the table to inspect it. Showing the first match by default.")
        selected = filtered.iloc[0]
    st.markdown("---")
    _detail_panel(ctx, selected)
