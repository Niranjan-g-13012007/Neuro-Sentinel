"""K-Means behavioral clusters page."""
from __future__ import annotations

import streamlit as st

from models.clustering import SILHOUETTE_TOLERANCE, cluster_label, describe_cluster
from utils import ui, visualization as viz


def render(ctx) -> None:
    ui.hero("Behavioral Clusters", "Discover - unsupervised K-Means",
            "K-Means groups activities with similar behavior without using any labels. "
            "Clusters describe how activities group together; they do not say whether a group is harmful.")

    clustering = ctx.clustering

    ui.section("Choosing the number of clusters")
    ui.kpi_row([
        ("Selected number of clusters", f"K = {clustering.k}",
         "manual override" if ctx.k_was_overridden else "chosen automatically", "primary"),
        ("Silhouette score", f"{clustering.silhouette:.3f}", "cohesion vs separation (-1 to 1)", "primary"),
        ("Davies-Bouldin index", f"{clustering.davies_bouldin:.3f}", "lower is better", "primary"),
        ("Inertia", f"{clustering.inertia:,.0f}", "within-cluster sum of squares", "primary"),
    ])
    st.markdown("&nbsp;", unsafe_allow_html=True)
    left, right = st.columns(2)
    with left:
        ui.show_plotly(viz.elbow_figure(ctx.k_evaluation, clustering.k), key="elbow")
    with right:
        ui.show_plotly(viz.silhouette_figure(ctx.k_evaluation, clustering.k), key="silhouette")

    best_row = ctx.k_evaluation.loc[ctx.k_evaluation["silhouette"].idxmax()]
    ui.note(
        f"Selection rule: the best silhouette score is {best_row['silhouette']:.3f} at K = {int(best_row['k'])}. "
        f"Scores within {SILHOUETTE_TOLERANCE} of the best are treated as a tie, and the largest such K is "
        f"chosen so that the segmentation is not reduced to 'one group vs everything else'. "
        f"The automatic recommendation is K = {ctx.recommended_k}. "
        f"The elbow curve is shown as supporting evidence; inertia always decreases as K grows, "
        f"so it is read by looking for where the improvement flattens.")

    ui.section("Cluster summary", "Mean feature values per cluster (z-scores: 0 = dataset average).")
    summary = ctx.cluster_summary.rename(columns={
        "cluster": "Cluster ID", "name": "Name", "activities": "Activities", "percent_of_dataset": "% of dataset"})
    feature_config = {c: st.column_config.NumberColumn(format="%.2f") for c in ctx.prep.feature_cols}
    ui.show_dataframe(summary, hide_index=True, column_config={
        "% of dataset": st.column_config.NumberColumn(format="%.2f%%"), **feature_config})

    left, right = st.columns([3, 2])
    with left:
        ui.show_plotly(viz.cluster_profile_heatmap(clustering.centers, ctx.prep.feature_cols), key="profile")
    with right:
        ui.show_plotly(viz.cluster_size_bar(ctx.cluster_summary), key="sizes")

    ui.section("What distinguishes each cluster")
    for row in ctx.cluster_summary.itertuples():
        description = describe_cluster(clustering.centers[row.cluster], ctx.prep.feature_cols, top_n=4)
        st.markdown(f"- **{cluster_label(row.cluster)}** - {row.activities:,} activities "
                    f"({row.percent_of_dataset:.1f}%). Most distinctive: {description}.")
    ui.note("Small clusters are not automatically suspicious and large clusters are not automatically safe. "
            "A cluster is only a group of similar behavior; any judgement about intent requires "
            "domain knowledge and further investigation.", caution=False)

    ui.section("PCA projection", "PCA compresses the features into two components for visualization only. "
               "Clustering itself uses all features, not the 2-D projection.")
    ui.show_plotly(viz.pca_scatter(ctx.table, ctx.pca_variance, highlight=False), key="clusters_pca")
    total_variance = ctx.pca_variance.sum() * 100
    st.caption(f"The two components together explain {total_variance:.1f}% of the variance. "
               "Clusters that overlap here may still be separable in the full feature space.")
