"""Plotly figure builders. Each function returns a figure; pages decide where to show it."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from models.anomaly_detection import FINAL_NORMAL, FINAL_UNUSUAL

PRIMARY = "#4F46E5"
SECONDARY = "#0EA5E9"
WARN = "#DC2626"
NEUTRAL = "#94A3B8"
# Cluster colours deliberately avoid red/orange, which are reserved for anomalies.
CLUSTER_COLORS = ["#4F46E5", "#0EA5E9", "#14B8A6", "#8B5CF6", "#64748B",
                  "#22C55E", "#06B6D4", "#A855F7", "#3B82F6", "#84CC16"]
DIVERGING = [[0.0, "#0F766E"], [0.5, "#FFFFFF"], [1.0, "#4338CA"]]
HOVER_FEATURES = ["src_bytes", "dst_bytes", "packets", "connections", "request_frequency"]


def _base_layout(fig: go.Figure, title: str | None = None, height: int = 380) -> go.Figure:
    fig.update_layout(
        template="plotly_white", height=height, title=dict(text=title, x=0.0, font=dict(size=15)) if title else None,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        font=dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", size=12, color="#0F172A"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.update_xaxes(gridcolor="#EEF2F7", zerolinecolor="#E2E8F0")
    fig.update_yaxes(gridcolor="#EEF2F7", zerolinecolor="#E2E8F0")
    return fig


def cluster_color(cluster_id: int) -> str:
    return CLUSTER_COLORS[int(cluster_id) % len(CLUSTER_COLORS)]


# ----------------------------------------------------------------- EDA
def feature_histogram(values: pd.Series, name: str) -> go.Figure:
    fig = go.Figure(go.Histogram(x=values, nbinsx=50, marker_color=PRIMARY, opacity=0.85))
    fig.update_layout(xaxis_title=f"{name} (z-score)", yaxis_title="Activities", bargap=0.04)
    return _base_layout(fig, f"Distribution of {name}", height=340)


def all_feature_histograms(frame: pd.DataFrame, columns: list[str], ncols: int = 4) -> go.Figure:
    nrows = int(np.ceil(len(columns) / ncols))
    fig = make_subplots(rows=nrows, cols=ncols, subplot_titles=columns, vertical_spacing=0.09)
    for i, col in enumerate(columns):
        fig.add_trace(go.Histogram(x=frame[col], nbinsx=30, marker_color=PRIMARY, showlegend=False),
                      row=i // ncols + 1, col=i % ncols + 1)
    fig.update_annotations(font_size=11)
    return _base_layout(fig, height=190 * nrows + 40)


def correlation_heatmap(corr: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.index, zmin=-1, zmax=1, colorscale=DIVERGING,
        text=np.round(corr.values, 2), texttemplate="%{text}", textfont=dict(size=9),
        colorbar=dict(title="r", thickness=12)))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return _base_layout(fig, "Feature correlation (Pearson)", height=560)


def protocol_distribution(counts: pd.Series) -> go.Figure:
    share = counts / counts.sum() * 100
    fig = go.Figure(go.Bar(
        x=counts.index, y=counts.values, marker_color=PRIMARY,
        text=[f"{c:,} ({s:.1f}%)" for c, s in zip(counts.values, share.values)], textposition="outside"))
    fig.update_layout(yaxis_title="Activities", xaxis_title="Protocol")
    fig.update_yaxes(range=[0, counts.max() * 1.15])
    return _base_layout(fig, "Protocol distribution", height=340)


def protocol_behavior(means: pd.DataFrame) -> go.Figure:
    """Grouped bars: mean z-score of key features for each protocol."""
    fig = go.Figure()
    for i, protocol in enumerate(means.index):
        fig.add_trace(go.Bar(name=str(protocol), x=means.columns, y=means.loc[protocol],
                             marker_color=CLUSTER_COLORS[i % len(CLUSTER_COLORS)]))
    fig.update_layout(barmode="group", yaxis_title="Mean z-score")
    return _base_layout(fig, "Average behaviour by protocol (z-scores)", height=340)


# ------------------------------------------------------- K-Means / PCA
def elbow_figure(k_eval: pd.DataFrame, selected_k: int) -> go.Figure:
    fig = go.Figure(go.Scatter(x=k_eval["k"], y=k_eval["inertia"], mode="lines+markers",
                               line=dict(color=PRIMARY, width=2), marker=dict(size=7)))
    chosen = k_eval[k_eval["k"] == selected_k]
    fig.add_trace(go.Scatter(x=chosen["k"], y=chosen["inertia"], mode="markers", name="Selected K",
                             marker=dict(size=14, color="rgba(0,0,0,0)", line=dict(color=WARN, width=2))))
    fig.update_layout(xaxis_title="Number of clusters (K)", yaxis_title="Inertia (within-cluster SSE)",
                      showlegend=False, xaxis=dict(dtick=1))
    return _base_layout(fig, "Elbow method", height=340)


def silhouette_figure(k_eval: pd.DataFrame, selected_k: int) -> go.Figure:
    colors = [PRIMARY if k == selected_k else "#C7D2FE" for k in k_eval["k"]]
    fig = go.Figure(go.Bar(x=k_eval["k"], y=k_eval["silhouette"], marker_color=colors,
                           text=k_eval["silhouette"].round(3), textposition="outside"))
    fig.update_layout(xaxis_title="Number of clusters (K)", yaxis_title="Silhouette score",
                      xaxis=dict(dtick=1))
    fig.update_yaxes(range=[0, max(k_eval["silhouette"].max() * 1.2, 0.1)])
    return _base_layout(fig, "Silhouette score", height=340)


def cluster_size_bar(summary: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Bar(
        x=summary["name"], y=summary["activities"],
        marker_color=[cluster_color(c) for c in summary["cluster"]],
        text=[f"{n:,} ({p:.1f}%)" for n, p in zip(summary["activities"], summary["percent_of_dataset"])],
        textposition="outside"))
    fig.update_layout(yaxis_title="Activities")
    fig.update_yaxes(range=[0, summary["activities"].max() * 1.18])
    return _base_layout(fig, "Cluster sizes", height=340)


def cluster_profile_heatmap(centers: np.ndarray, feature_cols: list[str]) -> go.Figure:
    names = [f"Cluster {i}" for i in range(len(centers))]
    limit = max(1.0, float(np.abs(centers).max()))
    fig = go.Figure(go.Heatmap(
        z=centers, x=feature_cols, y=names, zmin=-limit, zmax=limit, colorscale=DIVERGING,
        text=np.round(centers, 1), texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="mean z", thickness=12)))
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(tickangle=-40)
    return _base_layout(fig, "Cluster behaviour profile (centroid z-scores)", height=120 + 55 * len(centers))


def _hover_columns(table: pd.DataFrame) -> list[str]:
    return [c for c in HOVER_FEATURES if c in table.columns][:4]


def pca_scatter(table: pd.DataFrame, variance: np.ndarray, highlight: bool = True,
                height: int = 520) -> go.Figure:
    """PCA map coloured by cluster; potentially unusual activities get a red ring."""
    extra = _hover_columns(table)

    def hover(frame: pd.DataFrame) -> tuple[np.ndarray, str]:
        data = np.column_stack([frame["activity_id"], frame["cluster_name"], frame["final_status"],
                                frame["reconstruction_error"].round(4)] + [frame[c].round(2) for c in extra])
        template = ("<b>Activity %{customdata[0]}</b><br>%{customdata[1]}<br>%{customdata[2]}"
                    "<br>Reconstruction error: %{customdata[3]}")
        for i, col in enumerate(extra):
            template += f"<br>{col}: %{{customdata[{4 + i}]}}"
        return data, template + "<extra></extra>"

    fig = go.Figure()
    for cluster in sorted(table["cluster"].unique()):
        subset = table[table["cluster"] == cluster]
        data, template = hover(subset)
        fig.add_trace(go.Scatter(
            x=subset["pc1"], y=subset["pc2"], mode="markers", name=f"Cluster {cluster}",
            marker=dict(size=6, color=cluster_color(cluster), opacity=0.7),
            customdata=data, hovertemplate=template))
    if highlight:
        unusual = table[table["final_status"] == FINAL_UNUSUAL]
        if len(unusual):
            data, template = hover(unusual)
            fig.add_trace(go.Scatter(
                x=unusual["pc1"], y=unusual["pc2"], mode="markers", name="Potentially unusual",
                marker=dict(size=12, symbol="circle-open", color=WARN, line=dict(width=1.5)),
                customdata=data, hovertemplate=template))
    fig.update_layout(xaxis_title=f"PCA Component 1 ({variance[0] * 100:.1f}% variance)",
                      yaxis_title=f"PCA Component 2 ({variance[1] * 100:.1f}% variance)")
    return _base_layout(fig, height=height)


# ------------------------------------------------ Autoencoder / anomalies
def loss_curve(history: dict[str, list[float]], best_epoch: int) -> go.Figure:
    epochs = list(range(1, len(history["loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["loss"], name="Training loss", line=dict(color=PRIMARY, width=2)))
    fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], name="Validation loss",
                             line=dict(color=SECONDARY, width=2, dash="dash")))
    fig.add_vline(x=best_epoch, line_dash="dot", line_color=NEUTRAL,
                  annotation_text=f"best epoch {best_epoch}", annotation_position="top right")
    fig.update_layout(xaxis_title="Epoch", yaxis_title="Mean squared error")
    fig.update_yaxes(type="log")
    return _base_layout(fig, "Autoencoder training (log scale)", height=360)


def error_histogram(table: pd.DataFrame, threshold: float, percentile: float) -> go.Figure:
    """Distribution of log10 reconstruction error with the anomaly threshold."""
    fig = go.Figure()
    for split, color in (("Training", PRIMARY), ("Validation", SECONDARY)):
        values = np.log10(table.loc[table["split"] == split, "reconstruction_error"].clip(lower=1e-8))
        fig.add_trace(go.Histogram(x=values, name=f"{split} activities", histnorm="probability density",
                                   nbinsx=50, marker_color=color, opacity=0.6))
    fig.add_vline(x=np.log10(threshold), line_color=WARN, line_dash="dash",
                  annotation_text=f"threshold (P{percentile:g}) = {threshold:.3f}", annotation_position="top right")
    fig.update_layout(barmode="overlay", xaxis_title="log10(reconstruction error)", yaxis_title="Density")
    return _base_layout(fig, "Reconstruction error distribution", height=360)


def status_by_cluster(table: pd.DataFrame) -> go.Figure:
    counts = table.groupby(["cluster_name", "final_status"]).size().unstack(fill_value=0)
    for status in (FINAL_NORMAL, FINAL_UNUSUAL):
        if status not in counts:
            counts[status] = 0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=counts.index, y=counts[FINAL_NORMAL], name=FINAL_NORMAL, marker_color="#A5B4FC"))
    fig.add_trace(go.Bar(x=counts.index, y=counts[FINAL_UNUSUAL], name=FINAL_UNUSUAL, marker_color=WARN))
    fig.update_layout(barmode="stack", yaxis_title="Activities")
    return _base_layout(fig, "Behaviour status by cluster", height=360)


def signal_scatter(table: pd.DataFrame) -> go.Figure:
    """Autoencoder ratio vs cluster-distance ratio. Points right/above 1 exceed a threshold."""
    fig = go.Figure()
    for status, color, size in ((FINAL_NORMAL, "#A5B4FC", 5), (FINAL_UNUSUAL, WARN, 7)):
        subset = table[table["final_status"] == status]
        fig.add_trace(go.Scatter(
            x=subset["cluster_ratio"], y=subset["autoencoder_ratio"], mode="markers", name=status,
            marker=dict(size=size, color=color, opacity=0.7),
            customdata=np.column_stack([subset["activity_id"], subset["combined_score"].round(3)]),
            hovertemplate="Activity %{customdata[0]}<br>Cluster ratio: %{x:.2f}<br>"
                          "Autoencoder ratio: %{y:.2f}<br>Combined score: %{customdata[1]}<extra></extra>"))
    xs = np.linspace(0.02, 2.0, 100)
    fig.add_trace(go.Scatter(x=xs, y=2 - xs, mode="lines", name="Combined score = 1",
                             line=dict(color="#475569", dash="dot", width=1.5)))
    fig.add_vline(x=1, line_color=NEUTRAL, line_width=1)
    fig.add_hline(y=1, line_color=NEUTRAL, line_width=1)
    fig.update_xaxes(type="log", title="Cluster-distance ratio (distance / cluster threshold)")
    fig.update_yaxes(type="log", title="Autoencoder ratio (RMSE / threshold RMSE)")
    return _base_layout(fig, "Two independent signals", height=440)


def latent_scatter(table: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for status, color, size in ((FINAL_NORMAL, "#A5B4FC", 5), (FINAL_UNUSUAL, WARN, 7)):
        subset = table[table["final_status"] == status]
        fig.add_trace(go.Scatter(x=subset["latent_1"], y=subset["latent_2"], mode="markers", name=status,
                                 marker=dict(size=size, color=color, opacity=0.7),
                                 customdata=subset["activity_id"],
                                 hovertemplate="Activity %{customdata}<extra></extra>"))
    fig.update_layout(xaxis_title="Latent unit 1", yaxis_title="Latent unit 2")
    return _base_layout(fig, "Autoencoder latent space (2-D bottleneck)", height=380)


# --------------------------------------------------- activity investigation
def feature_comparison(explanation: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=explanation["feature"], y=explanation["z_value"], name="This activity",
                         marker_color=PRIMARY))
    fig.add_trace(go.Bar(x=explanation["feature"], y=explanation["cluster_mean"], name="Cluster average",
                         marker_color="#CBD5E1"))
    fig.update_layout(barmode="group", yaxis_title="z-score")
    fig.update_xaxes(tickangle=-40)
    return _base_layout(fig, "Feature values vs cluster average", height=360)


def error_share_bar(explanation: pd.DataFrame) -> go.Figure:
    ordered = explanation.sort_values("error_share_percent")
    fig = go.Figure(go.Bar(x=ordered["error_share_percent"], y=ordered["feature"], orientation="h",
                           marker_color=WARN, opacity=0.8))
    fig.update_layout(xaxis_title="Share of this activity's reconstruction error (%)")
    return _base_layout(fig, "Which features the autoencoder could not reconstruct", height=360)
