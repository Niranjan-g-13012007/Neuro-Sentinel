"""About the project page."""
from __future__ import annotations

import streamlit as st

from utils import ui


def render(ctx) -> None:
    ui.hero("About NeuroSentinel", "Unsupervised Behavioral Pattern Discovery and Neural Anomaly Detection System",
            "A college machine-learning project that combines clustering and a neural network to highlight "
            "unusual network behavior without labeled training data.")

    left, right = st.columns(2)
    with left:
        with st.container(border=True):
            st.markdown("### Problem")
            st.markdown("Network systems generate huge amounts of behavioral data, and almost none of it is "
                        "labeled. Manually marking every activity as normal or anomalous is slow and expensive, "
                        "and labels go stale as behavior changes.")
    with right:
        with st.container(border=True):
            st.markdown("### Solution")
            st.markdown("NeuroSentinel discovers behavioral groups with unsupervised learning (K-Means) and "
                        "detects unusual patterns with a neural-network autoencoder. The two views are combined "
                        "into one interpretable score that helps analysts decide what to look at first.")

    ui.section("Workflow")
    ui.workflow_steps([
        ("Observe", "NeuroSentinel receives network activity data."),
        ("Discover", "K-Means identifies groups of similar behavioral patterns."),
        ("Learn", "The autoencoder learns how typical activity can be represented and reconstructed."),
        ("Detect", "Activities with unusually high reconstruction error are flagged for investigation."),
        ("Investigate", "Analysts inspect individual activities and understand why they were flagged."),
    ])

    left, right = st.columns(2)
    with left:
        ui.section("Technologies")
        st.markdown("- Python\n- Streamlit\n- Scikit-learn\n- TensorFlow / Keras\n- Pandas and NumPy\n- Plotly")
    with right:
        ui.section("ML techniques")
        st.markdown("- K-Means clustering (elbow + silhouette)\n- PCA (visualization)\n"
                    "- Autoencoder neural network\n- Reconstruction error\n- Percentile-based anomaly thresholds")

    ui.section("Honest limitations")
    st.markdown(
        "- **No ground truth.** The dataset has no attack labels, so accuracy, precision or recall cannot be "
        "computed and none are claimed.\n"
        "- **Anomaly is not attack.** Unusual behavior can be benign (a backup job, a software update). "
        "Flags mean 'requires investigation'.\n"
        "- **Clustering is descriptive.** A cluster is a group of similar activities, not a verdict on intent.\n"
        "- **Percentile thresholds** flag a roughly fixed share of activities by design; they prioritize review "
        "rather than measure detection performance.\n"
        "- **Standardized inputs.** Feature values are z-scores, so original units (bytes, seconds) are not "
        "recoverable from this dataset.\n"
        "- **Training data may contain anomalies.** The autoencoder is trained on all available activity, "
        "including any unusual rows, which can make it slightly more tolerant of them.")
    st.caption(f"Dataset in use: {ctx.report.n_rows:,} activities, {len(ctx.prep.feature_cols)} model features.")
