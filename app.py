"""NeuroSentinel - Unsupervised Behavioral Pattern Discovery and Neural Anomaly Detection.

Run with:  streamlit run app.py
"""
import os

# Must be set before TensorFlow is imported (it is imported by the pipeline).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import streamlit as st

st.set_page_config(page_title="NeuroSentinel", page_icon=":shield:", layout="wide",
                   initial_sidebar_state="expanded")

from pipeline import DATA_PATH, SEED, Settings, build_context  # noqa: E402
from utils import ui  # noqa: E402
from views import about, clusters, eda, explorer, anomaly, overview  # noqa: E402

PAGES = {
    "Overview": overview,
    "Exploratory Analysis": eda,
    "Behavioral Clusters": clusters,
    "Neural Anomaly Detection": anomaly,
    "Anomaly Explorer": explorer,
    "About the Project": about,
}
DEFAULT_SETTINGS = {"k_choice": "Auto (silhouette)", "epochs": 100, "batch_size": 32, "percentile": 95.0}
K_CHOICES = ["Auto (silhouette)"] + list(range(2, 11))
BATCH_SIZES = [16, 32, 64, 128]


def render_sidebar() -> tuple[str, Settings]:
    """Navigation + model controls. Controls sit in a form so the (slow) autoencoder
    is only retrained when the user presses 'Apply settings'."""
    if "settings" not in st.session_state:
        st.session_state["settings"] = DEFAULT_SETTINGS.copy()
    current = st.session_state["settings"]

    with st.sidebar:
        st.markdown('<div class="ns-brand"><div class="ns-logo">N</div><div>'
                    '<div class="ns-brand-name">NeuroSentinel</div>'
                    '<div class="ns-brand-sub">Behavioral Intelligence Platform</div></div></div>',
                    unsafe_allow_html=True)
        st.markdown('<div class="ns-side-label">Navigation</div>', unsafe_allow_html=True)
        page = st.radio("Navigation", list(PAGES), label_visibility="collapsed", key="page")

        st.markdown('<div class="ns-side-label">Model controls</div>', unsafe_allow_html=True)
        with st.form("model_controls", border=False):
            k_choice = st.selectbox("Number of clusters (K)", K_CHOICES,
                                    index=K_CHOICES.index(current["k_choice"]),
                                    help="Auto picks K from silhouette scores. Choose a number to override.")
            epochs = st.slider("Autoencoder epochs (max)", 20, 300, current["epochs"], step=10,
                               help="Early stopping may end training sooner.")
            batch_size = st.select_slider("Batch size", BATCH_SIZES, value=current["batch_size"])
            percentile = st.slider("Anomaly percentile threshold", 90.0, 99.5, float(current["percentile"]),
                                   step=0.5, help="Higher = fewer, more extreme activities flagged.")
            submitted = st.form_submit_button("Apply settings", **ui.stretch_kwargs(st.form_submit_button))
        st.caption("Changing epochs or batch size retrains the autoencoder (about 30 s). "
                   "K and the percentile update instantly.")
        if submitted:
            st.session_state["settings"] = {"k_choice": k_choice, "epochs": epochs,
                                            "batch_size": batch_size, "percentile": percentile}
            current = st.session_state["settings"]

        st.markdown('<div class="ns-side-label">Dataset</div>', unsafe_allow_html=True)
        st.caption(f"{DATA_PATH.name}  \nSeed: {SEED}")

    k = None if current["k_choice"] == K_CHOICES[0] else int(current["k_choice"])
    return page, Settings(k=k, epochs=int(current["epochs"]), batch_size=int(current["batch_size"]),
                          percentile=float(current["percentile"]))


def main() -> None:
    ui.inject_css()
    page, settings = render_sidebar()
    try:
        with st.spinner("Preparing data and training models (cached after the first run)..."):
            context = build_context(settings)
    except (FileNotFoundError, ValueError) as error:
        st.error(f"NeuroSentinel could not start: {error}")
        st.stop()
    PAGES[page].render(context)


main()
