"""Exploratory data analysis page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from utils import ui, visualization as viz
from utils.preprocessing import NIGHT_COLUMN

BEHAVIOR_FEATURES = ["duration_sec", "src_bytes", "dst_bytes", "packets", "connections",
                     "request_frequency", "unique_destinations"]


def _feature_statistics(frame: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    stats = frame[cols].agg(["mean", "median", "std", "min", "max"]).T
    stats.index.name = "feature"
    return stats.reset_index()


def render(ctx) -> None:
    ui.hero("Exploratory Analysis", "Understand the data before modelling",
            "Data-quality checks, feature statistics and the main behavioral patterns in the dataset. "
            "All values are z-scores (standardized), so 0 is the dataset average.")

    prepared, report = ctx.prep, ctx.report
    frame, cols = prepared.frame, prepared.feature_cols

    ui.kpi_row([
        ("Records", f"{report.n_rows:,}", "rows in the CSV", "primary"),
        ("Model features", f"{len(cols)}", "activity_id excluded", "primary"),
        ("Missing values", f"{report.total_missing}", "across all columns",
         "primary" if report.total_missing == 0 else "warn"),
        ("Duplicate records", f"{report.duplicate_rows}", "fully identical rows",
         "primary" if report.duplicate_rows == 0 else "warn"),
    ])
    st.markdown("&nbsp;", unsafe_allow_html=True)

    tab_quality, tab_stats, tab_dist, tab_corr, tab_behavior = st.tabs(
        ["Data quality", "Feature statistics", "Distributions", "Correlations", "Network behavior"])

    with tab_quality:
        summary = report.summary_table()
        summary["Passed"] = summary["Passed"].map({True: "Pass", False: "Review"})
        ui.show_dataframe(summary, hide_index=True)
        if report.issues:
            ui.note("Findings: " + " ".join(report.issues), caution=True)
        else:
            ui.note("No data-quality problems were found.")
        st.markdown("**Pre-processing decisions**")
        for message in prepared.notes:
            st.markdown(f"- {message}")
        st.markdown("**Standardization check** (a standardized column has mean ~ 0 and std ~ 1)")
        check = prepared.standardization.copy()
        check["standardized"] = check["standardized"].map({True: "Yes", False: "No"})
        check.index.name = "feature"
        ui.show_dataframe(check.reset_index(), hide_index=True, column_config={
            "mean": st.column_config.NumberColumn(format="%.6f"),
            "std": st.column_config.NumberColumn(format="%.4f")})
        with st.expander("Column data types"):
            ui.show_dataframe(report.dtypes.rename("dtype").reset_index().rename(columns={"index": "column"}),
                              hide_index=True)

    with tab_stats:
        ui.show_dataframe(_feature_statistics(frame, cols), hide_index=True, column_config={
            c: st.column_config.NumberColumn(format="%.3f") for c in ["mean", "median", "std", "min", "max"]})
        st.caption("Because the data is standardized, the mean is ~0 and std ~1 by construction. "
                   "The median, min and max reveal skew and extreme values: for example, a median below the "
                   "mean together with a large maximum indicates a long right tail.")

    with tab_dist:
        feature = st.selectbox("Feature", cols, index=cols.index("src_bytes") if "src_bytes" in cols else 0)
        ui.show_plotly(viz.feature_histogram(frame[feature], feature), key="eda_hist")
        with st.expander("Show all feature distributions"):
            ui.show_plotly(viz.all_feature_histograms(frame, cols), key="eda_hist_all")

    with tab_corr:
        ui.show_plotly(viz.correlation_heatmap(frame[cols].corr()), key="eda_corr")
        st.caption("Strongly correlated features carry overlapping information. "
                   "Protocol columns are mutually exclusive one-hot indicators, so they are negatively related.")

    with tab_behavior:
        left, right = st.columns(2)
        with left:
            ui.show_plotly(viz.protocol_distribution(prepared.protocol.value_counts()), key="eda_protocol")
        with right:
            behavior = [c for c in BEHAVIOR_FEATURES if c in cols]
            means = frame[behavior].groupby(prepared.protocol.to_numpy()).mean()
            ui.show_plotly(viz.protocol_behavior(means), key="eda_protocol_behavior")
        if prepared.night_flag is not None:
            night_share = prepared.night_flag.mean() * 100
            st.markdown(f"**Night activity:** {int(prepared.night_flag.sum()):,} activities "
                        f"({night_share:.1f}%) are flagged as night-time activity "
                        f"(decoded from the standardized `{NIGHT_COLUMN}` column).")
        st.caption("Protocol labels are decoded from the one-hot columns for display only; "
                   "the original encoded columns are what the models use.")
