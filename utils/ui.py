"""Reusable Streamlit UI building blocks (theme, cards, helpers)."""
from __future__ import annotations

import html
import inspect

import pandas as pd
import streamlit as st

PRIMARY = "#4F46E5"
WARN = "#DC2626"
WARN_SOFT = "#FEF2F2"
TEXT = "#0F172A"
MUTED = "#64748B"
BORDER = "#E2E8F0"

_CSS = f"""
<style>
:root {{
  --ns-primary: {PRIMARY};
  --ns-text: {TEXT};
  --ns-muted: {MUTED};
  --ns-border: {BORDER};
  --ns-warn: {WARN};
}}
.stApp {{ background: #F8FAFC; }}
.block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1280px; }}
html, body, [class*="css"] {{ color: var(--ns-text); }}
h1, h2, h3, h4 {{ letter-spacing: -0.01em; color: var(--ns-text); }}
h2 {{ font-size: 1.35rem !important; margin-top: 1.6rem !important; }}
h3 {{ font-size: 1.1rem !important; }}
footer {{ visibility: hidden; }}

section[data-testid="stSidebar"] {{ background: #FFFFFF; border-right: 1px solid var(--ns-border); }}
.ns-brand {{ display:flex; align-items:center; gap:.65rem; margin-bottom:.2rem; }}
.ns-logo {{ width:34px; height:34px; border-radius:9px; background:var(--ns-primary); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; font-size:1.05rem; }}
.ns-brand-name {{ font-weight:700; font-size:1.15rem; line-height:1.1; }}
.ns-brand-sub {{ color:var(--ns-muted); font-size:.75rem; }}
.ns-side-label {{ text-transform:uppercase; letter-spacing:.06em; font-size:.68rem; font-weight:600;
  color:var(--ns-muted); margin:1.1rem 0 .3rem 0; }}

.ns-hero {{ background:#fff; border:1px solid var(--ns-border); border-left:4px solid var(--ns-primary);
  border-radius:12px; padding:1.3rem 1.5rem; margin-bottom:1.1rem; }}
.ns-hero h1 {{ margin:0; font-size:1.75rem; }}
.ns-hero .ns-tag {{ color:var(--ns-primary); font-weight:600; font-size:.9rem; margin-top:.15rem; }}
.ns-hero p {{ color:var(--ns-muted); margin:.55rem 0 0 0; max-width:62rem; font-size:.95rem; line-height:1.55; }}

.ns-card {{ background:#fff; border:1px solid var(--ns-border); border-radius:12px; padding:1rem 1.2rem;
  box-shadow:0 1px 2px rgba(15,23,42,.04); height:100%; }}
.ns-kpi-label {{ color:var(--ns-muted); font-size:.78rem; font-weight:600; text-transform:uppercase; letter-spacing:.04em; }}
.ns-kpi-value {{ font-size:1.9rem; font-weight:700; line-height:1.25; margin-top:.15rem; }}
.ns-kpi-sub {{ color:var(--ns-muted); font-size:.8rem; margin-top:.1rem; }}
.ns-kpi-warn .ns-kpi-value {{ color:var(--ns-warn); }}
.ns-kpi-warn {{ border-top:3px solid var(--ns-warn); }}
.ns-kpi-primary {{ border-top:3px solid var(--ns-primary); }}

.ns-steps {{ display:flex; gap:.7rem; margin:.2rem 0 1.1rem 0; flex-wrap:wrap; }}
.ns-step {{ flex:1 1 170px; background:#fff; border:1px solid var(--ns-border); border-radius:12px;
  padding:.8rem .95rem; position:relative; }}
.ns-step-num {{ display:inline-flex; width:22px; height:22px; border-radius:50%; background:#EEF2FF;
  color:var(--ns-primary); font-size:.75rem; font-weight:700; align-items:center; justify-content:center; }}
.ns-step-title {{ font-weight:700; margin-left:.45rem; font-size:.95rem; }}
.ns-step-text {{ color:var(--ns-muted); font-size:.8rem; margin-top:.4rem; line-height:1.45; }}

.ns-note {{ background:#EEF2FF; border:1px solid #C7D2FE; border-radius:10px; padding:.75rem 1rem;
  color:#312E81; font-size:.88rem; line-height:1.5; margin:.4rem 0 .8rem 0; }}
.ns-caution {{ background:{WARN_SOFT}; border:1px solid #FECACA; border-radius:10px; padding:.75rem 1rem;
  color:#7F1D1D; font-size:.88rem; line-height:1.5; margin:.4rem 0 .8rem 0; }}
.ns-pill {{ display:inline-block; padding:.12rem .55rem; border-radius:999px; font-size:.75rem; font-weight:600;
  border:1px solid var(--ns-border); background:#F1F5F9; color:#334155; margin-right:.35rem; }}
.ns-pill-ok {{ background:#ECFDF5; border-color:#A7F3D0; color:#065F46; }}
.ns-pill-warn {{ background:{WARN_SOFT}; border-color:#FECACA; color:#991B1B; }}
.ns-arch {{ display:flex; align-items:flex-end; gap:.35rem; flex-wrap:wrap; margin:.4rem 0 .2rem 0; }}
.ns-arch-node {{ text-align:center; }}
.ns-arch-box {{ background:#EEF2FF; border:1px solid #C7D2FE; color:#312E81; border-radius:8px;
  padding:.35rem .7rem; font-weight:700; font-size:.9rem; }}
.ns-arch-box-latent {{ background:var(--ns-primary); border-color:var(--ns-primary); color:#fff; }}
.ns-arch-label {{ color:var(--ns-muted); font-size:.68rem; margin-top:.15rem; }}
.ns-arch-arrow {{ color:#94A3B8; padding-bottom:1.1rem; }}

div[data-testid="stTabs"] button {{ font-weight:600; }}
div[data-testid="stDataFrame"] {{ border:1px solid var(--ns-border); border-radius:10px; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------
# HTML components (all dynamic text is escaped)
# --------------------------------------------------------------------------
def hero(title: str, tagline: str, description: str) -> None:
    st.markdown(
        f'<div class="ns-hero"><h1>{html.escape(title)}</h1>'
        f'<div class="ns-tag">{html.escape(tagline)}</div><p>{html.escape(description)}</p></div>',
        unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", tone: str = "primary") -> str:
    return (f'<div class="ns-card ns-kpi-{tone}"><div class="ns-kpi-label">{html.escape(label)}</div>'
            f'<div class="ns-kpi-value">{html.escape(value)}</div>'
            f'<div class="ns-kpi-sub">{html.escape(sub)}</div></div>')


def kpi_row(cards: list[tuple[str, str, str, str]]) -> None:
    """Render a row of KPI cards: (label, value, sub, tone)."""
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards):
        column.markdown(kpi_card(*card), unsafe_allow_html=True)


def workflow_steps(steps: list[tuple[str, str]]) -> None:
    blocks = "".join(
        f'<div class="ns-step"><span class="ns-step-num">{i}</span>'
        f'<span class="ns-step-title">{html.escape(title)}</span>'
        f'<div class="ns-step-text">{html.escape(text)}</div></div>'
        for i, (title, text) in enumerate(steps, start=1))
    st.markdown(f'<div class="ns-steps">{blocks}</div>', unsafe_allow_html=True)


def note(text: str, caution: bool = False) -> None:
    css = "ns-caution" if caution else "ns-note"
    st.markdown(f'<div class="{css}">{html.escape(text)}</div>', unsafe_allow_html=True)


def pills(items: list[tuple[str, bool]]) -> None:
    """Small status chips: (text, ok)."""
    chips = "".join(
        f'<span class="ns-pill {"ns-pill-ok" if ok else "ns-pill-warn"}">{html.escape(text)}</span>'
        for text, ok in items)
    st.markdown(chips, unsafe_allow_html=True)


def architecture_diagram(layer_sizes: list[int]) -> None:
    """Simple HTML diagram of the autoencoder layers."""
    middle = len(layer_sizes) // 2
    nodes = []
    for i, size in enumerate(layer_sizes):
        if i == 0:
            label = "input"
        elif i == len(layer_sizes) - 1:
            label = "output"
        elif i == middle:
            label = "latent"
        else:
            label = "encoder" if i < middle else "decoder"
        box_class = "ns-arch-box ns-arch-box-latent" if i == middle else "ns-arch-box"
        nodes.append(f'<div class="ns-arch-node"><div class="{box_class}">{size}</div>'
                     f'<div class="ns-arch-label">{label}</div></div>')
    st.markdown('<div class="ns-arch">' + '<span class="ns-arch-arrow">&rarr;</span>'.join(nodes) + '</div>',
                unsafe_allow_html=True)


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f"## {title}")
    if caption:
        st.caption(caption)


# --------------------------------------------------------------------------
# Version-tolerant Streamlit wrappers
# --------------------------------------------------------------------------
def _supports_width(func) -> bool:
    return "width" in inspect.signature(func).parameters


def stretch_kwargs(func) -> dict:
    """Keyword argument that makes a widget full-width on both old and new Streamlit."""
    return {"width": "stretch"} if _supports_width(func) else {"use_container_width": True}


def show_plotly(fig, key: str | None = None) -> None:
    """Full-width Plotly chart that works across Streamlit versions."""
    if _supports_width(st.plotly_chart):
        st.plotly_chart(fig, width="stretch", key=key)
    else:
        st.plotly_chart(fig, use_container_width=True, key=key)


def show_dataframe(data: pd.DataFrame | pd.io.formats.style.Styler, **kwargs):
    """Full-width dataframe that works across Streamlit versions."""
    if _supports_width(st.dataframe):
        return st.dataframe(data, width="stretch", **kwargs)
    return st.dataframe(data, use_container_width=True, **kwargs)
