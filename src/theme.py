"""The dashboard's look: page style, section headings, source lines, footer and warming stripes.

The layout follows the pattern of public climate-indicator portals: a solid blue
top bar, a full-width picture under it (here the warming stripes, drawn from
this dashboard's own data rather than a photo), a line of links to every
section, then sections with a centred green title, a centred description, the
controls on the right, charts with a short source line under them, and a blue
footer.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

BRAND = "#1d4a80"        # top bar, buttons, footer
BRAND_DARK = "#163a66"
ACCENT = "#5c9c2e"       # section titles (large bold text, 3:1 or better on the page background)
PAGE_BG = "#efefed"
TEXT = "#2b2b29"
COOL, NEUTRAL, WARM = "#2a78d6", "#f0efec", "#e34948"

PAGES = [  # (url path, label) for the link line under the top bar
    ("rate", "Warming rate"),
    ("enso", "El Niño and La Niña"),
    ("acceleration", "Has it sped up?"),
    ("conclusion", "Conclusion"),
]

_CSS = f"""
<style>
header[data-testid="stHeader"] {{ background: {BRAND}; height: 4rem; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] {{ background: transparent !important; border-radius: 0; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"] * {{ color: #ffffff !important; text-transform: uppercase;
  letter-spacing: 0.06em; font-weight: 600; font-size: 0.82rem; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"][aria-current="page"] {{
  box-shadow: inset 0 -3px 0 #ffffff; background: rgba(255,255,255,0.12) !important; }}
header[data-testid="stHeader"] a[data-testid="stTopNavLink"]:hover {{ background: rgba(255,255,255,0.10) !important; }}
header[data-testid="stHeader"] button, header[data-testid="stHeader"] button * {{ color: #ffffff !important; }}
header[data-testid="stHeader"] svg {{ fill: #ffffff; color: #ffffff; }}
.stMainBlockContainer {{ padding-top: 5.2rem; }}
div.stElementContainer:has(> div[data-testid="stDownloadButton"]) {{ align-self: flex-end; }}
div[data-testid="stDownloadButton"] button {{ background: {BRAND}; color: #ffffff; border: none; border-radius: 2px;
  text-decoration: underline; padding: 0.45rem 1rem; }}
div[data-testid="stDownloadButton"] button:hover {{ background: {BRAND_DARK}; color: #ffffff; }}
div[data-testid="stMetric"] {{ background: #ffffff; padding: 0.8rem 1.1rem; border-left: 4px solid {ACCENT}; }}
div[class*="st-key-panel_"] {{ background: #ffffff; padding: 1.3rem 1.5rem 0.5rem; margin: 0.3rem 0 0.6rem; }}
div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {{ height: 100%; }}
div[data-testid="stColumn"] div[data-testid="stLayoutWrapper"]:has(> div[class*="st-key-panel_"]) {{ flex: 1 1 auto; }}
div[data-testid="stColumn"] div[class*="st-key-panel_"] {{ height: 100%; }}
.stMainBlockContainer {{ padding-bottom: 0; }}
.stMainBlockContainer h3, .stMainBlockContainer h4 {{ color: {BRAND}; }}
.gw-links {{ text-align: center; font-size: 0.98rem; color: {TEXT}; margin: 0.2rem 0 1.4rem; line-height: 1.9; }}
.gw-links a {{ color: {BRAND}; text-decoration: underline; }}
.gw-links span.gw-sep {{ color: #8a8a86; margin: 0 0.45rem; }}
.gw-links a.gw-here {{ font-weight: 700; text-decoration: none; color: {TEXT}; }}
.gw-kicker {{ text-align: center; color: #6b6b67; letter-spacing: 0.14em; font-size: 0.78rem;
  text-transform: uppercase; margin-top: 1.2rem; }}
h2.gw-title {{ color: {ACCENT}; text-align: center; font-weight: 700; font-size: 2.15rem; line-height: 1.25;
  margin: 0.35rem auto 0.7rem; max-width: 1000px; padding: 0;
  text-wrap: balance; }}
p.gw-lead {{ text-align: center; max-width: 960px; margin: 0 auto 0.9rem; color: {TEXT}; font-size: 1.04rem;
  line-height: 1.65; }}
p.gw-meta {{ text-align: center; color: #6b6b67; font-size: 0.86rem; margin: 0 auto 0.1rem; max-width: 960px;
  text-wrap: balance; }}
.gw-metas {{ margin-bottom: 1.3rem; }}
p.gw-source {{ text-align: right; color: {TEXT}; font-size: 0.86rem; margin: -0.3rem 0 1rem; }}
.gw-footer {{ background: {BRAND}; color: #ffffff; text-align: center; padding: 1.6rem 1rem 2rem;
  margin: 2.2rem calc(50% - 50vw) 0; font-size: 0.95rem; line-height: 2; }}
.gw-footer a {{ color: #ffffff; text-decoration: none; margin: 0 1rem; white-space: nowrap; }}
.gw-hero {{ position: relative; overflow: hidden; min-height: 360px; margin: -2.3rem calc(50% - 50vw) 0.4rem; }}
.gw-hero svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
.gw-hero .gw-shade {{ position: absolute; inset: 0;
  background: linear-gradient(90deg, rgba(12,24,40,0.82) 0%, rgba(12,24,40,0.55) 55%, rgba(12,24,40,0.05) 100%); }}
.gw-hero .gw-hero-text {{ position: relative; padding: 4.2rem 3rem 3.2rem; max-width: 760px; }}
.gw-hero h1 {{ color: #ffffff; font-size: 2.7rem; line-height: 1.12; margin: 0 0 1rem; padding: 0;
  text-shadow: 0 1px 3px rgba(0,0,0,0.35); }}
.gw-hero p {{ color: #ffffff; font-size: 1.08rem; line-height: 1.55; margin: 0 0 0.8rem; font-weight: 600;
  text-shadow: 0 1px 2px rgba(0,0,0,0.4); }}
.gw-hero p.gw-for {{ font-weight: 400; font-size: 0.95rem; opacity: 0.95; }}
@media (max-width: 760px) {{
  .gw-hero .gw-hero-text {{ padding: 2.4rem 1.3rem 2rem; }}
  .gw-hero h1 {{ font-size: 2rem; }}
  h2.gw-title {{ font-size: 1.6rem; }}
  div[class*="st-key-panel_"] {{ padding: 1rem 0.7rem 0.4rem; }}
}}
</style>
"""


def apply_style() -> None:
    """Page-wide style; called once from app.py so it applies to every page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def link_line(current: str | None = None) -> None:
    """'Global warming data: page | page | page' links under the top bar."""
    links = []
    for path, label in PAGES:
        cls = ' class="gw-here"' if path == current else ""
        links.append(f'<a href="{path}" target="_self"{cls}>{html.escape(label)}</a>')
    st.markdown(f'<div class="gw-links">Global warming data: {"<span class=gw-sep>|</span>".join(links)}</div>',
                unsafe_allow_html=True)


def section_heading(title: str, lead: list[str] | str = (), kicker: str | None = None, meta: list[str] = ()) -> None:
    """Centred green title (a real <h2>), centred description paragraphs and a small meta line."""
    paragraphs = [lead] if isinstance(lead, str) else list(lead)
    parts = [f'<div class="gw-kicker">{html.escape(kicker)}</div>'] if kicker else []
    parts.append(f'<h2 class="gw-title">{html.escape(title)}</h2>')
    parts += [f'<p class="gw-lead">{html.escape(p)}</p>' for p in paragraphs]
    st.markdown("".join(parts), unsafe_allow_html=True)
    if meta:  # unit and latest observation, one small line each
        lines = "".join(f'<p class="gw-meta">{html.escape(line)}</p>' for line in meta)
        st.markdown(f'<div class="gw-metas">{lines}</div>', unsafe_allow_html=True)


def source_line(text: str) -> None:
    """Short source note, right-aligned under a chart."""
    st.markdown(f'<p class="gw-source">{html.escape(text)}</p>', unsafe_allow_html=True)


def footer() -> None:
    links = '<a href="./" target="_self">Overview</a>' + "".join(
        f'<a href="{path}" target="_self">{html.escape(label)}</a>' for path, label in PAGES)
    st.markdown(f'<div class="gw-footer">{links}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------- warming stripes

def _mix(a: str, b: str, t: float) -> str:
    """Blend two hex colours (t = 0 gives a, t = 1 gives b)."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def stripe_colour(value: float, limit: float = 1.0) -> str:
    """Diverging blue - neutral - red, centred on zero."""
    t = float(np.clip(value / limit, -1, 1))
    return _mix(NEUTRAL, WARM, t) if t >= 0 else _mix(NEUTRAL, COOL, -t)


def annual_means(data: pd.DataFrame, record_id: str = "noaa") -> pd.DataFrame:
    """Complete calendar years only, for one record."""
    part = data.loc[data["record_id"] == record_id]
    counts = part.groupby("year")["anomaly_c"].agg(["mean", "size"])
    return counts.loc[counts["size"] == 12, ["mean"]].reset_index()


def stripes_centre(years: pd.DataFrame) -> tuple[float, str]:
    """Stripes are centred on the 20th-century average (or on the years before 2001 that the data hold)."""
    century = years.loc[years["year"].between(1901, 2000)]
    first = int(century["year"].min()) if not century.empty else int(years["year"].min())
    return float(century["mean"].mean() if not century.empty else years["mean"].mean()), f"{first}–2000"


def stripes_svg(data: pd.DataFrame) -> str:
    """One vertical stripe per complete year of the NOAA record, filling its container."""
    years = annual_means(data)
    if years.empty:
        return ""
    centre, _ = stripes_centre(years)
    width = 1000 / len(years)
    rects = "".join(
        f'<rect x="{i * width:.2f}" y="0" width="{width + 0.6:.2f}" height="100" '
        f'fill="{stripe_colour(v - centre, 0.8)}"/>'
        for i, v in enumerate(years["mean"])
    )
    return f'<svg viewBox="0 0 1000 100" preserveAspectRatio="none" aria-hidden="true">{rects}</svg>'
