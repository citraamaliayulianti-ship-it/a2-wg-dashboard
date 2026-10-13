"""Shared page banners, so every page opens in the same visual language.

Each banner uses a shade from one dark slate ramp, carries a thin band of
"warming stripes" along its foot (one stripe per year, coloured by that year's
anomaly) and a faint backdrop drawn from the dashboard's own data: a miniature
of that page's main chart.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

# One slate ramp; darkest for the overview and the conclusion (the bookends).
BANNER = {
    "overview": "#1d2833",
    "rate": "#243444",
    "enso": "#2a3d50",
    "acceleration": "#30465b",
    "conclusion": "#1d2833",
}
COOL, NEUTRAL, WARM = "#2a78d6", "#f0efec", "#e34948"
_LINE = "#cfe0f5"
_W, _H = 520, 220
_STYLE = ("<style>.gw-banner-text{max-width:min(720px,60%)}.gw-banner-art{opacity:0.85}"
          "@media (max-width:760px){.gw-banner-text{max-width:100%}.gw-banner-art{opacity:0.3}}</style>")


def _mix(a: str, b: str, t: float) -> str:
    """Blend two hex colours (t = 0 gives a, t = 1 gives b)."""
    ca = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    cb = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(ca, cb))


def stripe_colour(value: float, limit: float = 1.0) -> str:
    """Diverging blue - neutral - red, centred on the 1991-2020 average (0 °C)."""
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


def stripes_svg(data: pd.DataFrame, height: int = 10, style: str = "") -> str:
    years = annual_means(data)
    if years.empty:
        return ""
    centre, _ = stripes_centre(years)
    width = 1000 / len(years)
    rects = "".join(
        f'<rect x="{i * width:.2f}" y="0" width="{width + 0.6:.2f}" height="{height}" '
        f'fill="{stripe_colour(v - centre, 0.8)}"/>'
        for i, v in enumerate(years["mean"])
    )
    return (f'<svg viewBox="0 0 1000 {height}" preserveAspectRatio="none" aria-hidden="true" '
            f'style="display:block;width:100%;height:{height}px;{style}">{rects}</svg>')


def _scale(values, low, high, out_low, out_high):
    return out_low + (np.clip(values, low, high) - low) / (high - low) * (out_high - out_low)


def _svg(body: str) -> str:
    return (f'<svg class="gw-banner-art" viewBox="0 0 {_W} {_H}" aria-hidden="true" style="position:absolute;'
            f'right:-6px;bottom:4px;width:min(38%,460px);height:auto">{body}</svg>')


def pattern_rate(data: pd.DataFrame) -> str:
    """12-month running mean of one record since 1950, with a straight line from 1970."""
    part = data.loc[(data["record_id"] == "noaa") & (data["year"] >= 1950)].sort_values("date")
    smooth = part["anomaly_c"].rolling(12).mean().to_numpy()
    xs = np.linspace(16, _W - 16, len(part))
    ys = _scale(smooth, -1.0, 1.0, _H - 16, 16)
    points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys) if np.isfinite(y))
    late = part["year"].to_numpy() >= 1970
    fit = np.polyfit(np.arange(len(part))[late], part["anomaly_c"].to_numpy()[late], 1)
    i0, i1 = np.where(late)[0][0], len(part) - 1
    y0, y1 = _scale(np.polyval(fit, [i0, i1]), -1.0, 1.0, _H - 16, 16)
    return _svg(f'<polyline points="{points}" fill="none" stroke="{_LINE}" stroke-opacity="0.55" stroke-width="1.6"/>'
                f'<line x1="{xs[i0]:.1f}" y1="{y0:.1f}" x2="{xs[i1]:.1f}" y2="{y1:.1f}" stroke="#ffffff" '
                f'stroke-opacity="0.6" stroke-width="2.2"/>')


def pattern_enso(data: pd.DataFrame) -> str:
    """The Oceanic Niño Index as thin bars: warm above the line, cool below."""
    oni = data.loc[(data["record_id"] == "noaa") & data["oni"].notna()].sort_values("date")
    oni = oni.loc[oni["year"] >= 1970]
    xs = np.linspace(16, _W - 16, len(oni))
    mid = _H / 2
    bars = []
    for x, v in zip(xs, oni["oni"]):
        h = abs(v) / 2.6 * (_H / 2 - 16)
        colour = "#f4a3a2" if v > 0 else "#9ec5f4"
        bars.append(f'<rect x="{x:.1f}" y="{mid - h if v > 0 else mid:.1f}" width="1.1" height="{h:.1f}" fill="{colour}" fill-opacity="0.6"/>')
    return _svg(f'<line x1="16" y1="{mid}" x2="{_W - 16}" y2="{mid}" stroke="#ffffff" stroke-opacity="0.45"/>' + "".join(bars))


def pattern_acceleration(data: pd.DataFrame) -> str:
    """Annual means since 1970 with separate straight lines before and after 2010."""
    years = annual_means(data)
    years = years.loc[years["year"] >= 1970]
    xs = _scale(years["year"].to_numpy(float), 1970, years["year"].max(), 16, _W - 16)
    ys = _scale(years["mean"].to_numpy(), -0.7, 0.8, _H - 16, 16)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.2" fill="{_LINE}" fill-opacity="0.5"/>' for x, y in zip(xs, ys))
    lines = ""
    for low, high in ((1970, 2009), (2010, int(years["year"].max()))):
        part = years.loc[years["year"].between(low, high)]
        if len(part) < 3:
            continue
        fit = np.polyfit(part["year"], part["mean"], 1)
        x0, x1 = _scale(np.array([low, high], float), 1970, years["year"].max(), 16, _W - 16)
        y0, y1 = _scale(np.polyval(fit, [low, high]), -0.7, 0.8, _H - 16, 16)
        lines += (f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="#ffffff" '
                  f'stroke-opacity="0.65" stroke-width="2.2"/>')
    return _svg(dots + lines)


def banner(kicker: str, title: str, subtitle: str, colour: str, data: pd.DataFrame, pattern: str = "") -> None:
    """A coloured page banner with a stripes band; the title is a real heading for screen readers."""
    st.markdown(
        f'{_STYLE}<div style="position:relative;overflow:hidden;border-radius:14px;background:{colour};'
        f'padding:28px 32px 36px;margin:2px 0 14px">{pattern}'
        f'<div class="gw-banner-text" style="position:relative">'
        f'<div style="font-size:12.5px;letter-spacing:0.08em;text-transform:uppercase;color:#c9d6e3">'
        f'{html.escape(kicker)}</div>'
        f'<h2 style="color:#ffffff;font-size:2rem;line-height:1.2;margin:8px 0 10px;padding:0">{html.escape(title)}</h2>'
        f'<p style="color:#e4ebf2;font-size:1.02rem;line-height:1.5;margin:0">{html.escape(subtitle)}</p>'
        f'</div>{stripes_svg(data, 10, "position:absolute;left:0;right:0;bottom:0")}</div>',
        unsafe_allow_html=True,
    )
