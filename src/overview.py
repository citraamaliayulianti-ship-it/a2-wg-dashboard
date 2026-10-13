"""Building blocks of the overview page: hero, introduction, current position,
about the data, question cards and data access.

The card pictures are miniature charts drawn from the dashboard's own data, and
the stripes in the hero are one stripe per year of the NOAA record.
"""

from __future__ import annotations

import html

import numpy as np
import pandas as pd
import streamlit as st

from src.data import ONI, RECORDS
from src.stories import (PAGE_ACCEL, PAGE_CONCLUSION, PAGE_ENSO, PAGE_RATE, above_preindustrial, latest_year,
                         rates_from)
from src.theme import annual_means, stripes_centre, stripes_svg
from src.views import COOL, FIRST_RATE_YEAR, INK, RECORD_COLOURS, RECORD_ORDER, SOURCES, WARM

PUBLISHERS = {record_id: SOURCES[i][0] for i, record_id in enumerate(RECORDS)}


# ---------------------------------------------------------------- hero banner

def hero(data: pd.DataFrame, title: str, question: str, audience: str) -> None:
    """Opening banner: what the dashboard asks, for whom, and where to start."""
    first = int(annual_means(data)["year"].min())
    kicker = f"Global surface temperature · {first}–{latest_year(data)} · four records"
    button = ("display:inline-block;padding:10px 18px;border-radius:8px;font-weight:600;font-size:15px;"
              "text-decoration:none;margin:0 10px 10px 0;")
    st.markdown(
        f'<div style="position:relative;overflow:hidden;border-radius:16px;background:#1d2833;'
        f'padding:44px 40px 64px;margin-bottom:8px">'
        f'<div style="position:relative;max-width:760px">'
        f'<div style="font-size:13px;letter-spacing:0.08em;text-transform:uppercase;color:#c9d6e3">'
        f'{html.escape(kicker)}</div>'
        f'<h1 style="color:#ffffff;font-size:2.6rem;line-height:1.15;margin:10px 0 14px;padding:0">'
        f'{html.escape(title)}</h1>'
        f'<p style="color:#e4ebf2;font-size:1.1rem;line-height:1.55;margin:0 0 16px">{html.escape(question)}</p>'
        f'<div style="margin:0 0 22px"><span style="display:inline-block;background:rgba(255,255,255,0.14);'
        f'color:#ffffff;border-radius:999px;padding:5px 14px;font-size:14px;line-height:1.45">'
        f'For: {html.escape(audience[0].lower() + audience[1:])}</span></div>'
        f'<a href="rate" target="_self" style="{button}background:#ffffff;color:#1d2833">Start with question 1 →</a>'
        f'<a href="#about-the-data" target="_self" style="{button}border:1.5px solid #ffffff;color:#ffffff">About the data</a>'
        f'<a href="#access-the-data" target="_self" style="{button}border:1.5px solid #ffffff;color:#ffffff">Access the data</a>'
        f'</div>{stripes_svg(data, 34, "position:absolute;left:0;right:0;bottom:0")}</div>',
        unsafe_allow_html=True,
    )
    _, centre = stripes_centre(annual_means(data))
    st.caption(f"Stripes along the foot: one stripe per complete year, {first}–{int(annual_means(data)['year'].max())}, "
               f"from the NOAA record; blue years were cooler and red years warmer than the {centre} average "
               "(after Ed Hawkins' warming stripes).")


# ---------------------------------------------------------------- introduction

def introduction() -> None:
    st.subheader("Why this question matters")
    st.markdown(
        "Temperature goals such as the Paris Agreement's 1.5 °C refer to long-term warming, but the numbers that "
        "reach the news are often single record years or rates for short recent periods. Three things decide "
        "whether such a number can be trusted, and each has its own page."
    )
    cols = st.columns(3)
    texts = [
        ("1 · The rate and the records", "How many °C per decade, and do NASA, NOAA, the Met Office and Berkeley "
         "Earth agree on it?"),
        ("2 · The Pacific's swings", "How much of the year-to-year ups and downs comes from El Niño and La Niña?"),
        ("3 · A faster pace?", "Once those swings are taken out, is the recent rate higher than before, and by more "
         "than the uncertainty?"),
    ]
    for col, (head, body) in zip(cols, texts):
        with col.container(border=True):
            st.markdown(f"**{head}**  \n{body}")
    st.markdown("The dashboard shows what the four records measure. It does not explain *why* the rate is what it "
                "is, and it makes no forecasts.")


def where_things_stand(data: pd.DataFrame) -> None:
    st.subheader("Where things stand")
    warm = above_preindustrial(data)
    rates = rates_from(data, (FIRST_RATE_YEAR,))
    cols = st.columns(3)
    if not warm.empty:
        cols[0].metric(f"Warming, {warm.iloc[0]['twenty_label']} average vs 1850–1900",
                       f"{warm['twenty_year'].mean():.2f} °C",
                       help=f"Range across the {len(warm)} records that start in 1850: "
                            f"{warm['twenty_year'].min():.2f}–{warm['twenty_year'].max():.2f} °C.")
        cols[1].metric("Latest 12 months vs 1850–1900", f"{warm['last_12'].mean():.2f} °C",
                       help="Each record's 12 most recent months: "
                            + "; ".join(f"{r}: {v:.2f} °C ({lbl})" for r, v, lbl in
                                        zip(warm["record"], warm["last_12"], warm["last_12_label"])))
    if not rates.empty:
        cols[2].metric(f"Warming rate since {FIRST_RATE_YEAR}", f"{rates['rate'].mean():.2f} °C per decade",
                       help=f"Average of the four records; range {rates['rate'].min():.2f}–{rates['rate'].max():.2f}.")
    st.caption("Values are averages of the records that cover each period (GISTEMP starts in 1880, so it is left out "
               "of the 1850–1900 comparisons). The Paris Agreement's 1.5 °C refers to a long-term average, so the "
               "20-year figure is the comparable one; single months or years can sit above or below it.")


# ---------------------------------------------------------------- about the data

def about_the_data(data: pd.DataFrame) -> None:
    st.subheader("About the data")
    st.markdown(
        "Four groups independently turn millions of thermometer readings from weather stations, ships and buoys "
        "into one global average per month. They use overlapping raw data but different methods to fill gaps "
        "(the Arctic, Africa and the Southern Ocean have few stations), to correct for changes in instruments, and "
        "to treat areas of sea ice. Comparing them shows how much those choices matter."
    )
    rows = []
    for record_id, spec in RECORDS.items():
        part = data.loc[data["record_id"] == record_id]
        if part.empty:
            continue
        rows.append({"Record": spec["label"], "Publisher": PUBLISHERS[record_id],
                     "Months": f"{part['date'].min():%b %Y}–{part['date'].max():%b %Y}",
                     "Published baseline": spec["baseline"],
                     "Uncertainty range": "Yes" if part["lower_95_c"].notna().any() else "Not in this file"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Berkeley Earth's original product stopped updating in 2025, so its new high-resolution product is "
               "used; Berkeley Earth labels that file preliminary and not yet peer reviewed. Its rates match the other "
               "records closely (page 1).")
    st.markdown(
        "- **Anomaly.** Each value is how much warmer or cooler a month was than the average for that calendar month "
        "over a reference period. Anomalies are used because they vary smoothly across large areas, unlike actual "
        "temperatures.\n"
        "- **One baseline for all.** The records use different reference periods, so this dashboard moves all of them "
        "to 1991–2020, the World Meteorological Organization's current standard. This shifts each line up or down "
        "but does not change any rate.\n"
        "- **El Niño and La Niña.** The Oceanic Niño Index (ONI) from NOAA's Climate Prediction Center measures how "
        "much warmer (El Niño, +0.5 °C or more) or cooler (La Niña, −0.5 °C or less) than normal the surface of the "
        "central equatorial Pacific is, as a three-month average. It starts in 1950."
    )


# ---------------------------------------------------------------- question cards

_W, _H, _P = 320, 170, 14
_FRAME = (f'<svg viewBox="0 0 {_W} {_H}" width="100%" role="img" aria-label="{{label}}" '
          f'style="background:#f6f6f3;border-radius:8px;display:block">{{body}}</svg>')


def _scale(values, low, high, out_low, out_high):
    return out_low + (np.clip(values, low, high) - low) / (high - low) * (out_high - out_low)


def _thumb_rate(data: pd.DataFrame) -> str:
    parts = []
    for record in RECORD_ORDER:
        part = data.loc[(data["record"] == record) & (data["year"] >= 1950)].sort_values("date")
        smooth = part["anomaly_c"].rolling(12).mean().to_numpy()
        xs = np.linspace(_P, _W - _P, len(part))
        ys = _scale(smooth, -1.0, 1.0, _H - _P, _P)
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys) if np.isfinite(y))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{RECORD_COLOURS[record]}" stroke-width="1.3" '
                     f'stroke-opacity="0.85"/>')
    return _FRAME.format(label="Miniature line chart of the four records since 1950", body="".join(parts))


def _thumb_enso(data: pd.DataFrame) -> str:
    oni = data.loc[(data["record_id"] == "noaa") & data["oni"].notna() & (data["year"] >= 1970)].sort_values("date")
    xs = np.linspace(_P, _W - _P, len(oni))
    mid = _H / 2
    bars = []
    for x, v in zip(xs, oni["oni"]):
        h = abs(v) / 2.7 * (_H / 2 - _P)
        bars.append(f'<rect x="{x:.1f}" y="{mid - h if v > 0 else mid:.1f}" width="0.8" height="{h:.1f}" '
                    f'fill="{WARM if v > 0 else COOL}"/>')
    return _FRAME.format(label="Miniature bar chart of the Oceanic Niño Index",
                         body=f'<line x1="{_P}" y1="{mid}" x2="{_W - _P}" y2="{mid}" stroke="{INK}"/>' + "".join(bars))


def _thumb_acceleration(data: pd.DataFrame) -> str:
    years = annual_means(data)
    years = years.loc[years["year"] >= 1970]
    hi = years["year"].max()
    xs = _scale(years["year"].to_numpy(float), 1970, hi, _P, _W - _P)
    ys = _scale(years["mean"].to_numpy(), -0.7, 0.8, _H - _P, _P)
    dots = "".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6" fill="#898781"/>' for x, y in zip(xs, ys))
    lines = ""
    for low, high, colour in ((1970, 2009, "#898781"), (2010, hi, "#2a78d6")):
        part = years.loc[years["year"].between(low, high)]
        fit = np.polyfit(part["year"], part["mean"], 1)
        x0, x1 = _scale(np.array([low, high], float), 1970, hi, _P, _W - _P)
        y0, y1 = _scale(np.polyval(fit, [low, high]), -0.7, 0.8, _H - _P, _P)
        lines += f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{colour}" stroke-width="2.4"/>'
    return _FRAME.format(label="Miniature chart of annual means with two trend lines", body=dots + lines)


def _thumb_conclusion(data: pd.DataFrame) -> str:
    rates = rates_from(data, (FIRST_RATE_YEAR,))
    value = rates["rate"].mean() if not rates.empty else float("nan")
    body = (f'<text x="{_W / 2}" y="84" text-anchor="middle" font-size="50" font-weight="700" fill="{WARM}">'
            f'{value:+.2f} °C</text>'
            f'<text x="{_W / 2}" y="116" text-anchor="middle" font-size="17" fill="{INK}">per decade since '
            f'{FIRST_RATE_YEAR}.</text><text x="{_W / 2}" y="138" text-anchor="middle" font-size="17" fill="{INK}">'
            f'What changes the picture?</text>')
    return _FRAME.format(label="Headline warming rate", body=body)


def question_cards(data: pd.DataFrame) -> None:
    st.subheader("Explore the questions")
    st.markdown("Each question has its own page with interactive charts, a short reading of what the data show, and "
                "a download of the selected data. The pictures are miniature versions of each page's chart.")
    cards = [
        (PAGE_RATE, "QUESTION 1", "How fast is the world warming, and do the records agree?", _thumb_rate(data)),
        (PAGE_ENSO, "QUESTION 2", "How much of the year-to-year ups and downs is El Niño and La Niña?", _thumb_enso(data)),
        (PAGE_ACCEL, "QUESTION 3", "Has warming sped up once El Niño and La Niña are set aside?", _thumb_acceleration(data)),
        (PAGE_CONCLUSION, "CONCLUSION", "What the evidence shows, and what it does not", _thumb_conclusion(data)),
    ]
    columns = st.columns(4)
    for column, (page, kicker, title, picture) in zip(columns, cards):
        with column.container(border=True):
            st.markdown(picture, unsafe_allow_html=True)
            st.caption(kicker)
            st.markdown(f"**{title}**")
            st.page_link(page, label="Open page", icon=":material/arrow_forward:")


# ---------------------------------------------------------------- data access

def access_data(data: pd.DataFrame) -> None:
    st.subheader("Access the data")
    left, middle, right = st.columns(3)
    with left:
        st.markdown("#### :blue[Download this dashboard's data]")
        st.markdown(
            "Each question page has a download button for the records and years you select. The complete table "
            f"behind this dashboard ({len(data):,} record-months, {data.shape[1]} columns: the anomaly on the common "
            "baseline, the value as published, the uncertainty range where given, and the ONI) is available here."
        )
        out = data.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        st.download_button("Download the full dataset (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="global_temperature_records.csv", mime="text/csv", key="full_dataset_download")
    with middle:
        st.markdown("#### :green[Original files]")
        st.markdown(
            "The five files were downloaded on 13 October 2026. The pipeline (`src/data.py`) repeats the downloads "
            "with `python -m src.data --acquire`."
        )
        st.markdown("\n".join(f"- [{spec['label']}]({spec['url']})" for spec in RECORDS.values())
                    + f"\n- [Oceanic Niño Index]({ONI['url']})")
    with right:
        st.markdown("#### :orange[Explore further]")
        st.markdown(
            "Each publisher explains its method and offers maps, regional series and gridded data that go beyond "
            "the global averages used here."
        )
        st.markdown(
            "- [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/)\n"
            "- [NOAA Climate at a Glance](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/)\n"
            "- [Met Office HadCRUT5](https://www.metoffice.gov.uk/hadobs/hadcrut5/)\n"
            "- [Berkeley Earth data](https://berkeleyearth.org/data/)\n"
            "- [NOAA ENSO: cold and warm episodes](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php)"
        )
