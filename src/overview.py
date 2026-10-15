"""Building blocks of the overview page.

Order on the page: a full-width picture with the title (the warming stripes,
one stripe per year of the NOAA record), the line of links to every section,
annual temperature compared with 1850-1900 for a record the reader selects,
the three questions, the data, and how to get it.
"""

from __future__ import annotations

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.data import ONI, RECORDS
from src.stories import PAGE_ACCEL, PAGE_CONCLUSION, PAGE_ENSO, PAGE_RATE, PREINDUSTRIAL, above_preindustrial, rates_from
from src.theme import annual_means, section_heading, source_line, stripes_centre, stripes_svg
from src.views import COOL, FIRST_RATE_YEAR, INK, MUTED, SOURCES, WARM, _style, _title

PUBLISHERS = {record_id: SOURCES[i][0] for i, record_id in enumerate(RECORDS)}
SHORT_SOURCE = {  # short form for the source line under the chart
    "NOAA GlobalTemp": "NOAA National Centers for Environmental Information",
    "Met Office HadCRUT5": "Met Office Hadley Centre and Climatic Research Unit",
    "Berkeley Earth": "Berkeley Earth",
}


# ---------------------------------------------------------------- picture with the title

def hero(data: pd.DataFrame, title: str, question: str, audience: str) -> None:
    """Full-width warming stripes behind the title, the question and the audience."""
    years = annual_means(data)
    st.markdown(
        f'<div class="gw-hero">{stripes_svg(data)}<div class="gw-shade"></div><div class="gw-hero-text">'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(question)}</p>'
        f'<p class="gw-for">For {html.escape(audience[0].lower() + audience[1:])}.</p></div></div>',
        unsafe_allow_html=True,
    )
    _, centre = stripes_centre(years)
    st.caption(f"Picture: one stripe per complete year, {int(years['year'].min())}–{int(years['year'].max())}, from the "
               f"NOAA record; blue years were cooler and red years warmer than the {centre} average (after Ed "
               "Hawkins' warming stripes).")


# ---------------------------------------------------------------- annual change since 1850-1900

def annual_vs_preindustrial(data: pd.DataFrame, record: str) -> pd.DataFrame:
    """Calendar-year means (complete years) of one record, measured from its own 1850-1900 average."""
    part = data.loc[data["record"] == record]
    offset = part.loc[part["year"].between(*PREINDUSTRIAL), "anomaly_c"].mean()
    years = part.groupby("year")["anomaly_c"].agg(["mean", "size"])
    years = years.loc[years["size"] == 12, "mean"].sub(offset).rename("change_c").reset_index()
    years["average_20_c"] = years["change_c"].rolling(20, min_periods=20).mean()
    years["kind"] = np.where(years["change_c"] >= 0, "Warmer year", "Cooler year")
    return years


def temperature_change(data: pd.DataFrame) -> None:
    warm = above_preindustrial(data)
    if warm.empty:
        return
    section_heading(
        "Annual global temperature compared with 1850–1900",
        lead=("Each bar is one calendar year's average global surface temperature, measured from the 1850–1900 "
              "average that the Paris Agreement uses as its pre-industrial reference. The line is the average of "
              "the 20 years up to each year, the kind of long-term figure the 1.5 °C goal refers to."),
        kicker="Where things stand",
    )
    options = list(warm["record"])
    _, pick = st.columns([3, 1.3])
    record = pick.selectbox("Select a temperature record", options, key="home_record",
                            index=options.index("NOAA GlobalTemp") if "NOAA GlobalTemp" in options else 0)
    years = annual_vs_preindustrial(data, record)
    row = warm.set_index("record").loc[record]
    rates = rates_from(data, (FIRST_RATE_YEAR,)).set_index("record")

    cols = st.columns(3)
    cols[0].metric(f"Warming, {row['twenty_label']} average", f"{row['twenty_year']:.2f} °C",
                   help=f"All {len(warm)} records that start in 1850: "
                        f"{warm['twenty_year'].min():.2f}–{warm['twenty_year'].max():.2f} °C.")
    cols[1].metric(f"Latest 12 months ({row['last_12_label']})", f"{row['last_12']:.2f} °C",
                   help=f"All {len(warm)} records: {warm['last_12'].min():.2f}–{warm['last_12'].max():.2f} °C.")
    if record in rates.index:
        cols[2].metric(f"Warming rate since {FIRST_RATE_YEAR}", f"{rates.loc[record, 'rate']:.2f} °C per decade",
                       help=f"All four records: {rates['rate'].min():.2f}–{rates['rate'].max():.2f} °C per decade.")

    kinds = ["Cooler year", "Warmer year", "20-year average"]
    colour = alt.Color("kind:N", title=None, scale=alt.Scale(domain=kinds, range=[COOL, WARM, "#2b2b29"]),
                       legend=alt.Legend(orient="bottom", labelFontSize=11, columnPadding=8))
    first, last = int(years["year"].min()), int(years["year"].max())
    x = alt.X("x0:Q", title=None, scale=alt.Scale(domain=[first, last + 1], nice=False, zero=False),
              axis=alt.Axis(format="d", tickCount=10))
    bars = alt.Chart(years.assign(x0=years["year"] + 0.08, x1=years["year"] + 0.92)).mark_bar().encode(
        x=x, x2="x1:Q", y=alt.Y("change_c:Q", title="°C, vs 1850–1900"), y2=alt.datum(0), color=colour,
        tooltip=[alt.Tooltip("year:Q", title="Year", format="d"),
                 alt.Tooltip("change_c:Q", title="Annual mean (°C vs 1850–1900)", format="+.2f"),
                 alt.Tooltip("average_20_c:Q", title="20-year average (°C)", format="+.2f")])
    line = alt.Chart(years.dropna(subset=["average_20_c"]).assign(x0=lambda f: f["year"] + 0.5,
                                                                   kind="20-year average")).mark_line(
        strokeWidth=2.6).encode(x=x, y="average_20_c:Q", color=colour)
    goal = pd.DataFrame({"y": [1.5], "x0": [first + 1], "text": ["1.5 °C"]})
    rule = alt.Chart(goal).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    note = alt.Chart(goal).mark_text(align="left", dy=-8, color=INK, fontSize=12).encode(x=x, y="y:Q", text="text:N")

    with st.container(key="panel_home_change"):
        chart = (rule + note + bars + line).properties(
            height=400, title=_title("Annual temperature vs 1850–1900", record))
        st.altair_chart(_style(chart), width="stretch")
        source_line(f"Source: {SHORT_SOURCE.get(record, record)}. Change from 1850–1900 calculated for this dashboard.")
        st.caption("Red bars: years warmer than the 1850–1900 average; blue bars: cooler years. Dark line: average of "
                   "the 20 years up to and including each year. Dashed line: 1.5 °C. Single years above it do not "
                   "mean the Paris goal has been passed, because the goal refers to the long-term average. "
                   "Only complete calendar years are shown.")
    out = years[["year", "change_c", "average_20_c"]].assign(record=record)
    st.download_button("Download these annual values (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name=f"annual_change_vs_1850_1900_{record.split()[0].lower()}.csv", mime="text/csv",
                       key="home_download")


# ---------------------------------------------------------------- the three questions

QUESTIONS = [
    (PAGE_RATE, "Question 1", "The rate and the records",
     "How many °C per decade, and do NASA, NOAA, the Met Office and Berkeley Earth agree on it?",
     "Open page 1"),
    (PAGE_ENSO, "Question 2", "The Pacific's swings",
     "How much of the year-to-year ups and downs comes from El Niño and La Niña?",
     "Open page 2"),
    (PAGE_ACCEL, "Question 3", "A faster pace?",
     "Once those swings are taken out, is the recent rate higher than before, and by more than the uncertainty?",
     "Open page 3"),
]


def questions() -> None:
    section_heading(
        "Why this question matters",
        lead=("Temperature goals such as the Paris Agreement's 1.5 °C refer to long-term warming, but the numbers that "
              "reach the news are often single record years or rates for short recent periods. Three things decide "
              "whether such a number can be trusted, and each has its own page. The dashboard shows what the four "
              "records measure; it does not explain why the rate is what it is, and it makes no forecasts."),
        kicker="Three questions",
    )
    for column, (page, kicker, head, body, link) in zip(st.columns(3), QUESTIONS):
        with column.container(key=f"panel_{kicker.replace(' ', '_').lower()}"):
            st.caption(kicker.upper())
            st.markdown(f"#### {head}")
            st.markdown(body)
            st.page_link(page, label=link, icon=":material/arrow_forward:")
    _, middle, _ = st.columns([1, 2, 1])
    middle.page_link(PAGE_CONCLUSION, label="Read the conclusion: what the evidence shows, and what it does not",
                     icon=":material/flag:")


# ---------------------------------------------------------------- about the data

def about_the_data(data: pd.DataFrame) -> None:
    section_heading(
        "About the data",
        lead=("Four groups independently turn millions of thermometer readings from weather stations, ships and buoys "
              "into one global average per month. They use overlapping raw data but different methods to fill gaps "
              "(the Arctic, Africa and the Southern Ocean have few stations), to correct for changes in instruments, "
              "and to treat areas of sea ice. Comparing them shows how much those choices matter."),
        kicker="Four records and one index",
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
    with st.container(key="panel_about"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Berkeley Earth's original product stopped updating in 2025, so its new high-resolution product is "
                   "used; Berkeley Earth labels that file preliminary and not yet peer reviewed. Its rates match the "
                   "other records closely (page 1).")
        st.markdown(
            "- **Anomaly.** Each value is how much warmer or cooler a month was than the average for that calendar "
            "month over a reference period. Anomalies are used because they vary smoothly across large areas, unlike "
            "actual temperatures.\n"
            "- **One baseline for all.** The records use different reference periods, so this dashboard moves all of "
            "them to 1991–2020, the World Meteorological Organization's current standard. This shifts each line up or "
            "down but does not change any rate. The overview chart above uses 1850–1900 instead, to match the Paris "
            "Agreement.\n"
            "- **El Niño and La Niña.** The Oceanic Niño Index (ONI) from NOAA's Climate Prediction Center measures how "
            "much warmer (El Niño, +0.5 °C or more) or cooler (La Niña, −0.5 °C or less) than normal the surface of "
            "the central equatorial Pacific is, as a three-month average. It starts in 1950."
        )


# ---------------------------------------------------------------- data access

def access_data(data: pd.DataFrame) -> None:
    section_heading(
        "Access the data",
        lead=("Every question page has a download of the records and years you select. The complete table and the "
              "original files are here."),
        kicker="Downloads and links",
    )
    left, middle, right = st.columns(3)
    with left.container(key="panel_access_download"):
        st.markdown("#### This dashboard's data")
        st.markdown(
            f"The complete table behind this dashboard: {len(data):,} record-months, {data.shape[1]} columns (the "
            "anomaly on the common baseline, the value as published, the uncertainty range where given, and the ONI)."
        )
        out = data.copy()
        out["date"] = out["date"].dt.strftime("%Y-%m-%d")
        st.download_button("Download the full dataset (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="global_temperature_records.csv", mime="text/csv", key="full_dataset_download")
    with middle.container(key="panel_access_files"):
        st.markdown("#### Original files")
        st.markdown(
            "The five files were downloaded on 13 October 2026. The pipeline (`src/data.py`) repeats the downloads "
            "with `python -m src.data --acquire`."
        )
        st.markdown("\n".join(f"- [{spec['label']}]({spec['url']})" for spec in RECORDS.values())
                    + f"\n- [Oceanic Niño Index]({ONI['url']})")
    with right.container(key="panel_access_more"):
        st.markdown("#### Explore further")
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
