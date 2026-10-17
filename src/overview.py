"""Building blocks of the overview page.

Order on the page: a full-width picture with the title (the warming stripes,
one stripe per year of the NOAA dataset), the line of links to every section,
annual anomalies relative to 1850-1900 for a dataset the reader selects, the
three research questions, the data, and how to get it.
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
        f'<p class="gw-for"><b>Audience:</b> {html.escape(audience)}.</p></div></div>',
        unsafe_allow_html=True,
    )
    _, centre = stripes_centre(years)
    st.caption(f"Warming stripes (after Hawkins, 2018): one stripe per complete year, {int(years['year'].min())}–"
               f"{int(years['year'].max())}, NOAA GlobalTemp. Blue: below the {centre} mean; red: above it.")


# ---------------------------------------------------------------- annual change since 1850-1900

def annual_vs_preindustrial(data: pd.DataFrame, record: str) -> pd.DataFrame:
    """Calendar-year means (complete years) of one dataset, relative to its own 1850-1900 mean."""
    part = data.loc[data["record"] == record]
    offset = part.loc[part["year"].between(*PREINDUSTRIAL), "anomaly_c"].mean()
    years = part.groupby("year")["anomaly_c"].agg(["mean", "size"])
    years = years.loc[years["size"] == 12, "mean"].sub(offset).rename("change_c").reset_index()
    years["average_20_c"] = years["change_c"].rolling(20, min_periods=20).mean()
    years["kind"] = np.where(years["change_c"] >= 0, "Above baseline", "Below baseline")
    return years


def temperature_change(data: pd.DataFrame) -> None:
    warm = above_preindustrial(data)
    if warm.empty:
        return
    section_heading(
        "Global surface temperature relative to 1850–1900",
        lead=("Bars show annual mean global surface temperature anomalies relative to 1850–1900, the period IPCC "
              "assessments use as an approximation of pre-industrial conditions. The line is the trailing 20-year "
              "mean, the averaging period IPCC AR6 uses to assess global warming levels such as 1.5 °C."),
        kicker="Current status",
    )
    options = list(warm["record"])
    _, pick = st.columns([3, 1.3])
    record = pick.selectbox("Dataset", options, key="home_record",
                            index=options.index("NOAA GlobalTemp") if "NOAA GlobalTemp" in options else 0)
    years = annual_vs_preindustrial(data, record)
    row = warm.set_index("record").loc[record]
    rates = rates_from(data, (FIRST_RATE_YEAR,)).set_index("record")

    cols = st.columns(3)
    cols[0].metric(f"20-year mean anomaly, {row['twenty_label']}", f"{row['twenty_year']:.2f} °C",
                   help=f"All {len(warm)} datasets that begin in 1850: "
                        f"{warm['twenty_year'].min():.2f}–{warm['twenty_year'].max():.2f} °C.")
    cols[1].metric(f"12-month mean anomaly, {row['last_12_label']}", f"{row['last_12']:.2f} °C",
                   help=f"All {len(warm)} datasets: {warm['last_12'].min():.2f}–{warm['last_12'].max():.2f} °C.")
    if record in rates.index:
        cols[2].metric(f"Linear trend since {FIRST_RATE_YEAR}", f"{rates.loc[record, 'rate']:.2f} °C per decade",
                       help=f"All four datasets: {rates['rate'].min():.2f}–{rates['rate'].max():.2f} °C per decade.")

    kinds = ["Below baseline", "Above baseline", "20-year mean"]
    colour = alt.Color("kind:N", title=None, scale=alt.Scale(domain=kinds, range=[COOL, WARM, "#2b2b29"]),
                       legend=alt.Legend(orient="bottom", labelFontSize=11, columnPadding=8))
    first, last = int(years["year"].min()), int(years["year"].max())
    x = alt.X("x0:Q", title=None, scale=alt.Scale(domain=[first, last + 1], nice=False, zero=False),
              axis=alt.Axis(format="d", tickCount=10))
    bars = alt.Chart(years.assign(x0=years["year"] + 0.08, x1=years["year"] + 0.92)).mark_bar().encode(
        x=x, x2="x1:Q", y=alt.Y("change_c:Q", title="°C, vs 1850–1900"), y2=alt.datum(0), color=colour,
        tooltip=[alt.Tooltip("year:Q", title="Year", format="d"),
                 alt.Tooltip("change_c:Q", title="Annual mean anomaly (°C vs 1850–1900)", format="+.2f"),
                 alt.Tooltip("average_20_c:Q", title="Trailing 20-year mean (°C)", format="+.2f")])
    line = alt.Chart(years.dropna(subset=["average_20_c"]).assign(x0=lambda f: f["year"] + 0.5,
                                                                   kind="20-year mean")).mark_line(
        strokeWidth=2.6).encode(x=x, y="average_20_c:Q", color=colour)
    goal = pd.DataFrame({"y": [1.5], "x0": [first + 1], "text": ["1.5 °C"]})
    rule = alt.Chart(goal).mark_rule(color=MUTED, strokeDash=[4, 4]).encode(y="y:Q")
    note = alt.Chart(goal).mark_text(align="left", dy=-8, color=INK, fontSize=12).encode(x=x, y="y:Q", text="text:N")

    with st.container(key="panel_home_change"):
        chart = (rule + note + bars + line).properties(
            height=400, title=_title("Annual anomaly relative to 1850–1900", record))
        st.altair_chart(_style(chart), width="stretch")
        source_line(f"Source: {SHORT_SOURCE.get(record, record)}. Anomalies relative to 1850–1900 calculated for "
                    "this dashboard.")
        st.caption("Red bars: annual means above the 1850–1900 mean; blue bars: below it. Dark line: trailing 20-year "
                   "mean. Dashed line: 1.5 °C. An individual year above 1.5 °C does not mean the Paris Agreement goal "
                   "has been exceeded, because the goal refers to long-term warming. Complete calendar years only.")
    out = years[["year", "change_c", "average_20_c"]].assign(record=record)
    st.download_button("Download annual anomalies (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                       file_name=f"annual_change_vs_1850_1900_{record.split()[0].lower()}.csv", mime="text/csv",
                       key="home_download")


# ---------------------------------------------------------------- the three questions

QUESTIONS = [
    (PAGE_RATE, "Question 1", "Warming rate",
     "What is the linear trend in global surface temperature, in °C per decade, and do NASA GISTEMP, NOAA "
     "GlobalTemp, HadCRUT5 and Berkeley Earth agree within uncertainty?",
     "Open page 1"),
    (PAGE_ENSO, "Question 2", "ENSO influence",
     "What share of interannual variability in global temperature is explained by the El Niño–Southern "
     "Oscillation?",
     "Open page 2"),
    (PAGE_ACCEL, "Question 3", "Acceleration",
     "After removing the ENSO signal, is the recent warming trend higher than the earlier trend by more than the "
     "uncertainty?",
     "Open page 3"),
]


def questions() -> None:
    section_heading(
        "Rationale",
        lead=("The Paris Agreement temperature goal refers to long-term warming, whereas public discussion often cites "
              "individual record-breaking years or trends over short recent periods. Three questions determine whether such "
              "figures are robust, and each is addressed on its own page. The dashboard describes what four "
              "observational datasets show; it does not attribute the causes of the trend and makes no projections."),
        kicker="Research questions",
    )
    for column, (page, kicker, head, body, link) in zip(st.columns(3), QUESTIONS):
        with column.container(key=f"panel_{kicker.replace(' ', '_').lower()}"):
            st.caption(kicker.upper())
            st.markdown(f"#### {head}")
            st.markdown(body)
            st.page_link(page, label=link, icon=":material/arrow_forward:")
    _, middle, _ = st.columns([1, 2, 1])
    middle.page_link(PAGE_CONCLUSION, label="Conclusion: findings, limitations and implications",
                     icon=":material/flag:")


# ---------------------------------------------------------------- about the data

def about_the_data(data: pd.DataFrame) -> None:
    section_heading(
        "About the data",
        lead=("Four independent groups produce monthly estimates of global mean surface temperature from "
              "instrumental observations (land weather stations, ships and buoys). They draw on overlapping raw data "
              "but differ in how they interpolate across data-sparse regions such as the Arctic, Africa and the "
              "Southern Ocean, homogenise for instrument changes, and treat sea-ice areas. Comparing them indicates "
              "the structural uncertainty arising from these methodological choices."),
        kicker="Four datasets and one index",
    )
    rows = []
    for record_id, spec in RECORDS.items():
        part = data.loc[data["record_id"] == record_id]
        if part.empty:
            continue
        rows.append({"Dataset": spec["label"], "Publisher": PUBLISHERS[record_id],
                     "Coverage": f"{part['date'].min():%b %Y}–{part['date'].max():%b %Y}",
                     "Reference period (as published)": spec["baseline"],
                     "Uncertainty estimates": "Yes" if part["lower_95_c"].notna().any() else "Not in this file"})
    with st.container(key="panel_about"):
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        st.caption("Berkeley Earth's original 1° product stopped updating in 2025, so its high-resolution product is "
                   "used; Berkeley Earth labels it preliminary and not yet peer reviewed. Its trends agree closely "
                   "with the other datasets (page 1).")
        st.markdown(
            "- **Temperature anomaly.** The departure of a month's temperature from the mean for that calendar month "
            "over a reference period. Anomalies are used because they are spatially coherent over large distances, "
            "unlike absolute temperatures.\n"
            "- **Common reference period.** The datasets are published relative to different reference periods, so "
            "this dashboard expresses all of them relative to 1991–2020, the current WMO climatological standard "
            "normal. This offsets each series but does not change any trend. The chart above uses 1850–1900 instead, "
            "the IPCC approximation of pre-industrial conditions.\n"
            "- **ENSO index.** The Oceanic Niño Index (ONI) from the NOAA Climate Prediction Center is the three-month "
            "running mean of sea-surface temperature anomalies in the Niño 3.4 region of the central equatorial "
            "Pacific. Values of +0.5 °C or more indicate El Niño conditions and −0.5 °C or less La Niña conditions. "
            "It begins in 1950."
        )


# ---------------------------------------------------------------- data access

def access_data(data: pd.DataFrame) -> None:
    section_heading(
        "Access the data",
        lead=("Each research question page offers a download of the selected datasets and period. The complete "
              "table and the original source files are available here."),
        kicker="Downloads and links",
    )
    left, middle, right = st.columns(3)
    with left.container(key="panel_access_download"):
        st.markdown("#### This dashboard's data")
        st.markdown(
            f"The complete table behind this dashboard: {len(data):,} dataset-months, {data.shape[1]} columns (the "
            "anomaly relative to 1991–2020, the anomaly as published, the uncertainty range where provided, and the "
            "ONI)."
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
            "Each publisher documents its methods and provides maps, regional series and gridded data beyond the "
            "global means used here."
        )
        st.markdown(
            "- [NASA GISTEMP](https://data.giss.nasa.gov/gistemp/)\n"
            "- [NOAA Climate at a Glance](https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/)\n"
            "- [Met Office HadCRUT5](https://www.metoffice.gov.uk/hadobs/hadcrut5/)\n"
            "- [Berkeley Earth data](https://berkeleyearth.org/data/)\n"
            "- [NOAA ENSO: cold and warm episodes](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php)"
        )
