"""Page texts that describe only what the data show, plus page navigation.

Every number in these summaries is computed from the loaded dataset, so the
text always matches the data the page is using.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import decimal_year, fit_trend, format_interval
from src.views import (DEFAULT_LAG, FIRST_RATE_YEAR, RECORD_ORDER, adjusted_series, enso_fits, record_trends)


HOME = "home.py"
PAGE_RATE = "sections/1_warming_rate.py"
PAGE_ENSO = "sections/2_el_nino.py"
PAGE_ACCEL = "sections/3_acceleration.py"
PAGE_CONCLUSION = "sections/4_conclusion.py"

PREINDUSTRIAL = (1850, 1900)


def page_nav(previous: tuple[str, str] | None, following: tuple[str, str] | None) -> None:
    st.divider()
    left, right = st.columns(2)
    if previous:
        left.page_link(previous[0], label=f"Previous: {previous[1]}", icon=":material/arrow_back:")
    if following:
        right.page_link(following[0], label=f"Next: {following[1]}", icon=":material/arrow_forward:")


def latest_year(data: pd.DataFrame) -> int:
    return int(data["year"].max())


# ---------------------------------------------------------------- shared calculations

@st.cache_data
def rates_from(data: pd.DataFrame, starts: tuple[int, ...]) -> pd.DataFrame:
    """Straight-line rate from each start year to the latest month, for every record."""
    rows = []
    for start in starts:
        if start < data["year"].min():
            continue
        trends = record_trends(data.loc[data["year"] >= start])
        for record, trend in trends.items():
            rows.append({"start": start, "record": record, "rate": trend.rate, "low": trend.low, "high": trend.high})
    return pd.DataFrame(rows)


@st.cache_data
def above_preindustrial(data: pd.DataFrame) -> pd.DataFrame:
    """Warming relative to 1850-1900 for the records that reach back that far."""
    rows = []
    for record in RECORD_ORDER:
        part = data.loc[data["record"] == record].sort_values("date")
        base = part.loc[part["year"].between(*PREINDUSTRIAL)]
        if base["year"].min() != PREINDUSTRIAL[0] or len(base) < 600:
            continue
        offset = base["anomaly_c"].mean()
        full_years = part.groupby("year").filter(lambda g: len(g) == 12)
        last_full = int(full_years["year"].max())
        twenty = full_years.loc[full_years["year"].between(last_full - 19, last_full), "anomaly_c"].mean() - offset
        last12 = part.tail(12)
        rows.append({"record": record, "twenty_year": twenty, "twenty_label": f"{last_full - 19}–{last_full}",
                     "last_12": last12["anomaly_c"].mean() - offset,
                     "last_12_label": f"{last12['date'].min():%b %Y}–{last12['date'].max():%b %Y}"})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- page 1

def rate_summary(data: pd.DataFrame) -> None:
    starts = (1880, 1950, 1970, 1990, 2000)
    rates = rates_from(data, starts)
    st.subheader("What the data show")
    if rates.empty:
        return
    summary = rates.groupby("start").agg(low=("rate", "min"), high=("rate", "max"), mean=("rate", "mean"),
                                         ci_low=("low", "min"), ci_high=("high", "max"), records=("record", "nunique"))
    end = latest_year(data)
    base_start = FIRST_RATE_YEAR if FIRST_RATE_YEAR in summary.index else summary.index[-1]
    base = summary.loc[base_start]
    earliest, latest_start = summary.index.min(), summary.index.max()
    rising = bool((summary["mean"].diff().dropna() > 0).all())
    narrowest = (rates.loc[rates["start"] == base_start, "high"] - rates.loc[rates["start"] == base_start, "low"]).min()
    spread = base["high"] - base["low"]
    first_width = summary.loc[earliest, "ci_high"] - summary.loc[earliest, "ci_low"]
    last_width = summary.loc[latest_start, "ci_high"] - summary.loc[latest_start, "ci_low"]
    st.markdown(
        ("- **The later the start, the faster the rate.** " if rising else "- **The rate depends on the start year.** ")
        + f"From {earliest} to {end} the records average {summary.loc[earliest, 'mean']:+.2f} °C per decade; from "
        f"{base_start} it is {base['mean']:+.2f}, and from {latest_start} it is {summary.loc[latest_start, 'mean']:+.2f}.\n"
        + (f"- **The records agree closely.** From {base_start}, the rates lie between {base['low']:+.2f} and "
           f"{base['high']:+.2f} °C per decade, a spread of {spread:.2f}, which is smaller than even the narrowest "
           f"95% interval ({narrowest:.2f} wide).\n" if spread < narrowest else
           f"- **The records differ.** From {base_start}, the rates lie between {base['low']:+.2f} and "
           f"{base['high']:+.2f} °C per decade, a spread of {spread:.2f}, wider than the narrowest 95% interval "
           f"({narrowest:.2f}).\n")
        + ("- **Shorter periods are less certain.** " if last_width > first_width else "- **Interval widths.** ")
        + f"Across records the 95% intervals span {summary.loc[earliest, 'ci_low']:+.2f} to "
        f"{summary.loc[earliest, 'ci_high']:+.2f} from {earliest}, and {summary.loc[latest_start, 'ci_low']:+.2f} to "
        f"{summary.loc[latest_start, 'ci_high']:+.2f} from {latest_start}."
    )
    table = summary.reset_index().rename(columns={"start": "From"})
    table["To"] = end
    table["Average rate (°C per decade)"] = table["mean"].round(3)
    table["Range across records"] = [format_interval(a, b) for a, b in zip(table["low"], table["high"])]
    table["95% intervals (all records)"] = [format_interval(a, b) for a, b in zip(table["ci_low"], table["ci_high"])]
    table["Records"] = table["records"]
    st.dataframe(table[["From", "To", "Records", "Average rate (°C per decade)", "Range across records",
                        "95% intervals (all records)"]], hide_index=True, width="stretch")
    st.caption("Each rate is a straight line fitted to every monthly value from January of the start year to the "
               "latest month. GISTEMP begins in 1880, so every record is included from 1880 onward.")


# ---------------------------------------------------------------- page 2

@st.cache_data
def strongest_events(data: pd.DataFrame, lag: int = DEFAULT_LAG, count: int = 6) -> pd.DataFrame:
    """The largest El Niño peaks and the global response in the following 12 months."""
    _, work = enso_fits(data, lag)
    oni = data.drop_duplicates("date")[["date", "oni"]].dropna().sort_values("date").reset_index(drop=True)
    peaks = []
    values = oni["oni"].to_numpy()
    for i in range(1, len(oni) - 1):
        if values[i] >= 1.5 and values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            if peaks and (oni.loc[i, "date"] - peaks[-1]["date"]).days < 365:
                if values[i] > peaks[-1]["oni"]:
                    peaks[-1] = {"date": oni.loc[i, "date"], "oni": values[i]}
                continue
            peaks.append({"date": oni.loc[i, "date"], "oni": values[i]})
    rows = []
    for peak in sorted(peaks, key=lambda p: -p["oni"])[:count]:
        window = work.loc[(work["date"] > peak["date"]) & (work["date"] <= peak["date"] + pd.DateOffset(months=12))]
        if window.groupby("record")["date"].count().min() < 12:
            continue
        rows.append({"El Niño peak": f"{peak['date']:%b %Y}", "Peak ONI (°C)": round(peak["oni"], 2),
                     "Global temperature above the long-term curve, next 12 months (°C)":
                         round(window.groupby("record")["deviation_c"].mean().mean(), 2)})
    return pd.DataFrame(rows)


def enso_summary(data: pd.DataFrame) -> None:
    with_oni = data.loc[data["oni"].notna()]
    fits, work = enso_fits(with_oni, DEFAULT_LAG)
    st.subheader("What the data show")
    if not fits:
        return
    effect = np.mean([f.enso_per_degree for f in fits.values()])
    annual = [f.annual_share for f in fits.values()]
    events = strongest_events(with_oni)
    latest = with_oni.sort_values("date").iloc[-1]
    higher = with_oni.drop_duplicates("date")
    higher = higher.loc[(higher["oni"] > latest["oni"]) & (higher["date"] < latest["date"])]
    since = f" — the highest since {higher['date'].max():%b %Y}" if not higher.empty else ""
    latest_part = work.sort_values("date").groupby("record").tail(1)["enso_part_c"].mean()
    bullets = [
        f"- **A steady link.** Over {int(with_oni['year'].min())}–{latest_year(data)}, each 1 °C of El Niño warming "
        f"goes with about {effect:.2f} °C of global warming {DEFAULT_LAG} months later, in every record.",
        f"- **A sizeable part of the year-to-year wobble.** El Niño and La Niña account for {min(annual):.0%}–"
        f"{max(annual):.0%} of the variation of yearly averages around the long-term curve.",
    ]
    if not events.empty:
        top = events.iloc[0]
        bullets.append(
            f"- **After big El Niños, years run warm.** In the 12 months after the {top['El Niño peak']} peak "
            f"(ONI {top['Peak ONI (°C)']:+.2f} °C), global temperature averaged "
            f"{top['Global temperature above the long-term curve, next 12 months (°C)']:+.2f} °C above its long-term curve."
        )
    bullets.append(
        f"- **Now.** The latest ONI value is {latest['oni']:+.2f} °C for the season centred on "
        f"{latest['date']:%B %Y}{since}. In the latest month, the El Niño part of global temperature is "
        f"{latest_part:+.2f} °C."
    )
    st.markdown("\n".join(bullets))
    if not events.empty:
        st.dataframe(events, hide_index=True, width="stretch")
        st.caption(f"El Niño peaks are the highest ONI values of events reaching at least +1.5 °C. The response "
                   f"averages the selected records over the 12 months after each peak, measured from each record's "
                   f"long-term curve (all records, delay {DEFAULT_LAG} months).")


# ---------------------------------------------------------------- page 3

@st.cache_data
def window_check(data: pd.DataFrame) -> pd.DataFrame:
    """Rates for commonly quoted windows, as published and with ENSO removed."""
    series = adjusted_series(data)
    end = latest_year(data)
    windows = [(1970, 2009), (2010, end), (2015, end), (2020, end)]
    rows = []
    for first, last in windows:
        for column, label in (("anomaly_c", "As published"), ("adjusted_c", "El Niño and La Niña removed")):
            rates = []
            for record in RECORD_ORDER:
                part = series.loc[(series["record"] == record) & series["year"].between(first, last)]
                if len(part) >= 24:
                    rates.append(fit_trend(decimal_year(part["year"], part["month"]), part[column].to_numpy(float)))
            if rates:
                rows.append({"Window": f"{first}–{last}", "Years": last - first + 1, "Temperatures": label,
                             "Average rate (°C per decade)": round(np.mean([t.rate for t in rates]), 2),
                             "Widest 95% interval": format_interval(min(t.low for t in rates), max(t.high for t in rates)),
                             "width": max(t.high for t in rates) - min(t.low for t in rates)})
    return pd.DataFrame(rows)


def acceleration_summary(data: pd.DataFrame) -> None:
    table = window_check(data)
    st.subheader("How much do short windows tell us?")
    if table.empty:
        return
    st.markdown(
        "Rates for recent short windows are often quoted to argue that warming has sped up. The table repeats that "
        "calculation for four windows, with each window's widest 95% interval across the records."
    )
    st.dataframe(table.drop(columns="width"), hide_index=True, width="stretch")
    raw = table.loc[table["Temperatures"] == "As published"].set_index("Window")
    adj = table.loc[table["Temperatures"] != "As published"].set_index("Window")
    longest, recent, shortest = raw.index[0], raw.index[1], raw.index[-1]
    narrower = adj.loc[shortest, "width"] < raw.loc[shortest, "width"]
    later = adj.loc[recent, "Average rate (°C per decade)"]
    earlier = adj.loc[longest, "Average rate (°C per decade)"]
    st.markdown(
        ("- **Short windows are uncertain.** " if raw.loc[shortest, "width"] > raw.loc[longest, "width"] else "- ")
        + f"As published, the {shortest} window's intervals span {raw.loc[shortest, 'width']:.2f} °C per decade, "
        f"against {raw.loc[longest, 'width']:.2f} for {longest}.\n"
        + ("- **Removing El Niño and La Niña narrows them.** " if narrower else
           "- **Removing El Niño and La Niña does not narrow them.** ")
        + f"For {shortest} the span becomes {adj.loc[shortest, 'width']:.2f} °C per decade.\n"
        + f"- **The rate since {recent[:4]} is {'higher' if later > earlier else 'not higher'} than in {longest}.** "
        f"With El Niño and La Niña removed, {recent} averages {later:+.2f} °C per decade, compared with "
        f"{earlier:+.2f} in {longest}."
    )


# ---------------------------------------------------------------- conclusion

def conclusion(data: pd.DataFrame) -> None:
    end = latest_year(data)
    rates = rates_from(data, (FIRST_RATE_YEAR,))
    with_oni = data.loc[data["oni"].notna()]
    fits, _ = enso_fits(with_oni, DEFAULT_LAG)
    windows = window_check(data)
    adj = windows.loc[windows["Temperatures"] != "As published"].set_index("Window")
    raw = windows.loc[windows["Temperatures"] == "As published"].set_index("Window")
    warm = above_preindustrial(data)
    events = strongest_events(with_oni)
    spread = rates["rate"].max() - rates["rate"].min()
    narrowest = (rates["high"] - rates["low"]).min()
    later = adj.loc[f"2010–{end}", "Average rate (°C per decade)"]
    earlier = adj.loc["1970–2009", "Average rate (°C per decade)"]

    st.subheader("What the data show")
    points = [
        f"1. **The world is warming at about {rates['rate'].mean():.2f} °C per decade since {FIRST_RATE_YEAR}**; "
        f"the {len(rates)} records' rates lie between {rates['rate'].min():.2f} and {rates['rate'].max():.2f} °C per "
        f"decade, a spread {'smaller' if spread < narrowest else 'larger'} than the narrowest 95% interval "
        f"({narrowest:.2f}).",
        f"2. **El Niño and La Niña explain a sizeable part of the year-to-year ups and downs** "
        f"({min(f.annual_share for f in fits.values()):.0%}–{max(f.annual_share for f in fits.values()):.0%} of "
        f"the variation of yearly averages around the long-term curve), with global temperature following the "
        f"Pacific by about {DEFAULT_LAG} months.",
        f"3. **With El Niño and La Niña removed, the rate since 2010 is {'higher' if later > earlier else 'not higher'} "
        f"than in 1970–2009**: {later:+.2f} against {earlier:+.2f} °C per decade on average. Page 3 shows whether "
        f"the 95% intervals of the two periods overlap for each record and split year.",
        f"4. **Short windows can mislead in either direction.** As published, the 2015–{end} rate averages "
        f"{raw.loc[f'2015–{end}', 'Average rate (°C per decade)']:+.2f} °C per decade with an interval "
        f"{raw.loc[f'2015–{end}', 'width']:.2f} °C per decade wide; removing El Niño and La Niña gives "
        f"{adj.loc[f'2015–{end}', 'Average rate (°C per decade)']:+.2f}.",
    ]
    if not warm.empty:
        points.append(
            f"5. **Relative to 1850–1900**, the {warm.iloc[0]['twenty_label']} average is "
            f"{warm['twenty_year'].min():.2f}–{warm['twenty_year'].max():.2f} °C warmer in the {len(warm)} records that "
            f"reach back that far, and the latest 12 months are {warm['last_12'].min():.2f}–{warm['last_12'].max():.2f} °C warmer."
        )
    st.markdown("\n".join(points))

    st.subheader("What the data cannot tell us")
    st.markdown(
        "- **Why the rate changed.** Only El Niño and La Niña are removed here. Volcanic eruptions, the solar cycle, "
        "changes in air pollution (aerosols) and greenhouse gases also shape the curve, and this dataset does not "
        "separate them.\n"
        "- **What comes next.** These are observations only. A straight line fitted to the past is not a forecast, "
        "so the dashboard draws no lines beyond the latest month.\n"
        "- **Exact values.** The four records are estimates built from incomplete coverage. Their published "
        "uncertainty (shown for HadCRUT5 and Berkeley Earth) is a few hundredths of a degree in recent decades and "
        "larger before 1900."
    )

    response = events["Global temperature above the long-term curve, next 12 months (°C)"] if not events.empty else None
    st.subheader("What this means for climate policy analysts")
    st.markdown(
        "- **Quote rates with their period and interval.** The same data give different rates for different start "
        "years; a rate without its period and 95% interval cannot be compared with another.\n"
        "- **Do not read one hot or cool year as a change in the trend.** "
        + (f"In the 12 months after the largest El Niño peaks, global temperature sat between {response.min():+.2f} "
           f"and {response.max():+.2f} °C from its long-term curve (page 2).\n" if response is not None else "\n")
        + f"- **Check more than one record.** A claim that holds in only one record deserves a second look; here the "
        f"records' rates since {FIRST_RATE_YEAR} differ by {spread:.2f} °C per decade.\n"
        "- **Single months or years above 1.5 °C are not the Paris Agreement threshold**, which refers to warming "
        "averaged over about 20 years. The 20-year figure above is the comparable one."
    )

    latest = with_oni.sort_values("date").iloc[-1]
    st.subheader("Next steps")
    st.markdown(
        "- Remove volcanic and solar effects as well (for example with aerosol optical depth and total solar "
        "irradiance series) and see whether the post-2010 change remains.\n"
        f"- Repeat the comparison as new months arrive. The latest ONI value is {latest['oni']:+.2f} °C "
        f"({latest['date']:%B %Y}), an El Niño; the months that follow it will become the end of every recent window."
    )


def methods_and_limitations() -> None:
    with st.expander("Methods and limitations", expanded=False):
        st.markdown(
            "**Common baseline.** Each record is published on its own baseline (GISTEMP and Berkeley Earth 1951–1980, "
            "NOAA 1901–2000, HadCRUT5 1961–1990). For every record and calendar month, the 1991–2020 mean is "
            "subtracted, so all four share the WMO's current 30-year normal. Rates do not depend on the baseline.\n\n"
            "**Rates.** A straight line is fitted by ordinary least squares to the monthly anomalies in a period. "
            "Monthly values are autocorrelated, so the 95% interval uses an effective sample size "
            "n(1−r)/(1+r), where r is the lag-1 autocorrelation of the residuals (Santer et al., 2008), with "
            "Student's t. With strong autocorrelation and short windows, such intervals can still be slightly too "
            "narrow.\n\n"
            "**El Niño and La Niña.** For each record, monthly anomalies are regressed on a quadratic curve in time "
            "plus the Oceanic Niño Index shifted by a delay in months. The ENSO part (coefficient × ONI) is "
            "subtracted to give the adjusted series. The effect per 1 °C of ONI barely changes if the curve is a "
            "straight line, but the share of variation explained is lower (because a straight line leaves the "
            "curve's bend in the residuals). A delay of 2 months fits best for all four records over 1950–2026. "
            "The ONI is itself a three-month running mean, dated here to its middle month.\n\n"
            "**Year-to-year share.** Calendar-year means of the deviation from the curve are compared with "
            "calendar-year means of the ENSO part; the share is 1 − (unexplained variance / total variance).\n\n"
            "**Records.** Berkeley Earth's original 1° product stopped updating in 2025 (its last month is December "
            "2024); this dashboard uses its high-resolution successor, which Berkeley Earth labels preliminary and "
            "not yet peer reviewed. In it, sea-ice areas use air temperatures when ice is present. Records end in "
            "different months; each rate uses every month a record has."
        )
