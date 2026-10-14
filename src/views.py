"""The three principal views. Each answers one part of the dashboard question.

Every view keeps its heading, unit, sources, latest-observation date, controls
and outputs inside ``st.container(key=<view id>)`` so they can be checked
against ``dashboard.yml``.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from src.analysis import (Trend, decimal_year, difference_interval, fit_enso, fit_trend, format_interval,
                          format_rate)
from src.theme import BANNER, banner, pattern_acceleration, pattern_enso, pattern_rate


DOWNLOAD_LABEL = "Download filtered data (CSV)"
DATE_LABEL = "Latest observation"
EMPTY_MESSAGE = "No temperature records match the selected filters."

SOURCES = [
    ("NASA Goddard Institute for Space Studies",
     "GISS Surface Temperature Analysis (GISTEMP v4): global-mean monthly land-ocean temperature index (GLB.Ts+dSST)"),
    ("NOAA National Centers for Environmental Information",
     "Climate at a Glance: Global Time Series, monthly global land and ocean temperature anomalies (NOAAGlobalTemp)"),
    ("Met Office Hadley Centre and Climatic Research Unit, University of East Anglia",
     "HadCRUT.5.2.0.0 global monthly summary series (analysis)"),
    ("Berkeley Earth",
     "Global monthly land and ocean temperature anomalies, high-resolution product (Global_TAVG_monthly)"),
    ("NOAA Climate Prediction Center",
     "Oceanic Niño Index (ONI), three-month running mean of ERSST.v5 sea-surface temperature anomalies in the Niño 3.4 region"),
]

# Colour follows the record (validated categorical slots 1-4, adjacent pairs, light mode).
RECORD_COLOURS = {
    "NASA GISTEMP v4": "#2a78d6",
    "NOAA GlobalTemp": "#eb6834",
    "Met Office HadCRUT5": "#1baf7a",
    "Berkeley Earth": "#eda100",
}
RECORD_ORDER = list(RECORD_COLOURS)
INK = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e0"
WARM, COOL, NEUTRAL_BAR = "#e34948", "#2a78d6", "#c3c2b7"
EARLIER, LATER = "#898781", "#2a78d6"

FIRST_RATE_YEAR = 1970   # start of the steady modern warming used as the default period
DEFAULT_LAG = 2          # months; the delay that fits best over 1950-2026 in all four records
SPLIT_YEARS = [1990, 1995, 2000, 2005, 2010, 2015]
SERIES = ["El Niño and La Niña removed", "As published"]


# ---------------------------------------------------------------- shared pieces

def _style(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_axis(gridColor=GRID, domainColor=GRID, tickColor=GRID, labelColor=INK,
                             titleColor=INK, titleFontWeight="normal", labelFontSize=12, titleFontSize=12)
        .configure_view(strokeWidth=0)
        .configure_legend(labelColor=INK, titleColor=INK, orient="top", labelFontSize=12, titleFontSize=12)
    )


def _record_colour(records: list[str]) -> alt.Scale:
    return alt.Scale(domain=RECORD_ORDER, range=[RECORD_COLOURS[r] for r in RECORD_ORDER])


def show_sources() -> None:
    lines = "  \n".join(f"{publisher}, {title}" for publisher, title in SOURCES)
    st.caption(f"**Sources:**  \n{lines}")


def _latest(data: pd.DataFrame) -> None:
    st.caption(f"{DATE_LABEL}: {data['date'].max().date().isoformat()}")


def _download(frame: pd.DataFrame, key: str, name: str) -> None:
    out = frame.copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    st.download_button(DOWNLOAD_LABEL, data=out.to_csv(index=False).encode("utf-8"), file_name=name,
                       mime="text/csv", key=key)


def _finish(selected: pd.DataFrame, key: str, name: str, data: pd.DataFrame | None = None, after=None) -> None:
    """End of a view: the download, then the page's own summary (if any), then the sources at the very bottom."""
    _download(selected, key, name)
    if after is not None and data is not None:
        st.divider()
        after(data)
    st.divider()
    show_sources()


def _records_picker(data: pd.DataFrame, key: str) -> list[str]:
    available = [r for r in RECORD_ORDER if r in set(data["record"])]
    return st.multiselect("Temperature records", available, default=available, key=key)


def _years_picker(data: pd.DataFrame, key: str, start: int) -> tuple[int, int]:
    low, high = int(data["year"].min()), int(data["year"].max())
    return st.slider("Years", min_value=low, max_value=high, value=(max(start, low), high), key=key)


def _period_text(frame: pd.DataFrame) -> str:
    first, last = frame["date"].min(), frame["date"].max()
    return f"{first:%b %Y}–{last:%b %Y}"


def running_mean(data: pd.DataFrame, column: str = "anomaly_c", months: int = 12) -> pd.Series:
    """Trailing 12-month mean within each record (needs a full 12 months)."""
    ordered = data.sort_values(["record_id", "date"])
    rolled = ordered.groupby("record_id")[column].transform(lambda s: s.rolling(months, min_periods=months).mean())
    return rolled.reindex(data.index)


def record_trends(frame: pd.DataFrame, column: str = "anomaly_c") -> dict[str, Trend]:
    """One straight-line trend per record over the rows given."""
    trends = {}
    for record in RECORD_ORDER:
        part = frame.loc[frame["record"] == record].dropna(subset=[column])
        if len(part) >= 24:
            trends[record] = fit_trend(decimal_year(part["year"], part["month"]), part[column].to_numpy(float))
    return trends


TITLES = {
    "rate": "How fast is the world warming, and do the records agree?",
    "enso": "How much of the year-to-year ups and downs is El Niño and La Niña?",
    "acceleration": "Has warming sped up once El Niño and La Niña are set aside?",
}
SUBTITLES = {
    "rate": "Four independent temperature records, one straight-line rate for any period you choose.",
    "enso": "Global temperature follows the Pacific's El Niño and La Niña cycle with a short delay.",
    "acceleration": "Compare the rate before and after a year you choose, with the El Niño swings taken out.",
}
PATTERNS = {"rate": pattern_rate, "enso": pattern_enso, "acceleration": pattern_acceleration}


def _heading(page: str, number: int, data: pd.DataFrame) -> None:
    banner(f"Question {number} of 3", TITLES[page], SUBTITLES[page], BANNER[page], data, PATTERNS[page](data))


# ---------------------------------------------------------------- view 1

def render_warming_rate(data: pd.DataFrame, intro: str | None = None, after=None) -> None:
    """View 1: the warming rate in each record over a chosen period."""
    _heading("rate", 1, data)
    st.caption("Unit: °C per decade. Temperatures are monthly anomalies in °C, each record measured from its own "
               "1991–2020 average.")
    _latest(data)
    if intro:
        st.markdown(intro)

    left, right = st.columns([2, 3])
    with left:
        first, last = _years_picker(data, "rate_period", FIRST_RATE_YEAR)
    with right:
        chosen = _records_picker(data, "rate_records")

    selected = data.loc[data["year"].between(first, last) & data["record"].isin(chosen)].copy()
    if selected.empty:
        st.info(EMPTY_MESSAGE)
        _finish(selected, "rate_download", "warming_rate_selection.csv")
        return

    trends = record_trends(selected)
    if not trends:
        st.warning("Choose a period of at least two years to calculate a rate.")
        _finish(selected, "rate_download", "warming_rate_selection.csv")
        return
    rates = pd.Series({r: t.rate for r, t in trends.items()})

    m1, m2, m3 = st.columns(3)
    m1.metric("Average rate across the selected records", f"{rates.mean():+.2f} °C per decade")
    m2.metric("Gap between the fastest and slowest record", f"{rates.max() - rates.min():.2f} °C per decade")
    m3.metric("Records shown", f"{len(trends)} of {data['record'].nunique()}")

    # Chart 1: 12-month running means with each record's straight line.
    smooth = data.assign(running=running_mean(data))
    smooth = smooth.loc[smooth.index.isin(selected.index)].dropna(subset=["running"])
    lines = []
    for record, trend in trends.items():
        part = selected.loc[selected["record"] == record]
        when = decimal_year(part["year"], part["month"])
        lines.append(pd.DataFrame({"record": record, "date": [part["date"].min(), part["date"].max()],
                                   "fitted": trend.value_at(np.array([when.min(), when.max()]))}))
    fitted = pd.concat(lines, ignore_index=True)
    colour = alt.Color("record:N", title="Temperature record", scale=_record_colour(chosen),
                       legend=alt.Legend(symbolType="stroke", symbolStrokeWidth=3, labelLimit=260))
    series = alt.Chart(smooth).mark_line(strokeWidth=1.6, opacity=0.9).encode(
        x=alt.X("date:T", title=None),
        y=alt.Y("running:Q", title="12-month mean anomaly (°C, vs 1991–2020)"),
        color=colour,
        tooltip=[alt.Tooltip("record:N", title="Record"), alt.Tooltip("date:T", title="Month ending", format="%b %Y"),
                 alt.Tooltip("running:Q", title="12-month mean (°C)", format="+.2f")],
    )
    trend_lines = alt.Chart(fitted).mark_line(strokeWidth=2.4, strokeDash=[6, 4]).encode(
        x="date:T", y="fitted:Q", color=colour)
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color=MUTED, strokeDash=[2, 3]).encode(y="y:Q")
    st.altair_chart(_style((zero + series + trend_lines).properties(height=340)), width="stretch")
    st.caption("Colours: blue NASA GISTEMP v4, orange NOAA GlobalTemp, green Met Office HadCRUT5, yellow Berkeley "
               "Earth. Solid lines: 12-month running means (each point averages that month and the 11 before it). "
               "Dashed lines: the straight-line trend through every monthly value in the period. "
               "The dotted line marks the 1991–2020 average.")

    # Chart 2 and table: the rates side by side with their 95% intervals.
    table = pd.DataFrame([
        {"Record": r, "From": _period_text(selected.loc[selected["record"] == r]).split("–")[0],
         "To": _period_text(selected.loc[selected["record"] == r]).split("–")[1],
         "Months": t.months, "Rate (°C per decade)": round(t.rate, 3),
         "95% interval": format_interval(t.low, t.high), "low": t.low, "high": t.high}
        for r, t in trends.items()
    ])
    dots = alt.Chart(table).mark_point(filled=True, size=110, opacity=1).encode(
        x=alt.X("Rate (°C per decade):Q", title=f"Warming rate, {first}–{last} (°C per decade)",
                scale=alt.Scale(zero=False), axis=alt.Axis(format=".2f", tickCount=6)),
        y=alt.Y("Record:N", sort=RECORD_ORDER, title=None, axis=alt.Axis(labelLimit=220)),
        color=alt.Color("Record:N", scale=_record_colour(chosen), legend=None),
        tooltip=["Record:N", alt.Tooltip("Rate (°C per decade):Q", format="+.3f"), "95% interval:N", "Months:Q"],
    )
    bars = alt.Chart(table).mark_rule(strokeWidth=2.5, color=INK).encode(
        x="low:Q", x2="high:Q", y=alt.Y("Record:N", sort=RECORD_ORDER))
    st.altair_chart(_style((bars + dots).properties(height=48 * len(table) + 50)), width="stretch")
    st.dataframe(table.drop(columns=["low", "high"]), hide_index=True, width="stretch")

    overlap = max(t.low for t in trends.values()) <= min(t.high for t in trends.values())
    st.markdown(
        f"**What the data show.** Between {first} and {last}, the selected records warm at between "
        f"{rates.min():+.2f} and {rates.max():+.2f} °C per decade. "
        + ("All of their 95% intervals overlap, so within the uncertainty of a straight-line fit the records "
           "agree on the rate." if overlap else
           "Not all of their 95% intervals overlap, so for this period the records differ by more than the "
           "uncertainty of a straight-line fit.")
        + f" The widest interval is ±{max(t.half_width for t in trends.values()):.2f} °C per decade."
    )
    _finish(selected, "rate_download", "warming_rate_selection.csv", data, after)


# ---------------------------------------------------------------- view 2

def enso_fits(selected: pd.DataFrame, lag: int) -> tuple[dict, pd.DataFrame]:
    fits, parts = {}, []
    for record in RECORD_ORDER:
        part = selected.loc[selected["record"] == record]
        if part["oni"].notna().sum() < 36:
            continue
        fit, work = fit_enso(part, lag)
        fits[record] = fit
        parts.append(work)
    return fits, (pd.concat(parts, ignore_index=True) if parts else pd.DataFrame())


def render_enso(data: pd.DataFrame, intro: str | None = None, after=None) -> None:
    """View 2: how much of the year-to-year variation follows El Niño and La Niña."""
    _heading("enso", 2, data)
    st.caption("Unit: °C. Global temperatures are monthly anomalies (vs 1991–2020); the Oceanic Niño Index (ONI) is the "
               "sea-surface temperature anomaly of the central equatorial Pacific (Niño 3.4 region), in °C.")
    _latest(data)
    if intro:
        st.markdown(intro)

    with_oni = data.loc[data["oni"].notna()]
    c1, c2, c3 = st.columns([2, 3, 2])
    with c1:
        first, last = _years_picker(with_oni, "enso_period", int(with_oni["year"].min()))
    with c2:
        chosen = _records_picker(data, "enso_records")
    with c3:
        lag = st.selectbox("Delay after the Pacific (months)", list(range(0, 7)), index=DEFAULT_LAG, key="enso_lag")

    selected = data.loc[data["year"].between(first, last) & data["record"].isin(chosen)].copy()
    if selected.empty:
        st.info(EMPTY_MESSAGE)
        _finish(selected, "enso_download", "el_nino_selection.csv")
        return

    fits, work = enso_fits(selected, lag)
    if not fits:
        st.warning("Choose a period of at least three years from 1950 onward, when the ONI begins.")
        _finish(selected, "enso_download", "el_nino_selection.csv")
        return
    effect = np.mean([f.enso_per_degree for f in fits.values()])
    annual = np.mean([f.annual_share for f in fits.values()])
    monthly = np.mean([f.monthly_share for f in fits.values()])

    m1, m2, m3 = st.columns(3)
    m1.metric(f"Global change per 1 °C of ONI, {lag} months later", f"{effect:+.3f} °C")
    m2.metric("Year-to-year variation explained", f"{annual:.0%}")
    m3.metric("Month-to-month variation explained", f"{monthly:.0%}")

    # Charts: the ONI on top; below, each record's deviation from its long-term curve and the ENSO part.
    # Separate charts (rather than one stacked chart) so each takes the full width on a phone; a fixed
    # time domain and y-axis width keep them aligned.
    ends = [selected["date"].min(), selected["date"].max()]
    span = alt.Scale(domain=[alt.DateTime(year=d.year, month=d.month, date=d.day) for d in ends])
    y_axis = {"minExtent": 44, "maxExtent": 44}
    oni = selected.drop_duplicates("date")[["date", "oni"]].dropna()
    oni["phase"] = np.select([oni["oni"] >= 0.5, oni["oni"] <= -0.5], ["El Niño", "La Niña"], "Neutral")
    oni_chart = alt.Chart(oni).mark_bar(width=1.2).encode(
        x=alt.X("date:T", title=None, scale=span),
        y=alt.Y("oni:Q", title="ONI (°C)", axis=alt.Axis(**y_axis)),
        color=alt.Color("phase:N", title="Pacific phase", scale=alt.Scale(
            domain=["El Niño", "Neutral", "La Niña"], range=[WARM, NEUTRAL_BAR, COOL])),
        tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"), alt.Tooltip("oni:Q", title="ONI (°C)", format="+.2f"),
                 alt.Tooltip("phase:N", title="Phase")],
    ).properties(height=240, title=alt.TitleParams("El Niño and La Niña (Oceanic Niño Index, three-month mean)",
                                                   fontSize=13, anchor="start", color=INK))
    st.altair_chart(_style(oni_chart), width="stretch")

    work = work.sort_values(["record_id", "date"])
    work["deviation_12"] = work.groupby("record_id")["deviation_c"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    work["enso_12"] = work.groupby("record_id")["enso_part_c"].transform(lambda s: s.rolling(12, min_periods=12).mean())
    long = work.melt(id_vars=["date", "record"], value_vars=["deviation_12", "enso_12"], var_name="part", value_name="value")
    long["part"] = long["part"].map({"deviation_12": "Deviation from long-term curve",
                                     "enso_12": "El Niño / La Niña part"})
    long = long.dropna(subset=["value"])
    parts = ["Deviation from long-term curve", "El Niño / La Niña part"]
    shown = [r for r in RECORD_ORDER if r in fits]
    for i, record in enumerate(shown):
        first_panel, last_panel = i == 0, i == len(shown) - 1
        panel = alt.Chart(long.loc[long["record"] == record]).mark_line(strokeWidth=1.6).encode(
            x=alt.X("date:T", title=None, scale=span, axis=alt.Axis(labels=last_panel, ticks=last_panel)),
            y=alt.Y("value:Q", title="°C", scale=alt.Scale(domain=[-0.3, 0.3], clamp=True), axis=alt.Axis(**y_axis)),
            color=alt.Color("part:N", title=None, scale=alt.Scale(domain=parts, range=[INK, WARM]),
                            legend=alt.Legend(labelLimit=400, symbolStrokeWidth=2) if first_panel else None),
            strokeDash=alt.StrokeDash("part:N", legend=None, scale=alt.Scale(domain=parts, range=[[1, 0], [5, 3]])),
            tooltip=[alt.Tooltip("date:T", title="Month ending", format="%b %Y"), alt.Tooltip("part:N", title="Line"),
                     alt.Tooltip("value:Q", title="12-month mean (°C)", format="+.2f")],
        ).properties(
            # Streamlit treats height as the whole chart (title, legend and axis included), so the panel
            # with the legend and the one with the time axis are taller to keep every plot the same size.
            height=150 + (45 if first_panel else 0) + (25 if last_panel else 0),
            title=alt.TitleParams(record, fontSize=13, fontWeight="normal", anchor="start", color=INK))
        st.altair_chart(_style(panel), width="stretch")
    st.caption("Lower panels: 12-month running means. The black line is how far each record sits above or below its "
               "own gently curving long-term trend; the red dashed line is the part of that wobble that matches the "
               "ONI from the chosen number of months earlier.")

    table = pd.DataFrame([
        {"Record": r, "Months used": f.months,
         "Change per 1 °C of ONI (°C)": round(f.enso_per_degree, 3),
         "95% interval": format_interval(f.enso_low, f.enso_high, "+.3f"),
         "Year-to-year variation explained": f"{f.annual_share:.0%}",
         "Month-to-month variation explained": f"{f.monthly_share:.0%}"}
        for r, f in fits.items()
    ])
    st.dataframe(table, hide_index=True, width="stretch")

    shares = [f.annual_share for f in fits.values()]
    st.markdown(
        f"**What the data show.** Between {first} and {last}, global temperature in the selected records is about "
        f"{effect:.2f} °C higher {lag} months after each 1 °C of El Niño warming in the central Pacific, and lower by "
        f"about the same after La Niña. El Niño and La Niña account for {min(shares):.0%}–{max(shares):.0%} of the "
        f"year-to-year variation around the long-term curve; the rest comes from other sources that this dataset "
        f"does not separate."
    )
    _finish(selected, "enso_download", "el_nino_selection.csv", data, after)


# ---------------------------------------------------------------- view 3

def adjusted_series(data: pd.DataFrame, lag: int = DEFAULT_LAG) -> pd.DataFrame:
    """Each record with the El Niño/La Niña part removed, fitted over every month with an ONI value."""
    parts = []
    for record in RECORD_ORDER:
        part = data.loc[data["record"] == record]
        if part["oni"].notna().sum() < 36:
            continue
        _, work = fit_enso(part, lag)
        parts.append(work[["record", "date", "year", "month", "anomaly_c", "adjusted_c"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def period_rates(series: pd.DataFrame, column: str, split: int, start: int = FIRST_RATE_YEAR) -> pd.DataFrame:
    rows = []
    for record in RECORD_ORDER:
        part = series.loc[series["record"] == record]
        before = part.loc[part["year"].between(start, split - 1)]
        after = part.loc[part["year"] >= split]
        if len(before) < 24 or len(after) < 24:
            continue
        t1 = fit_trend(decimal_year(before["year"], before["month"]), before[column].to_numpy(float))
        t2 = fit_trend(decimal_year(after["year"], after["month"]), after[column].to_numpy(float))
        diff, low, high = difference_interval(t1, t2)
        rows.append({"record": record, "before": t1, "after": t2, "change": diff, "change_low": low, "change_high": high,
                     "after_label": f"{after['date'].min():%Y}–{after['date'].max():%b %Y}"})
    return pd.DataFrame(rows)


def render_acceleration(data: pd.DataFrame, intro: str | None = None, after=None) -> None:
    """View 3: the rate before and after a chosen year, with or without the ENSO part."""
    _heading("acceleration", 3, data)
    st.caption("Unit: °C per decade, from monthly anomalies (vs 1991–2020). The earlier period always starts in "
               f"{FIRST_RATE_YEAR}; the later one runs to the latest month.")
    _latest(data)
    if intro:
        st.markdown(intro)

    c1, c2, c3 = st.columns([1, 2, 2])
    with c1:
        split = st.selectbox("Split the period in", SPLIT_YEARS, index=SPLIT_YEARS.index(2010), key="accel_split")
    with c2:
        chosen = _records_picker(data, "accel_records")
    with c3:
        which = st.radio("Temperatures", SERIES, index=0, key="accel_series")

    selected = data.loc[data["record"].isin(chosen)].copy()
    if selected.empty:
        st.info(EMPTY_MESSAGE)
        _finish(selected, "accel_download", "acceleration_selection.csv")
        return

    series = adjusted_series(selected)
    column = "adjusted_c" if which == SERIES[0] else "anomaly_c"
    rates = period_rates(series, column, split)
    if rates.empty:
        st.warning("Not enough months on one side of the split to calculate a rate.")
        _finish(selected, "accel_download", "acceleration_selection.csv")
        return

    rate_before = np.mean([t.rate for t in rates["before"]])
    rate_after = np.mean([t.rate for t in rates["after"]])
    m1, m2, m3 = st.columns(3)
    m1.metric(f"Rate {FIRST_RATE_YEAR}–{split - 1} (average of records)", f"{rate_before:+.2f} °C per decade")
    m2.metric(f"Rate since {split} (average of records)", f"{rate_after:+.2f} °C per decade")
    m3.metric("Change", f"{rate_after - rate_before:+.2f} °C per decade")

    points = []
    for _, row in rates.iterrows():
        for label, trend in ((f"{FIRST_RATE_YEAR}–{split - 1}", row["before"]), (f"Since {split}", row["after"])):
            points.append({"Record": row["record"], "Period": label, "Rate": trend.rate, "low": trend.low,
                           "high": trend.high, "95% interval": format_interval(trend.low, trend.high)})
    points = pd.DataFrame(points)
    periods = [f"{FIRST_RATE_YEAR}–{split - 1}", f"Since {split}"]
    colour = alt.Color("Period:N", title="Period", scale=alt.Scale(domain=periods, range=[EARLIER, LATER]))
    offset = alt.YOffset("Period:N", sort=periods)
    rules = alt.Chart(points).mark_rule(strokeWidth=2.5).encode(
        x=alt.X("low:Q", title="Warming rate (°C per decade)", axis=alt.Axis(format=".2f", tickCount=8)), x2="high:Q",
        y=alt.Y("Record:N", sort=RECORD_ORDER, title=None, axis=alt.Axis(labelLimit=220)), yOffset=offset,
        color=colour)
    dots = alt.Chart(points).mark_point(filled=True, size=100, opacity=1).encode(
        x="Rate:Q", y=alt.Y("Record:N", sort=RECORD_ORDER), yOffset=offset, color=colour,
        tooltip=["Record:N", "Period:N", alt.Tooltip("Rate:Q", title="Rate (°C per decade)", format="+.3f"),
                 "95% interval:N"])
    st.altair_chart(_style((rules + dots).properties(height=70 * len(rates) + 50)), width="stretch")
    st.caption("Dots are the straight-line rates; bars are their 95% intervals. Separate bars mean the two periods' "
               "rates differ by more than the fit's uncertainty.")

    table = pd.DataFrame([
        {"Record": r["record"],
         f"{FIRST_RATE_YEAR}–{split - 1} (°C per decade)": f"{format_rate(r['before'].rate)} ({format_interval(r['before'].low, r['before'].high)})",
         f"Since {split} (°C per decade)": f"{format_rate(r['after'].rate)} ({format_interval(r['after'].low, r['after'].high)})",
         "Change (95% interval)": f"{format_rate(r['change'])} ({format_interval(r['change_low'], r['change_high'])})"}
        for _, r in rates.iterrows()
    ])
    st.dataframe(table, hide_index=True, width="stretch")

    excludes = int(((rates["change_low"] > 0) | (rates["change_high"] < 0)).sum())
    st.markdown(
        f"**What the data show.** {'With El Niño and La Niña removed' if column == 'adjusted_c' else 'As published'}, "
        f"the selected records warm at {min(t.rate for t in rates['after']):+.2f} to "
        f"{max(t.rate for t in rates['after']):+.2f} °C per decade since {split}, compared with "
        f"{min(t.rate for t in rates['before']):+.2f} to {max(t.rate for t in rates['before']):+.2f} °C per decade "
        f"in {FIRST_RATE_YEAR}–{split - 1}. The 95% interval for the change excludes zero in {excludes} of "
        f"{len(rates)} records."
    )
    _finish(selected, "accel_download", "acceleration_selection.csv", data, after)
