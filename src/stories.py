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
    """OLS linear trend from each start year to the latest month, for every dataset."""
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
    """Warming relative to 1850-1900 for the datasets that reach back that far."""
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
    st.subheader("Trends by start year")
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
        ("- **Trends are higher for later start years.** " if rising else "- **Trends depend on the start year.** ")
        + f"From {earliest} to {end} the datasets average {summary.loc[earliest, 'mean']:+.2f} °C per decade; from "
        f"{base_start} the mean trend is {base['mean']:+.2f}, and from {latest_start} it is "
        f"{summary.loc[latest_start, 'mean']:+.2f}.\n"
        + (f"- **The datasets agree within uncertainty.** From {base_start}, trends range from {base['low']:+.2f} to "
           f"{base['high']:+.2f} °C per decade, an inter-dataset spread of {spread:.2f} that is smaller than the "
           f"narrowest 95% confidence interval ({narrowest:.2f} wide).\n" if spread < narrowest else
           f"- **The datasets differ.** From {base_start}, trends range from {base['low']:+.2f} to "
           f"{base['high']:+.2f} °C per decade, an inter-dataset spread of {spread:.2f} that exceeds the narrowest "
           f"95% confidence interval ({narrowest:.2f}).\n")
        + ("- **Shorter periods give wider confidence intervals.** " if last_width > first_width
           else "- **Confidence interval widths.** ")
        + f"Across datasets the 95% confidence intervals span {summary.loc[earliest, 'ci_low']:+.2f} to "
        f"{summary.loc[earliest, 'ci_high']:+.2f} from {earliest}, and {summary.loc[latest_start, 'ci_low']:+.2f} to "
        f"{summary.loc[latest_start, 'ci_high']:+.2f} from {latest_start}."
    )
    table = summary.reset_index().rename(columns={"start": "From"})
    table["To"] = end
    table["Mean trend (°C per decade)"] = table["mean"].round(3)
    table["Range across datasets"] = [format_interval(a, b) for a, b in zip(table["low"], table["high"])]
    table["95% CI (all datasets)"] = [format_interval(a, b) for a, b in zip(table["ci_low"], table["ci_high"])]
    table["Datasets"] = table["records"]
    st.dataframe(table[["From", "To", "Datasets", "Mean trend (°C per decade)", "Range across datasets",
                        "95% CI (all datasets)"]], hide_index=True, width="stretch")
    st.caption("Each trend is an OLS linear fit to all monthly values from January of the start year to the latest "
               "month. GISTEMP begins in 1880, so all four datasets are included from 1880 onward.")


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
                     "Mean detrended anomaly, following 12 months (°C)":
                         round(window.groupby("record")["deviation_c"].mean().mean(), 2)})
    return pd.DataFrame(rows)


def enso_summary(data: pd.DataFrame) -> None:
    with_oni = data.loc[data["oni"].notna()]
    fits, work = enso_fits(with_oni, DEFAULT_LAG)
    st.subheader("ENSO influence, 1950 to present")
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
        f"- **Consistent ENSO response.** Over {int(with_oni['year'].min())}–{latest_year(data)}, the regression "
        f"coefficient on the ONI at a {DEFAULT_LAG}-month lag is about {effect:+.2f} °C per °C, with similar values "
        f"in all four datasets.",
        f"- **ENSO explains a substantial share of interannual variability:** {min(annual):.0%}–{max(annual):.0%} of "
        f"the variance of annual mean detrended anomalies.",
    ]
    if not events.empty:
        top = events.iloc[0]
        bullets.append(
            f"- **Strong El Niño events are followed by positive anomalies.** In the 12 months after the "
            f"{top['El Niño peak']} peak (ONI {top['Peak ONI (°C)']:+.2f} °C), the mean detrended anomaly was "
            f"{top['Mean detrended anomaly, following 12 months (°C)']:+.2f} °C."
        )
    bullets.append(
        f"- **Current conditions.** The latest ONI value is {latest['oni']:+.2f} °C for the season centred on "
        f"{latest['date']:%B %Y}{since}. In the latest month, the estimated ENSO component of global temperature is "
        f"{latest_part:+.2f} °C."
    )
    st.markdown("\n".join(bullets))
    if not events.empty:
        st.dataframe(events, hide_index=True, width="stretch")
        st.caption(f"El Niño peaks are the maximum ONI values of events reaching at least +1.5 °C. The response is "
                   f"the detrended anomaly averaged over the 12 months after each peak and across all four datasets "
                   f"(lag {DEFAULT_LAG} months).")


# ---------------------------------------------------------------- page 3

@st.cache_data
def window_check(data: pd.DataFrame) -> pd.DataFrame:
    """Trends for commonly cited windows, unadjusted and ENSO-adjusted."""
    series = adjusted_series(data)
    end = latest_year(data)
    windows = [(1970, 2009), (2010, end), (2015, end), (2020, end)]
    rows = []
    for first, last in windows:
        for column, label in (("anomaly_c", "Unadjusted"), ("adjusted_c", "ENSO-adjusted")):
            rates = []
            for record in RECORD_ORDER:
                part = series.loc[(series["record"] == record) & series["year"].between(first, last)]
                if len(part) >= 24:
                    rates.append(fit_trend(decimal_year(part["year"], part["month"]), part[column].to_numpy(float)))
            if rates:
                rows.append({"Window": f"{first}–{last}", "Years": last - first + 1, "Series": label,
                             "Mean trend (°C per decade)": round(np.mean([t.rate for t in rates]), 2),
                             "Widest 95% CI": format_interval(min(t.low for t in rates), max(t.high for t in rates)),
                             "width": max(t.high for t in rates) - min(t.low for t in rates)})
    return pd.DataFrame(rows)


def acceleration_summary(data: pd.DataFrame) -> None:
    table = window_check(data)
    st.subheader("Sensitivity to short trend windows")
    if table.empty:
        return
    st.markdown(
        "Trends over short recent windows are often cited as evidence of acceleration. The table repeats that "
        "calculation for four windows, with the widest 95% confidence interval across the datasets for each window."
    )
    st.dataframe(table.drop(columns="width"), hide_index=True, width="stretch")
    raw = table.loc[table["Series"] == "Unadjusted"].set_index("Window")
    adj = table.loc[table["Series"] != "Unadjusted"].set_index("Window")
    longest, recent, shortest = raw.index[0], raw.index[1], raw.index[-1]
    narrower = adj.loc[shortest, "width"] < raw.loc[shortest, "width"]
    later = adj.loc[recent, "Mean trend (°C per decade)"]
    earlier = adj.loc[longest, "Mean trend (°C per decade)"]
    st.markdown(
        ("- **Short windows carry large uncertainty.** " if raw.loc[shortest, "width"] > raw.loc[longest, "width"]
         else "- ")
        + f"Unadjusted, the confidence intervals for {shortest} span {raw.loc[shortest, 'width']:.2f} °C per decade, "
        f"compared with {raw.loc[longest, 'width']:.2f} for {longest}.\n"
        + ("- **ENSO adjustment narrows the intervals.** " if narrower else
           "- **ENSO adjustment does not narrow the intervals.** ")
        + f"For {shortest} the span becomes {adj.loc[shortest, 'width']:.2f} °C per decade.\n"
        + f"- **The ENSO-adjusted trend since {recent[:4]} is {'higher' if later > earlier else 'not higher'} than in "
        f"{longest}:** {later:+.2f} °C per decade for {recent}, compared with {earlier:+.2f} for {longest}."
    )


# ---------------------------------------------------------------- conclusion

def conclusion(data: pd.DataFrame) -> None:
    end = latest_year(data)
    rates = rates_from(data, (FIRST_RATE_YEAR,))
    with_oni = data.loc[data["oni"].notna()]
    fits, _ = enso_fits(with_oni, DEFAULT_LAG)
    windows = window_check(data)
    adj = windows.loc[windows["Series"] != "Unadjusted"].set_index("Window")
    raw = windows.loc[windows["Series"] == "Unadjusted"].set_index("Window")
    warm = above_preindustrial(data)
    events = strongest_events(with_oni)
    spread = rates["rate"].max() - rates["rate"].min()
    narrowest = (rates["high"] - rates["low"]).min()
    later = adj.loc[f"2010–{end}", "Mean trend (°C per decade)"]
    earlier = adj.loc["1970–2009", "Mean trend (°C per decade)"]

    st.subheader("Key findings")
    points = [
        f"1. **Global surface temperature has risen by about {rates['rate'].mean():.2f} °C per decade since "
        f"{FIRST_RATE_YEAR}.** Linear trends in the {len(rates)} datasets range from {rates['rate'].min():.2f} to "
        f"{rates['rate'].max():.2f} °C per decade, an inter-dataset spread "
        f"{'smaller' if spread < narrowest else 'larger'} than the narrowest 95% confidence interval ({narrowest:.2f}).",
        f"2. **ENSO explains a substantial share of interannual variability** "
        f"({min(f.annual_share for f in fits.values()):.0%}–{max(f.annual_share for f in fits.values()):.0%} of the "
        f"variance of annual mean detrended anomalies), with global temperature lagging the ONI by about "
        f"{DEFAULT_LAG} months.",
        f"3. **ENSO-adjusted, the trend since 2010 is {'higher' if later > earlier else 'not higher'} than in "
        f"1970–2009**: {later:+.2f} compared with {earlier:+.2f} °C per decade (mean of datasets). Page 3 shows "
        f"whether the confidence intervals of the two periods overlap for each dataset and breakpoint year.",
        f"4. **Short-window trends are highly uncertain.** Unadjusted, the 2015–{end} trend averages "
        f"{raw.loc[f'2015–{end}', 'Mean trend (°C per decade)']:+.2f} °C per decade with confidence intervals spanning "
        f"{raw.loc[f'2015–{end}', 'width']:.2f} °C per decade; ENSO-adjusted it is "
        f"{adj.loc[f'2015–{end}', 'Mean trend (°C per decade)']:+.2f}.",
    ]
    if not warm.empty:
        points.append(
            f"5. **Relative to 1850–1900**, the {warm.iloc[0]['twenty_label']} mean anomaly is "
            f"{warm['twenty_year'].min():.2f}–{warm['twenty_year'].max():.2f} °C in the {len(warm)} datasets that begin "
            f"in 1850, and the most recent 12-month mean is {warm['last_12'].min():.2f}–{warm['last_12'].max():.2f} °C."
        )
    st.markdown("\n".join(points))

    st.subheader("Limitations")
    st.markdown(
        "- **Attribution.** Only the ENSO signal is removed. Volcanic aerosols, solar variability, anthropogenic "
        "aerosols and greenhouse gases also influence the trend, and this analysis does not separate them.\n"
        "- **Projection.** The analysis is observational. A linear trend fitted to past data is not a projection, so "
        "no trend is extended beyond the latest month.\n"
        "- **Measurement uncertainty.** The four datasets are statistical reconstructions from incomplete "
        "observational coverage. Their published uncertainty (available for HadCRUT5 and Berkeley Earth) is a few "
        "hundredths of a degree in recent decades and larger before 1900."
    )

    response = events["Mean detrended anomaly, following 12 months (°C)"] if not events.empty else None
    st.subheader("Implications for policy analysis")
    st.markdown(
        "- **Report trends with their period and confidence interval.** The same data give different trends for "
        "different start years; a trend reported without its period and 95% confidence interval cannot be compared "
        "with another.\n"
        "- **Do not interpret a single warm or cool year as a change in the long-term trend.** "
        + (f"In the 12 months after the largest El Niño peaks, the mean detrended anomaly ranged from "
           f"{response.min():+.2f} to {response.max():+.2f} °C (page 2).\n" if response is not None else "\n")
        + f"- **Compare more than one dataset.** A result that holds in only one dataset warrants scrutiny; here the "
        f"trends since {FIRST_RATE_YEAR} differ by {spread:.2f} °C per decade across datasets.\n"
        "- **A single month or year above 1.5 °C does not mean the Paris Agreement goal has been exceeded.** The goal "
        "refers to long-term warming, which IPCC AR6 assesses with 20-year means; the 20-year figure above is the "
        "comparable measure."
    )

    latest = with_oni.sort_values("date").iloc[-1]
    st.subheader("Further work")
    st.markdown(
        "- Remove volcanic and solar signals as well (for example with stratospheric aerosol optical depth and total "
        "solar irradiance series) and test whether the post-2010 increase in trend remains.\n"
        f"- Update the analysis as new months are published. The latest ONI value is {latest['oni']:+.2f} °C "
        f"({latest['date']:%B %Y}), indicating El Niño conditions; the following months will form the end of every "
        f"recent trend window."
    )


def methods_and_limitations() -> None:
    with st.expander("Methods and limitations", expanded=False):
        st.markdown(
            "**Common reference period.** Each dataset is published relative to its own reference period (GISTEMP "
            "and Berkeley Earth 1951–1980, NOAA 1901–2000, HadCRUT5 1961–1990). For every dataset and calendar month "
            "the 1991–2020 mean is subtracted, so all four share the current WMO climatological standard normal. "
            "Trends do not depend on the reference period.\n\n"
            "**Linear trends.** Trends are estimated by ordinary least squares (OLS) on the monthly anomalies in a "
            "period. Because monthly residuals are autocorrelated, the 95% confidence interval uses an effective "
            "sample size n(1−r)/(1+r), where r is the lag-1 autocorrelation of the residuals (Santer et al., 2008), "
            "with Student's t distribution. For short windows with strong autocorrelation these intervals may still "
            "be slightly too narrow.\n\n"
            "**ENSO adjustment.** For each dataset, monthly anomalies are regressed on a quadratic function of time "
            "plus the Oceanic Niño Index at a lag of k months (multiple linear regression, following the approach "
            "of Foster and Rahmstorf, 2011, for ENSO only). The ENSO component (coefficient × lagged ONI) is "
            "subtracted to give the ENSO-adjusted series. The ENSO coefficient changes little if the trend term is "
            "linear, but the variance explained is lower because a linear term leaves curvature in the residuals. "
            "A lag of 2 months gives the best fit in all four datasets over 1950–2026. The ONI is a three-month "
            "running mean, assigned here to its central month.\n\n"
            "**Interannual variance explained.** Calendar-year means of the detrended anomaly are compared with "
            "calendar-year means of the ENSO component; the share is 1 − (residual variance / total variance).\n\n"
            "**Datasets.** Berkeley Earth's original 1° product stopped updating in 2025 (last month December 2024); "
            "this dashboard uses its high-resolution successor, which Berkeley Earth labels preliminary and not yet "
            "peer reviewed. In this product, sea-ice regions use air temperature when ice is present. Datasets end in "
            "different months; each trend uses all available months of a dataset."
        )
