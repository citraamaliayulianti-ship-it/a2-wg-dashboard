# How fast is the world warming?

A Streamlit dashboard for **climate policy analysts in government and international agencies who track
progress against temperature goals**. It asks:

> How fast is the world warming once El Niño and La Niña are set aside, and do the major temperature
> records agree on the rate?

## Pages

| Page | File | What it shows |
|---|---|---|
| Overview | `home.py` | The question and audience over the warming stripes; annual global temperature compared with 1850–1900 for a selected record, with its 20-year average; the three questions; the four records and the ONI; data access |
| 1. Warming rate | `sections/1_warming_rate.py` | View `warming_rate`: 12-month means and straight-line trends for any period and set of records beside the rates with 95% intervals, and rates from different start years |
| 2. El Niño and La Niña | `sections/2_el_nino.py` | View `enso_effect`: the Oceanic Niño Index beside each record's deviation from its long-term curve, the effect per 1 °C of ONI, the share of variation explained, and the response after the largest El Niño peaks |
| 3. Has it sped up? | `sections/3_acceleration.py` | View `acceleration`: annual means with the two straight lines beside each record's rates before and after a chosen year, with or without the El Niño/La Niña part, and a check of commonly quoted short windows |
| Conclusion | `sections/4_conclusion.py` | What the data show, what they cannot tell us, what this means for policy analysts, next steps, methods and limitations |

`app.py` holds the navigation (`st.navigation`, menu at the top).

## Data

Five files, downloaded on 13 October 2026 and kept in `data/raw/`:

| Source | Publisher | Baseline as published | Licence |
|---|---|---|---|
| GISTEMP v4, global land-ocean monthly means | NASA Goddard Institute for Space Studies | 1951–1980 | Public domain |
| Climate at a Glance, global land and ocean (NOAAGlobalTemp) | NOAA National Centers for Environmental Information | 1901–2000 | Public domain |
| HadCRUT.5.2.0.0 global monthly summary series | Met Office Hadley Centre and CRU, University of East Anglia | 1961–1990 | Open Government Licence v3.0 |
| Global_TAVG_monthly (high-resolution product) | Berkeley Earth | 1951–1980 | CC BY-NC 4.0 |
| Oceanic Niño Index (ONI) | NOAA Climate Prediction Center | centred 30-year periods | Public domain |

Berkeley Earth's older 1° product (`Land_and_Ocean_complete.txt`) stopped updating in 2025 and ends in
December 2024, so its high-resolution successor is used. Berkeley Earth labels that file preliminary and not
yet peer reviewed; its rates match the other three records closely.

`data/processed.csv` has one row per record per month (8,000+ rows). `data/sample.csv` keeps every month from
1950 onward (the start of the ONI), enough to run every view offline.

## Running

```bash
python -m venv .venv && source .venv/bin/activate      # Python 3.11
pip install -r requirements.txt
python -m src.data --acquire        # optional: download the five raw files again
python -m src.data                  # rebuild and validate data/processed.csv from data/raw/
streamlit run app.py
```

Offline: `DASHBOARD_SAMPLE_MODE=1 python -m src.data` rebuilds `data/processed.csv` from the sample, and
`DASHBOARD_SAMPLE_MODE=1 streamlit run app.py` runs the app from it. `python -m src.data --refresh-sample`
rewrites the sample from the full data. Tests: `pytest`.

## Code

`src/data.py` holds acquisition, transformation (one reader per publisher's format, the common 1991–2020
baseline, the ONI join) and validation. `src/analysis.py` holds the calculations: straight-line rates with
autocorrelation-adjusted 95% intervals, and the El Niño regression. Presentation lives in `src/views.py`
(the three assessed views), `src/stories.py` (data-only page summaries, the conclusion and methods),
`src/overview.py` (overview page parts), `src/theme.py` (page style, section headings, source lines, footer and
warming stripes) and `src/loading.py` (cached data loading).

## Methods in brief

- **Common baseline.** For each record and calendar month the 1991–2020 mean is subtracted. Rates do not
  depend on the baseline.
- **Rates.** Ordinary least squares on monthly anomalies; the 95% interval uses an effective sample size
  n(1−r)/(1+r) from the lag-1 autocorrelation r of the residuals, with Student's t (Santer et al., 2008).
- **El Niño and La Niña.** Anomaly = quadratic curve in time + c × ONI shifted by a delay (default 2 months,
  the best fit for all four records over 1950–2026). The ENSO part c × ONI is subtracted for the adjusted
  series. Volcanic, solar and aerosol effects are not removed.
- **Charts.** Altair 6.3.0. Record colours follow the record across pages (blue GISTEMP, orange NOAA,
  aqua HadCRUT5, yellow Berkeley Earth), checked for colour-vision deficiency; every chart also has a
  legend and a table.

## Design

The layout follows the pattern of public climate-indicator portals such as the IMF Climate Change Indicators
Dashboard: a solid blue top bar, a full-width picture under it, a line of links to every section, sections with
a centred title and description, controls on the right above the charts, charts on white panels with a short
source line under them, a right-aligned download button, and a blue footer. Only the pattern is borrowed; every
picture and chart is drawn from this dashboard's own data (the picture is the warming stripes of the NOAA
record). The style lives in `src/theme.py` (CSS injected once from `app.py`) and the colours of the page in
`.streamlit/config.toml`.
