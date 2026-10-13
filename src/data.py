"""Acquire, transform and validate the dashboard's dataset.

One module holds the whole data pipeline so its lineage is inspectable:

* ``acquire()`` downloads five raw files into ``data/raw/``: four global surface
  temperature records (NASA GISTEMP v4, NOAA GlobalTemp, Met Office HadCRUT5 and
  Berkeley Earth) and NOAA's Oceanic Niño Index (ONI).
* ``transform()`` reads each record in its own published format, moves all four
  onto one common baseline (1991-2020) and joins the ONI, giving one tidy table
  with one row per record per month.
* ``validate()`` stops the pipeline if the table breaks a documented expectation.
* ``main()`` runs the steps. With ``DASHBOARD_SAMPLE_MODE=1`` it works offline:
  it rebuilds ``data/processed.csv`` from the bundled ``data/sample.csv``.

The trend and El Niño calculations live in ``analysis.py``; presentation in ``views.py``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
SAMPLE = ROOT / "data" / "sample.csv"
PROCESSED = ROOT / "data" / "processed.csv"

BASELINE = (1991, 2020)      # WMO climatological standard normal, covered by every record
FIRST_YEAR = 1850
SAMPLE_FIRST_YEAR = 1950     # the sample keeps every month from the start of the ONI record

# Each record: label, raw file, source URL and the baseline it is published on.
RECORDS = {
    "gistemp": {
        "label": "NASA GISTEMP v4",
        "file": "gistemp_v4_global.csv",
        "url": "https://data.giss.nasa.gov/gistemp/tabledata_v4/GLB.Ts+dSST.csv",
        "baseline": "1951-1980",
    },
    "noaa": {
        "label": "NOAA GlobalTemp",
        "file": "noaa_globaltemp_monthly.csv",
        "url": ("https://www.ncei.noaa.gov/access/monitoring/climate-at-a-glance/global/time-series/"
                "globe/land_ocean/1/0/1850-2026/data.csv"),
        "baseline": "1901-2000",
    },
    "hadcrut": {
        "label": "Met Office HadCRUT5",
        "file": "hadcrut5_global_monthly.csv",
        "url": ("https://www.metoffice.gov.uk/hadobs/hadcrut5/data/HadCRUT.5.2.0.0/analysis/diagnostics/"
                "HadCRUT.5.2.0.0.analysis.summary_series.global.monthly.csv"),
        "baseline": "1961-1990",
    },
    "berkeley": {
        "label": "Berkeley Earth",
        "file": "berkeley_hr_global_tavg_monthly.txt",
        "url": "https://storage.googleapis.com/berkeley-earth-temperature-hr/global/Global_TAVG_monthly.txt",
        "baseline": "1951-1980",
    },
}
ONI = {"file": "noaa_oni.txt", "url": "https://www.cpc.ncep.noaa.gov/data/indices/oni.ascii.txt"}
# ONI is a three-month running mean; each season is dated to its middle month.
SEASON_MONTH = {"DJF": 1, "JFM": 2, "FMA": 3, "MAM": 4, "AMJ": 5, "MJJ": 6,
                "JJA": 7, "JAS": 8, "ASO": 9, "SON": 10, "OND": 11, "NDJ": 12}

COLUMNS = [
    "date",
    "year",
    "month",
    "record_id",
    "record",
    "anomaly_c",
    "lower_95_c",
    "upper_95_c",
    "reported_anomaly_c",
    "reported_baseline",
    "oni",
]
STRING_COLUMNS = {"record_id": str, "record": str, "reported_baseline": str}
REQUIRED = ["date", "year", "month", "record_id", "record", "anomaly_c", "reported_anomaly_c", "reported_baseline"]
RANGES = {
    "anomaly_c": (-3.0, 3.0),
    "lower_95_c": (-3.5, 3.0),
    "upper_95_c": (-3.0, 3.5),
    "reported_anomaly_c": (-3.0, 3.0),
    "oni": (-3.5, 3.5),
}


def read_dataset(path: Path) -> pd.DataFrame:
    """Read a dataset CSV keeping identifiers as text; only empty fields are missing."""
    return pd.read_csv(path, dtype=STRING_COLUMNS, keep_default_na=False, na_values=[""])


# ---------------------------------------------------------------- acquisition

def _download(url: str, destination: Path) -> None:
    try:
        with urlopen(url, timeout=60) as response:
            payload = response.read()
    except (URLError, TimeoutError) as error:
        sys.exit(
            f"Could not fetch {url}: {error}. Check your connection, or download the URL "
            f"in a browser and save it as {destination}. To work offline, rerun with "
            "DASHBOARD_SAMPLE_MODE=1 to use the bundled sample."
        )
    if len(payload) < 1000:
        sys.exit(f"Unexpectedly short response from {url}: {payload[:200]!r}")
    destination.write_bytes(payload)
    print(f"Wrote {destination} ({len(payload):,} bytes)")


def acquire() -> None:
    """Download the five raw files to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for spec in [*RECORDS.values(), ONI]:
        _download(spec["url"], RAW_DIR / spec["file"])


# ---------------------------------------------------------------- readers
# Each reader returns year, month, reported anomaly (°C) and, where published,
# the lower and upper ends of a 95% uncertainty range on the record's own baseline.

def _read_gistemp(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, skiprows=1, na_values=["***", "****"])
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    long = table.melt(id_vars="Year", value_vars=months, var_name="name", value_name="reported")
    long["month"] = long["name"].map({name: i + 1 for i, name in enumerate(months)})
    long = long.rename(columns={"Year": "year"}).dropna(subset=["reported"])
    return long.assign(lower=np.nan, upper=np.nan)[["year", "month", "reported", "lower", "upper"]]


def _read_noaa(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, comment="#", dtype={"Date": str})
    table.columns = ["date", "reported"]
    return pd.DataFrame({
        "year": table["date"].str[:4].astype(int),
        "month": table["date"].str[4:6].astype(int),
        "reported": pd.to_numeric(table["reported"], errors="raise"),
        "lower": np.nan,
        "upper": np.nan,
    })


def _read_hadcrut(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path)
    table.columns = ["time", "reported", "lower", "upper"]
    return pd.DataFrame({
        "year": table["time"].str[:4].astype(int),
        "month": table["time"].str[5:7].astype(int),
        "reported": table["reported"],
        "lower": table["lower"],
        "upper": table["upper"],
    })


def _read_berkeley(path: Path) -> pd.DataFrame:
    """Monthly anomaly and its 95% uncertainty (columns 3-4).

    The high-resolution file has one section. Berkeley's older files had a second section
    (water temperature under sea ice) that starts again from 1850; reading stops there.
    """
    rows = []
    for line in path.read_text(encoding="latin-1").splitlines():
        parts = line.split()
        if not parts or line.lstrip().startswith("%"):
            continue
        year, month = int(parts[0]), int(parts[1])
        if rows and (year, month) <= (rows[-1][0], rows[-1][1]):
            break  # the second section (water temperature under sea ice) starts again from 1850
        if parts[2].lower() == "nan":
            continue
        anomaly, uncertainty = float(parts[2]), float(parts[3])
        rows.append((year, month, anomaly, anomaly - uncertainty, anomaly + uncertainty))
    return pd.DataFrame(rows, columns=["year", "month", "reported", "lower", "upper"])


READERS = {"gistemp": _read_gistemp, "noaa": _read_noaa, "hadcrut": _read_hadcrut, "berkeley": _read_berkeley}


def _read_oni(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=r"\s+")
    return pd.DataFrame({
        "year": table["YR"].astype(int),
        "month": table["SEAS"].map(SEASON_MONTH).astype(int),
        "oni": table["ANOM"].astype(float),
    })


# ---------------------------------------------------------------- transformation

def rebase(frame: pd.DataFrame) -> pd.DataFrame:
    """Move one record onto the 1991-2020 baseline, month by month.

    Subtracting each calendar month's 1991-2020 mean (rather than one number)
    also removes small seasonal differences left by the record's own baseline.
    """
    in_base = frame["year"].between(*BASELINE)
    offsets = frame.loc[in_base].groupby("month")["reported"].mean()
    if len(offsets) != 12 or frame.loc[in_base].groupby("month").size().min() < 30:
        raise ValueError("A record does not cover every month of the 1991-2020 baseline")
    shift = frame["month"].map(offsets)
    return frame.assign(
        anomaly_c=frame["reported"] - shift,
        lower_95_c=frame["lower"] - shift,
        upper_95_c=frame["upper"] - shift,
    )


def transform(raw_dir: Path) -> pd.DataFrame:
    """Read the raw files and build the tidy record-month table used by the app."""
    parts = []
    for record_id, spec in RECORDS.items():
        frame = READERS[record_id](raw_dir / spec["file"])
        frame = frame.loc[frame["year"] >= FIRST_YEAR]
        frame = rebase(frame).assign(record_id=record_id, record=spec["label"], reported_baseline=spec["baseline"])
        last = frame.sort_values(["year", "month"]).iloc[-1]
        print(f"{spec['label']}: {len(frame)} months, {int(frame['year'].min())}-01 to "
              f"{int(last['year'])}-{int(last['month']):02d}")
        parts.append(frame)
    table = pd.concat(parts, ignore_index=True)
    oni = _read_oni(raw_dir / ONI["file"])
    table = table.merge(oni, on=["year", "month"], how="left", validate="many_to_one")
    table["date"] = (pd.to_datetime(dict(year=table["year"], month=table["month"], day=1))
                     + pd.offsets.MonthEnd(0))
    table = table.rename(columns={"reported": "reported_anomaly_c"})
    for column in ["anomaly_c", "lower_95_c", "upper_95_c", "reported_anomaly_c"]:
        table[column] = table[column].round(4)
    return table[COLUMNS]


def finalise(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise").dt.strftime("%Y-%m-%d")
    frame["year"] = frame["year"].astype(int)
    frame["month"] = frame["month"].astype(int)
    order = {record_id: i for i, record_id in enumerate(RECORDS)}
    frame = frame.assign(_order=frame["record_id"].map(order))
    return frame.sort_values(["_order", "date"]).drop(columns="_order").reset_index(drop=True)[COLUMNS]


def validate(frame: pd.DataFrame, minimum_months: int = 600) -> None:
    """Stop the pipeline if the dataset breaks a documented expectation."""
    missing_columns = set(COLUMNS).difference(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")
    if frame.duplicated(["date", "record_id"]).any():
        raise ValueError("Duplicate date and record keys found")
    if frame[REQUIRED].isna().any().any():
        raise ValueError("Unexpected missing values in a required column")
    if not set(frame["record_id"]).issubset(RECORDS):
        raise ValueError(f"Unexpected record: {sorted(set(frame['record_id']) - set(RECORDS))}")
    labels = frame.groupby("record_id")["record"].unique()
    if any(len(v) != 1 or v[0] != RECORDS[k]["label"] for k, v in labels.items()):
        raise ValueError("Record labels do not match their identifiers")
    dates = pd.to_datetime(frame["date"], errors="raise")
    if not ((dates.dt.year == frame["year"]) & (dates.dt.month == frame["month"])).all():
        raise ValueError("Year and month must match the date")
    if not dates.dt.is_month_end.all():
        raise ValueError("Each date must be the last day of its month")
    if frame["year"].min() < FIRST_YEAR or dates.max() > pd.Timestamp.today() + pd.offsets.MonthEnd(0):
        raise ValueError("Dates outside the documented period")
    for column, (low, high) in RANGES.items():
        values = pd.to_numeric(frame[column], errors="raise").dropna()
        if not values.between(low, high).all():
            raise ValueError(f"{column} outside its plausible range {low} to {high}")
    bounded = frame.dropna(subset=["lower_95_c", "upper_95_c"])
    if not ((bounded["lower_95_c"] <= bounded["anomaly_c"] + 1e-9) & (bounded["anomaly_c"] <= bounded["upper_95_c"] + 1e-9)).all():
        raise ValueError("An anomaly lies outside its own uncertainty range")
    for record_id, part in frame.groupby("record_id"):
        if len(part) < minimum_months:
            raise ValueError(f"{record_id} has only {len(part)} months")
        steps = (part["year"] * 12 + part["month"]).sort_values().diff().dropna()
        if (steps != 1).any():
            raise ValueError(f"{record_id} has gaps between months")
        base = part.loc[part["year"].between(*BASELINE)]
        if base.groupby("month")["anomaly_c"].mean().abs().max() > 0.001:
            raise ValueError(f"{record_id} is not centred on the 1991-2020 baseline")
    oni = frame.loc[frame["year"] >= 1950].dropna(subset=["oni"])
    if oni.groupby("date")["oni"].nunique().max() > 1:
        raise ValueError("ONI differs between records for the same month")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--acquire", action="store_true", help="Refresh data/raw/ from the publishers and stop")
    parser.add_argument("--validate", action="store_true", help="Validate data/processed.csv and stop")
    parser.add_argument("--refresh-sample", action="store_true",
                        help="Live mode only: also rewrite data/sample.csv as every month from 1950 onward")
    args = parser.parse_args()
    sample_mode = os.getenv("DASHBOARD_SAMPLE_MODE", "0") == "1"
    if args.acquire:
        if sample_mode:
            parser.error("Unset DASHBOARD_SAMPLE_MODE before acquiring live data")
        acquire()
        return
    if args.validate:
        validate(read_dataset(PROCESSED))
        print("Processed data valid")
        return
    if sample_mode:
        if args.refresh_sample:
            parser.error("--refresh-sample needs live mode; unset DASHBOARD_SAMPLE_MODE")
        # The sample is already rebased and joined, so offline mode only re-checks and rewrites it.
        frame = finalise(read_dataset(SAMPLE))
    else:
        needed = [spec["file"] for spec in RECORDS.values()] + [ONI["file"]]
        if not all((RAW_DIR / name).exists() for name in needed):
            acquire()
        frame = finalise(transform(RAW_DIR))
    validate(frame)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(PROCESSED, index=False)
    print(f"Validated and wrote {len(frame)} rows to {PROCESSED}")
    if args.refresh_sample:
        subset = frame.loc[frame["year"] >= SAMPLE_FIRST_YEAR]
        validate(subset)
        subset.to_csv(SAMPLE, index=False)
        print(f"Wrote {len(subset)} sample rows ({SAMPLE_FIRST_YEAR} onward) to {SAMPLE}")


if __name__ == "__main__":
    main()
