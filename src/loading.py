"""Load the dashboard dataset once and share it across pages."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample.csv"
PROCESSED = ROOT / "data" / "processed.csv"

TITLE = "How fast is the world warming?"
QUESTION = (
    "How fast is the world warming once El Niño and La Niña are set aside, and do the major "
    "temperature records agree on the rate?"
)
AUDIENCE = (
    "Climate policy analysts in government and international agencies who track progress "
    "against temperature goals"
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    frame = pd.read_csv(
        path,
        dtype={"record_id": str, "record": str, "reported_baseline": str},
        keep_default_na=False,
        na_values=[""],
        parse_dates=["date"],
    )
    return frame.sort_values(["record_id", "date"]).reset_index(drop=True)


def data_path() -> Path:
    """The bundled sample in offline mode (or before the pipeline has run), otherwise the processed file."""
    sample_mode = os.getenv("DASHBOARD_SAMPLE_MODE", "0") == "1"
    if sample_mode or not PROCESSED.exists():
        return SAMPLE
    return PROCESSED


def get_data() -> pd.DataFrame:
    return load_data(str(data_path()))
