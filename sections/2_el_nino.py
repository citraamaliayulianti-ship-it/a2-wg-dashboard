"""Research question 2: how much interannual variability in global temperature does ENSO explain?"""

import streamlit as st

from src.loading import get_data
from src.stories import PAGE_ACCEL, PAGE_RATE, enso_summary, page_nav
from src.theme import footer, link_line
from src.views import render_enso


data = get_data()
link_line("enso")

with st.container(key="enso_effect"):
    render_enso(
        data,
        intro=(
            "The top panel shows the Oceanic Niño Index (ONI): red bars mark El Niño conditions and blue bars La "
            "Niña conditions in the equatorial Pacific. Below, for each dataset, the black line is the detrended "
            "global temperature anomaly and the red dashed line is its ENSO component, estimated by regression on "
            "the lagged ONI. Where the two lines coincide, ENSO accounts for the interannual variability."
        ),
        after=enso_summary,
    )
page_nav(previous=(PAGE_RATE, "Warming rate"), following=(PAGE_ACCEL, "Acceleration"))
footer()
