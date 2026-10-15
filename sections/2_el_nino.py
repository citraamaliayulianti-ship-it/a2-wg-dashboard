"""Question 2: how much of the year-to-year variation is El Niño and La Niña?"""

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
            "The top panel is the Oceanic Niño Index: red bars are El Niño (a warm central "
            "Pacific), blue bars La Niña (a cool one). Below, for each record, the black line shows how far global "
            "temperature sits above or below its long-term curve, and the red dashed line shows the part of that "
            "which matches the Pacific a few months earlier. Where the two lines move together, El Niño and La Niña "
            "explain the ups and downs."
        ),
        after=enso_summary,
    )
page_nav(previous=(PAGE_RATE, "Warming rate"), following=(PAGE_ACCEL, "Has it sped up?"))
footer()
