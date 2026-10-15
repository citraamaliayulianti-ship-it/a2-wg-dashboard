"""Question 1: how fast is the world warming, and do the records agree?"""

import streamlit as st

from src.loading import get_data
from src.stories import PAGE_ENSO, HOME, page_nav, rate_summary
from src.theme import footer, link_line
from src.views import render_warming_rate


data = get_data()
link_line("rate")

with st.container(key="warming_rate"):
    render_warming_rate(
        data,
        intro=(
            "Each coloured line is one temperature record. Choose a period: a straight line is fitted to each "
            "record's monthly values in it, and its slope is the warming rate in °C per decade. The chart on the "
            "right puts the rates side by side with their 95% intervals, so you can see whether the records agree."
        ),
        after=rate_summary,
    )
page_nav(previous=(HOME, "Overview"), following=(PAGE_ENSO, "El Niño and La Niña"))
footer()
