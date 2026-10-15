"""Question 3: has warming sped up once El Niño and La Niña are set aside?"""

import streamlit as st

from src.loading import get_data
from src.stories import PAGE_CONCLUSION, PAGE_ENSO, acceleration_summary, page_nav
from src.theme import footer, link_line
from src.views import render_acceleration


data = get_data()
link_line("acceleration")

with st.container(key="acceleration"):
    render_acceleration(
        data,
        intro=(
            "Choose a year to split the record. For each record, a straight line is fitted "
            "before and after that year, and the two rates are shown with their 95% intervals. By default the "
            "El Niño and La Niña part (page 2) is removed first, so that a warm El Niño year at the end of a short "
            "period does not inflate the rate. Switch to 'As published' to compare."
        ),
        after=acceleration_summary,
    )
page_nav(previous=(PAGE_ENSO, "El Niño and La Niña"), following=(PAGE_CONCLUSION, "Conclusion"))
footer()
