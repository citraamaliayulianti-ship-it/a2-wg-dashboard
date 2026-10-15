"""Research question 3: has global warming accelerated once the ENSO signal is removed?"""

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
            "Select a breakpoint year. For each dataset, separate OLS linear trends are fitted before and after "
            "that year and shown with their 95% confidence intervals. By default the ENSO component (page 2) is "
            "removed first, so that El Niño conditions near the end of a short window do not bias the trend "
            "upward. Select 'Unadjusted' to compare with the published series."
        ),
        after=acceleration_summary,
    )
page_nav(previous=(PAGE_ENSO, "ENSO influence"), following=(PAGE_CONCLUSION, "Conclusion"))
footer()
