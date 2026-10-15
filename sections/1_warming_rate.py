"""Research question 1: what is the rate of global surface warming, and do the datasets agree?"""

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
            "Each coloured line is one global surface temperature dataset. For the selected period, an ordinary "
            "least-squares linear trend is fitted to each dataset's monthly anomalies; its slope is the warming rate "
            "in °C per decade. The right-hand chart compares the trends and their 95% confidence intervals to show "
            "whether the datasets agree within uncertainty."
        ),
        after=rate_summary,
    )
page_nav(previous=(HOME, "Overview"), following=(PAGE_ENSO, "ENSO influence"))
footer()
