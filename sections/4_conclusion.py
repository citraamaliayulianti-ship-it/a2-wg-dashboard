"""Conclusion: what the evidence shows, and what it does not."""

import streamlit as st

from src.loading import QUESTION, get_data
from src.stories import PAGE_ACCEL, HOME, conclusion, methods_and_limitations, page_nav
from src.theme import footer, link_line, section_heading
from src.views import show_sources


data = get_data()
link_line("conclusion")
section_heading("What the evidence shows, and what it does not", lead=QUESTION, kicker="Conclusion")
with st.container(key="panel_conclusion"):
    conclusion(data)
methods_and_limitations()
st.divider()
show_sources()
page_nav(previous=(PAGE_ACCEL, "Has it sped up?"), following=(HOME, "Back to the overview"))
footer()
