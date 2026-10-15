"""Entry point: page navigation for the dashboard.

Each page is its own file. The overview introduces the question and the data;
pages 1-3 each answer one part of the question; the last page draws the conclusion.
"""

import streamlit as st

from src.stories import HOME, PAGE_ACCEL, PAGE_CONCLUSION, PAGE_ENSO, PAGE_RATE
from src.theme import apply_style


st.set_page_config(page_title="How fast is the world warming?", layout="wide")

navigation = st.navigation(
    [
        st.Page(HOME, title="Overview", icon=":material/home:", default=True),
        st.Page(PAGE_RATE, title="1. Warming rate", url_path="rate"),
        st.Page(PAGE_ENSO, title="2. El Niño and La Niña", url_path="enso"),
        st.Page(PAGE_ACCEL, title="3. Has it sped up?", url_path="acceleration"),
        st.Page(PAGE_CONCLUSION, title="Conclusion", url_path="conclusion"),
    ],
    position="top",
)
apply_style()
navigation.run()
