"""Overview page: the question, why it matters, where things stand, the data and how to access it."""

import streamlit as st

from src.loading import AUDIENCE, QUESTION, TITLE, get_data
from src.overview import about_the_data, access_data, hero, introduction, question_cards, where_things_stand


data = get_data()
hero(data, TITLE, QUESTION, AUDIENCE)
introduction()
st.divider()
where_things_stand(data)
st.divider()
about_the_data(data)
st.divider()
question_cards(data)
st.divider()
access_data(data)
