"""Overview page: the question, where things stand, the three questions, the data and how to access it."""

from src.loading import AUDIENCE, QUESTION, TITLE, get_data
from src.overview import about_the_data, access_data, hero, questions, temperature_change
from src.theme import footer, link_line


data = get_data()
hero(data, TITLE, QUESTION, AUDIENCE)
link_line()
temperature_change(data)
questions()
about_the_data(data)
access_data(data)
footer()
