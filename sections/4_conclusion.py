"""Conclusion: what the evidence shows, and what it does not."""

from src.loading import QUESTION, get_data
from src.stories import PAGE_ACCEL, HOME, conclusion, methods_and_limitations, page_nav
from src.theme import BANNER, banner, pattern_rate


data = get_data()
banner("Conclusion", "What the evidence shows, and what it does not", QUESTION, BANNER["conclusion"], data,
       pattern_rate(data))
conclusion(data)
methods_and_limitations()
page_nav(previous=(PAGE_ACCEL, "Has it sped up?"), following=(HOME, "Back to the overview"))
