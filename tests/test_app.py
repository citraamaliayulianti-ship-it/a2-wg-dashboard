from pathlib import Path

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from src.analysis import decimal_year, fit_enso, fit_trend


ROOT = Path(__file__).resolve().parents[1]
PAGES = [
    "sections/1_warming_rate.py",
    "sections/2_el_nino.py",
    "sections/3_acceleration.py",
    "sections/4_conclusion.py",
]


def _app(monkeypatch, page: str | None = None) -> AppTest:
    monkeypatch.setenv("DASHBOARD_SAMPLE_MODE", "1")
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    if page:
        app.switch_page(page).run()
    return app


def test_overview_renders(monkeypatch) -> None:
    app = _app(monkeypatch)
    assert not app.exception
    assert any("How fast is the world warming?" in item.value for item in app.markdown)


def test_every_page_renders(monkeypatch) -> None:
    for page in PAGES:
        app = _app(monkeypatch, page)
        assert not app.exception, page
        assert any("<h2" in item.value for item in app.markdown), page


def test_empty_record_selection_shows_message(monkeypatch) -> None:
    app = _app(monkeypatch, PAGES[0])
    app.multiselect(key="rate_records").set_value([]).run()
    assert any("No temperature records match" in item.value for item in app.info)


def test_switching_to_published_values_runs(monkeypatch) -> None:
    app = _app(monkeypatch, PAGES[2])
    app.radio(key="accel_series").set_value("As published").run()
    assert not app.exception


def test_trend_recovers_a_known_rate() -> None:
    """A noisy series warming at exactly 0.2 °C per decade gives that rate inside its interval."""
    rng = np.random.default_rng(0)
    when = 1970 + (np.arange(600) + 0.5) / 12
    values = 0.02 * (when - 1970) + rng.normal(0, 0.08, len(when))
    trend = fit_trend(when, values)
    assert abs(trend.rate - 0.2) < 0.02
    assert trend.low < 0.2 < trend.high


def test_enso_fit_recovers_a_known_effect() -> None:
    """A series built from a curve plus 0.1 x ONI (2 months earlier) returns about 0.1 at lag 2."""
    rng = np.random.default_rng(1)
    dates = pd.date_range("1950-01-31", periods=840, freq="ME")
    oni = np.zeros(len(dates))
    for i in range(1, len(dates)):
        oni[i] = 0.9 * oni[i - 1] + rng.normal(0, 0.3)
    when = decimal_year(dates.year, dates.month)
    anomaly = 0.015 * (when - 1950) + 0.1 * np.r_[np.zeros(2), oni[:-2]] + rng.normal(0, 0.05, len(dates))
    frame = pd.DataFrame({"date": dates, "year": dates.year, "month": dates.month, "record_id": "test",
                          "anomaly_c": anomaly, "oni": oni})
    fit, work = fit_enso(frame, lag=2)
    assert abs(fit.enso_per_degree - 0.1) < 0.02
    assert np.allclose(work["adjusted_c"], work["anomaly_c"] - work["enso_part_c"])
