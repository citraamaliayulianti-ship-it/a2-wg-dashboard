"""Trend and El Niño calculations shared by the pipeline and the views.

All calculations work on monthly global temperature anomalies (°C).

* ``fit_trend`` fits a straight line (ordinary least squares) and reports the
  rate in °C per decade with a 95% confidence interval. Monthly values are not
  independent - a warm month tends to be followed by another warm month - so the
  interval is widened using the lag-1 autocorrelation of the residuals
  (an AR(1) adjustment of the effective sample size, as in Santer et al. 2008).
* ``fit_enso`` adds the Oceanic Niño Index (ONI), shifted by a lag in months, to
  the straight line. Its coefficient says how many °C the global temperature
  moves for each 1 °C of Niño-3.4 sea-surface anomaly, and lets us subtract the
  El Niño/La Niña part to leave an "ENSO-adjusted" series.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

Z95 = 0.975  # two-sided 95% interval


@dataclass(frozen=True)
class Trend:
    """A straight-line trend over a period, in °C per decade."""

    rate: float          # °C per decade
    low: float           # lower end of the 95% confidence interval
    high: float          # upper end
    months: int          # number of monthly values used
    effective_n: float   # sample size after allowing for autocorrelation
    lag1: float          # lag-1 autocorrelation of the residuals
    intercept: float     # fitted anomaly at the middle of the period (°C)
    centre: float        # middle of the period (decimal year)

    @property
    def half_width(self) -> float:
        return (self.high - self.low) / 2

    def value_at(self, when: np.ndarray | float) -> np.ndarray | float:
        """Fitted anomaly (°C) at decimal year(s) ``when``."""
        return self.intercept + self.rate / 10 * (np.asarray(when) - self.centre)


def decimal_year(year: pd.Series | np.ndarray, month: pd.Series | np.ndarray) -> np.ndarray:
    """Middle of each month as a decimal year, e.g. January 2000 -> 2000.042."""
    return np.asarray(year, dtype=float) + (np.asarray(month, dtype=float) - 0.5) / 12


def _effective_n(residuals: np.ndarray) -> tuple[float, float]:
    n = len(residuals)
    r1 = float(np.corrcoef(residuals[:-1], residuals[1:])[0, 1]) if n > 3 else 0.0
    r1 = min(max(r1, 0.0), 0.99)  # negative autocorrelation is not used to narrow the interval
    return max(n * (1 - r1) / (1 + r1), 3.0), r1


def _slope_interval(x: np.ndarray, residuals: np.ndarray, slope: float, extra_params: int = 0
                    ) -> tuple[float, float, float, float]:
    """95% interval for a slope, widened for lag-1 autocorrelation."""
    n_eff, r1 = _effective_n(residuals)
    dof = max(n_eff - 2 - extra_params, 1.0)
    se = float(np.sqrt(np.sum(residuals ** 2) / dof / np.sum((x - x.mean()) ** 2)))
    half = float(stats.t.ppf(Z95, dof)) * se
    return slope - half, slope + half, n_eff, r1


def fit_trend(when: np.ndarray, anomaly: np.ndarray) -> Trend:
    """Straight line through monthly anomalies. ``when`` is in decimal years."""
    x = np.asarray(when, dtype=float)
    y = np.asarray(anomaly, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    if len(x) < 24:
        raise ValueError("A trend needs at least two years of monthly values.")
    centre = float(x.mean())
    slope, intercept = np.polyfit(x - centre, y, 1)
    residuals = y - (intercept + slope * (x - centre))
    low, high, n_eff, r1 = _slope_interval(x, residuals, slope)
    return Trend(rate=float(slope) * 10, low=low * 10, high=high * 10, months=len(x),
                 effective_n=n_eff, lag1=r1, intercept=float(intercept), centre=centre)


def lagged_oni(frame: pd.DataFrame, lag: int) -> pd.Series:
    """ONI from ``lag`` months earlier, matched month by month within each record."""
    ordered = frame.sort_values(["record_id", "date"])
    shifted = ordered.groupby("record_id", sort=False)["oni"].shift(lag)
    return shifted.reindex(frame.index)


TREND_DEGREE = 2  # the long-term curve is allowed to bend gently (a quadratic)


@dataclass(frozen=True)
class EnsoFit:
    """Temperature = gently curving long-term trend + ENSO effect (ONI shifted by ``lag`` months)."""

    lag: int
    enso_per_degree: float   # °C of global temperature per 1 °C of ONI
    enso_low: float          # 95% interval for that effect
    enso_high: float
    monthly_share: float     # share of month-to-month variation around the trend explained by ENSO
    annual_share: float      # the same, for calendar-year means ("year-to-year")
    months: int


def _curve(x: np.ndarray, degree: int) -> np.ndarray:
    return np.column_stack([x ** k for k in range(degree + 1)])


def fit_enso(frame: pd.DataFrame, lag: int, degree: int = TREND_DEGREE) -> tuple[EnsoFit, pd.DataFrame]:
    """Fit one record. ``frame`` holds one record_id with date, year, month, anomaly_c, oni.

    Returns the fit and a table with the long-term curve, the ENSO part and the
    ENSO-adjusted values (original minus ENSO part).
    """
    work = frame.sort_values("date").copy()
    work["oni_lagged"] = work["oni"].shift(lag)
    work = work.dropna(subset=["anomaly_c", "oni_lagged"])
    if len(work) < 36:
        raise ValueError("Not enough months with both temperature and ONI values.")
    x = decimal_year(work["year"], work["month"])
    x = (x - x.mean()) / 10  # decades from the middle of the period, for numerical stability
    y = work["anomaly_c"].to_numpy(float)
    oni = work["oni_lagged"].to_numpy(float)

    base = _curve(x, degree)
    design = np.column_stack([base, oni])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    residuals = y - design @ coef

    # The same curve without ENSO, to measure how much ENSO adds.
    base_coef, *_ = np.linalg.lstsq(base, y, rcond=None)
    curve_only = base @ base_coef
    monthly_share = 1 - float(np.sum(residuals ** 2) / np.sum((y - curve_only) ** 2))

    # 95% interval for the ENSO coefficient, widened for autocorrelation in the residuals.
    n_eff, _ = _effective_n(residuals)
    dof = max(n_eff - design.shape[1], 1.0)
    cov = np.linalg.inv(design.T @ design) * np.sum(residuals ** 2) / dof
    half = float(stats.t.ppf(Z95, dof)) * float(np.sqrt(cov[-1, -1]))

    work["curve_c"] = base @ coef[:-1]
    work["enso_part_c"] = coef[-1] * oni
    work["deviation_c"] = y - work["curve_c"]
    work["adjusted_c"] = y - work["enso_part_c"]

    # Year-to-year: calendar-year means of the deviation from the curve and of the ENSO part.
    full_years = work.groupby("year").filter(lambda g: len(g) == 12)
    annual = full_years.groupby("year")[["deviation_c", "enso_part_c"]].mean()
    if len(annual) >= 5:
        dev = annual["deviation_c"] - annual["deviation_c"].mean()
        part = annual["enso_part_c"] - annual["enso_part_c"].mean()
        annual_share = 1 - float(np.sum((dev - part) ** 2) / np.sum(dev ** 2))
    else:
        annual_share = float("nan")

    fit = EnsoFit(lag=lag, enso_per_degree=float(coef[-1]), enso_low=float(coef[-1]) - half,
                  enso_high=float(coef[-1]) + half, monthly_share=monthly_share,
                  annual_share=annual_share, months=len(work))
    return fit, work


def best_lag(frame: pd.DataFrame, lags: range = range(0, 7)) -> int:
    """The ONI lag (months) that explains the most variation around the trend."""
    scores = {lag: fit_enso(frame, lag)[0].monthly_share for lag in lags}
    return max(scores, key=scores.get)


def difference_interval(first: Trend, second: Trend) -> tuple[float, float, float]:
    """Change in rate between two periods (second minus first) with a 95% interval."""
    diff = second.rate - first.rate
    se = np.sqrt((first.half_width / stats.t.ppf(Z95, max(first.effective_n - 2, 1))) ** 2
                 + (second.half_width / stats.t.ppf(Z95, max(second.effective_n - 2, 1))) ** 2)
    dof = max(min(first.effective_n, second.effective_n) - 2, 1.0)
    half = float(stats.t.ppf(Z95, dof)) * float(se)
    return diff, diff - half, diff + half


def format_rate(value: float) -> str:
    return f"{value:+.2f}" if abs(value) >= 0.005 else "0.00"


def format_interval(low: float, high: float, spec: str = "+.2f") -> str:
    return f"{low:{spec}} to {high:{spec}}"
