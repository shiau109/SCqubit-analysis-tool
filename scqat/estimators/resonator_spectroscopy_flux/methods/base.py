"""Method strategy protocol for the resonator_spectroscopy_flux trace fit.

One estimator per experiment; when more than one approach can extract the same
flux-dependence physics, each approach is a *method strategy* implementing this
ABC. The trace-fit dispatcher owns the two-tier result contract:

* **Tier 1 — COMMON keys** (validated by :func:`..trace_fit.fit_flux_trace`,
  see ``trace_fit.COMMON_KEYS``): ``sweet_spot_flux`` (flux V at the maximum qubit
  frequency, i.e. the upper flux sweet spot), ``sweet_spot_res`` (resonator centre
  frequency there, Hz), ``dv_phi0`` (flux period, V), ``success`` (bool). Same
  name => same meaning and unit in every method.
* **Tier 2 — extras**: anything else the method naturally produces
  (``f_r0``/``g``/``f_q_max`` for the dispersive model; ``amp``/``offset`` for
  the sine model). Optional; downstream consumers use them only via
  ``if key in results``.

Both methods also return the same **structural** arrays so the composite
estimator can build plot data and the combined figure uniformly: ``flux_bias`` /
``center_freq`` (the full input trace) and ``fit_flux`` / ``fit_freq`` (the dense
fitted curve).

Plots are method-dependent, but the artifact contract is invariant: the
plot_data Dataset must be netCDF-safe and must stamp ``attrs["method"]`` so a
saved ``plotdata.nc`` replots with the right figure without re-fitting.
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

#: how many periodic images of the sweet spot to consider when mapping the
#: fitted phase into the swept flux range.
_PEAK_IMAGE_SPAN = 8


class FluxModel(ABC):
    """One extraction approach for the resonator centre-frequency(flux) trace."""

    #: registry key; also stamped into results["method"] / plot_data.attrs
    name: str = ""
    #: results keys dropped from the metadata JSON (bulky arrays belong in
    #: plot_data, not metadata)
    bulky_keys: frozenset = frozenset(
        {"flux_bias", "center_freq", "fit_flux", "fit_freq"}
    )

    @abstractmethod
    def extract(
        self,
        flux: np.ndarray,
        center: np.ndarray,
        flux_all: np.ndarray,
        center_all: np.ndarray,
        **kwargs,
    ) -> Dict[str, Any]:
        """Fit the model to the good ``(flux, center)`` points and return a
        results dict carrying the COMMON keys, the structural arrays and any
        method-specific extras.

        Parameters
        ----------
        flux, center
            The **good** points only (finite + success-masked), sorted by flux.
        flux_all, center_all
            The **full** input arrays (all flux points, NaN where undefined),
            reported verbatim so plots show every measured centre.
        """

    @abstractmethod
    def build_plot_data(self, results: Dict[str, Any]) -> xr.Dataset:
        """Bundle the data points and the dense fit curve so the figure redraws
        with zero recomputation. Must set ``attrs["method"] = self.name`` and
        contain no complex variables."""

    @abstractmethod
    def plot(self, plot_data: xr.Dataset) -> plt.Figure:
        """Draw this method's diagnostic figure using **only** ``plot_data``."""


# ----------------------------------------------------------------------
# Shared numeric helpers (used by every method — keep the math in one place)
# ----------------------------------------------------------------------
def median_smooth(v: np.ndarray, k: int = 3) -> np.ndarray:
    """Length-preserving running median (odd window ``k``). Robust to a single
    spurious spike capturing the seed, unlike a raw ``argmax``."""
    v = np.asarray(v, dtype=float)
    if len(v) < k or k < 2:
        return v
    pad = k // 2
    vp = np.pad(v, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(vp, k), axis=1)


def robust_peak_flux(flux: np.ndarray, y: np.ndarray) -> float:
    """Flux of the trace maximum, from the median-smoothed trace (so one
    spurious high point can't capture the seed)."""
    return float(flux[int(np.argmax(median_smooth(y)))])


def robust_trough_flux(flux: np.ndarray, y: np.ndarray) -> float:
    """Flux of the trace minimum, from the median-smoothed trace."""
    return float(flux[int(np.argmin(median_smooth(y)))])


def estimate_period(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate the flux period of the trace from its dominant FFT component.
    Falls back to twice the swept span when fewer than ~one period is visible."""
    n = len(x)
    span = float(x.max() - x.min()) or 1.0
    if n < 4:
        return 2.0 * span
    dx = float(np.median(np.diff(np.sort(x))))
    if dx <= 0:
        return 2.0 * span
    yd = (y - np.mean(y)) * np.hanning(n)
    amp = np.abs(np.fft.rfft(yd))
    freqs = np.fft.rfftfreq(n, d=dx)
    amp[0] = 0.0  # drop DC
    k = int(np.argmax(amp))
    f_peak = freqs[k]
    if f_peak <= 0:
        return 2.0 * span
    return 1.0 / f_peak


def period_candidates(flux: np.ndarray, y: np.ndarray) -> "list[float]":
    """Distinct period seeds for the flux-model fit.

    The FFT estimate locks onto the second harmonic when the good points are
    sparse or cover ~one arch (the classic half-period aliasing), so a single
    seed is not safe. Candidates: the FFT estimate, TWICE the FFT estimate, and
    the arch heuristic ``2 * |flux(max) - flux(min)|`` of the median-smoothed
    trace (max and min of one arch are half a period apart). Deduped within 5%;
    the caller fits every candidate and keeps the lowest-residual solution.
    """
    t_fft = estimate_period(flux, y)
    t_arch = 2.0 * abs(robust_peak_flux(flux, y) - robust_trough_flux(flux, y))
    out: "list[float]" = []
    for t in (t_fft, 2.0 * t_fft, t_arch):
        if not (np.isfinite(t) and t > 0):
            continue
        if any(abs(t - u) <= 0.05 * u for u in out):
            continue
        out.append(float(t))
    return out


def locate_extremum(
    phase: float,
    period: float,
    lo: float,
    hi: float,
    near_flux: float,
    model_at: Callable[[np.ndarray], np.ndarray],
) -> "tuple[float, float]":
    """Map a fitted extremum phase into the swept flux range.

    The flux dependence is periodic, so an extremum location (``phi_off`` for
    the arch top, ``phi_off + period/2`` for the bottom) is only defined modulo
    the period: the bare phase may be an out-of-range or arbitrary periodic
    image. Return the image that (a) lies in the swept ``[lo, hi]`` range and
    (b) is nearest the observed feature (``near_flux`` — the smoothed data max
    or min), together with the resonator frequency there (``model_at`` evaluated
    at that flux). When no image lands in range, wrap the phase to the nearest
    in-range value.
    """
    if not np.isfinite(period) or period <= 0:
        spot = float(np.clip(phase, lo, hi))
        return spot, float(model_at(np.array([spot]))[0])
    k = np.arange(-_PEAK_IMAGE_SPAN, _PEAK_IMAGE_SPAN + 1)
    cands = phase + k * period
    in_range = cands[(cands >= lo) & (cands <= hi)]
    if in_range.size:
        spot = float(in_range[int(np.argmin(np.abs(in_range - near_flux)))])
    else:
        mid = 0.5 * (lo + hi)
        spot = float(np.clip(phase - period * np.round((phase - mid) / period), lo, hi))
    return spot, float(model_at(np.array([spot]))[0])


def dense_flux(flux_all: np.ndarray, n: int = 400) -> np.ndarray:
    """Dense flux axis spanning the finite input range, for a smooth fit curve."""
    finite = flux_all[np.isfinite(flux_all)]
    return np.linspace(float(finite.min()), float(finite.max()), n)


def fail_result(
    flux_all: np.ndarray, center_all: np.ndarray, n_valid: int, extra: Dict[str, Any]
) -> Dict[str, Any]:
    """A uniform 'fit failed' results dict carrying the COMMON keys as NaN/False
    plus the structural arrays, merged with method-specific ``extra`` defaults."""
    nan = float("nan")
    base = {
        "flux_bias": flux_all,
        "center_freq": center_all,
        "fit_flux": np.asarray(flux_all, dtype=float),
        "fit_freq": np.full_like(np.asarray(flux_all, dtype=float), nan),
        "sweet_spot_flux": nan,
        "sweet_spot_res": nan,
        "sweet_spot_low_flux": nan,
        "sweet_spot_low_res": nan,
        "dv_phi0": nan,
        "sweet_spot_flux_err": nan,
        "dv_phi0_err": nan,
        "n_points": int(n_valid),
        "success": False,
    }
    base.update(extra)
    return base
