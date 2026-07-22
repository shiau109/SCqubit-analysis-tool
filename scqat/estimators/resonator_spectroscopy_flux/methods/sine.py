"""Sine method — model-light cosine approximation of the resonator flux arch.

Instead of the full dispersive + transmon-eigenfrequency model, fit the resonator
centre trace with a plain cosine of the flux::

    f_r(phi) = offset + amp * cos(2 * pi * (phi - phi_off) / T)

``amp`` is constrained non-negative, so the maximum sits at ``phi = phi_off``
(mod ``T``): for the usual readout configuration (qubit below the resonator, the
pull is largest where the qubit frequency is highest) that maximum marks the
**sweet-spot** flux, and ``T`` is the flux period.

This is far more robust than the dispersive model when only ~one arch is visible
or the trace is noisy — it has one fewer nonlinear parameter and no
``sqrt|cos|`` cusp — but it yields **no** ``f_r0`` / ``g`` physics. Use it to get
a trustworthy sweet-spot / period; use ``dispersive`` (with a known ``f_q_max``)
when the bare resonator and coupling are wanted. The reported ``sweet_spot_res`` is
the resonator maximum (the ``amp >= 0`` convention); on the rare occasion the
qubit sits *above* the resonator the sweet spot would instead be the trace
minimum — read the arch on the map to confirm the orientation.
"""

from typing import Any, Dict

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from lmfit import Model

from .base import (
    FluxModel,
    dense_flux,
    fail_result,
    locate_extremum,
    period_candidates,
    robust_peak_flux,
    robust_trough_flux,
)


def flux_cosine(flux, amp, T, phi_off, offset):
    """Plain cosine of the flux (see module docstring)."""
    return offset + amp * np.cos(2.0 * np.pi * (flux - phi_off) / T)


class SineMethod(FluxModel):
    """Fit ``center_frequency(flux)`` with a bare cosine of the flux."""

    name = "sine"

    def extract(self, flux, center, flux_all, center_all, **kwargs) -> Dict[str, Any]:
        n_valid = len(flux)
        extra_defaults = {
            "amp": float("nan"), "offset": float("nan"), "phi_off": float("nan"),
            "amp_err": float("nan"), "offset_err": float("nan"),
        }
        if n_valid < 5:
            return fail_result(flux_all, center_all, n_valid, extra_defaults)

        amp_guess = 0.5 * float(np.max(center) - np.min(center))
        offset_guess = float(np.median(center))
        phi_off_guess = robust_peak_flux(flux, center)
        dx = float(np.median(np.diff(flux)))
        span = float(flux.max() - flux.min())

        # Multi-seed period: the FFT estimate alone locks onto T/2 on sparse
        # traces — fit every candidate and keep the lowest-residual solution.
        model = Model(flux_cosine, independent_vars=["flux"])
        result = None
        for T_guess in period_candidates(flux, center):
            params = model.make_params(
                amp=amp_guess, T=T_guess, phi_off=phi_off_guess, offset=offset_guess
            )
            params["amp"].set(min=0.0, max=10 * abs(amp_guess) + 1.0)
            params["T"].set(min=2 * abs(dx) + 1e-12, max=1e4 * span + 1.0)
            params["phi_off"].set(min=flux.min() - T_guess, max=flux.max() + T_guess)
            try:
                cand = model.fit(center, params, flux=flux)
            except Exception:
                continue
            # Prefer converged solutions; among equals, the lowest residual.
            if (
                result is None
                or (cand.success and not result.success)
                or (cand.success == result.success and cand.redchi < result.redchi)
            ):
                result = cand
        if result is None:
            return fail_result(flux_all, center_all, n_valid, extra_defaults)
        converged = bool(result.success)

        p = result.params

        def _v(name):
            return float(p[name].value)

        def _e(name):
            return float(p[name].stderr) if p[name].stderr is not None else float("nan")

        amp, T, phi_off, offset = _v("amp"), _v("T"), _v("phi_off"), _v("offset")
        dense = dense_flux(flux_all)
        fit_freq = flux_cosine(dense, amp, T, phi_off, offset)

        # amp >= 0 => the cosine maximum is the upper sweet spot; the minimum
        # (lower sweet spot) sits half a period away.
        model_at = lambda ff: flux_cosine(ff, amp, T, phi_off, offset)  # noqa: E731
        # Extrema are mapped into the FULL swept flux range (flux_all), not the
        # good-point subset — an arch bottom often sits where per-slice fits were
        # rejected, and clamping it to the good range would misreport it.
        finite_all = flux_all[np.isfinite(flux_all)]
        lo, hi = float(finite_all.min()), float(finite_all.max())
        sweet_spot_flux, sweet_spot_res = locate_extremum(
            phi_off, T, lo, hi, robust_peak_flux(flux, center), model_at)
        sweet_spot_low_flux, sweet_spot_low_res = locate_extremum(
            phi_off + T / 2.0, T, lo, hi, robust_trough_flux(flux, center), model_at)

        in_range = flux.min() - T <= phi_off <= flux.max() + T
        success = bool(converged and np.isfinite(T) and T > 0 and amp > 0 and in_range)

        return {
            "flux_bias": flux_all,
            "center_freq": center_all,
            "fit_flux": dense,
            "fit_freq": fit_freq,
            "sweet_spot_flux": float(sweet_spot_flux),
            "sweet_spot_res": float(sweet_spot_res),
            "sweet_spot_low_flux": float(sweet_spot_low_flux),
            "sweet_spot_low_res": float(sweet_spot_low_res),
            "dv_phi0": float(T),
            "sweet_spot_flux_err": _e("phi_off"),
            "dv_phi0_err": _e("T"),
            "amp": amp, "offset": offset, "phi_off": phi_off,
            "amp_err": _e("amp"), "offset_err": _e("offset"),
            "n_points": int(n_valid),
            "success": success,
        }

    def build_plot_data(self, results: Dict[str, Any]) -> xr.Dataset:
        data_vars = {
            "center_freq": ("flux_bias", np.asarray(results["center_freq"], float)),
            "fit_freq": ("fit_flux", np.asarray(results["fit_freq"], float)),
        }
        coords = {
            "flux_bias": np.asarray(results["flux_bias"], float),
            "fit_flux": np.asarray(results["fit_flux"], float),
        }
        attrs = {
            "method": self.name,
            "sweet_spot_flux": float(results["sweet_spot_flux"]),
            "sweet_spot_res": float(results["sweet_spot_res"]),
            "sweet_spot_low_flux": float(results["sweet_spot_low_flux"]),
            "sweet_spot_low_res": float(results["sweet_spot_low_res"]),
            "dv_phi0": float(results["dv_phi0"]),
            "amp": float(results["amp"]),
            "offset": float(results["offset"]),
            "success": int(bool(results["success"])),
        }
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def plot(self, plot_data: xr.Dataset) -> plt.Figure:
        from ..visualization import plot_flux_model
        return plot_flux_model(plot_data)
