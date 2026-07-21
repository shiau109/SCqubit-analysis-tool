"""``circle`` method: Probst notch-model circle fit of complex S21.

Fits the raw complex data ``I + iQ`` with the full notch model — environment
(net amplitude, phase offset, cable delay) included — via
``scqat.tools.fit_notch_circle``. The transmission "baseline" is calibrated
out analytically instead of estimated, which also handles the Fano-asymmetric
magnitude shapes where any magnitude-only fit is biased. Extras: the coupling
and internal quality factors (``absQc``, ``Qc_dia_corr``, ``Qi_dia_corr``),
the impedance-mismatch angle ``phi0``, and the cable ``delay``.

Requires the **absolute** frequency axis (``full_freq``): the model term
``Ql*(f/fr - 1)`` is meaningless on a detuning axis that crosses zero.

Method knobs (kwargs): ``delay`` (fix the cable delay in seconds instead of
fitting it).
"""

from typing import Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.tools.fit_notch_circle import FitNotchCircle

from .base import ResonatorMethod
from ..visualization import plot_circle


class CircleMethod(ResonatorMethod):

    name = "circle"
    bulky_keys = frozenset({
        "raw_i", "raw_q", "fit_i", "fit_q",
        "norm_i", "norm_q", "norm_fit_i", "norm_fit_q",
    })

    def extract(self, detuning: np.ndarray, iq: np.ndarray,
                full_freq: Optional[np.ndarray] = None, **kwargs) -> Dict:
        if full_freq is None:
            raise ValueError(
                "method='circle' requires the 'full_freq' coordinate (absolute "
                "readout frequency in Hz): the notch model term Ql*(f/fr - 1) is "
                "meaningless on a detuning axis that crosses zero. Attach "
                "full_freq or use method='lorentzian'."
            )
        f = np.asarray(full_freq, dtype=float)

        fitter = FitNotchCircle(iq, x=f, delay=kwargs.get("delay"))
        res = fitter.fit()

        fr, ql = res["fr"], res["Ql"]
        fwhm = fr / ql if ql else np.nan
        # full_freq = detuning + const, so the detuning axis is linear in f and
        # the fr error maps 1:1 onto the detuning error
        order = np.argsort(f)
        det = float(np.interp(fr, f[order], np.asarray(detuning, dtype=float)[order]))
        rel_fr = res["fr_err"] / fr if fr else np.nan
        rel_ql = res["Ql_err"] / ql if ql else np.nan
        fwhm_err = abs(fwhm) * float(np.hypot(rel_fr, rel_ql))

        # Resolution floor: a fitted linewidth below the frequency grid step is
        # unresolvable — the degenerate fit phase-less (e.g. simulated) data
        # produces. Same convention as qubit_spectroscopy's min_fwhm_factor.
        step = float(np.median(np.abs(np.diff(f)))) if len(f) > 1 else 0.0
        success = (bool(res["success"]) and np.isfinite(fwhm)
                   and fwhm > 0.5 * step)

        return {
            "detuning": det,
            "fwhm": float(fwhm),
            "success": success,
            "full_freq": float(fr),      # the circle fit reports fr directly
            "detuning_err": float(res["fr_err"]),
            "fwhm_err": float(fwhm_err),
            # extras: quality factors + environment
            "Ql": float(ql),
            "absQc": float(res["absQc"]),
            "Qc_dia_corr": float(res["Qc_dia_corr"]),
            "Qi_dia_corr": float(res["Qi_dia_corr"]),
            "Qi_no_corr": float(res["Qi_no_corr"]),
            "phi0": float(res["phi0"]),
            "delay": float(res["delay"]),
            "a": float(res["a"]),
            "alpha": float(res["alpha"]),
            "Ql_err": float(res["Ql_err"]),
            "absQc_err": float(res["absQc_err"]),
            "phi0_err": float(res["phi0_err"]),
            "Qc_dia_corr_err": float(res["Qc_dia_corr_err"]),
            "Qi_dia_corr_err": float(res["Qi_dia_corr_err"]),
            "chi_square": float(res["chi_square"]),
            # float-pair arrays for the figure (netCDF-safe; bulky)
            "raw_i": np.real(iq).astype(float),
            "raw_q": np.imag(iq).astype(float),
            "fit_i": np.real(res["z_fit"]).astype(float),
            "fit_q": np.imag(res["z_fit"]).astype(float),
            "norm_i": np.real(res["z_norm"]).astype(float),
            "norm_q": np.imag(res["z_norm"]).astype(float),
            "norm_fit_i": np.real(res["z_norm_fit"]).astype(float),
            "norm_fit_q": np.imag(res["z_norm_fit"]).astype(float),
        }

    def build_plot_data(self, detuning: np.ndarray, iq: np.ndarray,
                        full_freq: Optional[np.ndarray], results: Dict) -> xr.Dataset:
        data_vars = {
            key: ("detuning", np.asarray(results[key], dtype=float))
            for key in ("raw_i", "raw_q", "fit_i", "fit_q",
                        "norm_i", "norm_q", "norm_fit_i", "norm_fit_q")
        }
        data_vars["full_freq"] = ("detuning", np.asarray(full_freq, dtype=float))
        coords = {"detuning": np.asarray(detuning, dtype=float)}
        attrs = {
            "method": self.name,
            "resonator_detuning": float(results["detuning"]),
            "fwhm": float(results["fwhm"]),
            "success": int(bool(results["success"])),
            "has_full_freq": 1,
            "resonator_frequency": float(results["full_freq"]),
        }
        for key in ("Ql", "absQc", "Qc_dia_corr", "Qi_dia_corr", "phi0", "delay",
                    "Ql_err", "absQc_err", "Qi_dia_corr_err"):
            attrs[key] = float(results[key])
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def plot(self, plot_data: xr.Dataset) -> plt.Figure:
        return plot_circle(plot_data)
