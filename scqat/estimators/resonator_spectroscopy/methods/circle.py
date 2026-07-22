"""``circle`` method: Probst notch-model circle fit of complex S21.

The fit itself lives in :func:`scqat.tools.dip_fit.fit_dip` — the reduction
shared by the whole resonator-dip experiment family (1-D spectroscopy, vs-flux,
vs-power). This method binds it into the 1-D estimator's method registry and
owns the per-method plot data / figure (IQ-plane panels).

Requires the **absolute** frequency axis (``full_freq``): the model term
``Ql*(f/fr - 1)`` is meaningless on a detuning axis that crosses zero.

Method knobs (kwargs): ``delay`` (fix the cable delay in seconds instead of
fitting it).
"""

from typing import Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.tools.dip_fit import fit_dip

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
        knobs = {k: kwargs[k] for k in ("delay",) if k in kwargs}
        return fit_dip(detuning, iq, full_freq=full_freq, method=self.name, **knobs)

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
