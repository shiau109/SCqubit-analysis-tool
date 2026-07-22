"""``lorentzian`` method: joint Lorentzian + polynomial-background fit of power.

The fit itself lives in :func:`scqat.tools.dip_fit.fit_dip` — the reduction
shared by the whole resonator-dip experiment family (1-D spectroscopy, vs-flux,
vs-power). This method binds it into the 1-D estimator's method registry and
owns the per-method plot data / figure.

Method knobs (kwargs): ``baseline_order`` (0/1/2 polynomial background order,
default 1).
"""

from typing import Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.tools.dip_fit import fit_dip

from .base import ResonatorMethod
from ..visualization import plot_lorentzian


class LorentzianMethod(ResonatorMethod):

    name = "lorentzian"
    bulky_keys = frozenset({
        "signal", "baseline", "signal_corrected", "fit_x", "fit_y", "seed_mask",
    })

    def extract(self, detuning: np.ndarray, iq: np.ndarray,
                full_freq: Optional[np.ndarray] = None, **kwargs) -> Dict:
        knobs = {k: kwargs[k] for k in ("baseline_order",) if k in kwargs}
        return fit_dip(detuning, iq, full_freq=full_freq, method=self.name, **knobs)

    def build_plot_data(self, detuning: np.ndarray, iq: np.ndarray,
                        full_freq: Optional[np.ndarray], results: Dict) -> xr.Dataset:
        power = np.asarray(results["signal"], dtype=float)
        baseline = np.asarray(results["baseline"], dtype=float)
        fit_component = np.asarray(results["fit_y"], dtype=float)

        data_vars = {
            "power": ("detuning", power),
            "power_baseline": ("detuning", baseline),
            "power_corrected": ("detuning", power - baseline),
            "power_fit": ("detuning", baseline + fit_component),
            "baseline_seed": ("detuning", np.asarray(results["seed_mask"], dtype=np.int8)),
        }
        coords = {"detuning": np.asarray(detuning, dtype=float)}
        attrs = {
            "method": self.name,
            "resonator_detuning": float(results["detuning"]),
            "fwhm": float(results["fwhm"]),
            "success": int(bool(results["success"])),
        }
        if full_freq is not None:
            data_vars["full_freq"] = ("detuning", np.asarray(full_freq, dtype=float))
            attrs["has_full_freq"] = 1
            if "full_freq" in results:
                attrs["resonator_frequency"] = float(results["full_freq"])
        else:
            attrs["has_full_freq"] = 0
        return xr.Dataset(data_vars, coords=coords, attrs=attrs)

    def plot(self, plot_data: xr.Dataset) -> plt.Figure:
        return plot_lorentzian(plot_data)
