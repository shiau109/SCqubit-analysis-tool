"""``lorentzian`` method: joint Lorentzian + polynomial-background fit of power.

Fits the readout **power** ``|IQ|^2`` — the quantity that is truly Lorentzian,
so the fitted FWHM equals the cavity linewidth kappa. The smooth background
(chain-gain slope, squared-amplitude curvature) is fitted **jointly** with the
dip in one model (``scqat.tools.fit_lorentzian_bg``), so the background cannot
bias the dip parameters the way a subtract-then-fit baseline did.

Method knobs (kwargs): ``baseline_order`` (0/1/2 polynomial background order,
default 1).
"""

from typing import Dict, Optional

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from scqat.tools.fit_lorentzian import lorentzian
from scqat.tools.fit_lorentzian_bg import FitLorentzianBG

from .base import ResonatorMethod
from ..visualization import plot_lorentzian


class LorentzianMethod(ResonatorMethod):

    name = "lorentzian"
    bulky_keys = frozenset({
        "signal", "baseline", "signal_corrected", "fit_x", "fit_y", "seed_mask",
    })

    def extract(self, detuning: np.ndarray, iq: np.ndarray,
                full_freq: Optional[np.ndarray] = None, **kwargs) -> Dict:
        baseline_order = int(kwargs.get("baseline_order", 1))

        signal = np.abs(iq) ** 2

        da = xr.DataArray(signal, coords={"x": detuning}, dims="x")
        gamma_max = float(detuning[-1] - detuning[0]) or 1.0
        fitter = FitLorentzianBG(
            da,
            inverted=True,
            background_order=baseline_order,
            bounds={"x0": (float(detuning.min()), float(detuning.max())),
                    "gamma": (0.0, abs(gamma_max))},
        )
        try:
            result = fitter.fit()
            p = result.params
            x0 = float(p["x0"].value)
            amplitude = float(p["amplitude"].value)
            gamma = float(p["gamma"].value)
            errs = {
                "detuning_err": float(p["x0"].stderr) if p["x0"].stderr is not None else np.nan,
                "amplitude_err": float(p["amplitude"].stderr) if p["amplitude"].stderr is not None else np.nan,
                "fwhm_err": 2 * float(p["gamma"].stderr) if p["gamma"].stderr is not None else np.nan,
            }
            baseline = np.asarray(fitter.baseline_values(), dtype=float)
            seed_mask = np.asarray(fitter.seed_mask, dtype=bool)
            bg = {f"bg_c{k}": float(p[f"c{k}"].value) for k in range(3)}
            converged = bool(getattr(result, "success", False))
        except Exception:
            # Fall back to the raw minimum against a flat median background
            med = float(np.median(signal))
            baseline = np.full_like(signal, med)
            idx = int(np.argmin(signal))
            x0 = float(detuning[idx])
            amplitude = float(signal[idx] - med)
            gamma = abs(detuning[1] - detuning[0]) * 5 if len(detuning) > 1 else 1.0
            errs = {"detuning_err": np.nan, "amplitude_err": np.nan, "fwhm_err": np.nan}
            seed_mask = np.ones(len(detuning), dtype=bool)
            bg = {"bg_c0": med, "bg_c1": 0.0, "bg_c2": 0.0}
            converged = False

        fwhm = 2.0 * abs(gamma)
        # Lorentzian component alone (zero offset): full model = baseline + this
        fit_y = lorentzian(detuning, x0, amplitude, gamma, 0.0)

        in_span = float(detuning.min()) <= x0 <= float(detuning.max())
        success = bool(converged and in_span and np.isfinite(fwhm) and fwhm > 0)

        return {
            "detuning": x0,
            "fwhm": float(fwhm),
            "success": success,
            "amplitude": amplitude,
            **errs,
            **bg,
            "baseline_order": baseline_order,
            "signal": signal,
            "baseline": baseline,
            "signal_corrected": signal - baseline,
            "fit_x": detuning,
            "fit_y": fit_y,
            "seed_mask": seed_mask,
        }

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
