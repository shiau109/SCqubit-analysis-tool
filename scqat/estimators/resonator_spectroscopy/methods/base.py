"""Method strategy protocol for the resonator_spectroscopy estimator.

One estimator per experiment; when more than one analysis approach can extract
the same physics, each approach is a *method strategy* implementing this ABC.
The estimator owns the two-tier result contract:

* **Tier 1 — COMMON keys** (validated by the estimator, see
  ``estimator.COMMON_KEYS``): ``detuning`` (fitted resonance position, Hz
  relative to LO), ``fwhm`` (total linewidth kappa_tot, Hz), ``success``
  (bool). ``detuning_err``/``fwhm_err`` are common-but-best-effort (NaN
  allowed). Same name => same meaning and unit in every method.
* **Tier 2 — extras**: anything else the method naturally produces (Qi, Qc,
  phi0, delay, background coefficients, ...). Optional; downstream consumers
  may use them only opportunistically (``if key in results``).

Plots are method-dependent, but the artifact contract is invariant: the
plot_data Dataset must be netCDF-safe (no complex variables — store I/Q float
pairs) and must stamp ``attrs["method"]`` so a saved ``plotdata.nc`` replots
with the right figure without re-fitting.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


class ResonatorMethod(ABC):
    """One extraction approach for resonator spectroscopy data."""

    #: registry key; also stamped into results["method"] / plot_data.attrs
    name: str = ""
    #: results keys dropped from the metadata JSON (bulky arrays belong in
    #: plot_data, not metadata)
    bulky_keys: frozenset = frozenset()

    @abstractmethod
    def extract(self, detuning: np.ndarray, iq: np.ndarray,
                full_freq: Optional[np.ndarray] = None, **kwargs) -> Dict:
        """Fit the data; return a results dict carrying the COMMON keys
        (plus any method-specific extras)."""

    @abstractmethod
    def build_plot_data(self, detuning: np.ndarray, iq: np.ndarray,
                        full_freq: Optional[np.ndarray], results: Dict) -> xr.Dataset:
        """Bundle the arrays needed to redraw this method's figure with zero
        recomputation. Must set ``attrs["method"] = self.name`` and contain no
        complex variables."""

    @abstractmethod
    def plot(self, plot_data: xr.Dataset) -> plt.Figure:
        """Draw this method's diagnostic figure using **only** ``plot_data``."""
