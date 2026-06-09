"""
Resonator flux-dispersion plotting helper.

``plot_dispersion`` consumes the **plot_data** Dataset built by
``ResonatorFluxDispersionEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``flux_bias``, ``fit_flux``
vars   : ``center_freq`` (flux_bias), ``fit_freq`` (fit_flux)
attrs  : ``f_r0``, ``g``, ``dv_phi0``, ``sweet_spot_flux``, ``sweet_spot_freq``,
         ``max_pull``, ``f_q_max``, ``f_q_max_fixed``, ``success``
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_dispersion(plot_data: xr.Dataset) -> plt.Figure:
    """Centre-frequency-vs-flux data points with the dispersive fit overlaid and
    the sweet spot marked, drawn entirely from ``plot_data``."""
    flux = plot_data.coords["flux_bias"].values.astype(float)
    center = plot_data["center_freq"].values.astype(float)
    fit_flux = plot_data.coords["fit_flux"].values.astype(float)
    fit_freq = plot_data["fit_freq"].values.astype(float)
    ss_flux = float(plot_data.attrs.get("sweet_spot_flux", np.nan))
    ss_freq = float(plot_data.attrs.get("sweet_spot_freq", np.nan))
    dv_phi0 = float(plot_data.attrs.get("dv_phi0", np.nan))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    ax.plot(flux, center / 1e9, "o", ms=4, color="C0", label="fitted centre (data)")
    if np.isfinite(fit_freq).any():
        ax.plot(fit_flux, fit_freq / 1e9, "-", lw=1.8, color="C1",
                label=f"dispersive fit (dv_phi0={dv_phi0:.4f} V)")
    if np.isfinite(ss_flux):
        ax.axvline(ss_flux, color="C3", ls=":", lw=1.0,
                   label=f"sweet spot @ {ss_flux:.4f} V, {ss_freq / 1e9:.4f} GHz")

    ax.set_xlabel("Flux bias (V)")
    ax.set_ylabel("Resonator centre frequency (GHz)")
    cond = "" if plot_data.attrs.get("f_q_max_fixed", 1) == 0 else " (g conditional on assumed f_q_max)"
    ax.set_title("Resonator flux dispersion" + cond)
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig
