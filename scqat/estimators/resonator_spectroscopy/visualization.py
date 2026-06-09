"""
Resonator spectroscopy plotting helper.

``plot_spectrum`` consumes the **plot_data** Dataset built by
``ResonatorSpectroscopyEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``detuning`` (+ optional ``full_freq`` data var)
vars   : ``amplitude`` / ``amplitude_baseline`` / ``amplitude_fit`` (detuning)
attrs  : ``resonator_detuning``, ``fwhm``, ``success``, ``has_full_freq``
         (+ optional ``resonator_frequency``)
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def plot_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """The |IQ| amplitude vs detuning with the (power-fitted) Lorentzian overlaid
    in the amplitude domain and a marker at the resonator dip, drawn entirely from
    ``plot_data``."""
    detuning = plot_data.coords["detuning"].values.astype(float)
    amplitude = plot_data["amplitude"].values
    amplitude_baseline = plot_data["amplitude_baseline"].values
    amplitude_fit = plot_data["amplitude_fit"].values
    det_res = float(plot_data.attrs.get("resonator_detuning", np.nan))
    fwhm = float(plot_data.attrs.get("fwhm", np.nan))

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)

    det_mhz = detuning / 1e6
    ax.plot(det_mhz, amplitude, "-", lw=0.8, color="C0", label="|IQ| amplitude")
    ax.plot(det_mhz, amplitude_baseline, "--", color="gray", lw=1.0, label="baseline")
    # Power-domain Lorentzian fit, shown as amplitude (sqrt of the power model)
    ax.plot(det_mhz, amplitude_fit, "-", lw=1.8, color="C1",
            label=f"Lorentzian fit (power; FWHM={fwhm / 1e6:.3f} MHz)")
    if np.isfinite(det_res):
        ax.axvline(det_res / 1e6, color="C3", ls=":", lw=1.0,
                   label=f"f_res @ {det_res / 1e6:.3f} MHz")

    ax.set_xlabel("Detuning (MHz)")
    ax.set_ylabel("Amplitude |IQ| (arb. u.)")
    ax.legend(fontsize=8)
    ax.set_title("Resonator spectroscopy")

    # Add absolute-frequency twin axis if available
    if plot_data.attrs.get("has_full_freq", 0):
        freq_vals = plot_data["full_freq"].values / 1e9
        ax_freq = ax.twiny()
        ax_freq.set_xlim(freq_vals[0], freq_vals[-1])
        ax_freq.set_xlabel("RF frequency (GHz)")

    fig.tight_layout()
    return fig
