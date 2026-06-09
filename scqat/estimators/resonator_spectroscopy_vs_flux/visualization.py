"""
Resonator-spectroscopy-vs-flux plotting helper.

``plot_flux_map`` consumes the **plot_data** Dataset built by
``ResonatorSpectroscopyVsFluxEstimator.build_plot_data`` and draws without any
recalculation.

plot_data layout
----------------
coords : ``flux_bias``, ``detuning`` (+ optional ``full_freq``)
vars   : ``amplitude`` (flux_bias, detuning); per-flux ``center_detuning`` /
         ``fwhm`` / ``dip_amplitude`` / ``success`` / ``good`` / ``outlier``
         (+ optional ``center_full_freq``)
attrs  : ``n_flux``, ``n_success``, ``n_good``, ``n_outlier``, ``has_full_freq``
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_flux_map(plot_data: xr.Dataset) -> plt.Figure:
    """The 2-D |IQ| amplitude map over (flux, frequency) with the fitted
    resonator-centre trace overlaid, drawn entirely from ``plot_data``."""
    flux = plot_data.coords["flux_bias"].values.astype(float)
    amplitude = plot_data["amplitude"].values  # (flux, detuning)

    use_full = bool(plot_data.attrs.get("has_full_freq", 0)) and "full_freq" in plot_data.coords
    if use_full:
        yvals = plot_data["full_freq"].values.astype(float) / 1e9
        center = plot_data["center_full_freq"].values / 1e9
        ylabel = "RF frequency (GHz)"
    else:
        yvals = plot_data.coords["detuning"].values.astype(float) / 1e6
        center = plot_data["center_detuning"].values / 1e6
        ylabel = "Detuning (MHz)"

    good = plot_data["good"].values.astype(bool)
    outlier = plot_data["outlier"].values.astype(bool)

    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    pcm = ax.pcolormesh(flux, yvals, amplitude.T, shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="Amplitude |IQ| (arb. u.)")
    # Kept centres form the trace; rejected (out-of-window / outlier) shown as red x.
    ax.plot(flux[good], center[good], ".-", color="C3", ms=5, lw=1.0, label="centre (kept)")
    if outlier.any():
        ax.plot(flux[outlier], center[outlier], "x", color="red", ms=7, mew=1.5,
                label="rejected (outlier)")
    ax.set_xlabel("Flux bias (V)")
    ax.set_ylabel(ylabel)
    n_good = int(plot_data.attrs.get("n_good", int(good.sum())))
    n_flux = int(plot_data.attrs.get("n_flux", len(flux)))
    ax.set_title(f"Resonator spectroscopy vs flux (kept {n_good}/{n_flux})")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig
