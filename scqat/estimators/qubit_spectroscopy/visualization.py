"""
Qubit spectroscopy plotting helpers.

``plot_spectrum`` consumes the **plot_data** Dataset built by
``QubitSpectroscopyEstimator.build_plot_data`` and draws without any
recalculation, so the figure stays reconstructable downstream.

plot_data layout
----------------
coords : ``detuning`` (+ optional ``peak``)
vars   : ``signal`` (detuning), ``baseline`` (detuning),
         ``signal_corrected`` (detuning), optional ``full_freq`` (detuning),
         and per-peak ``peak_fit`` (peak, detuning), ``peak_detuning`` (peak),
         ``peak_fwhm`` (peak)
attrs  : ``n_peaks``, ``has_full_freq``, ``inverted`` (+ optional ``ref_iq_*``)
"""

import matplotlib.pyplot as plt
import xarray as xr


def plot_spectrum(plot_data: xr.Dataset) -> plt.Figure:
    """Single figure showing the spectrum, baseline, and Lorentzian fits
    overlaid at each detected peak, drawn entirely from ``plot_data``."""
    detuning = plot_data.coords["detuning"].values.astype(float)
    signal = plot_data["signal"].values
    baseline = plot_data["baseline"].values
    corrected = plot_data["signal_corrected"].values
    n_peaks = int(plot_data.attrs.get("n_peaks", 0))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
        figsize=(10, 6), dpi=120,
    )

    # -- Top: raw signal + baseline + Lorentzian fits --
    det_mhz = detuning / 1e6
    ax_top.plot(det_mhz, signal, "-", lw=0.8, label="|IQdata - ref|")
    ax_top.plot(det_mhz, baseline, "--", color="gray", lw=1.0, label="baseline")

    for i in range(n_peaks):
        fit_y = plot_data["peak_fit"].isel(peak=i).values
        pk_det = float(plot_data["peak_detuning"].values[i])
        pk_fwhm = float(plot_data["peak_fwhm"].values[i])
        ax_top.plot(
            det_mhz, fit_y + baseline, "-", lw=1.5, color=f"C{i + 1}",
            label=f"peak {i}: {pk_det / 1e6:.2f} MHz, FWHM={pk_fwhm / 1e6:.2f} MHz",
        )
        ax_top.axvline(pk_det / 1e6, color=f"C{i + 1}", ls=":", lw=0.8, alpha=0.7)

    ax_top.set_ylabel("Signal (arb. u.)")
    ax_top.legend(fontsize=8)
    ax_top.set_title("Qubit spectroscopy")

    # Add absolute-frequency twin axis if available
    if plot_data.attrs.get("has_full_freq", 0):
        freq_vals = plot_data["full_freq"].values / 1e9
        ax_freq = ax_top.twiny()
        ax_freq.set_xlim(freq_vals[0], freq_vals[-1])
        ax_freq.set_xlabel("RF frequency (GHz)")

    # -- Bottom: baseline-subtracted + fits --
    ax_bot.plot(det_mhz, corrected, "-", lw=0.8, color="C0")
    for i in range(n_peaks):
        fit_y = plot_data["peak_fit"].isel(peak=i).values
        ax_bot.plot(det_mhz, fit_y, "-", lw=1.5, color=f"C{i + 1}")

    ax_bot.axhline(0, color="k", lw=0.5)
    ax_bot.set_xlabel("Detuning (MHz)")
    ax_bot.set_ylabel("Corrected signal")

    fig.tight_layout()
    return fig
