"""Resonator spectroscopy plotting helpers (one figure per method).

Both functions consume only the **plot_data** Dataset built by the matching
method's ``build_plot_data`` and draw with zero recalculation, so a saved
``plotdata.nc`` reconstructs the figure offline. ``plot_data.attrs["method"]``
decides which function the estimator dispatches to.

lorentzian plot_data layout
---------------------------
coords : ``detuning`` (+ optional ``full_freq`` data var)
vars   : ``power`` / ``power_baseline`` / ``power_corrected`` / ``power_fit``
         / ``baseline_seed`` (int8 mask), each over ``detuning``
attrs  : ``method``, ``resonator_detuning``, ``fwhm``, ``success``,
         ``has_full_freq`` (+ optional ``resonator_frequency``)

circle plot_data layout
-----------------------
coords : ``detuning``
vars   : ``raw_i``/``raw_q``, ``fit_i``/``fit_q``, ``norm_i``/``norm_q``,
         ``norm_fit_i``/``norm_fit_q``, ``full_freq``, each over ``detuning``
attrs  : ``method``, ``resonator_detuning``, ``fwhm``, ``success``,
         ``resonator_frequency``, ``Ql``/``absQc``/``Qc_dia_corr``/
         ``Qi_dia_corr``/``phi0``/``delay`` (+ ``*_err``)
"""

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


def _full_freq_twin(ax, plot_data):
    """Add an absolute-frequency (GHz) twin x-axis when available."""
    if plot_data.attrs.get("has_full_freq", 0):
        freq_vals = plot_data["full_freq"].values / 1e9
        ax_freq = ax.twiny()
        ax_freq.set_xlim(freq_vals[0], freq_vals[-1])
        ax_freq.set_xlabel("RF frequency (GHz)")


def plot_lorentzian(plot_data: xr.Dataset) -> plt.Figure:
    """Two-panel power-domain diagnostic of the joint background+Lorentzian fit.

    Top: raw power with the jointly-fitted background, the full fit, and the
    highlighted background-seed points — showing *how* the background was
    determined. Bottom: background-subtracted power with the Lorentzian
    component, i.e. what the extracted dip actually looks like.
    """
    detuning = plot_data.coords["detuning"].values.astype(float)
    power = plot_data["power"].values
    baseline = plot_data["power_baseline"].values
    corrected = plot_data["power_corrected"].values
    fit = plot_data["power_fit"].values
    seed = plot_data["baseline_seed"].values.astype(bool)
    det_res = float(plot_data.attrs.get("resonator_detuning", np.nan))
    fwhm = float(plot_data.attrs.get("fwhm", np.nan))
    success = bool(plot_data.attrs.get("success", 0))

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True,
        gridspec_kw={"height_ratios": [3, 2]},
        figsize=(10, 7), dpi=120,
    )

    mhz = detuning / 1e6
    ax_top.plot(mhz, power, "-", lw=0.9, color="C0", label="power $|IQ|^2$")
    ax_top.plot(mhz, baseline, "--", lw=1.6, color="crimson",
                label="background (joint fit)")
    ax_top.plot(mhz, fit, "-", lw=1.4, color="C1", alpha=0.9,
                label="full fit (background + Lorentzian)")
    ax_top.scatter(mhz[seed], power[seed], s=18, color="orange", zorder=5,
                   edgecolor="k", linewidth=0.3, label="background seed points")
    ax_top.set_ylabel("power $|IQ|^2$ (arb. u.)")
    title = "Resonator spectroscopy — joint background + Lorentzian fit"
    if not success:
        title += "  [FIT FAILED]"
    ax_top.set_title(title)
    ax_top.legend(fontsize=8, loc="best")
    _full_freq_twin(ax_top, plot_data)

    ax_bot.axhline(0, color="k", lw=0.5)
    ax_bot.plot(mhz, corrected, "-", lw=0.9, color="C0",
                label="power − background")
    ax_bot.plot(mhz, fit - baseline, "-", lw=1.8, color="C1",
                label=f"Lorentzian (FWHM = {fwhm / 1e6:.3f} MHz)")
    if np.isfinite(det_res):
        ax_bot.axvline(det_res / 1e6, color="C3", ls=":", lw=1.0,
                       label=f"f_res @ {det_res / 1e6:.3f} MHz")
    ax_bot.set_xlabel("Detuning (MHz)")
    ax_bot.set_ylabel("corrected power")
    ax_bot.legend(fontsize=8, loc="best")

    fig.tight_layout()
    return fig


def plot_circle(plot_data: xr.Dataset) -> plt.Figure:
    """Three-panel circle-fit diagnostic: normalized IQ-plane circle,
    |S21| magnitude, and phase, each with the notch-model fit overlaid."""
    detuning = plot_data.coords["detuning"].values.astype(float)
    mhz = detuning / 1e6
    raw = plot_data["raw_i"].values + 1j * plot_data["raw_q"].values
    fit = plot_data["fit_i"].values + 1j * plot_data["fit_q"].values
    norm = plot_data["norm_i"].values + 1j * plot_data["norm_q"].values
    norm_fit = plot_data["norm_fit_i"].values + 1j * plot_data["norm_fit_q"].values
    full_freq = plot_data["full_freq"].values.astype(float)

    fr = float(plot_data.attrs.get("resonator_frequency", np.nan))
    fwhm = float(plot_data.attrs.get("fwhm", np.nan))
    ql = float(plot_data.attrs.get("Ql", np.nan))
    qc = float(plot_data.attrs.get("Qc_dia_corr", np.nan))
    qi = float(plot_data.attrs.get("Qi_dia_corr", np.nan))
    qi_err = float(plot_data.attrs.get("Qi_dia_corr_err", np.nan))
    success = bool(plot_data.attrs.get("success", 0))

    fig = plt.figure(figsize=(11, 5.5), dpi=120)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.05, 1.45])
    ax_iq = fig.add_subplot(gs[:, 0])
    ax_mag = fig.add_subplot(gs[0, 1])
    ax_ph = fig.add_subplot(gs[1, 1], sharex=ax_mag)

    # -- IQ plane (normalized): the resonance circle --
    ax_iq.plot(norm.real, norm.imag, ".", ms=3.5, color="C0", alpha=0.7,
               label="data (normalized)")
    ax_iq.plot(norm_fit.real, norm_fit.imag, "-", lw=1.6, color="C1",
               label="notch fit")
    ax_iq.plot([1.0], [0.0], "k+", ms=9, label="off-resonant point")
    if np.isfinite(fr):
        idx = int(np.argmin(np.abs(full_freq - fr)))
        ax_iq.plot(norm_fit.real[idx], norm_fit.imag[idx], "o", ms=6,
                   color="C3", label="$f_r$")
    ax_iq.set_aspect("equal", adjustable="datalim")
    ax_iq.set_xlabel("Re $S_{21}$")
    ax_iq.set_ylabel("Im $S_{21}$")
    ax_iq.legend(fontsize=7, loc="best")
    box = (f"$f_r$ = {fr / 1e9:.6f} GHz\n"
           f"$\\kappa/2\\pi$ = {fwhm / 1e6:.3f} MHz\n"
           f"$Q_l$ = {ql:.0f}\n$Q_c$ = {qc:.0f}\n"
           f"$Q_i$ = {qi:.0f} ± {qi_err:.0f}")
    ax_iq.text(0.03, 0.03, box, transform=ax_iq.transAxes, fontsize=7,
               va="bottom", ha="left",
               bbox=dict(boxstyle="round", fc="white", alpha=0.75))

    # -- magnitude --
    ax_mag.plot(mhz, np.abs(raw), ".", ms=3, color="C0", alpha=0.7, label="data")
    ax_mag.plot(mhz, np.abs(fit), "-", lw=1.6, color="C1", label="fit")
    ax_mag.set_ylabel("$|S_{21}|$ (arb. u.)")
    title = "Resonator spectroscopy — circle fit (Probst notch model)"
    if not success:
        title += "  [FIT FAILED]"
    ax_mag.set_title(title)
    ax_mag.legend(fontsize=8, loc="best")

    # -- phase (the cable delay shows up as the overall slope) --
    ax_ph.plot(mhz, np.unwrap(np.angle(raw)), ".", ms=3, color="C0", alpha=0.7)
    ax_ph.plot(mhz, np.unwrap(np.angle(fit)), "-", lw=1.6, color="C1")
    ax_ph.set_xlabel("Detuning (MHz)")
    ax_ph.set_ylabel("arg $S_{21}$ (rad)")

    fig.tight_layout()
    return fig
