import matplotlib.pyplot as plt
import numpy as np
import xarray as xr


def plot_raw_2d_colormap(dataset: xr.Dataset) -> plt.Figure:
    """Plot the raw signal as a 2D colour-map (charge_gate vs idle_time)."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    idle_time = dataset.coords['idle_time'].values
    charge_gate = dataset.coords['charge_gate'].values

    X, Y = np.meshgrid(idle_time, charge_gate)
    im = ax.pcolormesh(X, Y, dataset['signal'].values, shading='auto', cmap='viridis')
    plt.colorbar(im, ax=ax)

    ax.set_xlabel('Idle Time')
    ax.set_ylabel('Charge Gate (V)')
    ax.set_title('Charge Gate Ramsey – Raw Signal')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_2d_spectrum(results: dict) -> plt.Figure:
    """
    Plot the FFT spectrum as a 2D colour-map with optional f_1/f_2 overlays.

    Parameters
    ----------
    results : dict
        Output of ChargeGateRamseyAnalyzer.extract_parameters.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    spectrum_ds = results.get('spectrum_dataset')
    if spectrum_ds is None:
        ax.text(0.5, 0.5, 'No spectrum data', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        plt.close(fig)
        return fig

    freq = spectrum_ds.coords['frequency'].values
    charge_gate = spectrum_ds.coords['charge_gate'].values
    X, Y = np.meshgrid(freq, charge_gate)

    im = ax.pcolormesh(X, Y, spectrum_ds['spectrum'].values, shading='auto', cmap='plasma')
    plt.colorbar(im, ax=ax, label='Spectrum Amplitude')

    # Overlay f_1 / f_2 markers
    f_1, f_2 = results['f_1'], results['f_2']
    cg = results['charge_gates']

    valid = ~np.isnan(f_1)
    if np.any(valid):
        ax.plot(f_1[valid], cg[valid], 'bo', markersize=4, label='f₁', alpha=0.8)

    valid = ~np.isnan(f_2)
    if np.any(valid):
        ax.plot(f_2[valid], cg[valid], 'ro', markersize=4, label='f₂', alpha=0.8)

    ax.set_xlabel('Frequency')
    ax.set_ylabel('Charge Gate (V)')
    ax.set_title('FFT Spectrum vs Charge Gate')
    if len(freq) > 1:
        ax.set_xlim(freq[1], freq[-1])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_2d_spectrum_with_fit(results: dict) -> plt.Figure:
    """
    Plot the FFT spectrum as a 2D colour-map with |cos| fit overlay.

    Like plot_2d_spectrum but replaces the f_1/f_2 scatter dots with the
    |cos| fit curve plotted as continuous lines (f_c ± fit_curve) vs cg_fine.

    Parameters
    ----------
    results : dict
        Output of ChargeGateRamseyAnalyzer.extract_parameters.
    """
    _FIG_WIDTH = 3.0
    _FIG_HEIGHT = 2.0
    _rc = {
        "figure.figsize": (_FIG_WIDTH, _FIG_HEIGHT),
        "figure.dpi": 300,
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "text.usetex": False,
        "mathtext.fontset": "stix",
        "lines.linewidth": 1.5,
        "lines.markersize": 4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
    }
    with plt.rc_context(_rc):
        return _plot_2d_spectrum_with_fit_inner(results)


def _plot_2d_spectrum_with_fit_inner(results: dict) -> plt.Figure:
    # FIG_WIDTH = 3.8
    # FIG_HEIGHT = 2.6
    # plt.rcParams.update({
    #     "figure.figsize": (FIG_WIDTH, FIG_HEIGHT),
    #     "figure.dpi": 300,
    #     "font.size": 10,
    #     "axes.labelsize": 10,
    #     "legend.fontsize": 10,
    #     "xtick.labelsize": 10,
    #     "ytick.labelsize": 10,
    #     "font.family": "serif",
    #     "text.usetex": True,
    #     "mathtext.fontset": "stix",
    #     "lines.linewidth": 1.5,
    #     "lines.markersize": 4,
    #     "xtick.direction": "in",
    #     "ytick.direction": "in",
    #     "xtick.top": True,
    #     "ytick.right": True,
    #     "xtick.major.size": 3,
    #     "ytick.major.size": 3,
    #     "xtick.minor.size": 2.5,
    #     "ytick.minor.size": 2.5,
    # })
    fig, ax = plt.subplots()

    spectrum_ds = results.get('spectrum_dataset')
    if spectrum_ds is None:
        ax.text(0.5, 0.5, 'No spectrum data', transform=ax.transAxes, ha='center')
        fig.tight_layout()
        plt.close(fig)
        return fig

    freq = spectrum_ds.coords['frequency'].values
    charge_gate = spectrum_ds.coords['charge_gate'].values
    X, Y = np.meshgrid(charge_gate, freq)

    im = ax.pcolormesh(X, Y, spectrum_ds['spectrum'].values.T, shading='auto', cmap='Purples')
    # plt.colorbar(im, ax=ax, label='Spectrum Amplitude')

    # Overlay |cos| fit as f_c ± fit_curve lines
    abscos_params = results.get('abscos_params')
    cg = results['charge_gates']
    if abscos_params is not None and abscos_params.get('success', False):
        amp = abscos_params['amplitude']
        fit_freq = abscos_params['frequency']
        phase = abscos_params['phase']
        f_c = results['f_c']

        cg_fine = np.linspace(cg.min(), cg.max(), 200)
        fit_curve = amp * np.cos(2 * np.pi * fit_freq * (cg_fine - phase))
        # fit_curve_m = -amp * np.cos(2 * np.pi * fit_freq * (cg_fine - phase))
        ax.plot(cg_fine, f_c + fit_curve, 'r--', linewidth=1, alpha=0.7, label='Even')
        ax.plot(cg_fine, f_c - fit_curve, 'b--', linewidth=1, alpha=0.7, label='Odd')

    ax.set_xlabel('$n_g$ (2e)', fontfamily='serif')
    ax.set_ylabel('Detuning (MHz)', fontfamily='serif')
    # ax.legend(loc='upper right', framealpha=0.8)
    # ax.set_title('FFT Spectrum vs Charge Gate (|cos| fit overlay)')
    if len(freq) > 1:
        ax.set_ylim(0, freq[-1] / 1.5)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_1d_frequencies(results: dict) -> plt.Figure:
    """
    Plot |f_1 − f_c| and |f_2 − f_c| vs charge_gate with |cos| fit overlay.

    Parameters
    ----------
    results : dict
        Output of ChargeGateRamseyAnalyzer.extract_parameters.
    """
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)

    cg = results['charge_gates']
    f_1, f_2 = results['f_1'], results['f_2']
    f_c = results['f_c']

    valid_f1 = ~np.isnan(f_1)
    if np.any(valid_f1):
        ax.scatter(cg[valid_f1], np.abs(f_1[valid_f1] - f_c),
                   c='blue', s=50, label='|f₁ − f_c|', alpha=0.8)

    valid_f2 = ~np.isnan(f_2)
    if np.any(valid_f2):
        ax.scatter(cg[valid_f2], np.abs(f_2[valid_f2] - f_c),
                   c='red', s=50, label='|f₂ − f_c|', alpha=0.8)

    # Overlay |cos| fit curve
    abscos_params = results.get('abscos_params')
    if abscos_params is not None and abscos_params.get('success', False):
        amp = abscos_params['amplitude']
        freq = abscos_params['frequency']
        phase = abscos_params['phase']

        cg_fine = np.linspace(cg.min(), cg.max(), 200)
        fit_curve = amp * np.abs(np.cos(2 * np.pi * freq * (cg_fine - phase)))
        ax.plot(cg_fine, fit_curve, 'g-', linewidth=2, label='|cos| fit', alpha=0.8)

        textstr = (
            f"|cos| fit:\n"
            f"  amplitude = {amp:.4g}\n"
            f"  frequency = {freq:.4g} V⁻¹\n"
            f"  phase = {phase:.4g} V\n"
            f"  χ²/dof = {abscos_params.get('redchi', np.nan):.4g}"
        )
        ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                va='top', ha='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    ax.set_xlabel('Charge Gate (V)')
    ax.set_ylabel('Frequency Difference')
    ax.set_title('Ramsey Frequency Dispersion vs Charge Gate')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.close(fig)
    return fig
