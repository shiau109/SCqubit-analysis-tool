"""
Qubit decoherence plotting helper.

``plot_decoherence`` consumes the **plot_data** Dataset built by
``QubitDecoherenceEstimator.build_plot_data`` and draws without any
recalculation — one figure per fitted variable (``rho_11`` / ``rho_10``).

plot_data layout
----------------
coords : ``time``
vars   : per present variable ``<v>_data`` / ``<v>_fit`` / ``<v>_residual`` (time)
attrs  : per present variable ``<v>_gamma`` / ``<v>_lambda`` / ``<v>_Delta`` /
         ``<v>_regime``, plus ``variables`` (comma-joined list)
"""

from typing import Dict

import matplotlib.pyplot as plt
import xarray as xr


def plot_decoherence(plot_data: xr.Dataset) -> Dict[str, plt.Figure]:
    """One figure per fitted variable: data + fit overlay with a residual subplot,
    drawn entirely from ``plot_data``."""
    t = plot_data.coords["time"].values
    figs: Dict[str, plt.Figure] = {}

    for var_name in ("rho_11", "rho_10"):
        if f"{var_name}_data" not in plot_data:
            continue
        y_data = plot_data[f"{var_name}_data"].values
        y_fit = plot_data[f"{var_name}_fit"].values
        residuals = plot_data[f"{var_name}_residual"].values
        label = r"$\rho_{11}$" if var_name == "rho_11" else r"$\rho_{10}$"

        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1]},
        )

        ax_top.plot(t, y_data, "o", ms=3, alpha=0.6, label=f"{label} data")
        ax_top.plot(
            t, y_fit, "-",
            label=(
                f"fit ("
                f"γ={plot_data.attrs[f'{var_name}_gamma']:.4g}, "
                f"λ={plot_data.attrs[f'{var_name}_lambda']:.4g}, "
                f"Δ={plot_data.attrs[f'{var_name}_Delta']:.4g})"
            ),
        )
        ax_top.set_ylabel(label)
        ax_top.legend()
        ax_top.set_title(
            f"{label}(t) decoherence fit  [{plot_data.attrs[f'{var_name}_regime']}]"
        )

        ax_bot.plot(t, residuals, ".-", ms=2)
        ax_bot.axhline(0, color="k", lw=0.5)
        ax_bot.set_xlabel("Time")
        ax_bot.set_ylabel("Residual")

        fig.tight_layout()
        figs[var_name] = fig

    return figs
