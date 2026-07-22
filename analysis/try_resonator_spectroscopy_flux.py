# %% [markdown]
# # Resonator spectroscopy vs flux — method comparison on saved SCQO runs
#
# Compares the composite estimator's stage-2 flux-models (`dispersive` — the full
# flux-tunable-transmon dispersive pull — vs `sine` — a bare cosine of the flux)
# on a real saved run, and shows the effect of the stage-1 `edge_margin_frac`
# gate that rejects dip centres pinned to the detuning-window edges on low-SNR
# slices.
#
# Data source: an SCQO **datastore run folder** (`dataset.nc` + `device_before.json`).
# Point `RUN` at any resonator_spectroscopy_flux run.
#
# The reference failure (run 20260721-190228, q1): the raw fit reported the
# sweet-spot flux at -0.176 V / 5.939 GHz because six edge-pinned
# spurious centres survived stage 1 and captured the fit seed. With the edge gate
# on, both methods recover ~+0.107 V / 5.936 GHz.

# %% Setup
import json
import os
import sys

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from scqat.estimators.resonator_spectroscopy_flux import ResonatorSpectroscopyFluxEstimator  # noqa: E402

RUN = r"D:\qpu_data_dev\5Q4C\2026-07-21\20260721-190228-5Q4C-resonator_spectroscopy_flux-01"
OUT_DIR = os.path.join(_HERE, "out", "resonator_spectroscopy_flux")
os.makedirs(OUT_DIR, exist_ok=True)

DS = xr.open_dataset(os.path.join(RUN, "dataset.nc"))
with open(os.path.join(RUN, "device_before.json"), encoding="utf-8") as fh:
    DEVICE = json.load(fh)
EST = ResonatorSpectroscopyFluxEstimator()
print(DS)


# %% Glue: prep a per-target slice into the estimator's (flux_bias, detuning) contract
def prep(target: str) -> xr.Dataset:
    """Rename the SCQO sweep axes and attach the absolute-frequency coord
    (detuning + this qubit's readout_freq at run time)."""
    sq = DS.sel(target=target) if "target" in DS.dims else DS
    sq = sq.rename({"flux_bias_v": "flux_bias", "detuning_hz": "detuning"})
    sq = sq.transpose("flux_bias", "detuning")
    f0 = float(DEVICE[target]["readout_freq"])
    return sq.assign_coords(full_freq=("detuning", sq["detuning"].values + f0))


TARGETS = [str(t) for t in DS["target"].values] if "target" in DS.dims else ["q1"]

# %% (A) dispersive vs sine on every target — print the COMMON physics + save figures
hdr = f"{'target':8}{'method':12}{'sweet_spot_flux/V':>16}{'sweet_spot_res/GHz':>18}{'dv_phi0/V':>12}{'n_good':>8}"
print(hdr)
print("-" * len(hdr))
for target in TARGETS:
    sq = prep(target)
    for method in ("dispersive", "sine"):
        unit_dir = os.path.join(OUT_DIR, f"{target}_{method}")
        results, figs = EST.analyze(sq, output_dir=unit_dir, method=method, edge_margin_frac=0.06)
        disp, vs = results["dispersion"], results["vs_flux"]
        print(f"{target:8}{method:12}{disp['sweet_spot_flux']:>16.4f}"
              f"{disp['sweet_spot_res'] / 1e9:>18.5f}{disp['dv_phi0']:>12.4f}"
              f"{vs['n_good']:>8d}")
        for fig in figs.values():
            plt.close(fig)

# %% (B) edge-gate on/off contrast (dispersive) — the reference failure mode
print(f"\n{'target':8}{'edge_margin_frac':>18}{'sweet_spot_flux/V':>16}{'sweet_spot_res/GHz':>18}{'n_good':>8}")
for target in TARGETS:
    sq = prep(target)
    for emf in (0.0, 0.06):
        r = EST.extract_parameters(sq, method="dispersive", edge_margin_frac=emf)
        disp, vs = r["dispersion"], r["vs_flux"]
        print(f"{target:8}{emf:>18.2f}{disp['sweet_spot_flux']:>16.4f}"
              f"{disp['sweet_spot_res'] / 1e9:>18.5f}{vs['n_good']:>8d}")

# %% (C) replot with NO re-fit from the plot data saved by (A)
# (attrs-based dispatch: the .nc file alone knows which method's figure to draw)
for target in TARGETS:
    for method in ("dispersive", "sine"):
        pd_path = os.path.join(OUT_DIR, f"{target}_{method}",
                               "resonator_spectroscopy_flux_plotdata.nc")
        plot_data = xr.load_dataset(pd_path)
        figs = EST.generate_figures(None, None, plot_data=plot_data)
        for name, fig in figs.items():
            fig.savefig(os.path.join(OUT_DIR, f"{target}_{method}__{name}.png"),
                        bbox_inches="tight")
            plt.close(fig)
        print(f"{target}/{method}: replotted from saved plot-data (no re-fit)")
