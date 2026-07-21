# %% [markdown]
# # Resonator spectroscopy — method comparison on saved SCQO runs
#
# Compares the estimator's extraction methods (`lorentzian` joint-background
# fit vs `circle` Probst notch fit) on a real saved run, plus a
# `baseline_order` sweep and a cross-check against the vendored NCU
# `resonator_tools.notch_port` reference implementation.
#
# Data source: an SCQO **datastore run folder** (`dataset.nc` — netCDF, opened
# with `xr.open_dataset`, *not* the LCHQM `ds_raw.h5` path used by other
# try-scripts). Point `RUN` at any resonator_spectroscopy run that contains
# `dataset.nc`.
#
# NOTE: simulated-backend runs have a noise-only Q quadrature, so the `circle`
# method may legitimately report `success=False` there — swap in a
# real-hardware run to see it shine (it calibrates cable delay / asymmetry).

# %% Setup
import json
import os
import sys

import numpy as np
import xarray as xr

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from _harness import slices, compare, estimator_method, replot  # noqa: E402
from scqat.estimators.resonator_spectroscopy import ResonatorSpectroscopyEstimator  # noqa: E402

RUN = r"D:\qpu_data_dev\chipA\2026-07-17\20260717-091946-chipA-resonator_spectroscopy-01"
OUT_DIR = os.path.join(_HERE, "out", "resonator_spectroscopy")
os.makedirs(OUT_DIR, exist_ok=True)

DS = xr.open_dataset(os.path.join(RUN, "dataset.nc"))
with open(os.path.join(RUN, "device_before.json"), encoding="utf-8") as fh:
    DEVICE = json.load(fh)
EST = ResonatorSpectroscopyEstimator()
print(DS)

# %% Glue: prep (raw run -> estimator contract) and adapt (results -> compare fields)
def prep(sq: xr.Dataset) -> xr.Dataset:
    """Rename the SCQO sweep axis and attach the absolute-frequency coord the
    circle method needs (detuning + this qubit's readout_freq at run time)."""
    qubit = sq["qubit"].values.item()
    sq = sq.rename({"detuning_hz": "detuning"})
    f0 = float(DEVICE[qubit]["readout_freq"])
    return sq.assign_coords(full_freq=("detuning", sq["detuning"].values + f0))


def adapt(res: dict, sq: xr.Dataset) -> dict | None:
    """Map estimator results to the normalized compare() fields, per method."""
    if not res.get("success", False):
        return None
    if res["method"] == "circle":
        y = np.hypot(res["raw_i"], res["raw_q"])
        fit_y = np.hypot(res["fit_i"], res["fit_q"])
        x = sq["detuning"].values.astype(float)
        return {"detuning": res["detuning"], "fwhm": res["fwhm"],
                "x": x, "y": y, "fit_x": x, "fit_y": fit_y}
    return {"detuning": res["detuning"], "fwhm": res["fwhm"],
            "x": res["fit_x"], "y": res["signal_corrected"],
            "fit_x": res["fit_x"], "fit_y": res["fit_y"]}


# %% (A) lorentzian vs circle on every qubit of the run
m_lorentzian = estimator_method(EST, adapt, label="lorentzian (joint bg)", method="lorentzian")
m_circle = estimator_method(EST, adapt, label="circle (Probst)", method="circle")
rows, fig = compare(
    slices(DS, prep=prep),
    [m_lorentzian, m_circle],
    out_png=os.path.join(OUT_DIR, "methods_compare.png"),
)

# %% (B) lorentzian background-order sweep
order_methods = [
    estimator_method(EST, adapt, label=f"baseline_order={k}",
                     method="lorentzian", baseline_order=k)
    for k in (0, 1, 2)
]
rows_b, fig_b = compare(
    slices(DS, prep=prep),
    order_methods,
    out_png=os.path.join(OUT_DIR, "baseline_order_sweep.png"),
)

# %% (C) cross-check the circle method against the vendored NCU notch_port
# (GPL reference implementation — imported here in analysis/ ONLY, never in
# the scqat package; our tools/fit_notch_circle is an independent
# implementation of the published algorithm)
NCU_REPO = r"D:\github\NCU-AS-program-cooperation"
try:
    if NCU_REPO not in sys.path:
        sys.path.insert(0, NCU_REPO)
    from resonator_tools.circuit import notch_port  # noqa: E402

    print(f"{'qubit':<8}{'source':<16}{'fr (GHz)':>12}{'Ql':>10}{'|Qc|':>10}{'Qi_dia':>10}")
    for name, sq in slices(DS, prep=prep):
        f = sq["full_freq"].values.astype(float)
        z = (sq["I"] + 1j * sq["Q"]).values.ravel()
        ours = EST.extract_parameters(sq, method="circle")
        port = notch_port(f_data=f, z_data_raw=z)
        port.autofit()
        ncu = port.fitresults
        print(f"{name:<8}{'scqat circle':<16}{ours['full_freq']/1e9:>12.6f}"
              f"{ours['Ql']:>10.0f}{ours['absQc']:>10.0f}{ours['Qi_dia_corr']:>10.0f}")
        print(f"{'':<8}{'NCU notch_port':<16}{ncu['fr']/1e9:>12.6f}"
              f"{ncu['Ql']:>10.0f}{ncu['absQc']:>10.0f}{ncu['Qi_dia_corr']:>10.0f}")
except Exception as err:  # simulated data / import trouble must not kill the script
    print(f"NCU cross-check skipped: {type(err).__name__}: {err}")

# %% (D) re-fit the saved run and write full estimator artifacts per qubit
# (metadata JSON + plotdata .nc + figure PNG — the same layout a datastore
# run's analysis/ folder gets)
import matplotlib.pyplot as plt  # noqa: E402

for name, sq in slices(DS, prep=prep):
    unit_dir = os.path.join(OUT_DIR, str(name))
    results, figs = EST.analyze(sq, output_dir=unit_dir, method="lorentzian")
    print(f"{name}: success={results['success']} "
          f"detuning={results['detuning'] / 1e6:.3f} MHz "
          f"fwhm={results['fwhm'] / 1e6:.3f} MHz -> {unit_dir}")
    for fig in figs.values():
        plt.close(fig)

# %% (E) replot with NO re-fit from the plot data saved by (D)
# (attrs-based dispatch: the .nc file alone knows which method's figure to draw)
plotdata_files = {
    str(name): os.path.join(OUT_DIR, str(name), "resonator_spectroscopy_plotdata.nc")
    for name, _ in slices(DS, prep=prep)
}
replot(EST, from_plotdata=plotdata_files, out_dir=os.path.join(OUT_DIR, "replot"))
