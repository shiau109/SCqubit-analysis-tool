"""
analysis/_harness.py — reusable offline harness for working with saved estimator data.

Why
---
``BaseEstimator.analyze(ds) -> (results, figs)`` is uniform across every estimator,
so ONE set of helpers covers the offline use cases:

* **try a new approach** — compare a custom method against the estimator (``compare``);
* **test parameters**    — run the estimator across a grid of kwargs (``estimator_method`` + ``compare``);
* **replot**             — regenerate the estimator's own figures from saved raw data
                           (e.g. an LCHQM run that stored ds_raw but skipped plotting) (``replot``).

The only per-experiment differences are how a saved ``ds_raw.h5`` is shaped into what the
estimator expects (``prep``) and how a ``results`` dict maps to normalized plot fields
(``adapt``) — both supplied by the thin per-experiment ``# %%`` script. Keep this module
logic-only so it stays git-friendly.

Normalized "method"
-------------------
A *method* is any callable ``sq (xr.Dataset) -> dict`` returning::

    {"label": str, "ok": bool,
     "detuning": float,  # Hz
     "fwhm": float,      # Hz
     "x", "y", "fit_x", "fit_y": 1-D arrays}   # x in Hz; omit the rest when ok is False

``adapt`` (per experiment)
--------------------------
``adapt(results: dict, sq: xr.Dataset) -> dict | None`` extracts the normalized fields
(detuning/fwhm/x/y/fit_x/fit_y) from an estimator ``results`` dict, or ``None`` if the
fit found nothing. ``estimator_method`` uses it to turn any estimator+kwargs into a method.
"""
from __future__ import annotations

import os
from typing import Callable

import xarray as xr
import matplotlib.pyplot as plt

from scqat.parsers import load_xarray_h5, repetition_data

__all__ = ["load", "slices", "compare", "estimator_method", "replot"]

Method = Callable[[xr.Dataset], dict]
Prep = Callable[[xr.Dataset], xr.Dataset]
Adapt = Callable[[dict, xr.Dataset], "dict | None"]


def load(path: str) -> xr.Dataset:
    """Load a saved ``ds_raw.h5`` (or any xarray HDF5) into memory."""
    return load_xarray_h5(path)


def slices(ds: xr.Dataset, prep: Prep | None = None, dim: str = "qubit"):
    """Split ``ds`` along ``dim`` into per-unit slices, applying ``prep`` (e.g. build
    IQdata / assign full_freq) to each. Returns ``[(name, sq), ...]``."""
    out = []
    for sq in repetition_data(ds, repetition_dim=dim):
        name = sq[dim].values.item()
        out.append((name, prep(sq) if prep is not None else sq))
    return out


def estimator_method(estimator, adapt: Adapt, label: str | None = None, **analyze_kwargs) -> Method:
    """Turn ``estimator`` + ``analyze_kwargs`` into a comparable *method* (``sq -> dict``).

    Use it for the baseline, and for parameter testing pass one per kwarg set, e.g.::

        [estimator_method(est, adapt, label=f"prom={p}", prominence=p) for p in (...)]
    """
    name = label or getattr(estimator, "estimator_name", type(estimator).__name__)

    def method(sq: xr.Dataset) -> dict:
        results = estimator.analyze(sq, output_dir=None, skip_figures=True, **analyze_kwargs)[0]
        fields = adapt(results, sq)
        if fields is None:
            return {"label": name, "ok": False}
        return {"label": name, "ok": True, **fields}

    method.__name__ = name
    return method


def compare(slices_, methods, out_png=None, freq_scale: float = 1e6, freq_unit: str = "MHz"):
    """Run each method on each ``(name, sq)`` slice; print a detuning/FWHM table and
    draw a rows=slices x cols=methods grid (signal + fit + centre line). Saves a PNG if
    ``out_png`` is given. Returns ``(rows, fig)``.

    ``methods`` is a list of callables ``sq -> normalized dict`` (see module docstring).
    """
    n_rows, n_cols = len(slices_), max(len(methods), 1)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.2 * n_rows),
                             squeeze=False, dpi=110)
    head = f"{'unit':6} {'method':30} {'detuning/' + freq_unit:>16} {'FWHM/' + freq_unit:>14}"
    print(head)
    print("-" * len(head))
    rows = []
    for r, (name, sq) in enumerate(slices_):
        for c, method in enumerate(methods):
            out = method(sq)
            label = out.get("label") or getattr(method, "__name__", f"method{c}")
            ax = axes[r][c]
            ax.set_xlabel(f"detuning ({freq_unit})")
            if not out.get("ok", False):
                ax.set_title(f"{name} — {label}: NO FIT")
                print(f"{name:6} {label:30} {'--':>16} {'--':>14}")
                rows.append({"unit": name, "method": label, "detuning": None, "fwhm": None})
                continue
            ax.plot(out["x"] / freq_scale, out["y"], ".", ms=3, label="signal")
            ax.plot(out["fit_x"] / freq_scale, out["fit_y"], "-", lw=1.6, color="C1", label="fit")
            ax.axvline(out["detuning"] / freq_scale, color="C3", ls=":", lw=1.0)
            ax.set_title(f"{name} — {label}")
            ax.legend(fontsize=8)
            print(f"{name:6} {label:30} {out['detuning'] / freq_scale:>16.3f} {out['fwhm'] / freq_scale:>14.3f}")
            rows.append({"unit": name, "method": label,
                         "detuning": out["detuning"], "fwhm": out["fwhm"]})
    fig.tight_layout()
    if out_png:
        fig.savefig(out_png, bbox_inches="tight")
        print(f"\nsaved comparison -> {out_png}")
    return rows, fig


def replot(estimator, slices_, out_dir=None, **analyze_kwargs):
    """Regenerate the estimator's OWN figures per slice from saved raw data — the
    figures an LCHQM run would have produced had ``plot`` not been skipped.

    For each ``(name, sq)``: re-run the (deterministic) fit, call
    ``estimator.generate_figures`` and, if ``out_dir`` is given, save each figure as
    ``<out_dir>/<name>__<figname>.png``. Also prints the persisted metadata keys so you
    can eyeball the extracted parameters. Returns ``{name: {figname: Figure}}``.
    """
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    figures = {}
    for name, sq in slices_:
        results = estimator.analyze(sq, output_dir=None, skip_figures=True, **analyze_kwargs)[0]
        figs = estimator.generate_figures(sq, results)
        figures[name] = figs
        meta = estimator.extract_metadata(results)
        print(f"{name}: figures={list(figs)}  metadata={meta}")
        if out_dir:
            for fig_name, fig in figs.items():
                path = os.path.join(out_dir, f"{name}__{fig_name}.png")
                fig.savefig(path, bbox_inches="tight")
                print(f"   saved {path}")
    return figures
