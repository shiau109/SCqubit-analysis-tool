"""Stage 2 of the resonator-vs-flux analysis: fit the centre-frequency trace.

Fit the 1-D ``center_frequency(flux)`` trace produced by :mod:`.dips` with a
selectable flux-dependence model ("method"):

* ``method="dispersive"`` (default) — the full flux-tunable-transmon dispersive
  model ``f_r = f_r0 + g^2 / (f_r0 - f_q(phi))`` with the transmon arch
  ``f_q(phi) = f_q_max * sqrt(|cos(pi (phi - phi_off) / phi0)|)``. Yields the
  bare resonator ``f_r0`` and (conditional) coupling ``g`` in addition to the
  sweet spot and period. See :mod:`.methods.dispersive` for the ``g``/``f_q_max``
  degeneracy.
* ``method="sine"`` — a bare cosine of the flux
  ``f_r = offset + amp cos(2 pi (phi - phi_off) / T)``. Model-light and robust
  when only ~one arch is visible or the trace is noisy, but yields no
  ``f_r0``/``g``. See :mod:`.methods.sine`.

Result contract (two tiers)
---------------------------
COMMON keys — validated here, identical meaning/unit in every method; the only
keys downstream orchestration may rely on:

    sweet_spot_flux     : flux (V) at the maximum qubit frequency (upper sweet
                          spot), reported inside the swept range
                          (periodic-image resolved)
    sweet_spot_res      : resonator centre frequency (Hz) at that flux
    sweet_spot_low_flux : flux (V) at the minimum qubit frequency (lower sweet
                          spot, half a period away), in-range likewise
    sweet_spot_low_res  : resonator centre frequency (Hz) there
    dv_phi0             : flux period (V)
    success             : bool — fit trustworthy

Method extras (``f_r0``/``g``/``f_q_max``/``max_pull`` for dispersive;
``amp``/``offset`` for sine) ride along in the results; consumers use them only
via ``if key in results``. ``results["method"]`` records provenance.

Like :mod:`.dips`, this is a plain function: inside scqat it is the second stage
of :class:`.estimator.ResonatorSpectroscopyFluxEstimator`; control repos (e.g.
the LCHQMDriver qualibrate node) call :func:`fit_flux_trace` directly on a trace
they assembled themselves.

Trace Dataset contract
----------------------
Coordinates:
    - flux_bias : 1-D float array – applied flux bias (V).
Data variables:
    - center_freq : (flux_bias,) – fitted resonator centre frequency (Hz).
    - success     : (flux_bias,) bool – optional per-point validity mask.
"""

from typing import Any, Dict

import numpy as np
import xarray as xr

from .methods import METHODS

#: Tier-1 keys every method must return — the only keys orchestration may rely
#: on. Same name => same meaning and unit across methods.
COMMON_KEYS = frozenset({
    "sweet_spot_flux", "sweet_spot_res",
    "sweet_spot_low_flux", "sweet_spot_low_res",
    "dv_phi0", "success",
})


def fit_flux_trace(trace: xr.Dataset, **kwargs) -> Dict[str, Any]:
    """Dispatch the trace to the selected method and enforce the COMMON-key
    contract (see module docstring).

    Keyword arguments
    -----------------
    method : str, optional
        Extraction approach: ``"dispersive"`` (default) or ``"sine"``.
    f_q_max, fit_f_q_max
        (dispersive) fix / free the qubit sweet-spot frequency; see
        :class:`.methods.dispersive.DispersiveMethod`.
    """
    if "flux_bias" not in trace.coords:
        raise ValueError("fit_flux_trace requires a 'flux_bias' coordinate.")
    if "center_freq" not in trace:
        raise ValueError("fit_flux_trace requires a 'center_freq' variable.")

    method_name = kwargs.pop("method", "dispersive")
    if method_name not in METHODS:
        raise ValueError(
            f"Unknown method {method_name!r}; available: {sorted(METHODS)}"
        )
    m = METHODS[method_name]

    flux_all = trace.coords["flux_bias"].values.astype(float)
    y_all = np.asarray(trace["center_freq"].values, dtype=float)

    mask = np.isfinite(flux_all) & np.isfinite(y_all)
    if "success" in trace:
        mask &= np.asarray(trace["success"].values, dtype=bool)
    flux = flux_all[mask]
    y = y_all[mask]
    order = np.argsort(flux)
    flux, y = flux[order], y[order]

    results = m.extract(flux, y, flux_all, y_all, **kwargs)

    missing = COMMON_KEYS - results.keys()
    if missing:  # a method bug, not a data problem — fail loudly
        raise RuntimeError(
            f"method {method_name!r} violated the flux-trace contract: "
            f"missing common keys {sorted(missing)}"
        )
    results["method"] = method_name
    return results
