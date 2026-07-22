"""Per-trace resonator dip fit — the reduction shared by an experiment FAMILY.

One spectrum trace (detuning axis + complex IQ) in, a few numbers out. This is
the pure-math reduction behind every resonator-dip experiment: 1-D resonator
spectroscopy uses it once, the vs-flux and vs-power maps call it once per slice.
Per the repo rule ("anything used by more than one estimator lives in tools/"),
it lives here — estimators compose it, they never call each other.

Methods
-------
``lorentzian`` (default)
    Joint Lorentzian + polynomial-background fit of the readout **power**
    ``|IQ|^2`` (:class:`scqat.tools.fit_lorentzian_bg.FitLorentzianBG`) — the
    quantity that is truly Lorentzian, so the fitted FWHM equals kappa. The
    smooth background is fitted jointly with the dip, so it cannot bias the
    centre. Knobs: ``baseline_order`` (0/1/2, default 1).
``circle``
    Probst notch-model circle fit of the raw complex S21
    (:class:`scqat.tools.fit_notch_circle.FitNotchCircle`) — calibrates the
    environment (amplitude, phase offset, cable delay) analytically and also
    yields Qi/Qc. Requires the **absolute** frequency axis (``full_freq``).
    Knobs: ``delay`` (fix the cable delay in seconds instead of fitting it).

Result contract (two tiers)
---------------------------
REQUIRED keys — identical meaning/unit in every method; the only keys a caller
may rely on:

    detuning : fitted dip centre, Hz relative to the axis given
    fwhm     : total linewidth kappa_tot, Hz
    success  : bool — fit trustworthy

``full_freq`` (absolute centre, Hz) is added whenever the ``full_freq`` axis is
supplied. Everything else is method-owned extras (``amplitude``/``bg_c*``/plot
arrays for lorentzian; ``Ql``/``absQc``/``Qi_dia_corr``/IQ-pair arrays for
circle) — consumers use them only via ``.get`` / ``if key in r``.

Callers that loop over slices should call :func:`validate_dip_kwargs` ONCE
before the loop, so a typo'd knob dies loudly instead of being swallowed by a
per-slice ``try/except``.
"""

from typing import Dict, Optional

import numpy as np
import xarray as xr

from .fit_lorentzian import lorentzian
from .fit_lorentzian_bg import FitLorentzianBG
from .fit_notch_circle import FitNotchCircle

#: caller-selectable knobs per method — the single source of truth callers
#: validate against (and SCQO's Literal-sync test reads the registry keys).
DIP_KNOBS = {
    "lorentzian": frozenset({"baseline_order"}),
    "circle": frozenset({"delay"}),
}


def validate_dip_kwargs(method: str, knobs: Dict) -> None:
    """Raise ValueError for an unknown method or knob — call BEFORE slice loops."""
    if method not in DIP_KNOBS:
        raise ValueError(
            f"Unknown dip method {method!r}; available: {sorted(DIP_KNOBS)}"
        )
    unknown = set(knobs) - DIP_KNOBS[method]
    if unknown:
        raise ValueError(
            f"Unknown knob(s) {sorted(unknown)} for dip method {method!r}; "
            f"valid: {sorted(DIP_KNOBS[method])}"
        )


def _fit_dip_lorentzian(detuning: np.ndarray, iq: np.ndarray, **knobs) -> Dict:
    """Joint Lorentzian+background fit of |IQ|^2 (see module docstring)."""
    baseline_order = int(knobs.get("baseline_order", 1))

    signal = np.abs(iq) ** 2

    da = xr.DataArray(signal, coords={"x": detuning}, dims="x")
    gamma_max = float(detuning[-1] - detuning[0]) or 1.0
    fitter = FitLorentzianBG(
        da,
        inverted=True,
        background_order=baseline_order,
        bounds={"x0": (float(detuning.min()), float(detuning.max())),
                "gamma": (0.0, abs(gamma_max))},
    )
    try:
        result = fitter.fit()
        p = result.params
        x0 = float(p["x0"].value)
        amplitude = float(p["amplitude"].value)
        gamma = float(p["gamma"].value)
        errs = {
            "detuning_err": float(p["x0"].stderr) if p["x0"].stderr is not None else np.nan,
            "amplitude_err": float(p["amplitude"].stderr) if p["amplitude"].stderr is not None else np.nan,
            "fwhm_err": 2 * float(p["gamma"].stderr) if p["gamma"].stderr is not None else np.nan,
        }
        baseline = np.asarray(fitter.baseline_values(), dtype=float)
        seed_mask = np.asarray(fitter.seed_mask, dtype=bool)
        bg = {f"bg_c{k}": float(p[f"c{k}"].value) for k in range(3)}
        converged = bool(getattr(result, "success", False))
    except Exception:
        # Fall back to the raw minimum against a flat median background
        med = float(np.median(signal))
        baseline = np.full_like(signal, med)
        idx = int(np.argmin(signal))
        x0 = float(detuning[idx])
        amplitude = float(signal[idx] - med)
        gamma = abs(detuning[1] - detuning[0]) * 5 if len(detuning) > 1 else 1.0
        errs = {"detuning_err": np.nan, "amplitude_err": np.nan, "fwhm_err": np.nan}
        seed_mask = np.ones(len(detuning), dtype=bool)
        bg = {"bg_c0": med, "bg_c1": 0.0, "bg_c2": 0.0}
        converged = False

    fwhm = 2.0 * abs(gamma)
    # Lorentzian component alone (zero offset): full model = baseline + this
    fit_y = lorentzian(detuning, x0, amplitude, gamma, 0.0)

    in_span = float(detuning.min()) <= x0 <= float(detuning.max())
    success = bool(converged and in_span and np.isfinite(fwhm) and fwhm > 0)

    return {
        "detuning": x0,
        "fwhm": float(fwhm),
        "success": success,
        "amplitude": amplitude,
        **errs,
        **bg,
        "baseline_order": baseline_order,
        "signal": signal,
        "baseline": baseline,
        "signal_corrected": signal - baseline,
        "fit_x": detuning,
        "fit_y": fit_y,
        "seed_mask": seed_mask,
    }


def _fit_dip_circle(detuning: np.ndarray, iq: np.ndarray,
                    full_freq: Optional[np.ndarray], **knobs) -> Dict:
    """Probst notch-model circle fit of complex S21 (see module docstring)."""
    if full_freq is None:
        raise ValueError(
            "method='circle' requires the 'full_freq' coordinate (absolute "
            "readout frequency in Hz): the notch model term Ql*(f/fr - 1) is "
            "meaningless on a detuning axis that crosses zero. Attach "
            "full_freq or use method='lorentzian'."
        )
    f = np.asarray(full_freq, dtype=float)

    fitter = FitNotchCircle(iq, x=f, delay=knobs.get("delay"))
    res = fitter.fit()

    fr, ql = res["fr"], res["Ql"]
    fwhm = fr / ql if ql else np.nan
    # full_freq = detuning + const, so the detuning axis is linear in f and
    # the fr error maps 1:1 onto the detuning error
    order = np.argsort(f)
    det = float(np.interp(fr, f[order], np.asarray(detuning, dtype=float)[order]))
    rel_fr = res["fr_err"] / fr if fr else np.nan
    rel_ql = res["Ql_err"] / ql if ql else np.nan
    fwhm_err = abs(fwhm) * float(np.hypot(rel_fr, rel_ql))

    # Resolution floor: a fitted linewidth below the frequency grid step is
    # unresolvable — the degenerate fit phase-less (e.g. simulated) data
    # produces. Same convention as qubit_spectroscopy's min_fwhm_factor.
    step = float(np.median(np.abs(np.diff(f)))) if len(f) > 1 else 0.0
    success = (bool(res["success"]) and np.isfinite(fwhm)
               and fwhm > 0.5 * step)

    return {
        "detuning": det,
        "fwhm": float(fwhm),
        "success": success,
        "full_freq": float(fr),      # the circle fit reports fr directly
        "detuning_err": float(res["fr_err"]),
        "fwhm_err": float(fwhm_err),
        # extras: quality factors + environment
        "Ql": float(ql),
        "absQc": float(res["absQc"]),
        "Qc_dia_corr": float(res["Qc_dia_corr"]),
        "Qi_dia_corr": float(res["Qi_dia_corr"]),
        "Qi_no_corr": float(res["Qi_no_corr"]),
        "phi0": float(res["phi0"]),
        "delay": float(res["delay"]),
        "a": float(res["a"]),
        "alpha": float(res["alpha"]),
        "Ql_err": float(res["Ql_err"]),
        "absQc_err": float(res["absQc_err"]),
        "phi0_err": float(res["phi0_err"]),
        "Qc_dia_corr_err": float(res["Qc_dia_corr_err"]),
        "Qi_dia_corr_err": float(res["Qi_dia_corr_err"]),
        "chi_square": float(res["chi_square"]),
        # float-pair arrays for the figure (netCDF-safe; bulky)
        "raw_i": np.real(iq).astype(float),
        "raw_q": np.imag(iq).astype(float),
        "fit_i": np.real(res["z_fit"]).astype(float),
        "fit_q": np.imag(res["z_fit"]).astype(float),
        "norm_i": np.real(res["z_norm"]).astype(float),
        "norm_q": np.imag(res["z_norm"]).astype(float),
        "norm_fit_i": np.real(res["z_norm_fit"]).astype(float),
        "norm_fit_q": np.imag(res["z_norm_fit"]).astype(float),
    }


DIP_METHODS = {
    "lorentzian": _fit_dip_lorentzian,
    "circle": _fit_dip_circle,
}


def fit_dip(detuning: np.ndarray, iq: np.ndarray,
            full_freq: Optional[np.ndarray] = None,
            method: str = "lorentzian", **knobs) -> Dict:
    """Fit the resonator dip in one spectrum trace (see module docstring).

    Parameters
    ----------
    detuning : 1-D float array
        Readout-frequency detuning axis (Hz, relative to the LO).
    iq : 1-D complex array
        Demodulated signal I + iQ on that axis.
    full_freq : 1-D float array, optional
        Absolute readout frequency (Hz). Required by ``method="circle"``;
        when present, the result always carries the absolute ``full_freq``
        centre too.
    method : str
        ``"lorentzian"`` (default) or ``"circle"``.
    **knobs
        Method knobs, validated against :data:`DIP_KNOBS` (unknown -> ValueError).
    """
    validate_dip_kwargs(method, knobs)
    detuning = np.asarray(detuning, dtype=float)
    iq = np.asarray(iq).ravel()

    if method == "circle":
        results = _fit_dip_circle(detuning, iq, full_freq, **knobs)
    else:
        results = DIP_METHODS[method](detuning, iq, **knobs)
    results["method"] = method

    # Common post-step: absolute centre frequency (method-agnostic; a method
    # that already knows it, e.g. circle, reports it itself).
    if full_freq is not None and "full_freq" not in results:
        f = np.asarray(full_freq, dtype=float).ravel()
        order = np.argsort(detuning)
        results["full_freq"] = float(
            np.interp(results["detuning"], detuning[order], f[order])
        )
    return results
