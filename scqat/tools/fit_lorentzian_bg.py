"""Joint Lorentzian + polynomial-background fitter.

Fits the peak/dip and the smooth background **simultaneously** in one model,
instead of subtracting a pre-estimated baseline and fitting the residual.
A two-stage subtract-then-fit freezes whatever slope error the baseline stage
made — the Lorentzian can never recover it and absorbs it into ``x0`` and
``gamma`` instead. The joint fit lets the background and the line trade off
correctly and propagates their covariance into the reported errors.

Model::

    c0 + c1*u + c2*u**2 + amplitude / (1 + ((x - x0)/gamma)**2)

where ``u = (x - x_mid) / x_halfspan`` is the sweep coordinate rescaled to
[-1, 1]. The polynomial lives on ``u`` so its coefficients stay O(1) even for
x in Hz (~1e6-1e9) — plain ``c1*x`` would be catastrophically ill-conditioned.
The Lorentzian keeps physical ``x`` so ``x0``/``gamma`` are directly in data
units. ``x_mid``/``x_halfspan`` are carried as fixed (non-varying) parameters,
making every saved fit result self-describing.
"""

from xarray import DataArray
from lmfit import Model
from lmfit.model import ModelResult
import numpy as np

from .function_fitting import FunctionFitting, register_fitter, parse_xy


def lorentzian_bg(x, x0, amplitude, gamma, c0, c1, c2, x_mid, x_halfspan):
    """Lorentzian peak/dip on a polynomial background (see module docstring)."""
    u = (x - x_mid) / x_halfspan
    return c0 + c1 * u + c2 * u**2 + amplitude / (1 + ((x - x0) / gamma) ** 2)


@register_fitter('lorentzian_bg')
class FitLorentzianBG(FunctionFitting):
    """
    Jointly fit a Lorentzian peak/dip and a polynomial background:
        c0 + c1*u + c2*u**2 + amplitude / (1 + ((x - x0)/gamma)**2),
        u = (x - x_mid)/x_halfspan  (sweep rescaled to [-1, 1])

    Input DataArray must have a coordinate named 'x'.

    Parameters
    ----------
    data : xarray.DataArray
        1-D signal with coordinate 'x' (or raw arrays, see ``parse_xy``).
    inverted : bool, optional
        If True, the initial amplitude guess is forced negative
        (dip rather than peak). Default False.
    background_order : int, optional
        Polynomial background order: 0 (constant), 1 (slope, default) or
        2 (curvature). Higher coefficients are fixed at 0, not fitted.
    bounds : dict, optional
        Optional per-parameter ``{'name': (min, max)}`` overrides, e.g.
        ``{'x0': (xmin, xmax), 'gamma': (0, gamma_max)}``.

    Attributes
    ----------
    seed_mask : numpy.ndarray of bool
        The off-resonance mask used to seed the background polynomial (set by
        ``guess``). Diagnostic: shows *where* the initial background came from.

    Notes
    -----
    Seeding is dip-aware, in two stages: (1) an edge-anchored linear detrend
    locates the line robustly even on steep backgrounds (a value-quantile mask
    fails there: it keeps a slope in the residual, so the extremum lands at a
    sweep edge and the fit falls into a local minimum); (2) the background
    polynomial is seeded by a polyfit that *excludes* the ±3*gamma window
    around the located line, so the line cannot drag the seed.
    """

    def __init__(self, data: DataArray = None, inverted: bool = False,
                 background_order: int = 1, bounds: dict = None, x=None):
        if background_order not in (0, 1, 2):
            raise ValueError(f"background_order must be 0, 1 or 2, got {background_order}")
        self._data_parser(data, x)
        self.inverted = inverted
        self.background_order = background_order
        self.bounds = bounds or {}
        self.seed_mask = None
        self.model = Model(self.model_function)
        self.params = None

    def _data_parser(self, data: DataArray, x=None):
        self.x, self.y = parse_xy(data, x)

    @staticmethod
    def model_function(x, x0, amplitude, gamma, c0, c1, c2, x_mid, x_halfspan):
        return lorentzian_bg(x, x0, amplitude, gamma, c0, c1, c2, x_mid, x_halfspan)

    def guess(self):
        x = self.x
        y = self.y

        x_lo, x_hi = float(x.min()), float(x.max())
        x_mid = 0.5 * (x_lo + x_hi)
        x_halfspan = 0.5 * (x_hi - x_lo) or 1.0
        u = (x - x_mid) / x_halfspan

        # --- Stage 1: locate the line via an edge-anchored linear detrend ---
        # A line through the sweep-edge medians removes the bulk of any slope,
        # so the line extremum dominates the residual (a value-quantile mask
        # does not: it leaves the slope in, and the extremum lands at an edge).
        k = max(3, len(x) // 10)
        x_l, x_r = float(np.median(x[:k])), float(np.median(x[-k:]))
        y_l, y_r = float(np.median(y[:k])), float(np.median(y[-k:]))
        slope_e = (y_r - y_l) / (x_r - x_l) if x_r != x_l else 0.0
        resid_e = y - (y_l + slope_e * (x - x_l))
        if self.inverted:
            idx = int(np.argmin(resid_e))
        else:
            idx = int(np.argmax(np.abs(resid_e)))
        x0_guess = float(x[idx])
        amp_e = float(resid_e[idx])
        if amp_e == 0:
            amp_e = float(np.max(np.abs(resid_e))) or 1.0

        # FWHM rough estimate: width where |resid_e| > |amp|/2
        half_mask = np.abs(resid_e) >= abs(amp_e) / 2
        if half_mask.sum() >= 2:
            x_in = x[half_mask]
            gamma_guess = float((x_in.max() - x_in.min()) / 2)
        else:
            gamma_guess = float(abs(x[1] - x[0]) * 5) if len(x) > 1 else 1.0
        if gamma_guess <= 0:
            gamma_guess = float(x_hi - x_lo) / 10 if x_hi > x_lo else 1.0

        # --- Stage 2: background seed from the off-resonance points only ---
        mask = np.abs(x - x0_guess) > 3 * gamma_guess
        if mask.sum() < self.background_order + 2:  # line fills the sweep
            mask = np.ones(len(x), dtype=bool)
        self.seed_mask = mask
        coeffs = np.polyfit(u[mask], y[mask], deg=self.background_order)
        c_seed = np.zeros(3)
        c_seed[:self.background_order + 1] = coeffs[::-1]  # polyfit is highest-first

        # Refine the amplitude seed against the background seed
        amp_guess = float((y - np.polyval(coeffs, u))[idx])
        if amp_guess == 0:
            amp_guess = amp_e

        x_span = (x_hi - x_lo) if x_hi > x_lo else 1.0
        x0_bounds = self.bounds.get('x0', (x_lo, x_hi))
        gamma_bounds = self.bounds.get('gamma', (0.0, x_span))

        self.params = self.model.make_params(
            x0=dict(value=x0_guess, min=x0_bounds[0], max=x0_bounds[1]),
            amplitude=dict(value=amp_guess),
            gamma=dict(value=gamma_guess, min=gamma_bounds[0], max=gamma_bounds[1]),
            c0=dict(value=float(c_seed[0])),
            c1=dict(value=float(c_seed[1]), vary=self.background_order >= 1),
            c2=dict(value=float(c_seed[2]), vary=self.background_order >= 2),
            x_mid=dict(value=x_mid, vary=False),
            x_halfspan=dict(value=x_halfspan, vary=False),
        )
        # Generic bound overrides for the remaining free parameters
        for name in ('amplitude', 'c0', 'c1', 'c2'):
            if name in self.bounds:
                lo, hi = self.bounds[name]
                self.params[name].min = lo
                self.params[name].max = hi
        return self.params

    def fit(self, data: DataArray = None, x=None) -> ModelResult:
        if data is not None:
            self._data_parser(data, x)
            self.params = None  # stale scaling/seeds: re-guess on the new data
        if self.params is None:
            self.guess()
        result = self.model.fit(self.y, self.params, x=self.x)
        self.result = result
        return result

    def baseline_values(self, x=None, params=None) -> np.ndarray:
        """Evaluate only the polynomial background component.

        Uses the fitted parameters when ``fit()`` has run, else the initial
        guess. Pass ``x`` to evaluate on a custom grid (default: the data grid).
        """
        p = params
        if p is None:
            p = self.result.params if getattr(self, 'result', None) is not None else self.params
        if p is None:
            raise RuntimeError("Call guess() or fit() before baseline_values().")
        x_arr = self.x if x is None else np.asarray(x, dtype=float)
        u = (x_arr - p['x_mid'].value) / p['x_halfspan'].value
        return p['c0'].value + p['c1'].value * u + p['c2'].value * u**2
