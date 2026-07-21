import numpy as np
import pytest
import xarray as xr

from scqat.tools.fit_lorentzian_bg import FitLorentzianBG, lorentzian_bg
from scqat.tools import get_fitter


def _make_data(x0=1.6e6, amplitude=-0.5, gamma=1.4e6, coeffs=(1.0, 0.0, 0.0),
               n_points=241, x_lo=-6e6, x_hi=6e6, noise_std=0.0, seed=42):
    """Lorentzian on a polynomial background over a Hz-scale sweep.

    ``coeffs = (c0, c1, c2)`` are on the scaled coordinate u = (x - mid)/halfspan,
    matching the fitter's parameterization.
    """
    x = np.linspace(x_lo, x_hi, n_points)
    x_mid = 0.5 * (x_lo + x_hi)
    x_halfspan = 0.5 * (x_hi - x_lo)
    y = lorentzian_bg(x, x0, amplitude, gamma, *coeffs, x_mid, x_halfspan)
    if noise_std > 0:
        rng = np.random.default_rng(seed)
        y = y + rng.normal(0, noise_std, size=y.shape)
    return xr.DataArray(y, coords={'x': x}, dims='x')


class TestFitLorentzianBG:

    def test_noiseless_dip_on_slope(self):
        """A sloped background must not bias the dip centre or width."""
        x0, amp, gamma = 1.6e6, -0.5, 1.4e6
        da = _make_data(x0=x0, amplitude=amp, gamma=gamma, coeffs=(1.0, 0.3, 0.0))
        result = FitLorentzianBG(da, inverted=True).fit()
        assert result.success
        assert result.params['x0'].value == pytest.approx(x0, abs=1e3)
        assert result.params['amplitude'].value == pytest.approx(amp, rel=0.05)
        assert result.params['gamma'].value == pytest.approx(gamma, rel=0.05)
        assert result.params['c1'].value == pytest.approx(0.3, rel=0.05)

    def test_noiseless_peak_on_slope(self):
        x0, amp, gamma = -2.0e6, 0.8, 0.9e6
        da = _make_data(x0=x0, amplitude=amp, gamma=gamma, coeffs=(0.2, -0.4, 0.0))
        result = FitLorentzianBG(da).fit()
        assert result.success
        assert result.params['x0'].value == pytest.approx(x0, abs=1e3)
        assert result.params['amplitude'].value == pytest.approx(amp, rel=0.05)

    def test_noisy_curved_background(self):
        """The approved mockup scenario: slope + curvature + noise. The old
        subtract-then-fit was off by 0.34 MHz in x0 and -37% in FWHM here; the
        joint fit must recover both."""
        x0, amp, gamma = 1.6e6, -0.55, 1.4e6
        da = _make_data(x0=x0, amplitude=amp, gamma=gamma,
                        coeffs=(1.25, 0.36, 0.11), noise_std=0.015)
        result = FitLorentzianBG(da, inverted=True, background_order=2).fit()
        assert result.success
        assert result.params['x0'].value == pytest.approx(x0, abs=0.1 * gamma)
        fwhm = 2 * abs(result.params['gamma'].value)
        assert fwhm == pytest.approx(2 * gamma, rel=0.15)

    def test_flat_background_slope_stays_zero(self):
        """On genuinely flat background the fitted slope must be ~0 (no
        spurious tilt invented by the extra freedom)."""
        x0, amp, gamma = 0.0, -0.6, 1.0e6
        da = _make_data(x0=x0, amplitude=amp, gamma=gamma, coeffs=(1.0, 0.0, 0.0),
                        noise_std=0.01)
        result = FitLorentzianBG(da, inverted=True).fit()
        assert result.success
        assert result.params['x0'].value == pytest.approx(x0, abs=0.1 * gamma)
        assert abs(result.params['c1'].value) < 0.02

    def test_background_order_fixes_higher_coeffs(self):
        da = _make_data()
        fitter0 = FitLorentzianBG(da, inverted=True, background_order=0)
        fitter0.guess()
        assert not fitter0.params['c1'].vary
        assert not fitter0.params['c2'].vary
        fitter1 = FitLorentzianBG(da, inverted=True, background_order=1)
        fitter1.guess()
        assert fitter1.params['c1'].vary
        assert not fitter1.params['c2'].vary
        with pytest.raises(ValueError):
            FitLorentzianBG(da, background_order=3)

    def test_seed_mask_diagnostic(self):
        x0 = 1.6e6
        da = _make_data(x0=x0)
        fitter = FitLorentzianBG(da, inverted=True)
        fitter.guess()
        assert fitter.seed_mask is not None
        assert fitter.seed_mask.dtype == bool
        assert fitter.seed_mask.shape == fitter.y.shape
        # off-resonance seeding: the point nearest the line centre is excluded,
        # but enough background points remain for the polyfit
        idx_line = int(np.argmin(np.abs(fitter.x - x0)))
        assert not fitter.seed_mask[idx_line]
        assert fitter.seed_mask.sum() >= 4

    def test_baseline_values_matches_truth(self):
        c0, c1, c2 = 1.0, 0.4, 0.15
        da = _make_data(coeffs=(c0, c1, c2), noise_std=0.01)
        fitter = FitLorentzianBG(da, inverted=True, background_order=2)
        fitter.fit()
        x = fitter.x
        u = (x - 0.5 * (x.min() + x.max())) / (0.5 * (x.max() - x.min()))
        true_bg = c0 + c1 * u + c2 * u**2
        assert np.allclose(fitter.baseline_values(), true_bg, atol=0.03)

    def test_factory_registration(self):
        da = _make_data()
        fitter = get_fitter('lorentzian_bg', data=da, inverted=True)
        assert isinstance(fitter, FitLorentzianBG)

    def test_accepts_raw_xy(self):
        x = np.linspace(0.0, 1.0, 5)
        y = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
        fitter = FitLorentzianBG(y, x=x)
        assert np.allclose(fitter.x, x)
        assert np.allclose(fitter.y, y)
