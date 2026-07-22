"""Unit tests for the family-shared per-trace dip fit (tools/dip_fit).

Synthetic notch-model S21 with real phase content, so BOTH methods can fit —
the simulated-backend datasets (noise-only Q quadrature) cannot exercise the
circle method.
"""

import numpy as np
import pytest

from scqat.tools.dip_fit import DIP_KNOBS, DIP_METHODS, fit_dip, validate_dip_kwargs


def _notch(f: np.ndarray, fr: float, ql: float, qc_abs: float, a: float = 1.0) -> np.ndarray:
    """Ideal notch-resonator S21 (no mismatch, no delay)."""
    return a * (1.0 - (ql / qc_abs) / (1.0 + 2j * ql * (f / fr - 1.0)))


class TestFitDip:
    def test_registry_and_knobs_declared(self):
        assert set(DIP_METHODS) == {"lorentzian", "circle"}
        assert set(DIP_KNOBS) == set(DIP_METHODS)

    def test_methods_agree_on_synthetic_notch(self):
        f0 = 7.0e9
        detuning = np.linspace(-3e6, 3e6, 201)
        f = f0 + detuning
        fr = f0 + 0.4e6
        iq = _notch(f, fr, ql=10_000, qc_abs=20_000)

        r_lor = fit_dip(detuning, iq, full_freq=f, method="lorentzian")
        r_cir = fit_dip(detuning, iq, full_freq=f, method="circle")

        assert r_lor["success"] and r_cir["success"]
        # Same physics from both methods (a method changes robustness, not physics).
        assert r_lor["detuning"] == pytest.approx(0.4e6, abs=50e3)
        assert r_cir["detuning"] == pytest.approx(0.4e6, abs=5e3)
        assert r_cir["full_freq"] == pytest.approx(fr, abs=5e3)
        # Provenance + the two-tier contract: amplitude is lorentzian-owned.
        assert r_lor["method"] == "lorentzian" and r_cir["method"] == "circle"
        assert "amplitude" in r_lor and "amplitude" not in r_cir

    def test_full_freq_post_step_for_lorentzian(self):
        detuning = np.linspace(-3e6, 3e6, 201)
        f = 7.0e9 + detuning
        iq = _notch(f, 7.0e9 - 0.5e6, ql=10_000, qc_abs=20_000)
        r = fit_dip(detuning, iq, full_freq=f, method="lorentzian")
        assert r["full_freq"] == pytest.approx(7.0e9 + r["detuning"], abs=1.0)

    def test_validation_fails_loudly(self):
        detuning = np.linspace(-1e6, 1e6, 51)
        iq = np.ones(51, dtype=complex)
        with pytest.raises(ValueError, match="available"):
            fit_dip(detuning, iq, method="nope")
        with pytest.raises(ValueError, match="delay"):
            fit_dip(detuning, iq, method="lorentzian", delay=1e-9)  # circle-only knob
        with pytest.raises(ValueError, match="full_freq"):
            fit_dip(detuning, iq, method="circle")  # needs the absolute axis
        # validate_dip_kwargs is the pre-loop guard callers use directly.
        with pytest.raises(ValueError, match="basline_order"):
            validate_dip_kwargs("lorentzian", {"basline_order": 2})
