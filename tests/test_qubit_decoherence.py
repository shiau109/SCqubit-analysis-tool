import numpy as np
import xarray as xr
import pytest

from scqat.protocols.qubit_decoherence import (
    QubitDecoherenceAnalyzer,
    _decoherence_G,
)


# ------------------------------------------------------------------
# Unit tests for G(t)
# ------------------------------------------------------------------

class TestDecoherenceG:
    """Unit tests for the decoherence function G(t)."""

    def test_G_at_t0_is_one(self):
        """G(0) = 1 regardless of parameters."""
        assert _decoherence_G(0.0, Gamma=0.5, Lambda=1.0) == pytest.approx(1.0)
        assert _decoherence_G(0.0, Gamma=3.0, Lambda=2.0) == pytest.approx(1.0)
        # Critical: Lambda = 2*Gamma -> d=0
        assert _decoherence_G(0.0, Gamma=1.5, Lambda=3.0) == pytest.approx(1.0)

    def test_G_decays_overdamped(self):
        """G(t) decays towards zero in the overdamped regime (Lambda > 2*Gamma)."""
        t = np.linspace(0, 20, 200)
        G = _decoherence_G(t, Gamma=0.5, Lambda=2.0)
        assert np.abs(G[-1]) < 0.01 * np.abs(G[0])

    def test_G_oscillates_underdamped(self):
        """G(t) should oscillate when Lambda < 2*Gamma (underdamped)."""
        t = np.linspace(0, 20, 500)
        G = _decoherence_G(t, Gamma=3.0, Lambda=0.5)
        # Check that sign changes occur (oscillation)
        sign_changes = np.sum(np.diff(np.sign(G)) != 0)
        assert sign_changes >= 2

    def test_G_critical_damping_no_nan(self):
        """No NaN at critical damping (Lambda = 2*Gamma -> d=0)."""
        t = np.linspace(0, 10, 100)
        G = _decoherence_G(t, Gamma=1.0, Lambda=2.0)
        assert not np.any(np.isnan(G))
        assert G[0] == pytest.approx(1.0)

    def test_G_returns_real(self):
        """G(t) must be real for both overdamped and underdamped cases."""
        t = np.linspace(0, 5, 50)
        assert _decoherence_G(t, Gamma=0.3, Lambda=2.0).dtype == np.float64
        assert _decoherence_G(t, Gamma=3.0, Lambda=1.0).dtype == np.float64


# ------------------------------------------------------------------
# Integration tests for the analyzer
# ------------------------------------------------------------------

class TestQubitDecoherenceAnalyzer:
    """Fit-recovery tests using synthetic data."""

    def test_rho11_overdamped(self):
        """Recover Gamma, Lambda, rho_0 from clean overdamped rho_11 data."""
        Gamma_true, Lambda_true, rho0_true = 0.5, 2.0, 0.95
        t = np.linspace(0, 5, 200)
        G = _decoherence_G(t, Gamma_true, Lambda_true)
        rho11 = np.abs(G) ** 2 * rho0_true

        ds = xr.Dataset({"rho_11": ("time", rho11)}, coords={"time": t})
        analyzer = QubitDecoherenceAnalyzer()
        results, figs = analyzer.analyze(ds)

        r = results["rho_11"]
        assert r["Gamma"] == pytest.approx(Gamma_true, rel=0.1)
        assert r["Lambda"] == pytest.approx(Lambda_true, rel=0.1)
        assert r["rho_0"] == pytest.approx(rho0_true, rel=0.05)
        assert r["regime"] == "overdamped"
        assert "rho_11" in figs

    def test_rho10_underdamped(self):
        """Recover parameters from underdamped (oscillatory) rho_10 data."""
        Gamma_true, Lambda_true, rho0_true = 3.0, 1.0, 0.5
        t = np.linspace(0, 10, 300)
        rho10 = _decoherence_G(t, Gamma_true, Lambda_true) * rho0_true

        ds = xr.Dataset({"rho_10": ("time", rho10)}, coords={"time": t})
        analyzer = QubitDecoherenceAnalyzer()
        results, figs = analyzer.analyze(ds)

        r = results["rho_10"]
        assert r["Gamma"] == pytest.approx(Gamma_true, rel=0.1)
        assert r["Lambda"] == pytest.approx(Lambda_true, rel=0.1)
        assert r["rho_0"] == pytest.approx(rho0_true, rel=0.1)
        assert r["regime"] == "underdamped"
        assert "rho_10" in figs

    def test_both_variables(self):
        """Both rho_11 and rho_10 are fitted when both present."""
        Gamma, Lambda = 0.5, 1.5
        t = np.linspace(0, 8, 250)
        G = _decoherence_G(t, Gamma, Lambda)

        ds = xr.Dataset(
            {"rho_11": ("time", np.abs(G) ** 2 * 1.0), "rho_10": ("time", G * 0.48)},
            coords={"time": t},
        )
        analyzer = QubitDecoherenceAnalyzer()
        results, figs = analyzer.analyze(ds)

        assert "rho_11" in results and "rho_10" in results
        assert "rho_11" in figs and "rho_10" in figs

    def test_fit_residuals_small(self):
        """Residuals should be negligible for noiseless data."""
        t = np.linspace(0, 6, 200)
        G = _decoherence_G(t, Gamma=0.5, Lambda=1.5)
        rho11 = np.abs(G) ** 2 * 1.0

        ds = xr.Dataset({"rho_11": ("time", rho11)}, coords={"time": t})
        results, _ = QubitDecoherenceAnalyzer().analyze(ds)

        assert np.max(np.abs(results["rho_11"]["residuals"])) < 1e-6

    # ------------------------------------------------------------------
    # Error-raising tests
    # ------------------------------------------------------------------
    def test_missing_time_raises(self):
        ds = xr.Dataset({"rho_11": ("idx", [1, 2, 3])}, coords={"idx": [0, 1, 2]})
        with pytest.raises(ValueError, match="time"):
            QubitDecoherenceAnalyzer().analyze(ds)

    def test_missing_variables_raises(self):
        ds = xr.Dataset({"voltage": ("time", [1, 2, 3])}, coords={"time": [0, 1, 2]})
        with pytest.raises(ValueError, match="rho_11"):
            QubitDecoherenceAnalyzer().analyze(ds)
