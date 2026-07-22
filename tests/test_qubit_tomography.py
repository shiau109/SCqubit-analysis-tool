"""Qubit-tomography estimator over the flat ``tools.discriminate`` reduction.

First direct coverage for this estimator — including the degenerate-training
regression: ragged ``direct_counts`` from a shot-less GMM centre used to raise
``IndexError`` at ``counts[1, 1]``; the fixed-shape reduction must instead
yield a graceful ``success=False``.
"""

import numpy as np
import pytest
import xarray as xr

from scqat.estimators import QubitTomographyEstimator
from scqat.estimators.state_discrimination import StateDiscriminationEstimator


def _tomo_dataset(n_train=300, n_shot=200, sep=8.0, sigma=1.0, seed=0,
                  degenerate=False):
    """Minimal tomography dataset: z-basis only, one gate count, two training
    states. ``degenerate=True`` puts BOTH training states in the ground blob
    (e.g. the pi pulse was dead) so one GMM centre captures every shot."""
    rng = np.random.default_rng(seed)
    centers = np.array([[0.0, 0.0], [sep, 0.0]])

    I_train = np.empty((2, n_train))
    Q_train = np.empty((2, n_train))
    for s in range(2):
        c = centers[0] if degenerate else centers[s]
        I_train[s] = c[0] + sigma * rng.standard_normal(n_train)
        Q_train[s] = c[1] + sigma * rng.standard_normal(n_train)

    # Tomography shots: all in the excited blob (population_z -> 1).
    I_tomo = centers[1][0] + sigma * rng.standard_normal((1, 1, 1, n_shot))
    Q_tomo = centers[1][1] + sigma * rng.standard_normal((1, 1, 1, n_shot))

    return xr.Dataset(
        {
            "I_tomo": (("basis", "sym", "gate_count", "shot_idx"), I_tomo),
            "Q_tomo": (("basis", "sym", "gate_count", "shot_idx"), Q_tomo),
            "I_train": (("prepared_state", "train_shot_idx"), I_train),
            "Q_train": (("prepared_state", "train_shot_idx"), Q_train),
        },
        coords={
            "basis": ["z"], "sym": ["reg"], "gate_count": [0],
            "shot_idx": np.arange(n_shot), "prepared_state": [0, 1],
            "train_shot_idx": np.arange(n_train),
        },
    )


def test_healthy_training_classifies_and_succeeds():
    ds = _tomo_dataset()
    res = QubitTomographyEstimator().extract_parameters(ds)
    assert res["success"]
    assert res["readout_fidelity"] > 0.9
    assert np.asarray(res["confusion_matrix"]).shape == (2, 2)
    # All tomo shots were prepared in |1>.
    assert res["population_z"][0] == pytest.approx(1.0, abs=0.05)


def test_degenerate_training_fails_gracefully():
    """One GMM centre captures no shot: previously an IndexError at
    counts[1, 1]; now a well-formed unsuccessful result."""
    ds = _tomo_dataset(degenerate=True)
    res = QubitTomographyEstimator().extract_parameters(
        ds, user_mean=[[0.0, 0.0], [50.0, 50.0]], user_std=1.0
    )
    assert res["success"] is False
    assert res["readout_fidelity"] == pytest.approx(0.5, abs=0.05)
    assert np.asarray(res["confusion_matrix"]).shape == (2, 2)


def test_unknown_kwarg_raises():
    ds = _tomo_dataset(n_train=50, n_shot=20)
    with pytest.raises(ValueError, match="user_stdd"):
        QubitTomographyEstimator().extract_parameters(ds, user_stdd=1.0)


def test_state_discrimination_wrapper_contract():
    """The thin StateDiscriminationEstimator keeps its public result keys and
    its artifacts (metadata JSON-shape + plot data + 4 figures) intact."""
    ds = _tomo_dataset()
    sd_ds = xr.Dataset(
        {"I": ds["I_train"], "Q": ds["Q_train"]}
    ).rename({"train_shot_idx": "shot_idx"})

    est = StateDiscriminationEstimator()
    res = est.extract_parameters(sd_ds)
    assert set(res) == {
        "trained_paras", "fitted_paras", "gaussian_norms", "direct_counts",
        "state_label", "outlier_mask", "outlier_probability", "norm_res",
        "fit_residues", "hist_dataset",
    }
    assert res["direct_counts"].shape == (2, 2)
    assert res["fit_residues"].dims == ("prepared_state", "y", "x")

    meta = est.extract_metadata(res)
    assert set(meta) == {
        "trained_paras", "fitted_paras", "gaussian_norms",
        "direct_counts", "outlier_probability", "norm_res",
    }
    pd = est.build_plot_data(sd_ds, res)
    assert set(est.generate_figures(sd_ds, res, plot_data=pd)) == {
        "raw", "2DHist", "outliers", "fit_residue",
    }
