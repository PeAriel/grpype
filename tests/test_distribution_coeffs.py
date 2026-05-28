"""Tests for grpype.detection.distribution_coeffs -- pure functions."""
import numpy as np
import pytest
from scipy.optimize import curve_fit
from scipy.stats import norm

from grpype.detection.distribution_coeffs import (
    burst_duration_for_index,
    calibrate_dist,
    calculate_coefficients,
    counts_save_path,
    fit_tail,
)
from grpype.detection.utils import exp2_tail


# ---------------------------------------------------------------------------
# burst_duration_for_index
# ---------------------------------------------------------------------------
class TestBurstDurationForIndex:
    def test_base_case_001(self):
        assert burst_duration_for_index(0.01, 0) == 0.04

    def test_base_case_0001(self):
        assert burst_duration_for_index(0.001, 0) == 0.002

    def test_scaling_001(self):
        d1 = burst_duration_for_index(0.01, 1)
        d2 = burst_duration_for_index(0.01, 2)
        ratio = d2 / d1
        assert ratio == pytest.approx(1.35, rel=0.01)

    def test_unsupported_binning(self):
        with pytest.raises(ValueError, match="Unsupported binning"):
            burst_duration_for_index(0.1, 0)


# ---------------------------------------------------------------------------
# counts_save_path
# ---------------------------------------------------------------------------
class TestCountsSavePath:
    def test_path_format(self):
        p = counts_save_path(0.04, 3)
        assert "counts0.040" in str(p)
        assert str(p).endswith("3.npy")


# ---------------------------------------------------------------------------
# fit_tail
# ---------------------------------------------------------------------------
class TestFitTail:
    def test_gaussian_tail(self):
        bins = np.linspace(-5, 10, 201)
        centers = (bins[:-1] + bins[1:]) / 2
        counts = norm.pdf(centers)

        popt = fit_tail(exp2_tail, bins, counts, tail_start=2)
        assert len(popt) == 3
        assert all(np.isfinite(popt))

    def test_recovers_known_function(self):
        bins = np.linspace(-5, 10, 201)
        centers = (bins[:-1] + bins[1:]) / 2
        true_params = (0.5, 0.3, 0.02)
        counts = exp2_tail(centers, *true_params)
        counts = np.maximum(counts, 0)

        popt = fit_tail(exp2_tail, bins, counts, tail_start=2, p0=true_params)
        np.testing.assert_allclose(popt, true_params, rtol=0.1)


# ---------------------------------------------------------------------------
# calculate_coefficients
# ---------------------------------------------------------------------------
class TestCalculateCoefficients:
    def test_returns_coefficients(self):
        rng = np.random.default_rng(77)
        mf = rng.normal(0, 1, (500, 10)).astype(np.float32)
        popt = calculate_coefficients(exp2_tail, mf, 0.04, save=False)
        assert len(popt) == 3
        assert all(np.isfinite(popt))


# ---------------------------------------------------------------------------
# calibrate_dist
# ---------------------------------------------------------------------------
class TestCalibrateDist:
    def test_gaussian_identity(self):
        params = fit_tail(
            exp2_tail,
            np.linspace(-5, 10, 201),
            norm.pdf((np.linspace(-5, 10, 201)[:-1] + np.linspace(-5, 10, 201)[1:]) / 2),
            tail_start=2,
        )
        calibrated = calibrate_dist(4.0, exp2_tail, params)
        assert np.isfinite(calibrated[0])
        assert calibrated[0] > 0
