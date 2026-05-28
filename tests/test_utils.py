"""Tests for grpype.detection.utils -- pure math utility functions."""
from datetime import datetime

import numpy as np
import pytest
from scipy.stats import norm

from grpype.detection.global_params import echans, ndetectors, EPS
from grpype.detection.utils import (
    calc_best_amp,
    calc_glitch_statistic,
    calc_mf,
    calibrate_from_norm,
    exp2_tail,
    fibonacci_sphere,
    find_peaks_2d,
    fit_bkg,
    generate_filtered_combinations,
    generate_seed,
    inner,
    logmatch,
    match,
    quad_rolling_mean_padded,
    rolling_mean_padded,
    sf,
    simple_mf,
    square_convolve,
    t90_to_sigma,
    xyz2thetahpi,
)


# ---------------------------------------------------------------------------
# calc_mf
# ---------------------------------------------------------------------------
class TestCalcMf:
    def test_known_snr(self):
        bkg = np.full(100, 10.0)
        amp = 5.0
        template = np.random.default_rng(0).exponential(1, 100)
        template /= template.sum()
        d = bkg + amp * template
        snr = calc_mf(d, bkg, template)
        assert snr.shape == (1,)
        assert snr[0] > 0

    def test_2d_templates(self):
        bkg = np.full(100, 10.0)
        templates = np.random.default_rng(1).exponential(1, (5, 100))
        templates /= templates.sum(axis=1, keepdims=True)
        d = bkg + 3.0 * templates[2]
        snr = calc_mf(d, bkg, templates)
        assert snr.shape == (5,)
        assert np.argmax(snr) == 2

    def test_noise_only_low_snr(self):
        bkg = np.full(200, 50.0)
        templates = np.ones((1, 200)) / 200.0
        d = bkg.copy()
        snr = calc_mf(d, bkg, templates)
        assert abs(snr[0]) < 5


# ---------------------------------------------------------------------------
# calc_best_amp
# ---------------------------------------------------------------------------
class TestCalcBestAmp:
    def test_amplitude_recovery(self):
        bkg = np.full(100, 20.0)
        template = np.random.default_rng(2).exponential(1, 100)
        template /= template.sum()
        true_amp = 7.0
        d = bkg + true_amp * template
        recovered = calc_best_amp(d, bkg, template)
        assert recovered.shape == (1,)
        np.testing.assert_allclose(recovered[0], true_amp, rtol=0.1)

    def test_1d_and_2d_consistency(self):
        bkg = np.full(50, 15.0)
        template = np.ones(50) / 50.0
        d = bkg + 3.0 * template
        amp_1d = calc_best_amp(d, bkg, template)
        amp_2d = calc_best_amp(d, bkg, template[None, :])
        np.testing.assert_allclose(amp_1d, amp_2d)


# ---------------------------------------------------------------------------
# calc_glitch_statistic
# ---------------------------------------------------------------------------
class TestCalcGlitchStatistic:
    def test_return_best_false(self):
        bkg = np.full(100, 10.0)
        templates = np.random.default_rng(3).exponential(1, (3, 100))
        templates /= templates.sum(axis=1, keepdims=True)
        d = bkg + 5.0 * templates[0]
        result = calc_glitch_statistic(d, bkg, templates, bkg, return_best=False)
        assert np.isscalar(result) or result.ndim == 0

    def test_return_best_true(self):
        bkg = np.full(100, 10.0)
        templates = np.random.default_rng(4).exponential(1, (3, 100))
        templates /= templates.sum(axis=1, keepdims=True)
        d = bkg + 5.0 * templates[1]
        mf_stat, best_mf, best_idx = calc_glitch_statistic(
            d, bkg, templates, bkg, return_best=True
        )
        assert best_idx == 1


# ---------------------------------------------------------------------------
# simple_mf
# ---------------------------------------------------------------------------
class TestSimpleMf:
    def test_shape(self):
        rng = np.random.default_rng(5)
        n, c = 50, 80
        data = rng.poisson(10, (n, c)).astype(float)
        template = rng.exponential(1, c)
        template /= template.sum()
        bkg = np.full(c, 10.0)
        result = simple_mf(data, template, bkg, 1.0)
        assert result.shape == (n,)


# ---------------------------------------------------------------------------
# square_convolve
# ---------------------------------------------------------------------------
class TestSquareConvolve:
    def test_nonsplit_shape(self):
        rng = np.random.default_rng(6)
        T, C = 200, ndetectors * echans
        data = rng.poisson(5, (T, C)).astype(np.float32)
        fltr = rng.exponential(1, (3, C)).astype(np.float32)
        window = 10
        result = square_convolve(data, window, fltr)
        assert result.shape == (T + 1 - window, 3)

    def test_split_per_detector(self):
        rng = np.random.default_rng(7)
        T, C = 200, ndetectors * echans
        data = rng.poisson(5, (T, C)).astype(np.float32)
        fltr = rng.exponential(1, (3, C)).astype(np.float32)
        window = 10
        result = square_convolve(data, window, fltr, split=True)
        assert result.shape == (ndetectors, T + 1 - window, 3)
        nonsplit = square_convolve(data, window, fltr, split=False)
        np.testing.assert_allclose(result.sum(axis=0), nonsplit, atol=1e-2)

    def test_1d_filter(self):
        rng = np.random.default_rng(8)
        T, C = 100, ndetectors * echans
        data = rng.poisson(5, (T, C)).astype(np.float32)
        fltr = rng.exponential(1, C).astype(np.float32)
        window = 5
        result = square_convolve(data, window, fltr)
        assert result.shape == (T + 1 - window,)


# ---------------------------------------------------------------------------
# rolling_mean_padded / quad_rolling_mean_padded
# ---------------------------------------------------------------------------
class TestRollingMean:
    def test_constant_returns_constant_1d(self):
        data = np.full(500, 7.0)
        result = rolling_mean_padded(data, window=50)
        np.testing.assert_allclose(result, 7.0, atol=0.5)

    def test_constant_returns_constant_2d(self):
        data = np.full((500, 10), 3.0, dtype=np.float32)
        result = rolling_mean_padded(data, window=50)
        np.testing.assert_allclose(result, 3.0, atol=0.5)

    def test_preserves_length(self):
        data = np.random.default_rng(9).normal(0, 1, 300)
        result = rolling_mean_padded(data, window=40, gap=10)
        assert result.shape == data.shape

    def test_gap_parameter(self):
        data = np.zeros(500)
        data[250] = 1000.0
        result_no_gap = rolling_mean_padded(data, window=20, gap=0)
        result_gap = rolling_mean_padded(data, window=20, gap=30)
        peak_no_gap = np.max(np.abs(result_no_gap))
        peak_gap = np.max(np.abs(result_gap))
        assert peak_gap < peak_no_gap


class TestQuadRollingMean:
    def test_constant_input(self):
        data = np.full((300, 20), 5.0, dtype=np.float32)
        result = quad_rolling_mean_padded(data, window=30, gap=10)
        np.testing.assert_allclose(result, 5.0, atol=0.5)

    def test_clipneg(self):
        data = np.full((300, 20), 0.1, dtype=np.float32)
        result = quad_rolling_mean_padded(data, window=30, gap=10, clipneg=True)
        assert np.all(result >= 0)


# ---------------------------------------------------------------------------
# fit_bkg
# ---------------------------------------------------------------------------
class TestFitBkg:
    def test_smooth_passthrough(self):
        bkg = np.full((50, ndetectors * echans), 10.0, dtype=np.float64)
        result = fit_bkg(bkg)
        assert result.shape == bkg.shape

    def test_1d_input(self):
        bkg = np.full(ndetectors * echans, 10.0, dtype=np.float64)
        result = fit_bkg(bkg)
        assert result.ndim == 2


# ---------------------------------------------------------------------------
# find_peaks_2d
# ---------------------------------------------------------------------------
class TestFindPeaks2d:
    def test_isolated_peaks(self):
        mf = np.zeros((1000, 5))
        mf[200, 1] = 15.0
        mf[600, 3] = 20.0
        times, temps = find_peaks_2d(mf, min_peak_dist=100, min_peak_height=10)
        assert len(times) == 2
        assert set(times) == {200, 600}

    def test_min_distance_enforcement(self):
        mf = np.zeros((1000, 3))
        mf[100, 0] = 12.0
        mf[110, 1] = 15.0
        mf[500, 2] = 11.0
        times, temps = find_peaks_2d(mf, min_peak_dist=50, min_peak_height=10)
        assert 500 in times
        found_near_100 = [t for t in times if 90 <= t <= 120]
        assert len(found_near_100) == 1

    def test_cross_template_max(self):
        mf = np.zeros((500, 4))
        mf[200, 0] = 11.0
        mf[200, 2] = 18.0
        times, temps = find_peaks_2d(mf, min_peak_dist=50, min_peak_height=10)
        idx = np.where(times == 200)[0]
        assert len(idx) == 1
        assert temps[idx[0]] == 2


# ---------------------------------------------------------------------------
# exp2_tail / sf / calibrate_from_norm
# ---------------------------------------------------------------------------
class TestCalibration:
    def test_exp2_tail_evaluation(self):
        x = np.array([0.0, 1.0, 2.0])
        result = exp2_tail(x, 1.0, 0.5, 0.1)
        assert result[0] == pytest.approx(1.0)
        assert all(result[1:] < 1.0)

    def test_sf_positive(self):
        val = sf(exp2_tail, (1.0, 0.5, 0.1), 0.0)
        assert val > 0

    def test_calibrate_from_norm_roundtrip(self):
        x0 = 5.0
        norm_sf_val = norm.sf(x0)
        params = (norm.pdf(5.0), 0.5, 0.05)
        result = calibrate_from_norm(exp2_tail, params, x0)
        assert isinstance(result, float) or result.ndim == 0


# ---------------------------------------------------------------------------
# generate_seed
# ---------------------------------------------------------------------------
class TestGenerateSeed:
    def test_deterministic(self):
        dt = datetime(2017, 8, 17, 12)
        s1 = generate_seed(dt, 0.04)
        s2 = generate_seed(dt, 0.04)
        assert s1 == s2

    def test_different_dates(self):
        s1 = generate_seed(datetime(2017, 8, 17, 12), 0.04)
        s2 = generate_seed(datetime(2018, 1, 1, 0), 0.04)
        assert s1 != s2

    def test_string_input(self):
        s = generate_seed("2017-08-17 12:00:00.000000", 0.04)
        assert isinstance(s, (int, np.integer))

    def test_different_burst_duration(self):
        dt = datetime(2017, 8, 17, 12)
        s1 = generate_seed(dt, 0.04)
        s2 = generate_seed(dt, 1.0)
        assert s1 != s2


# ---------------------------------------------------------------------------
# fibonacci_sphere / xyz2thetahpi
# ---------------------------------------------------------------------------
class TestCoordinates:
    def test_fibonacci_sphere_count(self):
        x, y, z = fibonacci_sphere(100)
        assert len(x) == 100

    def test_fibonacci_sphere_unit(self):
        x, y, z = fibonacci_sphere(200)
        r2 = x ** 2 + y ** 2 + z ** 2
        np.testing.assert_allclose(r2, 1.0, atol=1e-10)

    def test_xyz2thetahpi_north_pole(self):
        theta, phi = xyz2thetahpi(0, 0, 1)
        assert theta == pytest.approx(0.0, abs=1e-10)

    def test_xyz2thetahpi_equator(self):
        theta, phi = xyz2thetahpi(1, 0, 0)
        assert theta == pytest.approx(np.pi / 2, abs=1e-10)

    def test_xyz2thetahpi_azel(self):
        el, az = xyz2thetahpi(0, 0, 1, azel=True)
        assert el == pytest.approx(90.0, abs=1e-5)


# ---------------------------------------------------------------------------
# match / logmatch / inner
# ---------------------------------------------------------------------------
class TestTemplateMatching:
    def test_match_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        sigma = np.ones(3)
        assert match(a, a, sigma) == pytest.approx(1.0)

    def test_match_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sigma = np.ones(2)
        assert match(a, b, sigma) == pytest.approx(0.0, abs=1e-10)

    def test_logmatch_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        sigma = np.full(3, 10.0)
        result = logmatch(a, a, sigma)
        assert result == pytest.approx(1.0, abs=1e-5)

    def test_inner_1d(self):
        a = np.array([2.0, 3.0])
        b = np.array([4.0, 5.0])
        sigma = np.array([1.0, 1.0])
        assert inner(a, b, sigma) == pytest.approx(23.0)

    def test_inner_2d(self):
        a = np.array([[1.0, 2.0], [3.0, 4.0]])
        b = np.array([1.0, 1.0])
        sigma = np.array([1.0, 1.0])
        result = inner(a, b, sigma)
        np.testing.assert_allclose(result, [3.0, 7.0])


# ---------------------------------------------------------------------------
# t90_to_sigma
# ---------------------------------------------------------------------------
class TestDurationConversions:
    def test_t90_coverage(self):
        t90 = 2.0
        sigma = t90_to_sigma(t90)
        coverage = norm.cdf(t90, 0, sigma) - norm.cdf(-t90, 0, sigma)
        assert coverage == pytest.approx(0.9, abs=0.01)


# ---------------------------------------------------------------------------
# generate_filtered_combinations
# ---------------------------------------------------------------------------
class TestFilteredCombinations:
    def test_alpha_gt_beta(self):
        alpha = np.array([-0.5, 0.0, 0.5])
        beta = np.array([-2.0, -1.0, 0.0])
        combos = generate_filtered_combinations(alpha, beta)
        assert all(combos[:, 0] > combos[:, 1])

    def test_no_valid(self):
        alpha = np.array([-5.0])
        beta = np.array([0.0])
        combos = generate_filtered_combinations(alpha, beta)
        assert len(combos) == 0
