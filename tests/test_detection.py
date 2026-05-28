"""Tests for grpype.detection.pipeline -- Detection class core methods."""
import numpy as np
import pytest

from grpype.detection.global_params import echans, ndetectors, EPS
from grpype.detection.pipeline import Detection


DATALEN = ndetectors * echans


# ---------------------------------------------------------------------------
# __init__ / set_burst_duration / set_rolling_gap
# ---------------------------------------------------------------------------
class TestDetectionInit:
    def test_burst_duration_samples(self):
        det = Detection(binning=0.01, burst_duration=0.1)
        assert det.bdur_samp == 10
        assert det.bdur_sec == pytest.approx(0.1)

    def test_bkg_window(self):
        det = Detection(binning=0.01, burst_duration=0.04, rolling_window_sec=10)
        assert det.bkg_window == int(10 // 0.01)

    def test_rolling_gap(self):
        det = Detection(binning=0.01, burst_duration=0.04, rolling_gap_sec=5)
        assert det.bkg_window_gap == np.int16(5 // 0.01)

    def test_set_burst_duration(self):
        det = Detection(binning=0.01, burst_duration=0.04)
        det.set_burst_duration(0.5)
        assert det.bdur_samp == 50
        assert det.bdur_sec == pytest.approx(0.5)

    def test_set_rolling_gap(self):
        det = Detection(binning=0.01, burst_duration=0.04)
        det.set_rolling_gap(2.0)
        assert det.bkg_window_gap == np.int16(2.0 // 0.01)

    def test_reset_params(self):
        det = Detection(binning=0.01, burst_duration=0.04)
        assert det.trigtimes == []
        assert det.snrs == []
        assert det.sharptimes == ""


# ---------------------------------------------------------------------------
# roll_mf
# ---------------------------------------------------------------------------
class TestRollMf:
    @pytest.fixture
    def detection(self):
        return Detection(binning=0.01, burst_duration=0.1, rolling_window_sec=5, rolling_gap_sec=0.5)

    def test_output_shape_no_drift(self, detection):
        rng = np.random.default_rng(10)
        T = 2000
        n_templates = 3
        data = rng.poisson(10, (T, DATALEN)).astype(np.float32)
        bkg = np.full((T, DATALEN), 10.0, dtype=np.float32)
        fltr_bkg = np.full(DATALEN, 10.0, dtype=np.float32)
        templates = rng.exponential(1, (n_templates, DATALEN)).astype(np.float32)
        templates /= templates.sum(axis=1, keepdims=True) + EPS

        mf, numers, varz, zvar0, zvar = detection.roll_mf(
            data, templates, bkg, fltr_bkg, drift_corr=False, split=False
        )
        expected_len = T + 1 - detection.bdur_samp
        assert mf.shape == (expected_len, n_templates)
        np.testing.assert_array_equal(zvar0, 0)
        np.testing.assert_array_equal(zvar, 1)

    def test_output_shape_with_drift(self, detection):
        rng = np.random.default_rng(11)
        T = 2000
        n_templates = 3
        data = rng.poisson(10, (T, DATALEN)).astype(np.float32)
        bkg = np.full((T, DATALEN), 10.0, dtype=np.float32)
        fltr_bkg = np.full(DATALEN, 10.0, dtype=np.float32)
        templates = rng.exponential(1, (n_templates, DATALEN)).astype(np.float32)
        templates /= templates.sum(axis=1, keepdims=True) + EPS

        mf, numers, varz, zvar0, zvar = detection.roll_mf(
            data, templates, bkg, fltr_bkg, drift_corr=True, split=False
        )
        expected_len = T + 1 - detection.bdur_samp
        assert mf.shape == (expected_len, n_templates)
        assert not np.all(zvar0 == 0)

    def test_split_returns_per_detector(self, detection):
        rng = np.random.default_rng(12)
        T = 2000
        n_templates = 3
        data = rng.poisson(10, (T, DATALEN)).astype(np.float32)
        bkg = np.full((T, DATALEN), 10.0, dtype=np.float32)
        fltr_bkg = np.full(DATALEN, 10.0, dtype=np.float32)
        templates = rng.exponential(1, (n_templates, DATALEN)).astype(np.float32)
        templates /= templates.sum(axis=1, keepdims=True) + EPS

        mf, numers, varz, zvar0, zvar = detection.roll_mf(
            data, templates, bkg, fltr_bkg, drift_corr=False, split=True
        )
        assert numers.shape[0] == ndetectors
        assert varz.shape[0] == ndetectors


# ---------------------------------------------------------------------------
# get_glitch_times
# ---------------------------------------------------------------------------
class TestGetGlitchTimes:
    def test_detects_known_glitch(self):
        det = Detection(binning=0.01, burst_duration=0.1)
        T, n_templates, n_glitch = 500, 3, 2
        mf = np.random.default_rng(20).normal(0, 1, (T, n_templates)).astype(np.float32)
        glitch_mf = np.zeros((T, n_glitch), dtype=np.float32)

        glitch_loc = 200
        glitch_mf[glitch_loc, 0] = 20.0
        mf[glitch_loc, :] = 5.0

        iconv, fconv = 50, 450
        times = det.get_glitch_times(
            mf, glitch_mf, iconv, fconv, glitch_threshold=10, glitch_extend=0
        )
        assert glitch_loc - iconv in times

    def test_extend_widens(self):
        det = Detection(binning=0.01, burst_duration=0.1)
        T, n_templates, n_glitch = 500, 3, 2
        mf = np.random.default_rng(21).normal(0, 1, (T, n_templates)).astype(np.float32)
        glitch_mf = np.zeros((T, n_glitch), dtype=np.float32)

        glitch_loc = 250
        glitch_mf[glitch_loc, 0] = 20.0
        mf[glitch_loc, :] = 5.0

        iconv, fconv = 50, 450
        times_no_ext = det.get_glitch_times(
            mf, glitch_mf, iconv, fconv, glitch_threshold=10, glitch_extend=0
        )
        times_ext = det.get_glitch_times(
            mf, glitch_mf, iconv, fconv, glitch_threshold=10, glitch_extend=2
        )
        assert len(times_ext) >= len(times_no_ext)
