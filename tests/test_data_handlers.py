"""Tests for grpype.data_io.data_handlers -- pure array logic methods."""
from datetime import datetime

import numpy as np
import pytest

from grpype.detection.global_params import echans, ndetectors
from grpype.data_io.data_handlers import DataLoaders, TTEData


DATALEN = ndetectors * echans


def _make_bare_tte(**overrides):
    """Create a TTEData-like object without running __init__."""
    obj = object.__new__(TTEData)
    defaults = dict(
        ndetectors=ndetectors,
        binning=0.01,
        cut_len_seconds=10,
        cut_len_samples=1000,
        data=np.random.default_rng(99).poisson(10, (10000, DATALEN)).astype(np.int16),
        time=np.linspace(500_000_000, 500_000_000 + 100, 10000),
        gti=None,
        gti_inds=None,
        full_gti_inds=[(0, 10000)],
        full_time_data=None,
        full_time=None,
        full_det_data=None,
        full_chan_data=None,
        sharptime_reduce=0,
        has_bkg=False,
        bkgs=None,
        slice_time=None,
        burst_duration_sec=0.04,
        burst_duration_samp=4,
    )
    defaults.update(overrides)
    for k, v in defaults.items():
        setattr(obj, k, v)
    return obj


# ---------------------------------------------------------------------------
# DataLoaders._fix_time
# ---------------------------------------------------------------------------
class TestFixTime:
    def test_zero_padding(self):
        date = datetime(2017, 1, 5, 3)
        year, month, day, hour = DataLoaders._fix_time(date)
        assert year == "2017"
        assert month == "01"
        assert day == "05"
        assert hour == "03"

    def test_no_padding_needed(self):
        date = datetime(2023, 12, 25, 14)
        year, month, day, hour = DataLoaders._fix_time(date)
        assert month == "12"
        assert day == "25"
        assert hour == "14"


# ---------------------------------------------------------------------------
# TTEData.merge_gtis
# ---------------------------------------------------------------------------
class TestMergeGtis:
    def test_disjoint_stay_separate(self):
        tte = _make_bare_tte()
        gtis = [[100, 200], [500, 600]]
        result = tte.merge_gtis(gtis, gtype="seconds")
        assert len(result) == 2

    def test_overlapping_merge(self):
        tte = _make_bare_tte()
        gtis = [[100, 200], [105, 300], [110, 250]]
        result = tte.merge_gtis(gtis, gtype="seconds")
        assert len(result) < 3

    def test_single_gti(self):
        tte = _make_bare_tte()
        result = tte.merge_gtis([[100, 200]])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# TTEData.get_gti_inds
# ---------------------------------------------------------------------------
class TestGetGtiInds:
    def test_trims_by_cut_length(self):
        tte = _make_bare_tte()
        tte.gti = [[tte.time[100], tte.time[9900]]]
        inds = tte.get_gti_inds(cut_length_seconds=10)
        for start, stop in inds:
            assert start >= 100
            assert stop <= 9900

    def test_short_gti_filtered(self):
        tte = _make_bare_tte()
        tte.gti = [[tte.time[100], tte.time[110]]]
        inds = tte.get_gti_inds(cut_length_seconds=10)
        assert len(inds) == 0


# ---------------------------------------------------------------------------
# TTEData.zoomin / recover_time
# ---------------------------------------------------------------------------
class TestZoomRecover:
    def test_zoomin_reduces_data(self):
        tte = _make_bare_tte()
        original_len = len(tte.time)
        tte.zoomin(5000, 2000)
        assert len(tte.time) == 2000
        assert tte.data.shape[0] == 2000

    def test_recover_restores(self):
        tte = _make_bare_tte()
        original_data = tte.data.copy()
        original_time = tte.time.copy()
        tte.zoomin(5000, 2000)
        tte.recover_time()
        np.testing.assert_array_equal(tte.data, original_data)
        np.testing.assert_array_equal(tte.time, original_time)


# ---------------------------------------------------------------------------
# TTEData.remove_dets / keep_chans / keep1det + recovery
# ---------------------------------------------------------------------------
class TestDetChanOps:
    def test_remove_dets(self):
        tte = _make_bare_tte()
        tte.remove_dets([0, 3])
        assert tte.data.shape[1] == (ndetectors - 2) * echans
        assert tte.ndetectors == ndetectors - 2

    def test_recover_dets(self):
        tte = _make_bare_tte()
        original = tte.data.copy()
        tte.remove_dets([1])
        tte.recover_dets()
        assert tte.ndetectors == ndetectors
        np.testing.assert_array_equal(tte.data, original)

    def test_keep1det(self):
        tte = _make_bare_tte()
        original = tte.data.copy()
        tte.keep1det(5)
        assert tte.data.shape[1] == echans
        expected = original[:, 5 * echans : 6 * echans]
        np.testing.assert_array_equal(tte.data, expected)

    def test_keep_chans(self):
        tte = _make_bare_tte()
        chans = [0, 10, 50]
        tte.keep_chans(chans)
        assert tte.data.shape[1] == len(chans) * ndetectors

    def test_keep_chans_recover(self):
        tte = _make_bare_tte()
        original = tte.data.copy()
        tte.keep_chans([0, 10])
        tte.recover_chans()
        np.testing.assert_array_equal(tte.data, original)


# ---------------------------------------------------------------------------
# TTEData.interp_bkg
# ---------------------------------------------------------------------------
class TestInterpBkg:
    def test_rescales_by_binning_ratio(self):
        tte = _make_bare_tte(binning=0.001)
        old_binning = 0.01
        T_old = 1000
        bkg_old = np.full((T_old, DATALEN), 10.0, dtype=np.float32)
        old_time = np.linspace(tte.time[0], tte.time[-1], T_old)
        tte.interp_bkg(bkg_old, old_binning, old_time)
        assert tte.has_bkg is True
        expected_scale = 0.001 / 0.01
        np.testing.assert_allclose(
            tte.bkgs.mean(), 10.0 * expected_scale, rtol=0.1
        )


# ---------------------------------------------------------------------------
# TTEData.apply_timeslides
# ---------------------------------------------------------------------------
class TestApplyTimeslides:
    def test_known_slides(self):
        tte = _make_bare_tte()
        slides_str = "-".join(["0"] * ndetectors)
        tte.apply_timeslides(minutes=1.0, slides=slides_str)
        assert len(tte.full_gti_inds) >= 1

    def test_gti_shrinks(self):
        tte = _make_bare_tte()
        original_len = tte.full_gti_inds[0][1] - tte.full_gti_inds[0][0]
        slides_str = "-".join([str(i * 0.05) for i in range(ndetectors)])
        tte.apply_timeslides(minutes=1.0, slides=slides_str)
        if len(tte.full_gti_inds) > 0:
            new_len = tte.full_gti_inds[0][1] - tte.full_gti_inds[0][0]
            assert new_len < original_len


# ---------------------------------------------------------------------------
# TTEData.total_time_used
# ---------------------------------------------------------------------------
class TestTotalTimeUsed:
    def test_basic(self):
        tte = _make_bare_tte()
        total = tte.total_time_used
        expected = (tte.full_gti_inds[0][1] - tte.full_gti_inds[0][0]) * tte.binning
        assert total == pytest.approx(expected, rel=0.01)

    def test_with_sharptime_reduction(self):
        tte = _make_bare_tte(sharptime_reduce=5.0)
        total = tte.total_time_used
        gti_time = (tte.full_gti_inds[0][1] - tte.full_gti_inds[0][0]) * tte.binning
        assert total == pytest.approx(gti_time - 5.0, rel=0.01)


# ---------------------------------------------------------------------------
# TTEData.cut_ppu_glitch
# ---------------------------------------------------------------------------
class TestCutPpuGlitch:
    def test_spike_splits_gti(self):
        tte = _make_bare_tte()
        rng = np.random.default_rng(88)
        data = rng.poisson(1, (10000, DATALEN)).astype(np.int16)
        spike_loc = 5000
        data[spike_loc - 5 : spike_loc + 5, :] = 100
        gtis = [(500, 9500)]
        result = tte.cut_ppu_glitch(data, gtis)
        assert len(result) >= 1


# ---------------------------------------------------------------------------
# TTEData.cut_deadtime
# ---------------------------------------------------------------------------
class TestCutDeadtime:
    def test_no_deadtime_preserves_gti(self):
        tte = _make_bare_tte()
        data = np.random.default_rng(89).poisson(5, (10000, DATALEN)).astype(np.int16)
        gtis = [(500, 9500)]
        result = tte.cut_deadtime(data, gtis)
        assert len(result) == 1
        assert tuple(result[0]) == gtis[0]

    def test_zero_block_returns_list(self):
        T = 30000
        tte = _make_bare_tte(
            data=np.random.default_rng(89).poisson(5, (T, DATALEN)).astype(np.int16),
            time=np.linspace(0, T * 0.01, T),
            full_gti_inds=[(0, T)],
        )
        data = tte.data.copy()
        data[14000:15000, :] = 0
        gtis = [(1000, 29000)]
        result = tte.cut_deadtime(data, gtis)
        assert isinstance(result, list)
