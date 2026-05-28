"""Tests for grpype.detection.templates -- TemplateBank, TemplateGrid, GlitchTemplates."""
import numpy as np
import pytest

from grpype.detection.global_params import echans, ndetectors, EPS
from grpype.detection.templates import TemplateBank, TemplateGrid, GlitchTemplates


DATALEN = ndetectors * echans


def _make_template_bank(ntemplates=10, rng=None):
    """Build a TemplateBank with synthetic data, bypassing file I/O."""
    rng = rng or np.random.default_rng(42)
    tb = object.__new__(TemplateBank)
    tb.ndetectors = ndetectors
    tb.binning = 0.01
    tb.fulltemplates = None
    tb.fullchantemplates = None
    tb.quantile = 0
    tb.hasamps = False
    tb.alltemplates = True

    raw = rng.exponential(1.0, (ntemplates, DATALEN)).astype(np.float32)
    raw /= raw.sum(axis=1, keepdims=True) + EPS
    tb.templates = raw

    tb.alphas = rng.uniform(-2, 1, ntemplates).astype(np.float32)
    tb.betas = rng.uniform(-5, -1, ntemplates).astype(np.float32)
    tb.epeaks = rng.uniform(50, 1000, ntemplates).astype(np.float32)
    tb.phis = rng.uniform(0, 360, ntemplates).astype(np.float32)
    tb.thetas = rng.uniform(0, 180, ntemplates).astype(np.float32)
    tb.ntemplates = ntemplates
    return tb


def _make_template_grid(nspec=4, nsky=6, rng=None):
    """Build a TemplateGrid with synthetic data, bypassing file I/O."""
    rng = rng or np.random.default_rng(43)
    tg = object.__new__(TemplateGrid)
    tg.ndetectors = ndetectors
    tg.binning = 0.01
    tg.fulltemplates = None
    tg.hasamps = False

    raw = rng.exponential(1.0, (DATALEN, nspec, nsky)).astype(np.float32)
    sums = raw.sum(axis=0, keepdims=True) + EPS
    raw /= sums
    tg.templates = raw

    tg.alphas = rng.uniform(-2, 1, nspec).astype(np.float32)
    tg.betas = rng.uniform(-5, -1, nspec).astype(np.float32)
    tg.epeaks = rng.uniform(50, 1000, nspec).astype(np.float32)
    tg.phis = rng.uniform(0, 360, nsky).astype(np.float32)
    tg.thetas = rng.uniform(0, 180, nsky).astype(np.float32)
    tg.amps = np.ones((nspec, nsky), dtype=np.float32)
    tg.ntemplates = nspec * nsky
    return tg


def _make_glitch_templates(rng=None):
    """Build GlitchTemplates with synthetic data, bypassing file I/O."""
    rng = rng or np.random.default_rng(44)
    gt = object.__new__(GlitchTemplates)
    gt.ndetectors = ndetectors
    gt.binning = 0.01
    gt.fullglitch1d = None
    gt.fullchanglitch1d = None
    gt.hasamps = False
    nglitch = ndetectors
    raw = rng.exponential(1.0, (nglitch, DATALEN)).astype(np.float32)
    raw /= raw.sum(axis=1, keepdims=True) + EPS
    gt.glitch1d = raw
    return gt


# ---------------------------------------------------------------------------
# TemplateBank
# ---------------------------------------------------------------------------
class TestTemplateBankNormalize:
    def test_rows_sum_to_one(self):
        tb = _make_template_bank()
        raw = np.random.default_rng(50).exponential(1, (5, DATALEN)).astype(np.float32)
        normed = tb.normalize_templates(raw)
        sums = normed.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_zero_row(self):
        tb = _make_template_bank()
        raw = np.zeros((3, DATALEN), dtype=np.float32)
        raw[1] = np.random.default_rng(51).exponential(1, DATALEN)
        normed = tb.normalize_templates(raw)
        assert normed.shape == raw.shape


class TestTemplateBankClean:
    def test_quantile_zero_keeps_all(self):
        tb = _make_template_bank(ntemplates=20)
        n_before = tb.templates.shape[0]
        tb._clean_bad_templates(0)
        assert tb.templates.shape[0] == n_before

    def test_quantile_half_removes(self):
        tb = _make_template_bank(ntemplates=20)
        tb._clean_bad_templates(0.5)
        assert tb.templates.shape[0] < 20


class TestTemplateBankDetOps:
    def test_remove_dets_shape(self):
        tb = _make_template_bank()
        tb.remove_dets([0, 3])
        assert tb.templates.shape[1] == (ndetectors - 2) * echans

    def test_recover_dets(self):
        tb = _make_template_bank()
        original = tb.templates.copy()
        tb.remove_dets([1])
        tb.recover_dets()
        np.testing.assert_array_equal(tb.templates, original)

    def test_keep1det(self):
        tb = _make_template_bank()
        original = tb.templates.copy()
        tb.keep1det(5)
        assert tb.templates.shape[1] == echans
        expected = original[:, 5 * echans : 6 * echans]
        np.testing.assert_array_equal(tb.templates, expected)

    def test_keep1det_recover(self):
        tb = _make_template_bank()
        original = tb.templates.copy()
        tb.keep1det(5)
        tb.recover_dets()
        np.testing.assert_array_equal(tb.templates, original)

    def test_keep_chans(self):
        tb = _make_template_bank()
        chans = [0, 10, 50]
        tb.keep_chans(chans)
        assert tb.templates.shape[1] == len(chans) * ndetectors

    def test_keep_chans_recover(self):
        tb = _make_template_bank()
        original = tb.templates.copy()
        tb.keep_chans([0, 10])
        tb.recover_chans()
        np.testing.assert_allclose(tb.templates, original, atol=1e-6)


class TestTemplateBankCartesian:
    def test_unit_vectors(self):
        tb = _make_template_bank()
        x, y, z = tb.to_cartesian()
        r2 = x ** 2 + y ** 2 + z ** 2
        assert r2.shape == (tb.ntemplates,)


# ---------------------------------------------------------------------------
# TemplateGrid
# ---------------------------------------------------------------------------
class TestTemplateGridNormalize:
    def test_axis0_normalization(self):
        tg = _make_template_grid()
        raw = np.random.default_rng(60).exponential(1, tg.templates.shape).astype(np.float32)
        normed = tg.normalize_templates(raw)
        sums = normed.sum(axis=0)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)


class TestTemplateGridCalcAmps:
    def test_output_shape(self):
        tg = _make_template_grid(nspec=4, nsky=6)
        d = np.random.default_rng(61).poisson(10, DATALEN).astype(np.float64)
        bkg = np.full(DATALEN, 10.0)
        amps = tg.calc_amps(d, bkg[:, None, None])
        assert amps.shape == (4, 6)


class TestTemplateGridCalcPosterior:
    def test_output_shape(self):
        tg = _make_template_grid(nspec=4, nsky=6)
        d = np.random.default_rng(62).poisson(10, DATALEN).astype(np.float64)
        bkg = np.full(DATALEN, 10.0)
        post = tg.calc_posterior(d, bkg, slc=6)
        assert post.shape == (4, 6)

    def test_posterior_peaks_at_signal(self):
        rng = np.random.default_rng(63)
        nspec, nsky = 3, 5
        tg = _make_template_grid(nspec=nspec, nsky=nsky, rng=rng)
        bkg = np.full(DATALEN, 10.0)
        target_spec, target_sky = 1, 3
        signal = tg.templates[:, target_spec, target_sky] * 50.0
        d = (bkg + signal).astype(np.float64)
        post = tg.calc_posterior(d, bkg, slc=nsky)
        best_spec, best_sky = np.unravel_index(np.argmax(post), post.shape)
        assert best_spec == target_spec
        assert best_sky == target_sky


class TestTemplateGridDetOps:
    def test_remove_dets_shape(self):
        tg = _make_template_grid()
        tg.remove_dets([0, 1])
        assert tg.templates.shape[0] == (ndetectors - 2) * echans

    def test_recover_dets(self):
        tg = _make_template_grid()
        original = tg.templates.copy()
        tg.remove_dets([2])
        tg.recover_dets()
        np.testing.assert_allclose(tg.templates, original)

    def test_keep1det(self):
        tg = _make_template_grid()
        tg.keep1det(3)
        assert tg.templates.shape[0] == echans


class TestTemplateGridCartesian:
    def test_unit_vectors(self):
        tg = _make_template_grid()
        x, y, z = tg.to_cartesian()
        assert x.shape == tg.phis.shape


# ---------------------------------------------------------------------------
# GlitchTemplates
# ---------------------------------------------------------------------------
class TestGlitchTemplates:
    def test_normalize(self):
        gt = _make_glitch_templates()
        raw = np.random.default_rng(70).exponential(1, (5, DATALEN)).astype(np.float32)
        normed = gt.normalize_template(raw)
        sums = normed.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)

    def test_remove_dets(self):
        gt = _make_glitch_templates()
        gt.remove_dets([0])
        assert gt.glitch1d.shape[1] == (ndetectors - 1) * echans

    def test_recover_dets(self):
        gt = _make_glitch_templates()
        original = gt.glitch1d.copy()
        gt.remove_dets([0])
        gt.recover_dets()
        np.testing.assert_array_equal(gt.glitch1d, original)

    def test_keep_chans(self):
        gt = _make_glitch_templates()
        gt.keep_chans([0, 5, 10])
        assert gt.glitch1d.shape[1] == 3 * ndetectors

    def test_keep1det(self):
        gt = _make_glitch_templates()
        gt.keep1det(2)
        assert gt.glitch1d.shape[1] == echans
