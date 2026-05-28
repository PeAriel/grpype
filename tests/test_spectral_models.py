"""Tests for grpype.templates.spectral_models -- spectral model classes."""
import numpy as np
import pytest

from grpype.templates.spectral_models import (
    BandFunction,
    Blackbody,
    CutoffPowerLaw,
    PowerLaw,
    band_function,
    cutoff_powerlaw,
)


ENERGIES = np.logspace(0, 4, 200)


# ---------------------------------------------------------------------------
# BandFunction (class)
# ---------------------------------------------------------------------------
class TestBandFunction:
    @pytest.fixture
    def band(self):
        return BandFunction()

    def test_positive_output(self, band):
        spec = band([0.0, -2.5, 300.0], ENERGIES)
        assert np.all(spec >= 0)

    def test_low_and_high_energy_branches(self, band):
        alpha, beta, epeak = -1.0, -2.5, 300.0
        e_break = epeak * (alpha - beta) / (alpha + 2)
        spec = band([alpha, beta, epeak], ENERGIES)
        low_mask = ENERGIES < e_break * 0.5
        high_mask = ENERGIES > e_break * 2.0
        assert np.all(spec[low_mask] > 0)
        assert np.all(spec[high_mask] > 0)

    def test_continuity_at_break(self, band):
        alpha, beta, epeak = -1.0, -2.5, 300.0
        e_break = epeak * (alpha - beta) / (alpha + 2)
        fine_e = np.array([e_break - 0.01, e_break + 0.01])
        spec = band([alpha, beta, epeak], fine_e)
        np.testing.assert_allclose(spec[0], spec[1], rtol=0.01)

    def test_vectorized_batch(self, band):
        params = np.array([[-1.0, -2.5, 300.0], [0.0, -3.0, 500.0]])
        spec = band(params, ENERGIES)
        assert spec.shape == (2, len(ENERGIES))

    def test_alpha_lt_beta_gives_zeros(self, band):
        spec = band([-3.0, -1.0, 300.0], ENERGIES)
        np.testing.assert_array_equal(spec, 0.0)

    def test_epeak_zero_gives_zeros(self, band):
        spec = band([-1.0, -2.5, 0.0], ENERGIES)
        np.testing.assert_array_equal(spec, 0.0)

    def test_alpha_minus2_gives_zeros(self, band):
        spec = band([-2.0, -3.0, 300.0], ENERGIES)
        np.testing.assert_array_equal(spec, 0.0)

    def test_log_prior_inside(self, band):
        assert band.log_prior([-1.0, -2.5, 300.0]) == 0.0

    def test_log_prior_alpha_lt_beta(self, band):
        assert band.log_prior([-3.0, -1.0, 300.0]) == -np.inf

    def test_log_prior_outside_bounds(self, band):
        assert band.log_prior([5.0, -2.5, 300.0]) == -np.inf


# ---------------------------------------------------------------------------
# band_function (standalone)
# ---------------------------------------------------------------------------
class TestBandFunctionStandalone:
    def test_matches_class(self):
        params = [-1.0, -2.5, 300.0]
        standalone = band_function(params, ENERGIES)
        cls_result = BandFunction()(params, ENERGIES)
        np.testing.assert_allclose(standalone, cls_result, rtol=1e-10)


# ---------------------------------------------------------------------------
# CutoffPowerLaw
# ---------------------------------------------------------------------------
class TestCutoffPowerLaw:
    @pytest.fixture
    def cpl(self):
        return CutoffPowerLaw()

    def test_positive_output(self, cpl):
        spec = cpl([-1.0, 300.0], ENERGIES)
        assert np.all(spec >= 0)

    def test_exponential_cutoff(self, cpl):
        spec = cpl([-1.0, 100.0], ENERGIES)
        high_e = ENERGIES > 500
        low_e = ENERGIES < 50
        assert spec[high_e].max() < spec[low_e].max()

    def test_log_prior_inside(self, cpl):
        assert cpl.log_prior([-1.0, 300.0]) == 0.0

    def test_log_prior_outside(self, cpl):
        assert cpl.log_prior([10.0, 300.0]) == -np.inf


# ---------------------------------------------------------------------------
# PowerLaw
# ---------------------------------------------------------------------------
class TestPowerLaw:
    @pytest.fixture
    def pl(self):
        return PowerLaw()

    def test_scalar_input(self, pl):
        spec = pl(-2.0, ENERGIES)
        assert spec.shape == ENERGIES.shape

    def test_array_input(self, pl):
        spec = pl(np.array([[-2.0], [-1.0]]), ENERGIES)
        assert spec.shape == (2, len(ENERGIES))

    def test_slope(self, pl):
        spec = pl(-2.0, ENERGIES)
        log_ratio = np.log(spec[-1] / spec[0]) / np.log(ENERGIES[-1] / ENERGIES[0])
        assert log_ratio == pytest.approx(-2.0, abs=0.01)

    def test_log_prior_inside(self, pl):
        assert pl.log_prior(-2.0) == 0.0

    def test_log_prior_outside(self, pl):
        assert pl.log_prior(20.0) == -np.inf


# ---------------------------------------------------------------------------
# Blackbody
# ---------------------------------------------------------------------------
class TestBlackbody:
    @pytest.fixture
    def bb(self):
        return Blackbody()

    def test_non_negative(self, bb):
        spec = bb(100.0, ENERGIES)
        assert np.all(spec >= 0)

    def test_peak_shifts_with_kT(self, bb):
        spec_low = bb(50.0, ENERGIES)
        spec_high = bb(500.0, ENERGIES)
        peak_low = ENERGIES[np.argmax(spec_low)]
        peak_high = ENERGIES[np.argmax(spec_high)]
        assert peak_high > peak_low

    def test_log_prior_inside(self, bb):
        assert bb.log_prior(100.0) == 0.0

    def test_log_prior_outside(self, bb):
        assert bb.log_prior(0.5) == -np.inf


# ---------------------------------------------------------------------------
# SpectralModel.fold_model
# ---------------------------------------------------------------------------
class TestFoldModel:
    def test_identity_response(self):
        nchan = 10
        ndet = 14
        nphoton = nchan
        rsp = np.zeros((ndet, nchan, nphoton))
        for i in range(ndet):
            rsp[i] = np.eye(nchan)

        model = BandFunction()
        nai_centroids = np.linspace(10, 900, nchan)
        bgo_centroids = np.linspace(200, 40000, nchan)
        nai_widths = np.ones(nchan)
        bgo_widths = np.ones(nchan)

        template = model.fold_model(
            rsp, [-1.0, -2.5, 300.0],
            nai_centroids, nai_widths, bgo_centroids, bgo_widths,
            normalize=True,
        )
        assert template.shape == (ndet * nchan,)
        np.testing.assert_allclose(template.sum(), 1.0, atol=1e-6)

    def test_normalize_false(self):
        nchan = 10
        ndet = 14
        rsp = np.zeros((ndet, nchan, nchan))
        for i in range(ndet):
            rsp[i] = np.eye(nchan)

        model = PowerLaw()
        centroids = np.linspace(10, 1000, nchan)
        widths = np.ones(nchan)

        template = model.fold_model(
            rsp, -2.0, centroids, widths, centroids, widths, normalize=False,
        )
        assert template.sum() != pytest.approx(1.0, abs=0.01)
