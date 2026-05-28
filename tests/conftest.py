import numpy as np
import pytest

from grpype.detection.global_params import echans, ndetectors, DATALEN, EPS


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_background(rng):
    """Flat background: 10 counts per channel per time bin."""
    return np.full(DATALEN, 10.0, dtype=np.float32)


@pytest.fixture
def synthetic_templates(rng):
    """Small set of 5 normalised templates, shape (5, DATALEN)."""
    raw = rng.exponential(1.0, size=(5, DATALEN)).astype(np.float32)
    raw /= raw.sum(axis=1, keepdims=True) + EPS
    return raw


@pytest.fixture
def synthetic_timeseries(rng):
    """2-D count data, shape (1000, DATALEN), Poisson around background=10."""
    bkg = np.full(DATALEN, 10.0, dtype=np.float32)
    data = rng.poisson(bkg, size=(1000, DATALEN)).astype(np.float32)
    return data, bkg
