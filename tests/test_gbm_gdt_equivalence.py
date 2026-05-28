"""
Numerical equivalence tests for the gbm -> gdt migration.

These tests compare current outputs against stored golden values in
``tests/golden_values.npz``.
"""
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

GOLDEN_PATH = Path(__file__).parent / "golden_values.npz"

# ── Test MET values and datetimes used across all validations ──────────────
MET_VALUES = np.array([252460800.0, 500000000.0, 599529605.0, 700000000.0])
REF_DATETIMES = [
    datetime(2010, 1, 1),
    datetime(2017, 8, 17, 12, 41, 4),
    datetime(2020, 1, 1),
]

# ── Haversine test cases (lon1, lat1, lon2, lat2) in degrees ──────────────
HAVERSINE_CASES = np.array([
    [0.0, 0.0, 0.0, 0.0],         # same point
    [0.0, 0.0, 180.0, 0.0],       # antipodal
    [0.0, 90.0, 0.0, 0.0],        # pole to equator
    [100.0, -30.0, 200.0, 45.0],  # typical sky pair 1
    [350.0, 80.0, 10.0, -80.0],   # typical sky pair 2
    [45.0, 45.0, 225.0, -45.0],   # antipodal (general)
    [0.0, 0.0, 1e-8, 0.0],        # near-zero separation
])

# ── Synthetic prob array for find_greedy_credible_levels ──────────────────
PROB_ARRAY = np.random.default_rng(42).dirichlet(np.ones(192))


# ═══════════════════════════════════════════════════════════════════════════
# Phase V – verify new gdt implementations against golden values
# ═══════════════════════════════════════════════════════════════════════════
@pytest.fixture(scope="module")
def golden():
    if not GOLDEN_PATH.exists():
        pytest.skip(f"Golden values not found at {GOLDEN_PATH}; run generate_golden_values() first")
    return np.load(GOLDEN_PATH, allow_pickle=True)


class TestTimeConversionEquivalence:
    """MET <-> datetime round-trips must be identical."""

    def test_met_to_iso(self, golden):
        from gdt.missions.fermi.time import Time

        for i, m in enumerate(MET_VALUES):
            t = Time(m, format="fermi", scale="utc")
            iso_new = t.utc.datetime.strftime("%Y-%m-%d %H:%M:%S.%f")
            expected = golden["met_iso"][i]
            # Strip timezone suffix from old gbm output for comparison
            expected_stripped = expected.split("+")[0]
            assert iso_new == expected_stripped, (
                f"MET {m}: old='{expected}' new='{iso_new}'"
            )

    def test_datetime_to_met(self, golden):
        from gdt.missions.fermi.time import Time

        for i, dt in enumerate(REF_DATETIMES):
            t = Time(dt, scale="utc")
            new_met = t.fermi
            expected = golden["dt_to_met"][i]
            np.testing.assert_allclose(
                new_met, expected, atol=1e-3,
                err_msg=f"datetime {dt}: old={expected} new={new_met}",
            )


class TestAngularSeparationEquivalence:
    """haversine replacement must match to float64 precision."""

    def test_haversine_values(self, golden):
        from astropy.coordinates import angular_separation

        expected = golden["haversine"]
        for i, row in enumerate(HAVERSINE_CASES):
            lon1, lat1, lon2, lat2 = np.radians(row)
            new_val = np.degrees(angular_separation(lon1, lat1, lon2, lat2))
            np.testing.assert_allclose(
                new_val, expected[i], atol=1e-10,
                err_msg=f"Case {i}: {row}",
            )


class TestSunPositionEquivalence:
    """get_sun_loc replacement must agree within 0.01 degrees."""

    def test_sun_ra_dec(self, golden):
        from astropy.coordinates import get_sun
        from gdt.missions.fermi.time import Time

        expected_ra = golden["sun_ra"]
        expected_dec = golden["sun_dec"]

        for i, m in enumerate(MET_VALUES):
            t = Time(m, format="fermi", scale="utc")
            sun = get_sun(t)
            np.testing.assert_allclose(
                sun.ra.deg, expected_ra[i], atol=0.01,
                err_msg=f"Sun RA at MET={m}",
            )
            np.testing.assert_allclose(
                sun.dec.deg, expected_dec[i], atol=0.01,
                err_msg=f"Sun Dec at MET={m}",
            )


class TestCredibleLevelsEquivalence:
    """find_greedy_credible_levels must produce identical output."""

    def test_credible_levels(self, golden):
        from grpype._compat import find_greedy_credible_levels

        expected = golden["credible_levels"]
        result = find_greedy_credible_levels(PROB_ARRAY)
        np.testing.assert_array_equal(
            result, expected,
            err_msg="find_greedy_credible_levels output differs",
        )
