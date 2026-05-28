"""
Compatibility helpers for the gbm -> gdt migration.

Provides drop-in replacements for gbm functions that have no direct
equivalent in gdt-core / gdt-fermi.
"""
from pathlib import Path

import numpy as np
from astropy.coordinates import SkyCoord, angular_separation as _angular_separation
from astropy.coordinates import get_sun as _get_sun
import astropy.units as u

from gdt.missions.fermi.time import Time


# ---------------------------------------------------------------------------
# Patch gdt HealPix._mesh_grid to avoid RA=0/360 wrapping artifact.
#
# The upstream grid uses np.linspace(0, 2pi, N) which includes both 0 and
# 2pi.  Astropy wraps RA=360 to 0, so pcolormesh sees a coordinate jump
# across the full sky and draws a giant quad that appears as a horizontal
# smear whenever the localization has non-zero probability near RA~0/360.
# Using endpoint=False eliminates the duplicate point.
# ---------------------------------------------------------------------------
def _patched_mesh_grid(self, num_phi, num_theta):
    import healpy as hp

    numpts_phi = int(num_phi)
    numpts_theta = int(num_theta)
    if numpts_phi <= 0 or numpts_theta <= 0:
        raise ValueError('num_phi and num_theta must be positive')

    theta = np.linspace(np.pi, 0.0, numpts_theta)
    phi = np.linspace(0.0, 2 * np.pi, numpts_phi, endpoint=False)
    phi_grid, theta_grid = np.meshgrid(phi, theta)
    grid_pix = hp.ang2pix(self.nside, theta_grid, phi_grid)
    return (grid_pix, phi, theta)


from gdt.core.healpix import HealPix as _HealPix  # noqa: E402
_HealPix._mesh_grid = _patched_mesh_grid


# ---------------------------------------------------------------------------
# Patch HealPixLocalization.confidence_region_path to smooth the contour grid.
#
# The default gdt code evaluates the map on a coarse RA/Dec grid using
# nearest-neighbor lookups, which creates blocky, jagged "staircase" contours,
# especially for large localizations. Applying a small Gaussian filter to the
# grid before contouring smooths out the pixel edges and produces clean curves
# similar to the old gbm_data_tools behavior.
# ---------------------------------------------------------------------------
def _patched_confidence_region_path_smooth(self, clevel, numpts_ra=360,
                                           numpts_dec=180):
    from matplotlib.pyplot import contour as Contour
    from scipy.ndimage import gaussian_filter

    if clevel < 0.0 or clevel > 1.0:
        raise ValueError('clevel must be between 0 and 1')

    # create the grid and integrated probability array
    grid_pix, phi, theta = self._mesh_grid(numpts_ra, numpts_dec)
    sig_arr = 1.0 - self.sig[grid_pix]

    # Smooth the grid to remove jagged pixel edges
    # mode='wrap' for RA (axis 1), 'nearest' for DEC (axis 0)
    sig_arr = gaussian_filter(sig_arr, sigma=1.5, mode=['nearest', 'wrap'])

    ra = self._phi_to_ra(phi)
    dec = self._theta_to_dec(theta)

    # use matplotlib contour to produce a path object
    contour = Contour(ra, dec, sig_arr, [clevel])

    # get the contour path, which is made up of segments
    paths = contour.collections[0].get_paths()

    # extract all the vertices
    pts = [path.vertices for path in paths]

    # unfortunately matplotlib will plot this, so we need to remove
    for c in contour.collections:
        c.remove()

    return pts


from gdt.core.healpix import HealPixLocalization as _HealPixLoc  # noqa: E402
_HealPixLoc.confidence_region_path = _patched_confidence_region_path_smooth


def haversine(lon1, lat1, lon2, lat2, deg=True):
    """Drop-in replacement for ``gbm.coords.haversine``.

    Returns the angular separation in degrees (when *deg=True*).
    """
    if deg:
        lon1, lat1, lon2, lat2 = (
            np.radians(lon1), np.radians(lat1),
            np.radians(lon2), np.radians(lat2),
        )
    result = _angular_separation(lon1, lat1, lon2, lat2)
    if deg:
        return np.degrees(result)
    return result


def get_sun_loc(met):
    """Drop-in replacement for ``gbm.coords.get_sun_loc``.

    Parameters
    ----------
    met : float
        Fermi Mission Elapsed Time.

    Returns
    -------
    (ra, dec) : tuple of float
        Sun RA and Dec in degrees (ICRS).
    """
    t = Time(met, format="fermi", scale="utc")
    sun = _get_sun(t)
    return (sun.ra.deg, sun.dec.deg)


def find_greedy_credible_levels(p):
    """Drop-in replacement for ``gbm.data.localization.find_greedy_credible_levels``.

    Calculate the credible values of a probability array using a greedy
    algorithm.

    Parameters
    ----------
    p : np.ndarray
        Probability array (should sum to ~1).

    Returns
    -------
    np.ndarray
        Credible-level array (same shape as *p*).
    """
    p = np.asarray(p, dtype=np.float64)
    i = np.flipud(np.argsort(p))
    sorted_credible = np.cumsum(p[i])
    credible_levels = np.empty_like(p)
    credible_levels[i] = sorted_credible
    return credible_levels



class PosHistCompat:
    """Thin wrapper around ``GbmPosHist`` that provides the old ``PosHist`` API.

    Only the methods actually used in this codebase are implemented:
    ``time_range``, ``to_equatorial``, ``get_geocenter_radec``,
    ``get_earth_radius``, and ``full_path`` (as ``filename``).
    """

    def __init__(self, gbm_poshist, filepath=None):
        self._ph = gbm_poshist
        self._frame = gbm_poshist.get_spacecraft_frame()
        self._filepath = Path(filepath) if filepath is not None else None

    # ── proxy attributes ──────────────────────────────────────────────
    @property
    def full_path(self):
        if self._filepath is not None:
            return self._filepath
        return Path(self._ph.filename)

    @property
    def filename(self):
        return self._ph.filename

    @property
    def time_range(self):
        return (self._frame.obstime[0].fermi,
                self._frame.obstime[-1].fermi)

    # ── coordinate helpers ────────────────────────────────────────────
    def _frame_at(self, met):
        t = Time(met, format="fermi", scale="utc")
        return self._frame.at(t)

    def to_equatorial(self, fermi_az, fermi_zen, met):
        """Convert spacecraft (az, zenith) to equatorial (RA, Dec) in degrees.

        Matches old gbm behaviour: scalar inputs (or single-element arrays)
        produce scalar float outputs; array inputs produce 1-D numpy arrays.
        """
        single = self._frame_at(met)
        az_arr = np.atleast_1d(np.asarray(fermi_az, dtype=float))
        el_arr = 90.0 - np.atleast_1d(np.asarray(fermi_zen, dtype=float))
        sc = SkyCoord(az=az_arr, el=el_arr, frame=single, unit="deg")
        icrs = sc.icrs
        ra = np.asarray(icrs.ra.deg).ravel()
        dec = np.asarray(icrs.dec.deg).ravel()
        if ra.size == 1:
            return float(ra[0]), float(dec[0])
        return ra, dec

    def get_geocenter_radec(self, met):
        single = self._frame_at(met)
        gc = single.geocenter
        return (gc.ra.deg, gc.dec.deg)

    def get_earth_radius(self, met):
        single = self._frame_at(met)
        return single.earth_angular_radius.to(u.deg).value

    def to_fermi_frame(self, ra, dec, met):
        """Convert equatorial (RA, Dec) to spacecraft (az, zenith) in degrees.

        Matches old gbm behaviour: scalar/single-element -> scalar float out.
        """
        single = self._frame_at(met)
        ra_arr = np.atleast_1d(np.asarray(ra, dtype=float))
        dec_arr = np.atleast_1d(np.asarray(dec, dtype=float))
        icrs = SkyCoord(ra=ra_arr, dec=dec_arr, unit="deg", frame="icrs")
        sc = icrs.transform_to(single)
        az = np.asarray(sc.az.deg).ravel()
        zen = 90.0 - np.asarray(sc.el.deg).ravel()
        if az.size == 1:
            return float(az[0]), float(zen[0])
        return az, zen

    def get_saa_passage(self, met):
        """Return True if Fermi is in the SAA at the given MET."""
        if self._saa_interp is None:
            self._init_saa_interp()
        return bool(self._saa_interp(met) > 0.5)

    def get_mcilwain_l(self, met):
        """Approximate McIlwain L from lat/lon (same algorithm as old gbm)."""
        if self._geo_interp is None:
            self._init_geo_interp()
        lat = float(self._geo_interp[0](met))
        lon = float(self._geo_interp[1](met))
        cos_lat = np.cos(np.radians(lat))
        return 1.0 / cos_lat ** 2

    # ── lazy interpolation setup ──────────────────────────────────────
    _saa_interp = None
    _geo_interp = None

    def _init_saa_interp(self):
        from scipy.interpolate import interp1d
        states = self._ph.get_spacecraft_states()
        times = np.array(states["time"].fermi)
        saa = np.array(states["saa"], dtype=float)
        self._saa_interp = interp1d(
            times, saa, kind="nearest", bounds_error=False, fill_value=0.0,
        )

    def _init_geo_interp(self):
        from scipy.interpolate import interp1d
        frame = self._frame
        times = np.array(frame.obstime.fermi)
        itrs = frame.sc_itrs
        lats = itrs.earth_location.geodetic.lat.deg
        lons = itrs.earth_location.geodetic.lon.deg
        self._geo_interp = (
            interp1d(times, lats, kind="linear", bounds_error=False, fill_value="extrapolate"),
            interp1d(times, lons, kind="linear", bounds_error=False, fill_value="extrapolate"),
        )

    # ── delegated FITS methods ────────────────────────────────────────
    def __getattr__(self, name):
        return getattr(self._ph, name)
