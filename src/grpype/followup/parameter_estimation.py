import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta
from time import time, sleep
import tracemalloc
import gc

import numpy as np
import pandas as pd
import healpy as hp
import emcee

from gdt.missions.fermi.time import Time
from gdt.missions.fermi.gbm.response import GbmRsp as RSP
from gdt.core.healpix import HealPixLocalization
from gdt.missions.fermi.gbm.localization import GbmHealPix
from gdt.core.plot.sky import EquatorialPlot
from gdt.core.plot.plot import SkyPoints
from grpype._compat import get_sun_loc, haversine, find_greedy_credible_levels

from grpype.detection.pipeline import Detection
from grpype.detection.utils import rolling_mean_padded
from grpype.data_io.data_handlers import DataLoaders, TTEData
from grpype.detection.utils import square_convolve, xyz2thetahpi, fibonacci_sphere
from grpype.detection.global_params import DATAPATH, EPS, detectors

import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings


burst_dict = {
    0.002: (10, 30),
    0.003: (10, 30),
    0.004: (10, 30),
    0.005: (10, 30),
    0.007: (10, 30),
    0.009: (10, 30),
    0.012: (10, 30),
    0.016: (10, 30),
    0.022: (10, 30),
    0.03: (40, 120),
    0.04: (40, 120),
    0.054: (40, 120),
    0.073: (40, 120),
    0.098: (40, 120),
    0.133: (60, 180),
    0.179: (60, 180),
    0.242: (60, 180),
    0.327: (60, 180),
    0.441: (60, 180),
    0.596: (60, 180),
    0.804: (60, 180),
    1.086: (60, 180),
    1.466: (100, 270),
    1.979: (100, 270),
    2.671: (100, 270),
    3.606: (100, 300),
    4.869: (100, 300),
    6.573: (100, 300),
}

slcs = [
    [0.0, 0.053],
    [0.048, 0.105],
    [0.09999999999999999, 0.158],
    [0.153, 0.211],
    [0.206, 0.263],
    [0.258, 0.316],
    [0.311, 0.368],
    [0.363, 0.421],
    [0.416, 0.474],
    [0.469, 0.526],
    [0.521, 0.579],
    [0.574, 0.632],
    [0.627, 0.684],
    [0.679, 0.737],
    [0.732, 0.789],
    [0.784, 0.842],
    [0.837, 0.895],
    [0.89, 0.947],
    [0.942, 1.0]
]

def get_closest_burst(t, burst_dict):
    closest_key = min(burst_dict.keys(), key=lambda k: abs(k - t))
    return burst_dict[closest_key]

def check_mem(text=''):
    current, peak = tracemalloc.get_traced_memory()
    current_mb = current / 10**6
    peak_mb = peak / 10**6
    print(f"{text} {current_mb}MB; Peak was {peak_mb}MB")


class SpectralModel:
    def __call__(self, params, energy_bin_centroids):
        raise NotImplementedError

    def log_prior(self, params):
        raise NotImplementedError


class BandFunction(SpectralModel):
    def __init__(self, epiv=100, A=1, alpha_prior=[-2, 3], beta_prior=[-7, 1], epeak_prior=[8, 5000]):
        self.epiv = epiv
        self.A = A
        
        self.min_alpha = alpha_prior[0]
        self.max_alpha = alpha_prior[1]
        
        self.min_beta = beta_prior[0]
        self.max_beta = beta_prior[1]
        
        self.min_epeak = epeak_prior[0]
        self.max_epeak = epeak_prior[1]

        self.param_bounds = [alpha_prior, beta_prior, epeak_prior]
        self.param_labels = ['alpha', 'beta', 'epeak']

    def __call__(self, params, energy_bin_centroids):
        alpha, beta, epeak = params
        epiv, A = self.epiv, self.A

        if alpha < beta:
            return np.zeros_like(energy_bin_centroids)
        if epeak <= 0 or epiv <= 0 or not np.isfinite(epeak):
            return np.zeros_like(energy_bin_centroids)
        if abs(alpha + 2.0) < 1e-3:
            return np.zeros_like(energy_bin_centroids)

        spectrum = np.zeros(energy_bin_centroids.shape)
        
        threshold = epeak*(alpha - beta)/(alpha + 2)

        low_e = A*(energy_bin_centroids/epiv)**alpha*np.exp(-(alpha + 2)*energy_bin_centroids/epeak)
        high_e = A*(energy_bin_centroids/epiv)**beta*np.exp(beta - alpha)*((alpha - beta)*epeak/(epiv*(alpha + 2)))**(alpha - beta)

        spectrum[energy_bin_centroids < threshold] = low_e[energy_bin_centroids < threshold]
        spectrum[energy_bin_centroids >= threshold] = high_e[energy_bin_centroids >= threshold]
        spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)

        return spectrum

    def log_prior(self, params):
        alpha, beta, epeak = params
        if (self.min_alpha <= alpha <= self.max_alpha and
             self.min_beta <= beta <= self.max_beta and
               self.min_epeak <= epeak <= self.max_epeak and
                 alpha >= beta):
            return 0.0
        return -np.inf


class CutoffPowerLaw(SpectralModel):
    def __init__(self, epiv=100, A=1, alpha_prior=[-2, 3], epeak_prior=[8, 5000]):
        self.epiv = epiv
        self.A = A

        self.min_alpha = alpha_prior[0]
        self.max_alpha = alpha_prior[1]
        
        self.min_epeak = epeak_prior[0]
        self.max_epeak = epeak_prior[1]

        self.param_bounds = [alpha_prior, epeak_prior]
        self.param_labels = ['alpha', 'epeak']

    def __call__(self, params, bin_centroids):
        alpha, epeak = params
        x = bin_centroids
        return self.A * (x / self.epiv)**alpha * np.exp(-(alpha + 2) * x / epeak)

    def log_prior(self, params):
        alpha, epeak = params
        if (self.min_alpha <= alpha <= self.max_alpha and
            self.min_epeak <= epeak <= self.max_epeak):
            return 0.0
        return -np.inf


class PowerLaw(SpectralModel):
    def __init__(self, epiv=100, A=1, alpha_prior=[-10, 10]):
        self.epiv = epiv
        self.A = A

        self.min_alpha = alpha_prior[0]
        self.max_alpha = alpha_prior[1]

        self.param_bounds = [alpha_prior]
        self.param_labels = ['alpha']

    def __call__(self, alpha, bin_centroids):
        x = bin_centroids
        return self.A * (x / self.epiv)**alpha

    def log_prior(self, alpha):
        if (self.min_alpha <= alpha <= self.max_alpha):
            return 0.0
        return -np.inf


class FullResponseHandler:
    def __init__(self, burstdata, trigger_met, timescale, binning, spec_model, trig_offset=None, rsp_folder='PE_231027145949', rsp_cache_max=1024):
        self.metobj = trigger_met
        self.rsps_folder = DATAPATH / 'rsp' / rsp_folder

        self.timescale = timescale
        self.binning = binning

        self.upper_factor = {
            0.002: 1.5,
            0.003: 1.5,
            0.004: 1.5,
            0.005: 1.5,
            0.007: 1.5,
            0.009: 1.5,
            0.012: 1.5,
            0.016: 1.5,
            0.022: 1.5,
            0.03:  1.5,
            0.04:  1.5,
            0.054: 1.5,
            0.073: 1.5,
            0.098: 1.5,
            0.133: 1.5,
            0.179: 1.5,
            0.242: 1.5,
            0.327: 1.5,
            0.441: 1.5,
            0.596: 1.5,
            0.804: 1.5,
            1.086: 1.5,
            1.466: 1.5,
            1.979: 1.5,
            2.671: 1.5,
            3.606: 1.5,
            4.869: 1.05,
            6.573: 1.25,
        }

        if self.metobj.fermi >= burstdata.poshist.time_range[0] and self.metobj.fermi <= burstdata.poshist.time_range[1]:
            poshist = burstdata.poshist
        elif self.metobj.fermi > burstdata.poshist.time_range[1]:
            loader = DataLoaders()
            date = self.metobj.utc.datetime + pd.Timedelta(minutes=5)
            poshist = loader.open_poshist_by_date(date)
        elif self.metobj.fermi < burstdata.poshist.time_range[0]:
            loader = DataLoaders()
            date = self.metobj.utc.datetime - pd.Timedelta(minutes=5)
            poshist = loader.open_poshist_by_date(date)
        
        self.poshist = poshist
        
        self.phis = np.load(self.rsps_folder / 'phis.npy')
        self.thetas = np.load(self.rsps_folder / 'thetas.npy')
        self.ras, self.decs = self.poshist.to_equatorial(self.phis, self.thetas, self.metobj.fermi)
        
        self.spec_model = spec_model  # callable: params, bin_centroids -> spectrum

        self.rsp_cache = {}
        self.rsp_cache_counter = {}
        self.rsp_cache_max = rsp_cache_max
        
        self.nai_bin_centroids = np.load(self.rsps_folder / 'nai_photon_bin_centroids.npy')
        self.nai_bin_widths = np.load(self.rsps_folder / 'nai_photon_bin_widths.npy')
        self.bgo_bin_centroids = np.load(self.rsps_folder / 'bgo_photon_bin_centroids.npy')
        self.bgo_bin_widths = np.load(self.rsps_folder / 'bgo_photon_bin_widths.npy')
        
        self.echans = 128  # Set according to your response

        self.amp_cache = {}

        rolling_window_sec, slice_seconds = get_closest_burst(timescale, burst_dict)
        min_dist_sec = 30 if binning < 0.01 else 60
        detection = Detection(binning, timescale, rolling_window_sec, 3*timescale)
        self.d_t, self.bkg_t, self.slice_trig_ind = detection.fast_d_bkg_slice(burstdata, self.metobj.fermi, slice_seconds, min_dist_sec)
        # del burstdata.data
        # del burstdata.bkgs
        if self.d_t is None or self.bkg_t is None or self.slice_trig_ind is None:
            return

        self.bdur_samp = burstdata.burst_duration_samp
        self.bkg_window = detection.bkg_window
        self.bkg_window_gap = detection.bkg_window_gap

        if trig_offset is None:
            trig_offset = 0
        else:
            trig_offset = int(np.abs(trig_offset) / binning * np.sign(trig_offset))

        self.tleft = self.slice_trig_ind - self.bdur_samp//2 + (self.bdur_samp+1)%2 + trig_offset
        self.tright = self.slice_trig_ind + self.bdur_samp//2 + 1 + trig_offset

        self.d = self.d_t[self.tleft:self.tright].sum(axis=0)
        self.bkg = self.bkg_t[self.tleft:self.tright].sum(axis=0)
        self.fltr_bkg = self.bkg_t.mean(axis=0)

    def load_rsp(self, ra, dec):
        """Load response for closest grid point to (ra, dec)"""

        dra = np.mod(self.ras - ra + 180, 360) - 180
        ddec = self.decs - dec

        idx = np.argmin(dra**2 + ddec**2)

        i = int(idx)

        if i in self.rsp_cache:
            self.rsp_cache_counter[i] += 1
            return self.rsp_cache[i]

        rsp = np.load(self.rsps_folder / f'rsp_{i}.npy')
        self.rsp_cache[i] = rsp
        self.rsp_cache_counter[i] = 1

        if len(self.rsp_cache) > self.rsp_cache_max:
            # Remove the least used response from cache
            min_key = sorted(self.rsp_cache_counter, key=self.rsp_cache_counter.get)
            for key in min_key[:min(self.rsp_cache_max//10, len(self.rsp_cache))]:
                del self.rsp_cache[key]
                del self.rsp_cache_counter[key]
        
        return rsp

    def fold_model(self, rsp, spec_params, normalize=True):
        spec_nai = self.spec_model(spec_params, self.nai_bin_centroids)
        spec_bgo = self.spec_model(spec_params, self.bgo_bin_centroids)

        folded_nai = np.einsum('ijk,j->ik', rsp[:12], spec_nai * self.nai_bin_widths)
        folded_bgo = np.einsum('ijk,j->ik', rsp[12:], spec_bgo * self.bgo_bin_widths)

        folded = np.row_stack([folded_nai, folded_bgo])
        template = folded.reshape(-1)

        if normalize:
            template = template / (np.sum(template) + EPS)

        return template

    def calc_amp(self, data, bkg, template, psd_drift=None):
        bkg = np.copy(bkg) + EPS
        fltr = np.log1p(template / bkg)

        if psd_drift is None:
            b1 = 0.0
            b2 = 1.0
        else:
            b1 = np.clip(psd_drift[0], -0.25, 0.25)
            b2 = np.sqrt(np.clip(psd_drift[1], 0.85**2, 1.25**2))

        numer = b2 * np.sum((data - bkg) * fltr) + b1 * np.sqrt(np.sum(bkg * fltr**2))
        denom = np.sum(template * fltr) + EPS
        return numer / denom

    def log_likelihood(self, theta, d, bkg, psd_drift=None):
        ra, dec = theta[:2]
        spec_params = theta[2:]

        rsp = self.load_rsp(ra, dec)
        template = self.fold_model(rsp, spec_params)

        if not np.all(np.isfinite(template)) or np.any(template < 0):
            return -np.inf

        amp = self.calc_amp(d, bkg, template, psd_drift=psd_drift)

        # template = amp * template
        # psd_drift = self.calc_drift(amp*template)
        # amp = self.calc_amp(d, bkg, template, psd_drift=psd_drift)
        self.amp_cache[tuple(theta)] = amp

        ll = np.sum(d * np.log(1 + amp * template / bkg) - amp * template)
        # ll = np.sum((d-bkg) * amp * template / bkg)  # Gauss

        if not np.isfinite(ll):
            return -np.inf

        return ll

    def calc_drift(self, template):
        fltr = np.log1p(template/(self.fltr_bkg + EPS))
        mf_numers = square_convolve(self.d_t - self.bkg_t, self.bdur_samp, fltr)
        mf_vars = square_convolve(self.bkg_t, self.bdur_samp, fltr**2)

        temp_mf = mf_numers / (np.sqrt(mf_vars) + EPS)

        zvar0 = rolling_mean_padded(temp_mf, self.bkg_window, gap=self.bkg_window_gap)
        zvar = rolling_mean_padded(temp_mf**2, self.bkg_window, gap=self.bkg_window_gap)
        zvar -= zvar0**2

        return zvar0[self.slice_trig_ind], zvar[self.slice_trig_ind]

    def log_prior(self, theta):
        ra, dec = theta[:2]
        spec_params = theta[2:]

        if not (self.ras.min() <= ra <= self.ras.max() and self.decs.min() <= dec <= self.decs.max()):
            return -np.inf

        spec_prior = self.spec_model.log_prior(spec_params)
        if not np.isfinite(spec_prior):
            return -np.inf

        spatial_prior = np.log(np.cos(np.deg2rad(dec)))

        return spec_prior + spatial_prior

    def log_posterior(self, theta, d, bkg, psd_drift=None):
        lp = self.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + self.log_likelihood(theta, d, bkg, psd_drift=psd_drift)

    def run_mcmc(self, initial_guess, nwalkers=32, nsteps=1000, progress=True):
        ndim = len(initial_guess)
        p0 = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers, ndim, self.log_posterior, args=(self.d, self.bkg))
        sampler.run_mcmc(p0, nsteps, progress=progress)
        
        return sampler
    
    def plot_skymap(self, samples, fwhm_deg=3.0, points=None, clevels=None, title=None, **plot_kwargs):
        if clevels is None:
            clevels = [0.9]
        elif not type(clevels) == list:
            clevels = [clevels]

        nside = 64
        npix = hp.nside2npix(nside)

        theta_samps = np.deg2rad(90.0 - samples[:, 1])
        phi_samps = np.deg2rad(samples[:, 0] % 360.0)

        pix = hp.ang2pix(nside, theta_samps, phi_samps, nest=False)
        counts = np.bincount(pix, minlength=npix).astype(float)

        probmap = counts / counts.sum() if counts.sum() > 0 else np.ones(npix)/npix

        if fwhm_deg > 0:
            probmap = hp.smoothing(probmap, fwhm=np.deg2rad(fwhm_deg), verbose=False)
            probmap = np.maximum(0, probmap)
            probmap /= probmap.sum()

        theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)

        dec = 90 - np.degrees(theta)
        ra = np.degrees(phi)

        # Build a GBM-aware localization map so sky plotting can access frame
        # metadata (detector pointings, Sun/Earth overlays).
        try:
            frame = self.poshist._frame_at(self.metobj.fermi)
            hpmap = GbmHealPix.from_data(
                probmap,
                trigtime=self.metobj.fermi,
                quaternion=frame.quaternion,
                scpos=frame.obsgeoloc,
            )
        except Exception:
            hpmap = HealPixLocalization.from_data(probmap)

        fermiplot = EquatorialPlot()
        add_kwargs = dict(gradient=True, clevels=clevels, sun=True, earth=True)
        if not hasattr(hpmap, "frame"):
            add_kwargs.update(detectors=[], sun=False, earth=False)
        add_kwargs.update(plot_kwargs)

        # Contour paths returned by gdt can contain Dec values that are
        # numerically just outside [-90, 90] (e.g. 90.00000000000001) when a
        # contour touches a pole. astropy.SkyCoord then raises. Wrap the
        # method on this instance to clip latitudes into the valid range.
        _orig_confidence_region_path = hpmap.confidence_region_path

        def _clipped_confidence_region_path(*args, **kwargs):
            paths = _orig_confidence_region_path(*args, **kwargs)
            return [
                np.column_stack([p[:, 0], np.clip(p[:, 1], -90.0, 90.0)])
                for p in paths
            ]

        hpmap.confidence_region_path = _clipped_confidence_region_path
        try:
            fermiplot.add_localization(hpmap, **add_kwargs)
        finally:
            hpmap.confidence_region_path = _orig_confidence_region_path
        fermiplot.ax.set_facecolor('white')
        fermiplot.ax.set_yticks(np.deg2rad(np.arange(-75, 76, 15)))
        fermiplot.ax.set_yticklabels([f'{d}°' for d in range(-75, 76, 15)])
        
        default_title = f'{hpmap.area(clevels[0]):.2f} square degrees at {clevels[0]:.3f}% confidence level. Best fit: RA={ra[np.argmax(probmap)]:.2f}, DEC={dec[np.argmax(probmap)]:.2f}'
        if title is None:
            title = default_title
        
        fermiplot.ax.set_title(title)

        return fermiplot, hpmap

    def calc_diff_photon_flux(self, ra, dec, spec_params, ebins=None):
        """
        Calculate the photon flux for a given position and spectral parameters.
        params
        -------
        ra : float
            The right ascension of the source.
        dec : float
            The declination of the source.
        spec_params : array-like
            The spectral parameters to use for the model.

        Returns
        -------
        flux : float
            The calculated photon flux [photons/cm^2/s/keV].
        """
        rsp = self.load_rsp(ra, dec)
        folded_model = self.fold_model(rsp, spec_params, normalize=True)
        amp = self.calc_amp(self.d, self.bkg, folded_model)

        folded_model_not_normed = self.fold_model(rsp, spec_params, normalize=False)

        expected_counts = (folded_model_not_normed * self.timescale).sum()
        Aphys = amp / expected_counts
        
        if ebins is None:
            ebins = 10**np.linspace(np.log10(5), np.log10(30000), int(1e6))
    
        spec = Aphys * self.spec_model(spec_params, ebins)

        return spec

    def calc_localization_stats(self, ra_samps, dec_samps, inflate_sun=1, fwhm_deg=3.0):
        nside = 64
        npix = hp.nside2npix(nside)

        theta_samps = np.deg2rad(90.0 - dec_samps)
        phi_samps = np.deg2rad(ra_samps % 360.0)

        pix = hp.ang2pix(nside, theta_samps, phi_samps, nest=False)
        counts = np.bincount(pix, minlength=npix).astype(float)
        
        sky_posterior = counts / counts.sum() if counts.sum() > 0 else np.ones(npix)/npix
        if fwhm_deg > 0:
            sky_posterior = hp.smoothing(sky_posterior, fwhm=np.deg2rad(fwhm_deg), verbose=False)
            sky_posterior = np.maximum(0, sky_posterior)
            sky_posterior /= sky_posterior.sum()

        theta, phi = hp.pix2ang(nside, np.arange(npix), nest=False)
        dec = 90 - np.degrees(theta)
        ra = np.degrees(phi)
        
        sun_mask = self._is_occulted(ra, dec, 'sun', inflate=inflate_sun)
        earth_mask = self._is_occulted(ra, dec, 'earth')
        no_es_mask = ~(sun_mask | earth_mask)

        res_e = np.sum(sky_posterior[earth_mask]/earth_mask.sum())
        res_s = np.sum(sky_posterior[sun_mask]/sun_mask.sum()/inflate_sun)
        res_nes = np.sum(sky_posterior[no_es_mask]/no_es_mask.sum())

        sun_statistic = res_s/(res_nes + EPS)
        earth_statistic = res_e/(res_nes + EPS)

        return earth_statistic, sun_statistic

    def _is_occulted(self, ras, decs, body, inflate=1):
        if body == 'earth':
            center = np.array(self.poshist.get_geocenter_radec(self.metobj.fermi)).reshape(2, -1)
            angular_radius = self.poshist.get_earth_radius(self.metobj.fermi) * inflate
        elif body == 'sun':
            center = np.array(get_sun_loc(self.metobj.fermi)).reshape(2, -1)
            angular_radius = 0.5 * inflate

        angle = haversine(center[0, :], center[1, :], ras, decs).astype(np.float32)
        occulted = (angle <= angular_radius)
        return occulted

    @classmethod
    def from_gbm_resps_bank(cls, ang_res, burstdata, metobj, timescale, binning, spec_model, radec_bounds=None, rsp_cache_max=1024, verbose=False):
        """
        You must generate the responses in the spacecraft coordinates if you use
        a fixed respone for the parameter estimation!
        """
        loader = DataLoaders()
        while True:
            try:
                # cspec = loader.open_cspec_by_date(metobj.utc.datetime, verbose=False)
                poshist = loader.open_poshist_by_date(metobj.utc.datetime, verbose=False)
                break
            except:
                sleep(10)
                continue

        year, month, day, hour = loader._fix_time(metobj.utc.datetime)

        rsps_path = DATAPATH / f'rsp/{metobj.utc.datetime.strftime("%y%m%d%H%M%S")}'
        os.makedirs(rsps_path, exist_ok=True)
        shutil.copy(poshist.full_path, rsps_path / poshist.full_path.name)

        nsky = int(4*np.pi * (180 / np.pi)**2 / ang_res)
        thetas, phis = xyz2thetahpi(*fibonacci_sphere(nsky), azel=True)
        thetas = (-thetas + 90) % 180
        phis = (phis + 360) % 360
        ras, decs = poshist.to_equatorial(phis, thetas, metobj.fermi)

        if radec_bounds is not None:
            radec_mask = (ras >= radec_bounds[0]) & (ras <= radec_bounds[1]) & (decs >= radec_bounds[2]) & (decs <= radec_bounds[3])
            ras, decs = ras[radec_mask], decs[radec_mask]
            thetas, phis = thetas[radec_mask], phis[radec_mask]

        np.save(rsps_path / 'thetas', thetas)
        np.save(rsps_path / 'phis', phis)
        
        rsps_path = DATAPATH / f'rsp/{metobj.utc.datetime.strftime("%y%m%d%H%M%S")}'
        os.makedirs(rsps_path, exist_ok=True)
        shutil.copy(poshist.full_path, rsps_path / poshist.full_path.name)

        for detector in detectors:
            fname = loader.cspec_template.format(detector, metobj.utc.datetime.strftime('%y%m%d'), "00")
            cspec_saved_folder = loader.cspec_folder / str(year) / str(month) / str(day) / fname
            cspec_move_name = rsps_path / fname
            if not cspec_saved_folder.exists():
                loader.download_cspec(metobj.utc.datetime, detector, verbose=False)
                
            shutil.copy(cspec_saved_folder, cspec_move_name)
            os.makedirs(rsps_path / detector, exist_ok=True)

        stime = metobj.fermi

        t0 = time()
        for i in range(len(ras)):
            t1 = time()

            gen_resp_command = [
                "SA_GBM_RSP_Gen.pl",
                f"-S{stime}",
                f"-R{ras[i]}",
                f"-D{decs[i]}",
                "-Ccspec",
                str(rsps_path) + '/.'
                ]
            
            result = subprocess.run(gen_resp_command, check=True, capture_output=True, text=True)

            rsp_all = []
            for j, detector in enumerate(detectors):
                rsp_name = f"glg_cspec_{detector}*.rsp"
                filelist = list(rsps_path.glob(rsp_name))
                while len(filelist) == 0:
                    filelist = list(rsps_path.glob(rsp_name))
                
                shutil.move(filelist[0], rsps_path / detector / f'detector_{detector}_index_{i}.rsp')
                
                rsp = RSP.open(rsps_path / f'{detector}/detector_{detector}_index_{i}.rsp')
                drm = rsp.drm
                rsp_all.append(drm.matrix)

                if i == 0:
                    if j == 0:
                        np.save(rsps_path / 'nai_photon_bin_centroids', drm.photon_bin_centroids)
                        np.save(rsps_path / 'nai_photon_bin_widths', drm.photon_bin_widths)
                    if j == 13:
                        np.save(rsps_path / 'bgo_photon_bin_centroids', drm.photon_bin_centroids)
                        np.save(rsps_path / 'bgo_photon_bin_widths', drm.photon_bin_widths)

            rsp_stack = np.stack(rsp_all, axis=0)
            np.save(rsps_path / f'rsp_{i}', rsp_stack)

            log_name = "*.logfile"
            for log_path in list(rsps_path.glob(log_name)):
                os.remove(log_path)

            if verbose:
                print(f"Response {i+1}/{len(ras)} done in {time()-t1:.2f} s, total time: {time()-t0:.2f} s")

        return cls(
            burstdata=burstdata,
            trigger_met=metobj,
            timescale=timescale,
            binning=binning,
            spec_model=spec_model,
            rsp_folder=rsps_path.name,
            rsp_cache_max=rsp_cache_max
        )

    def optimize_duration(self, init_guess):
        lower = 0.03 if self.binning == 0.01 else 0.003
        # upper = 1.5 * self.timescale if self.binning == 0.01 else 3 * self.timescale
        upper = self.upper_factor[self.timescale] * self.timescale

        durations_sec = np.logspace(np.log10(lower), np.log10(upper), 100)
        durations_samp = np.unique((durations_sec / self.binning).astype(int))
        durations_sec = durations_samp * self.binning

        ra, dec, alpha, beta, epeak = init_guess[:5]
        rsp = self.load_rsp(ra, dec)
        template = self.fold_model(rsp, (alpha, beta, epeak), normalize=True)

        lefts = self.slice_trig_ind - durations_samp//2 + (durations_samp+1)%2
        rights = self.slice_trig_ind + durations_samp//2 + 1

        bkg = self.bkg * durations_samp[:, None] / self.bdur_samp

        d_ts = np.array([self.d_t[left:right].sum(axis=0) for left, right in zip(lefts, rights)])
        amp_arr = np.array([self.calc_amp(d, b_i, template) for d, b_i in zip(d_ts, bkg)])

        ll = d_ts * np.log1p(amp_arr[:, None] * template[None, :] / bkg) - amp_arr[:, None] * template[None, :]
        ll = ll.sum(axis=1)

        best_idx = np.argmax(ll)
        best_duration = durations_sec[best_idx]

        return best_duration, ll[best_idx]

    def calc_tot_flux_statistics(self, samples, nsamples=2000, ebins=None):
        kev2erg = 1.60218e-9  # 1 keV in erg

        indices = np.arange(len(samples))
        indices = np.random.choice(indices, nsamples, replace=True)

        if ebins is None:
            nebins = int(1e4)
            ebins = 10**np.linspace(np.log10(5), np.log10(40_000), nebins)

        tot_energy_flux = np.zeros(nsamples)

        for i, ind in enumerate(indices):
            ra = samples[ind, 0]
            dec = samples[ind, 1]
            alpha = samples[ind, 2]
            beta = samples[ind, 3]
            epeak = samples[ind, 4]
            
            diff_photon_flux = self.calc_diff_photon_flux(ra, dec, [alpha, beta, epeak], ebins=ebins)

            tot_energy_flux[i] = np.sum((ebins * diff_photon_flux)[:-1] * np.diff(ebins)) * kev2erg

        mean_tot_energy_flux = np.median(tot_energy_flux)
        errp_tot_energy_flux = np.percentile(tot_energy_flux, 84) - mean_tot_energy_flux
        errm_tot_energy_flux = mean_tot_energy_flux - np.percentile(tot_energy_flux, 16)

        return mean_tot_energy_flux, errp_tot_energy_flux, errm_tot_energy_flux


def extract_stats(samples):
    # Return ML + 1-sigma confidence intervals
    percentiles = [16, 50, 84]
    results = {}
    labels = ['ra', 'dec', 'alpha', 'beta', 'epeak']
    for i, label in enumerate(labels):
        counts, bins = np.histogram(samples[:, i], bins=100)
        bin_centroids = (bins[1:] + bins[:-1]) / 2
        
        q16, q50, q84 = np.percentile(samples[:, i], percentiles)

        results[f'{label}_max'] = bin_centroids[np.argmax(counts)]
        results[f'{label}_med'] = q50
        results[f'{label}_errm'] = q50 - q16
        results[f'{label}_errp'] = q84 - q50
    return pd.DataFrame([results])


def run_pe_chunk(
    chunk_id,
    df_chunk,
    timeslides,
    output_dir,
    save_every=100,
    progress=True,
    rsp_root=None,
):
    tracemalloc.start()

    buffer = []

    log_path = os.path.join(output_dir, f"chunk_{chunk_id}_log.txt")  # Just for tracking how much time every trigger takes
    save_log_path = os.path.join(output_dir, f"save_chunk_{chunk_id}_log.txt")  # For the saving
    
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')
    
    if not os.path.exists(save_log_path):
        with open(save_log_path, 'w') as f:
            f.write('')
        last_saved = -1
    else:
        with open(save_log_path, 'r') as f:
            lines = f.readlines()
            if len(lines) > 0:
                last_saved = int(lines[-1].strip())
            else:
                last_saved = -1
            
    for count, (idx, row) in enumerate(df_chunk.iterrows()):
        t0 = time()
        index = row.name
        if index < last_saved:
            continue

        binning = row['binning']
        burst_duration = row['timescale']
        date = datetime.fromisoformat(row['trigtime'])
        trigmet = Time(date, scale='utc')
        slc_ind = row['slc_ind']

        init_guess = [row['ra'], row['dec'], row['alpha'], row['beta'], row['epeak']]
        
        if binning < 0.01:
            cut_len_seconds = 0
            slc = slcs[int(slc_ind)]
        else:
            cut_len_seconds = 10
            slc = None

        print(f"Processing trigger {idx}, date {row['trigtime']}, binning {binning}")
        try:
            if timeslides is not None:
                slides = row['timeslides']
                burstdata = TTEData(date, binning, burst_duration, debug_slides=slides, cut_len_seconds=cut_len_seconds, slice_time=slc, old_load=True)
            else:
                burstdata = TTEData(date, binning, burst_duration, cut_len_seconds=cut_len_seconds, slice_time=slc)

            if len(burstdata.full_gti_inds) == 0:
                with open(output_dir / 'failed.txt', 'a') as f:
                    f.write(f"Trigger {index} has a gti reconstruction error\n")
                continue
        
            trigger_time_samp = np.searchsorted(burstdata.time, trigmet.fermi)
            if trigger_time_samp < burstdata.full_gti_inds[0][0]:
                date = date - timedelta(hours=1)
                shiftflag = True
            elif trigger_time_samp > burstdata.full_gti_inds[-1][-1]:
                date = date + timedelta(hours=1)
                shiftflag = True
            else:
                shiftflag = False
            
            if shiftflag:
                del burstdata
                if timeslides is not None:
                    slides = row['timeslides']
                    burstdata = TTEData(date, binning, burst_duration, debug_slides=slides, cut_len_seconds=cut_len_seconds, slice_time=slc, old_load=True)
                else:
                    burstdata = TTEData(date, binning, burst_duration, cut_len_seconds=cut_len_seconds, slice_time=slc)
        except Exception as e:
            with open(output_dir / 'failed.txt', 'a') as f:
                f.write(f"Trigger {index} has a data loading error: {e}\n")
                continue

        spec_model = BandFunction()
        rsp_base = Path(rsp_root) if rsp_root is not None else DATAPATH
        rsp_folder = rsp_base / 'rsp/PE_231027145949'
        handler = FullResponseHandler(burstdata, trigmet, burst_duration, binning, spec_model, rsp_folder=rsp_folder)

        if handler.d_t is None or handler.bkg_t is None or handler.slice_trig_ind is None:
            with open(output_dir / 'failed.txt', 'a') as f:
                f.write(f"Trigger {index} has a data loading error\n")
            continue    

        if not (output_dir / 'samples' / f'trigger_{index}_samples.npy').exists():
            sampler = handler.run_mcmc(initial_guess=init_guess, nwalkers=32, nsteps=1000, progress=progress)
            samples = sampler.get_chain(discard=100, flat=True)
            if row.snr**2 >= 50:
                np.save(output_dir / 'samples' / f'trigger_{index}_samples.npy', samples)
        else:
            samples = np.load(output_dir / 'samples' / f'trigger_{index}_samples.npy')

        param_stats = extract_stats(samples)

        best_t, loglike = handler.optimize_duration(init_guess)
        param_stats['pe_best_duration'] = best_t
        param_stats['pe_best_duration_loglike'] = loglike
        handler.d_t = None  # Free memory
        handler.bkg_t = None  # Free memory

        earth_stat, sun_stat = handler.calc_localization_stats(samples[:, 0], samples[:, 1], inflate_sun=1.5, fwhm_deg=3.0)
        param_stats['pe_earth_stat'] = earth_stat
        param_stats['pe_sun_stat'] = sun_stat

        tot_flux, tot_flux_errp, tot_flux_errm = handler.calc_tot_flux_statistics(samples)
        param_stats['pe_tot_energy_flux'] = tot_flux
        param_stats['pe_tot_energy_flux_errp'] = tot_flux_errp
        param_stats['pe_tot_energy_flux_errm'] = tot_flux_errm

        param_stats['trigger_index'] = index
        param_stats['trigtime_'] = row['trigtime']

        buffer.append(param_stats)

        if isinstance(handler, FullResponseHandler):
            # drop big caches
            handler.rsp_cache.clear()
            handler.rsp_cache_counter.clear()
            del handler, burstdata, spec_model, samples
            gc.collect()

        if (count + 1) % save_every == 0 or count == len(df_chunk) - 1:
            combined_df = pd.concat(buffer, ignore_index=True)
            save_path = output_dir / f'pe_chunk_{chunk_id}_triggers_{idx}.csv'
            combined_df.to_csv(save_path, index=False)
            buffer.clear()

            with open(save_log_path, 'a') as f:
                f.write(f"{index}\n")

        current, peak = tracemalloc.get_traced_memory()
        with open(log_path, 'a') as f:
            f.write(f"Trigger {index}. Finished {count + 1}/{len(df_chunk)} triggers in chunk {chunk_id}. Took {time()-t0:.1f} sec. Max mem {peak / 10**6}MB\n")
