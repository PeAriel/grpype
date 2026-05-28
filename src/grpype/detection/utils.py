from datetime import datetime
from hashlib import sha256
from itertools import product
from time import time

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.stats import norm
from scipy.signal import find_peaks

from gdt.missions.fermi.time import Time

from grpype.detection.global_params import detectors, echans, ndetectors, EPS


def gaussian(x, amp, mean, sigma):
    return amp/np.sqrt(2*np.pi*sigma**2) * np.exp(-(x - mean)**2 / (2 * sigma**2))

def t90_to_sigma(t90, resolution=1000):
    """
    Get the standard deviation of a gaussian distribution that has 90% of its area within t90
    :param t90: The time interval that contains 90% of the area under the gaussian
    :param resolution: The number of points to use in the calculation
    """
    sigma = np.linspace(1e-2, t90, resolution)

    dist = norm(0, sigma)
    equation = dist.cdf(t90) - dist.cdf(-t90) - 0.9

    return sigma[np.argmin(np.abs(equation))]

def t50_to_sigma(t50, resolution=1000):
    """
    Get the standard deviation of a gaussian distribution that has 50% of its area within t50
    :param t90: The time interval that contains 90% of the area under the gaussian
    :param resolution: The number of points to use in the calculation
    """
    sigma = np.linspace(1e-2, t50, resolution)

    dist = norm(0, sigma)
    equation = dist.cdf(t50) - dist.cdf(-t50) - 0.5

    return sigma[np.argmin(np.abs(equation))]


def generate_filtered_combinations(alpha, beta):
    """
    Generate filtered combinations of 'alpha' and 'beta' where 'alpha' is always larger than 'beta'.

    Parameters:
        alpha (ndarray): 1-dimensional NumPy array representing the values of Band function parameter 'alpha'.
        beta (ndarray): 1-dimensional NumPy array representing the values of Band function parameter 'beta'.

    Returns:
        ndarray: 2-dimensional NumPy array containing a list of the filtered combinations.
    """
    combinations = np.array(list(product(alpha, beta)))

    mask = combinations[:, 0] > combinations[:, 1]
    filtered_combinations = combinations[mask]

    return filtered_combinations

def exp2_tail(x, a, b, c):
    return a * np.exp(-b * x - c * x**2)

def exp_tail(x, a, b, c):
    return a * np.exp(-b * x)

def sf(fitting_func, parmas, x0, mx=20, res=int(1e4)):
    x = np.linspace(x0, mx, res)
    dx = x[1] - x[0]

    return np.sum(fitting_func(x, *parmas), axis=0) * dx

def calibrate_from_norm(fitting_func, params, x0, mx=20, res=int(1e4)):
    """
    Find the threshold of the fitting function that gives the same survival function as the normal distribution at the given x0.
    """
    r = np.linspace(x0-5, x0+5, res)
    x = np.linspace(r, mx, res)
    dx = x[1, 0] - x[0, 0]

    vals = np.sum(fitting_func(x, *params), axis=0) * dx
    vals = np.log(vals)

    normsf = np.log(norm(0, 1).sf(x0))

    idx = np.argmin(np.abs(vals - normsf))
    
    return r[idx]


def generate_seed(date_time, burst_duration, base_seed=42):
    if type(date_time) == str:
        date_time = datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S.%f")
    else:
        date_time = datetime(date_time.year, date_time.month, date_time.day, date_time.hour)

    date_str = date_time.strftime("%Y-%m-%d %H")
    combined_str = f"{date_str}{burst_duration}"

    date_hash = int(sha256(combined_str.encode()).hexdigest(), 16)
    seed = (base_seed + date_hash) % (2**32)

    return seed


def calc_mf(d, bkg, templates):
    if templates.ndim == 1:
        templates = templates[None, :]

    fltr = np.log1p(templates / (bkg + EPS))
    mf_numer = np.dot(fltr, d - bkg)
    mf_var = np.dot(fltr**2, bkg)

    mf = mf_numer / np.sqrt(mf_var + EPS)

    return mf


def calc_best_amp(d, bkg, bf_template):
    if bf_template.ndim == 1:
        bf_template = bf_template[None, :]

    fltr = np.log1p(bf_template / (bkg + EPS))
    amp = fltr.dot(d.T - bkg.T) / fltr.dot(bf_template.T)

    return amp.flatten()


def calc_glitch_statistic(d, bkg, templates, glitches, return_best=False):
    basemf = calc_mf(d, bkg, templates)
    bestmf_ind = np.argmax(basemf)
    best_template = templates[bestmf_ind]
    amp = calc_best_amp(d, bkg, best_template)

    fltr = np.log1p(glitches / (bkg + EPS)) - np.log1p(amp * best_template / (bkg + EPS))

    mf_numer = np.sum((d - amp * best_template - bkg) * fltr)
    mf_var = np.sum((amp * best_template + bkg) * fltr**2)
    mf = mf_numer / np.sqrt(mf_var + EPS)

    if return_best:
        return mf, basemf[bestmf_ind], bestmf_ind

    return mf


def simple_mf(data, template, bkg, amp):
    logterm = np.log(1 + amp * template / bkg)
    return np.sum(data * logterm, axis=1)


def find_threshold(template, bkg, nsamples, amp_init=1, bins=500, niter=20, prec=1e-3, verbose=False):
    t0 = time()

    datalen = len(template)

    amp = amp_init
    amps = []

    counter = 0
    while counter <= niter:
        dataGnull = stats.poisson(bkg).rvs(size=[nsamples, datalen]) - bkg
        dataGburst = stats.poisson(bkg + amp * template).rvs(size=[nsamples, datalen]) - bkg

        testGnull = simple_mf(dataGnull, template, bkg, amp)
        testGburst = simple_mf(dataGburst, template, bkg, amp)

        null_counts, null_edges = np.histogram(testGnull, bins=bins, density=True)
        burst_counts, burst_edges = np.histogram(testGburst, bins=bins, density=True)

        null_cdf = np.cumsum(null_counts * np.diff(null_edges))
        burst_cdf = np.cumsum(burst_counts * np.diff(burst_edges))

        thresh = np.searchsorted(null_cdf, 1 - 1e-6)
        burst_center = np.searchsorted(burst_cdf, 0.5)

        prob2flux = np.sum(template * np.log(1 + amp * template / bkg))
        thresh_amp = null_edges[thresh] / prob2flux
        burst_center_amp = burst_edges[burst_center] / prob2flux

        cond = np.abs(burst_center_amp - thresh_amp) / thresh_amp
        if cond < prec:
            amps.append(thresh_amp)
            if len(amps) == 10:
                break

        counter += 1
        amp = thresh_amp

    if len(amps) == 0:
        if verbose:
            print(f"Solution was not found with {niter} iterations and {prec} precision")
        prec = prec * 10
        if verbose:
            print(f"Lowering precision to {prec}")
        return find_threshold(template, bkg, nsamples, amp_init, bins=500, niter=100, prec=prec)

    if verbose:
        print(f"finished in: {time() - t0:.2f}s")

    return np.mean(amps)

def fibonacci_sphere(samples):
    """
    Evenly spaces points on a unit sphere using the Fibonacci algorithm.
    """
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians
    
    i = np.arange(samples)

    y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
    radius = np.sqrt(1 - y * y)  # radius at y

    theta = phi * i  # golden angle increment

    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    return x, y, z

def xyz2thetahpi(x, y, z, azel=False):
    theta = np.arccos(z)
    phi = np.arctan2(y, x)
    if azel:
        return np.degrees(np.pi/2-theta), np.degrees(phi)
    return theta, phi

def ang2cart(theta, phi):
    """
    Convert angles to a unit vector
    """
    sf = np.arctan(1.0) / 45.0
    plat = theta * sf
    plon = phi * sf
    x, y, z = np.array([np.cos(plat)*np.cos(plon), np.cos(plat)*np.sin(plon), np.sin(plat)])

    return x, y, z

def time_string_to_met(time_string, format_string='%Y-%m-%d %H:%M:%S.%f'):
    return Time(datetime.strptime(time_string, format_string), scale='utc')

def load_templates(path, nrandom):
    templates = np.zeros([len(detectors), echans, nrandom])
    for d, det in enumerate(detectors):
        templates[d, :, :] = np.load(path.format(det, nrandom))
    return templates

def inner(a, b, sigma):
    if a.ndim == 1:
        return np.sum(a * b / sigma**2)
    return np.sum(a * b / sigma**2, axis=1)

def match(a, b, sigma):
    return inner(a, b, sigma) / np.sqrt(inner(a, a, sigma) * inner(b, b, sigma))

def loginner(a, b, sigma):
    a = np.array(a)
    b = np.array(b)
    if a.ndim == 1:
        return np.sum(a * np.log(1 + b/sigma))
    return np.sum(a * np.log(1 + b/sigma), axis=1)

def logmatch(a, b, sigma):
    eps = 1e-10
    a = np.array(a)
    b = np.array(b)
    inform_ab = loginner(a, b, sigma) / loginner(sigma*np.log(1 + b/(sigma + eps)), b, sigma)**(1/2)
    inform_aa = loginner(a, a, sigma) / loginner(sigma*np.log(1 + a/(sigma + eps)), a, sigma)**(1/2)
    return inform_ab / inform_aa

def spectrum2square(spectrum, width_seconds, time_bins):
    """
    Transform the spectral template into a square pulse at each energy channel.
    Probably gaussian is a good choice.
    :param spectrum (array like): counts vs energy channel array
    :param width_t90 (float): duration of the pulse in terms of T90 (seconds)
    :param time_bins (array like): the width of the time bins in seconds
    """
    width_samples = int(width_seconds/time_bins)
    square = np.ones(width_samples)

    if spectrum.ndim == 1:
        return np.outer(square, spectrum)
    
    pulse = np.zeros([spectrum.shape[0], width_samples, spectrum.shape[1]])
    for i in range(spectrum.shape[0]):
        pulse[i, :, :] = np.outer(square, spectrum[i, :])

    return pulse

def spectrum2pulse(spectrum, width_t90, time_bins, time_length_seconds=None):
    """
    Transform the spectral template into a pulse at each energy channel. 
    Probably gaussian is a good choice.
    :param spectrum (array like): counts vs energy channel array
    :param width_t90 (float): duration of the pulse in terms of T90 (seconds)
    :param time_bins (array like): the width of the time bins in seconds
    :param time_length_seconds (float): length of the pulse (seconds)
    """
    spec_len = len(spectrum)
    sigma_seconds = t90_to_sigma(width_t90)
    
    assert sigma_seconds > time_bins, "sigma must be larger than the time bins in order to detect it"

    sigma_samples = int(sigma_seconds/time_bins)
    if time_length_seconds is None:
        time_length_samples = 10*sigma_samples  # Make the signal 10 times sigma long 
    if time_length_seconds is not None:
        time_length_samples = int(time_length_seconds/time_bins)
    
    pulse = np.zeros([time_length_samples, spec_len])
    for t in range(spec_len):
        x = np.arange(time_length_samples)
        pulse[:, t] = gaussian(x, spectrum[t], time_length_samples//2, sigma_samples)

    return pulse

def square_convolve(data, window, fltr, split=False, channels=echans, numdetectors=ndetectors, jump=1):
    """
    Convolve data with a square window of size window and filter fltr. Uses the cumsum trick which is O(n) instead of O(nlogn).
    params:
        data: 2d array - first axis is time, second is energy
        window: int
        fltr: 1d array - energy filter
        split: wheater to return the individual components of the variances (to get the combined values just
               sum over axis=0).
        jump: int - the number of samples to jump in the cumsum trick, to account for the overlap in the convolution
        
    returns:
        mf: 1d array - matched filter at each time
    """
    fltrdim = fltr.ndim
    cdata = np.concatenate(([np.zeros(data.shape[1])], data), axis=0)
    cs = np.cumsum(cdata, axis=0)

    if fltrdim == 3:
        cswin = cs[window:, :] - cs[:-window, :]
        mf = np.einsum('ij,klj->ikl', cswin, fltr)
        return mf

    if fltrdim == 1:
        fltr = fltr[np.newaxis, :]

    if split:
        individs = np.zeros([numdetectors, cs.shape[0]-window, fltr.shape[0]], dtype=np.float32)
        for i in range(numdetectors):
            subt = cs[window::jump, channels*i:channels*(i+1)] - cs[:-window:jump, channels*i:channels*(i+1)]
            fl = fltr[:, channels*i:channels*(i+1)]
            individs[i, ::jump] = (fl @ subt.T).T

            # individs[i, ::jump] = (cs[window::jump, channels*i:channels*(i+1)] - cs[:-window:jump, channels*i:channels*(i+1)]).dot(fltr[:, channels*i:channels*(i+1)].T)

        return individs
    
    mf = np.zeros([cs.shape[0]-window, fltr.shape[0]], dtype=np.float32)
    subt = cs[window::jump, :] - cs[:-window:jump, :]
    mf[::jump] = (fltr @ subt.T).T

    # mf[::jump] = (cs[window::jump, :] - cs[:-window:jump, :]).dot(fltr.T)

    if fltrdim == 1:
        mf = mf[:, 0]
    
    return mf

def rolling_mean(data, window):
    if data.ndim == 1:
        cs = np.cumsum(data)
    else:
        cs = np.cumsum(data, axis=0)
    
    mn = (cs[window:] - cs[:-window]) / window

    return mn


def rolling_mean_padded(data, window, gap=0):
    """
    Calculates the rolling mean of the data with a window and a gap. The gap is a region in the middle of the window
    that is not used to calculate the mean. This is useful to avoid using the burst in the background calculation.
    This method also pads the data with the mean of the first and last window//4 samples to avoid edge effects
    and keep the same length of the data.

    params:
    -------
    data (np.ndarray): the data to calculate the rolling mean
    window (int): the window size in samples
    gap (int): the gap size in samples
    """
    if gap != 0:
        window = 2 * window + gap
    if window % 2 == 1:
        window += 1

    spadidx = window // 4
    fpadidx = window // 2
    half = window // 2
    if data.ndim == 1:
        mean_first = np.mean(data[spadidx:fpadidx])
        mean_last = np.mean(data[-spadidx:])
        padded_data = np.pad(data, (half, half), mode="constant", constant_values=(mean_first, mean_last))
        cs = np.cumsum(padded_data)
    else:
        mean_first = np.mean(data[spadidx:fpadidx], axis=0).astype(np.float32)
        mean_last = np.mean(data[-spadidx:], axis=0).astype(np.float32)
        padded_data = np.pad(data.astype(np.float32), ((half, half), (0, 0)), mode="constant")
        if half > 0:
            padded_data[:half] = mean_first
            padded_data[-half:] = mean_last
        cs = np.cumsum(padded_data, axis=0)

    if gap > 0:
        mn = (cs[window:] - cs[:-window])
        mn_sub = (cs[gap:] - cs[:-gap])
        mn -= mn_sub[(window - gap) // 2 : -(window - gap) // 2]
        mn = mn / (window - gap)
    else:
        mn = (cs[window:] - cs[:-window]) / window

    return mn


def quad_rolling_mean_padded(data, window, gap=0, clipneg=False):
    w1 = rolling_mean_padded(data, window, 3 * gap)
    w2 = rolling_mean_padded(data, window, gap)

    f = (4 * w2 - w1) / 3
    if clipneg:
        f = np.clip(f, 0, np.inf)

    return f

def rolling_safe_mean(data, window, sigma=4):
    if window % 2 == 1:
        window += 1

    spadidx = window//4
    fpadidx = window//2
    half = window // 2
    if data.ndim == 1:
        mean_first = np.mean(data[spadidx:fpadidx])
        mean_last = np.mean(data[-spadidx:])
        data = np.pad(data, (half, half), mode='constant', constant_values=(mean_first, mean_last))
    else:
        mean_first = np.mean(data[spadidx:fpadidx], axis=0)
        mean_last = np.mean(data[-spadidx:], axis=0)
        data = np.pad(data, ((half, half), (0, 0)), mode='constant')
        if half > 0:
            data[:half] = mean_first
            data[-half:] = mean_last

    roll_mn = rolling_mean(data, window)
    roll_mn_sq = rolling_mean(data**2, window)
    roll_std = np.sqrt(roll_mn_sq - roll_mn**2)
    
    out = np.clip(roll_mn, roll_mn - sigma*roll_std, roll_mn + sigma*roll_std)

    return out

def rolling_double_mean(data, window, delta=0):
    if delta != 0:
        window = 2*window + delta
    if window % 2 == 1:
        window += 1
    
    spadidx = window//4
    fpadidx = window//2
    half = window // 2
    if data.ndim == 1:
        mean_first = np.mean(data[spadidx:fpadidx])
        mean_last = np.mean(data[-spadidx:])
        data = np.pad(data, (half, half), mode='constant', constant_values=(mean_first, mean_last))
        cs = np.cumsum(data)
    else:
        mean_first = np.mean(data[spadidx:fpadidx], axis=0)
        mean_last = np.mean(data[-spadidx:], axis=0)
        data = np.pad(data, ((half, half), (0, 0)), mode='constant')
        if half > 0:
            data[:half] = mean_first
            data[-half:] = mean_last
        cs = np.cumsum(data, axis=0)
    
    if delta > 0:
        mn = (cs[window:] - cs[:-window])
        mn_sub = (cs[delta:] - cs[:-delta])
        mn -= mn_sub[(window-delta)//2:-(window-delta)//2]
        mn = mn / (window-delta)
    else:
        mn = (cs[window:] - cs[:-window]) / window

    return mn


def psd_drift_svd(bank, interval, burstdata, bkg_window, bkg_window_gap, trigtime, nsing):
    iconv = interval[0] + burstdata.burst_duration_samp // 2 - (burstdata.burst_duration_samp + 1) % 2
    tleft = trigtime + iconv - burstdata.burst_duration_samp // 2 + (burstdata.burst_duration_samp + 1) % 2
    tright = trigtime + iconv + burstdata.burst_duration_samp // 2 + 1

    fltr = bank.templates
    fltr = np.log1p(bank.templates / (burstdata.fltr_bkgs[-1] + EPS), out=fltr)
    var = np.dot(fltr**2, burstdata.bkgs[tleft:tright].sum(axis=0))
    f = fltr / (np.sqrt(var) + EPS)[:, None]
    del fltr, var
    subset = np.random.choice(bank.ntemplates, size=50 * nsing, replace=False)
    u, s, vt = np.linalg.svd(f[subset], full_matrices=False)
    del u

    x = np.cumsum(s[::-1] ** 2)
    x = x[::-1] / x[-1]
    th = 1e-3
    _ = th

    vt = vt[:nsing]

    gauge = np.sign(vt[:, 0])
    vt *= gauge[:, None]

    coeffs = f @ vt.T
    del f

    dtwid = burstdata.data[interval[0] : interval[1]] - burstdata.bkgs[interval[0] : interval[1]]
    svdmf = square_convolve(dtwid, burstdata.burst_duration_samp, vt)
    del vt
    vmf_tavg = rolling_mean_padded(svdmf, bkg_window, gap=bkg_window_gap)[trigtime]
    drift1 = np.dot(coeffs, vmf_tavg)
    del vmf_tavg

    vmf2_tavg = np.zeros([nsing, nsing])
    for k in range(nsing):
        for j in range(k, nsing):
            vmf2_kj = svdmf[:, k] * svdmf[:, j]
            vmf2_tavg[k, j] = rolling_mean_padded(vmf2_kj, bkg_window, gap=bkg_window_gap)[trigtime]
    del svdmf

    vmf2_tavg = vmf2_tavg + vmf2_tavg.T - np.diag(vmf2_tavg.diagonal())

    drift2 = np.zeros_like(drift1)
    slc = 100
    for i in range(0, bank.ntemplates, slc):
        drift2[i : i + slc] = np.einsum("ij,jk,ik->i", coeffs[i : i + slc], vmf2_tavg, coeffs[i : i + slc])

    drift2 -= drift1**2
    return drift1, drift2


def interp_posterior(tgrid, burstdata, trig_met, posterior, nside=128):
    from grpype.data_io.data_handlers import DataLoaders
    import healpy as hp
    import pandas as pd
    from scipy.interpolate import griddata

    if trig_met >= burstdata.poshist.time_range[0] and trig_met <= burstdata.poshist.time_range[1]:
        poshist = burstdata.poshist
    elif trig_met > burstdata.poshist.time_range[1]:
        loader = DataLoaders()
        date = Time(trig_met, format='fermi', scale='utc').utc.datetime + pd.Timedelta(minutes=5)
        poshist = loader.open_poshist_by_date(date)
    elif trig_met < burstdata.poshist.time_range[0]:
        loader = DataLoaders()
        date = Time(trig_met, format='fermi', scale='utc').utc.datetime - pd.Timedelta(minutes=5)
        poshist = loader.open_poshist_by_date(date)

    ra, dec = poshist.to_equatorial(tgrid.phis, tgrid.thetas, trig_met)

    pix_indices = hp.ang2pix(nside, np.radians(90 - dec), np.radians(ra))

    n_pixels = hp.nside2npix(nside)
    probmap = np.zeros(n_pixels)

    probmap = np.full(n_pixels, np.nan)
    for pix, prob in zip(pix_indices, posterior):
        probmap[pix] = prob

    theta, phi = hp.pix2ang(nside, np.arange(n_pixels))
    ra_pix = np.degrees(phi)
    dec_pix = 90 - np.degrees(theta)

    known_points = np.column_stack((ra, dec))
    pixel_centers = np.column_stack((ra_pix, dec_pix))

    interpolated_probabilities = griddata(known_points, posterior, pixel_centers, method="nearest")

    probmap[np.isnan(probmap)] = interpolated_probabilities[np.isnan(probmap)]

    probmap /= np.sum(probmap)

    return probmap, ra_pix, dec_pix


def fit_bkg(bkg, degs=[2, 5]):
    if bkg.ndim == 1:
        bkg = bkg[None, :]

    fit_chans_nai = [[45, 63], [75, 95], [110, 123]]
    fit_chans_bgo = [[60, 120]]

    new_bkg = bkg.copy()

    for j in range(ndetectors):
        fit_chans = fit_chans_nai if j < 12 else fit_chans_bgo
        deg = degs[0] if j < 12 else degs[1]

        for a, b in fit_chans:
            x = np.arange(a, b)
            y = bkg[:, echans * j + a : echans * j + b]

            coeffs = np.polyfit(x, y.T, deg)

            new_bkg[:, echans * j + a : echans * j + b] = np.polyval(coeffs, x[:, None]).T

    return new_bkg

def find_peaks_2d(mf_mat, min_peak_dist, min_peak_height, verbose=False):
    """
    Given a 2d matched filter matrix, find the peaks in each template and return the time and template indices.
    It allows for a minimum peak distance and height to be specified, and the distances are accounted for, cross template.

    To algorithm works as follows:
    - Find all the unique times and save the indices and peak values.
    - Find the maximal value for each unique time and save it with its time index.
    - Create a zeros array and fill it with the maximal values at the time indices.
    - Use find peaks again.
    - The template index is found by the argmax along the template dimension of the match filter at the time index of the peak.

    params:
        mf_mat: 2d array - first axis is time, second is template
        min_peak_dist: int - minimum distance between peaks in samples
        min_peak_height: float - minimum peak height
        verbose: bool - whether to print progress

    returns:
        maxtimes: 1d array - times of peaks
        maxtemps: 1d array - template indices of peaks
    """
    peak_loc = np.array([], dtype=int)
    peak_val = np.array([])

    for i in range(mf_mat.shape[1]):
        pks = find_peaks(mf_mat[:, i], height=min_peak_height, distance=min_peak_dist)
        peak_loc = np.concatenate([peak_loc, pks[0]])
        peak_val = np.concatenate([peak_val, pks[1]['peak_heights']])

        if verbose:
            print(f'finished finding peaks in template: {i+1}/{mf_mat.shape[1]}', end='\r')

    uniques = np.unique(peak_loc)
    maxmap = np.zeros([len(uniques), 2])
    for i, unique in enumerate(uniques):
        maxmap[i] = [unique, np.max(peak_val[peak_loc == unique])]

    spaced_peaks = np.zeros(len(mf_mat))
    spaced_peaks[maxmap[:, 0].astype(int)] = maxmap[:, 1]
    pkspks = find_peaks(spaced_peaks, height=min_peak_height, distance=min_peak_dist)

    maxtemps = np.argmax(mf_mat[pkspks[0]], axis=1)
    maxtimes = pkspks[0]

    return maxtimes, maxtemps


def main():
    pass

if __name__ == '__main__':
    main()
