import argparse
from datetime import datetime
from time import time

import numpy as np
from scipy.optimize import curve_fit, fsolve
from scipy.stats import norm, poisson

from grpype.detection.global_params import DATAPATH, EPS
from grpype.detection.pipeline import Detection, TTEData
from grpype.detection.templates import GlitchTemplates, TemplateBank
from grpype.detection.utils import sf, square_convolve

reference_date = datetime(2014, 1, 1, 0)
match_filter_threshold = 10
slice_seconds = 0.5 * 60
rolling_window_sec = 10
rolling_gap_multiplier = 3
background_burst_duration = 0.04

dist_coeffs_dir = DATAPATH / "dist_coeffs"
arrays_dir = dist_coeffs_dir / "arrays"
nbins = 200
bin_edges = np.linspace(-6, 9, nbins)
default_averages = 10_000


def get_typical_background(date, binning):
    template_bank = TemplateBank(binning)
    template_bank.load_templates()

    glitches = GlitchTemplates(binning, 3)
    burst_data = TTEData(
        date,
        binning,
        background_burst_duration,
        timeslides=None,
        simulate=False,
        slice_time=[0.0, 0.1],
        cut_len_seconds=5,
    )
    detection = Detection(
        binning,
        background_burst_duration,
        rolling_window_sec=rolling_window_sec,
        rolling_gap_sec=rolling_gap_multiplier * background_burst_duration,
    )
    detection.match_filter(
        burst_data,
        template_bank,
        glitches,
        slice_seconds=slice_seconds,
        mf_threshold=match_filter_threshold,
        min_dist_sec=60,
        glitch_threshold=match_filter_threshold,
    )

    return burst_data.fltr_bkgs[1]


def calculate_distribution(binning, burst_duration, background, n_random=int(1e5)):
    simdata = poisson(background[None, :]).rvs(size=[n_random, background.size])

    template_bank = TemplateBank(binning, alltemplates=False)
    template_bank.load_templates()

    fltr = np.log(1 + template_bank.templates / (background + EPS))
    window = int(burst_duration // binning)
    mf_numer = square_convolve(simdata - background, window, fltr)
    mf_var = square_convolve(np.ones_like(simdata) * background, window, fltr**2)

    return mf_numer / np.sqrt(mf_var + EPS)


def calculate_coefficients(fitting_func, mf, burst_duration, save=False):
    counts, bins = np.histogram(mf.flatten(), bins=100, density=True)
    popt = fit_tail(fitting_func, bins, counts, 3)

    if save:
        save_path = dist_coeffs_dir / f"coefficients{burst_duration:.3f}.npy"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        # np.save(save_path, popt)

    return popt


def calibrate_dist(val, fitting_func, params):
    return fsolve(
        lambda x: np.log(norm.sf(x)) - np.log(sf(fitting_func, params, val, res=int(1e6))),
        val,
    )


def fit_tail(fitting_func, bins, counts, tail_start, p0=(1, 0.001, 0.001)):
    centers = (bins[:-1] + bins[1:]) / 2
    tail_mask = centers > tail_start
    bin_centers_tail = centers[tail_mask]
    counts_tail = counts[tail_mask]

    popt, _pcov = curve_fit(fitting_func, bin_centers_tail, counts_tail, p0=p0)

    return popt


def burst_duration_for_index(binning, index):
    if np.isclose(binning, 0.01):
        return round(0.04 * 1.35**index, 3)
    if np.isclose(binning, 0.001):
        return round(0.002 * 1.35**index, 3)
    raise ValueError(f"Unsupported binning: {binning}")


def counts_save_path(burst_duration, par_idx):
    return arrays_dir / f"counts{burst_duration:.3f}/{par_idx}.npy"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("binning", type=float)
    parser.add_argument("bdur_ind", type=int)
    parser.add_argument("par_idx", type=int)
    parser.add_argument("--navg", type=int, default=default_averages)
    return parser.parse_args()


def main():
    args = parse_args()

    burst_duration = burst_duration_for_index(args.binning, args.bdur_ind)
    save_arr_path = counts_save_path(burst_duration, args.par_idx)
    save_arr_path.parent.mkdir(parents=True, exist_ok=True)

    template_bank = TemplateBank(args.binning)
    template_bank.load_templates()
    background = get_typical_background(reference_date, args.binning)

    if save_arr_path.exists():
        counts = np.load(save_arr_path)
    else:
        counts = np.zeros([template_bank.ntemplates, nbins - 1])

    for iteration in range(args.navg):
        start = time()
        mf = calculate_distribution(args.binning, burst_duration, background)
        for template_index in range(template_bank.ntemplates):
            counts_, _ = np.histogram(mf[:, template_index], bins=bin_edges)
            counts[template_index] += counts_

        print(f"{iteration + 1}/{args.navg} took {time() - start:.2f} sec")

        counts_save = counts / np.diff(bin_edges) / np.sum(counts, axis=1)[:, None]
        if iteration % 100 == 0:
            np.save(save_arr_path, counts_save)


if __name__ == "__main__":
    main()

