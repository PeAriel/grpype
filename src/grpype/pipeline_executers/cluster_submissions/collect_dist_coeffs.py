import argparse

import numpy as np

from grpype.detection.global_params import DATAPATH

arrays_dir = DATAPATH / "dist_coeffs" / "arrays"
nbins = 200
bin_edges = np.linspace(-6, 9, nbins)


def burst_durations():
    fine = np.round(0.002 * 1.35 ** np.arange(8), 3)
    coarse = np.round(0.04 * 1.35 ** np.arange(-2, 20), 3)
    return np.concatenate([fine, coarse])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("npar", type=int)
    return parser.parse_args()


def main():
    args = parse_args()

    for burst_duration in burst_durations():
        counts_dir = arrays_dir / f"counts{burst_duration:.3f}"
        counts = np.zeros_like(np.load(counts_dir / "0.npy"))
        for par_idx in range(args.npar):
            counts += np.load(counts_dir / f"{par_idx}.npy")

        counts_save = counts / np.diff(bin_edges) / np.sum(counts, axis=1)[:, None]
        np.save(arrays_dir / f"counts{burst_duration:.3f}.npy", counts_save)

        print(f"Finished {burst_duration:.3f}")


if __name__ == "__main__":
    main()
