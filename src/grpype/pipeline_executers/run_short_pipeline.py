from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from time import time

import numpy as np

from grpype.detection.global_params import DATAPATH
from grpype.detection.pipeline import Detection
from grpype.detection.templates import GlitchTemplates, TemplateBank
from grpype.data_io.data_handlers import TTEData
from grpype.detection.utils import generate_seed
from grpype.pipeline_executers.exec_utils import already_done


def run_short_pipeline(date, delta, timeslides=None, simulate=False, save=False, jobname="", tte_npar=1):
    for j in range(delta):
        binning = 0.01
        tbank = TemplateBank(binning)
        tbank.load_templates()
        glitches = GlitchTemplates(binning, 3)
        burst_duration = 0.04
        rolling_window_sec = 40.0
        rolling_gap_sec = burst_duration * 3
        mf_threshold = 6.3
        glitch_threshold = 6.3
        slice_seconds = 120
        min_dist_sec = 30
        overlap = 1
        glitch_extend = 2

        curdate = date + timedelta(hours=j)

        try:
            burstdata = TTEData(
                curdate,
                binning,
                burst_duration,
                timeslides=timeslides,
                simulate=simulate,
                cut_len_seconds=10,
                save_obj=False,
                npar=tte_npar,
            )
        except OSError:
            res_str = f"results/{curdate.year}/corrupted.txt" if timeslides is None else f"results_timeslides/{curdate.year}/corrupted.txt"
            res_str = f"results_simul/{curdate.year}/corrupted.txt" if simulate else res_str
            with open(DATAPATH / res_str, "a") as f:
                f.write(f"corrupted date {curdate}\n")
            continue

        t0 = time()

        detection = Detection(binning, burst_duration, rolling_window_sec, rolling_gap_sec)
        detection.just_bkg = True

        detection.match_filter(
            burstdata,
            tbank,
            glitches,
            slice_seconds,
            mf_threshold,
            min_dist_sec,
            glitch_threshold,
            glitch_extend,
            overlap,
        )

        res_str = f"results/{curdate.year}/finished_{jobname}.txt" if timeslides is None else f"results_timeslides/{curdate.year}/finished_{jobname}.txt"
        res_str = f"results_simul/{curdate.year}/finished_{jobname}.txt" if simulate else res_str
        used_data = burstdata.total_time_used / 60

        tot_time = time() - t0

        del detection

        bkg = burstdata.bkgs.copy()
        old_binning = binning
        old_time = burstdata.time.copy()
        del burstdata

        binning = 0.001
        tbank = TemplateBank(binning)
        tbank.load_templates()
        glitches = GlitchTemplates(binning, 3)
        nbursts = 6
        burst_durations = np.around(0.002 * 1.35 ** np.arange(nbursts), 3)
        rolling_gap_sec = 0.004 * 3
        rolling_window_sec = 10.0
        slice_seconds = 15
        mf_threshold = 6.3
        glitch_threshold = 6.3

        seed = generate_seed(curdate, burst_durations[0])
        np.random.seed(seed)

        slc = np.linspace(0, 1, 20)
        d = slc[1] - slc[0]
        for si, s in enumerate(slc[:-1]):
            s = [round(s, 3), round(s + d, 3)]
            if s[0] > 0:
                s[0] -= 0.005
            try:
                burstdata = TTEData(
                    curdate,
                    binning,
                    burst_durations[0],
                    timeslides=timeslides,
                    simulate=simulate,
                    cut_len_seconds=0,
                    save_obj=False,
                    slice_time=s,
                    npar=tte_npar,
                )
                burstdata.interp_bkg(bkg, old_binning, old_time)
            except OSError:
                res_str = f"results/{curdate.year}/corrupted.txt" if timeslides is None else f"results_timeslides/{curdate.year}/corrupted.txt"
                res_str = f"results_simul/{curdate.year}/corrupted.txt" if simulate else res_str
                with open(DATAPATH / res_str, "a") as f:
                    f.write(f"corrupted date {curdate}\n")
                continue
            except ValueError:
                continue
            except IndexError:
                continue

            for i in range(nbursts):
                if already_done(curdate, burst_durations[i], res_str, slice_ind=si):
                    continue

                t0_short = time()
                seed = generate_seed(curdate, burst_durations[i])
                np.random.seed(seed)

                if simulate:
                    burstdata.interp_bkg(bkg, old_binning, old_time)

                detection = Detection(binning, burst_durations[i], rolling_window_sec, rolling_gap_sec)
                mf, maxtimes, maxtemps, triggers_met = detection.match_filter(
                    burstdata,
                    tbank,
                    glitches,
                    slice_seconds,
                    mf_threshold,
                    min_dist_sec,
                    glitch_threshold,
                    glitch_extend,
                    overlap,
                )

                used_data_short = burstdata.total_time_used / 60
                tot_time_short = time() - t0_short
                print(
                    f"finished date {curdate}, bdur: {burst_durations[i]}, slice {si}. Took: {tot_time_short:.2f} sec, triggers: {len(maxtimes)}, used data {used_data_short:.2f} min"
                )
                print(f"mf: {mf[maxtimes, maxtemps]}\n")
                if save:
                    detection.save_triggers(
                        curdate,
                        burstdata.timeslides_minutes,
                        simulate,
                        clean=True,
                        filename=f"triggers_{jobname}",
                        slc_ind=si,
                    )
                    with open(DATAPATH / res_str, "a") as f:
                        f.write(
                            f"finished date {curdate}, bdur: {burst_durations[i]}, slice {si}. Took: {tot_time_short:.2f} sec, triggers: {len(maxtimes)}, used data {used_data_short:.2f} min\n"
                        )

                del detection
            del burstdata
        del bkg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("year", type=int)
    parser.add_argument("month", type=int)
    parser.add_argument("day", type=int)
    parser.add_argument("hour", type=int)
    parser.add_argument("delta_hours", type=int)
    parser.add_argument("--jobname", type=str, default="")
    parser.add_argument("--timeslides", type=float, default=None)
    parser.add_argument("--tte-npar", type=int, default=1)
    parser.add_argument("--simulate", type=bool, default=False)
    parser.add_argument("--save", type=bool, default=False)
    args = parser.parse_args()

    date = datetime(args.year, args.month, args.day, args.hour)
    run_short_pipeline(
        date,
        args.delta_hours,
        args.timeslides,
        args.simulate,
        save=args.save,
        jobname=args.jobname,
        tte_npar=args.tte_npar,
    )


if __name__ == "__main__":
    main()
