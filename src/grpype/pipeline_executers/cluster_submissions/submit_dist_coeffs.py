import argparse
import os
from pathlib import Path
from time import sleep

SUBMISSION = """\
#BSUB -J dist_{bdur_idx}_{par_idx}
#BSUB -n 1
#BSUB -oo {out_dir}/distout_{bdur_idx}_{par_idx}.txt
#BSUB -eo {out_dir}/disterr_{bdur_idx}_{par_idx}.txt
#BSUB -q {queue}
#BSUB -R 'rusage[mem=18500]'
#BSUB -R "select[model==AMD_EPYC&&type!=X86_GPU]"
#BSUB -R 'span[ptile=1]'

#RUN YOUR CODE
python ../../detection/distribution_coeffs.py {binning} {bdur_idx} {par_idx}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("npar", type=int)
    parser.add_argument("--queue", type=str, default="long")
    args = parser.parse_args()

    binnings = [0.01, 0.001]
    ranges = [range(-2, 16), range(8)]
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    for binning, idx_range in zip(binnings, ranges):
        for bdur_idx in idx_range:
            for par_idx in range(args.npar):
                script_name = "run_dist_coeffs.sh"
                with open(script_name, "w") as handle:
                    handle.write(
                        SUBMISSION.format(
                            bdur_idx=bdur_idx,
                            par_idx=par_idx,
                            out_dir=out_dir,
                            queue=args.queue,
                            binning=binning,
                        )
                    )
                sleep(0.01)
                os.system(f"bsub < {script_name}")
                os.system(f"rm {script_name}")


if __name__ == "__main__":
    main()
