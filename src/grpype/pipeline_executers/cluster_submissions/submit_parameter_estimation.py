from __future__ import annotations

import argparse
import os
from pathlib import Path
from time import sleep

from grpype.detection.global_params import DATAPATH

SUBMISSION = """\
#BSUB -J {trigtype}pe_{year}_{idx}
#BSUB -n 1
#BSUB -oo {logdir_abs}/{trigtype}pe_out{idx}.txt
#BSUB -eo {logdir_abs}/{trigtype}pe_err{idx}.txt
#BSUB -q {queue}
#BSUB -R 'rusage[mem={mem}]'
#BSUB -R "select[model==AMD_EPYC&&type!=X86_GPU]"
#BSUB -R 'span[ptile=1]'

#RUN YOUR CODE
cd {run_dir_abs}
python ../run_parameter_estimation.py {trigtype} --year {year} --n_jobs {n_jobs} --execution cluster --chunk-index {idx}{output_root_arg}
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trigtype", choices=["p", "ts"], type=str)
    parser.add_argument("--year", type=int, default=None)
    parser.add_argument("--n_jobs", type=int, default=96)
    parser.add_argument("--queue", type=str, default="short")
    parser.add_argument("--mem", type=int, default=12500, help="Memory in MB (LSF rusage[mem=...])")
    parser.add_argument("--output-root", type=str, default=None)
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent
    year = str(args.year) if args.year is not None else "all"
    suffix = "_timeslides" if args.trigtype == "ts" else ""
    output_root = Path(args.output_root) if args.output_root is not None else DATAPATH
    output_dir = output_root / f"pe_results{suffix}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logdir = run_dir / "outputs" / year
    logdir.mkdir(parents=True, exist_ok=True)
    run_dir_abs = str(run_dir)
    logdir_abs = str(logdir)

    for idx in range(args.n_jobs):
        script_name = f"run_pe_{args.trigtype}_{year}_{idx}.sh"
        script_path = run_dir / script_name
        with open(script_path, "w") as handle:
            output_root_arg = f" --output-root {output_root}" if args.output_root is not None else ""
            handle.write(
                SUBMISSION.format(
                    trigtype=args.trigtype,
                    year=year,
                    idx=idx,
                    queue=args.queue,
                    mem=args.mem,
                    n_jobs=args.n_jobs,
                    output_root_arg=output_root_arg,
                    run_dir_abs=run_dir_abs,
                    logdir_abs=logdir_abs,
                )
            )
        sleep(0.01)
        os.system(f"bsub < {script_path}")
        script_path.unlink()


if __name__ == "__main__":
    main()
