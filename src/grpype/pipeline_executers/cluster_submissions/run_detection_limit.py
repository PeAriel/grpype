import argparse
import os
from pathlib import Path
from time import sleep

import numpy as np

submission = """\
#BSUB -J detlim{idx}
#BSUB -n 1
#BSUB -oo {out_dir}/out{idx}.txt
#BSUB -eo {out_dir}/err{idx}.txt
#BSUB -q {queue}
#BSUB -R 'rusage[mem=19000]'
#BSUB -R "select[model==AMD_EPYC&&type!=X86_GPU]"
#BSUB -R 'span[ptile=1]'

#RUN YOUR CODE
python ../../templates/detection_limit.py {binning} {idx0} {idx1}{template_arg}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binning", type=float)
    parser.add_argument("npar", type=int)
    parser.add_argument("--queue", type=str, default="short")
    parser.add_argument("--templates", type=str, default="")
    args = parser.parse_args()

    ntemplates = int(2500 * 180)
    inds = np.linspace(0, ntemplates, args.npar + 1).astype(int)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    template_arg = f" --templates {args.templates}" if args.templates else ""

    for idx in range(args.npar):
        if idx == 0:
            sleep(5)
        script_name = "run_detlim.sh"
        with open(script_name, "w") as handle:
            handle.write(
                submission.format(
                    idx=idx,
                    out_dir=out_dir,
                    queue=args.queue,
                    binning=args.binning,
                    idx0=inds[idx],
                    idx1=inds[idx + 1],
                    template_arg=template_arg,
                )
            )

        os.system(f"echo {idx}")
        os.system(f"bsub < {script_name}")
        os.system(f"rm {script_name}")


if __name__ == "__main__":
    main()

