"""
Resubmit PE jobs that failed with exit 137 (TERM_MEMLIMIT) using 2x memory.
Scans outputs/<year>/*pe_out*.txt for failed jobs and resubmits with same parameters.
"""
from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from time import sleep

# Same template as submit_parameter_estimation (must stay in sync)
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

FAILURE_MARKERS = ("Exited with exit code 137", "TERM_MEMLIMIT")


def parse_failed_job(out_path: Path) -> dict | None:
    """Parse a job output file. Return params dict if it failed with 137/TERM_MEMLIMIT, else None."""
    text = out_path.read_text()
    if not any(m in text for m in FAILURE_MARKERS):
        return None

    # From path: outputs/2023/ppe_out697.txt -> year=2023, stem=ppe_out697
    parts = out_path.parts
    year = parts[-2] if len(parts) >= 2 else "all"
    stem = out_path.stem  # e.g. ppe_out697
    match = re.match(r"^(.+)pe_out(\d+)$", stem)
    if not match:
        return None
    trigtype = match.group(1)  # p or ts
    idx = int(match.group(2))

    # From content: rusage[mem=12500], --n_jobs 2000, --output-root ...
    mem_match = re.search(r"rusage\[mem=(\d+)\]", text)
    n_jobs_match = re.search(r"--n_jobs\s+(\d+)", text)
    queue_match = re.search(r"#BSUB -q\s+(\S+)", text)
    output_root_match = re.search(r"--output-root\s+(\S+)", text)

    if not mem_match or not n_jobs_match:
        return None

    params = {
        "trigtype": trigtype,
        "year": year,
        "idx": idx,
        "mem": int(mem_match.group(1)),
        "n_jobs": int(n_jobs_match.group(1)),
        "queue": queue_match.group(1) if queue_match else "short",
        "output_root": output_root_match.group(1) if output_root_match else None,
    }
    return params


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resubmit PE jobs that failed with exit 137 (memory limit) using 2x memory."
    )
    parser.add_argument(
        "year",
        nargs="?",
        default=None,
        help="Year subfolder under outputs/ to scan (e.g. 2023). If omitted, scan all output subdirs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print which jobs would be resubmitted and with what memory.",
    )
    parser.add_argument(
        "--memory-multiplier",
        type=float,
        default=2.0,
        help="Multiply original memory by this factor (default: 2.0).",
    )
    args = parser.parse_args()

    run_dir = Path(__file__).resolve().parent
    outputs_dir = run_dir / "outputs"

    if not outputs_dir.is_dir():
        print("No outputs/ directory found.")
        return

    if args.year is not None:
        year_dirs = [outputs_dir / str(args.year)]
    else:
        year_dirs = [d for d in outputs_dir.iterdir() if d.is_dir()]

    failed = []
    for year_dir in year_dirs:
        if not year_dir.is_dir():
            continue
        for out_file in year_dir.glob("*pe_out*.txt"):
            p = parse_failed_job(out_file)
            if p is not None:
                failed.append(p)

    if not failed:
        print("No failed jobs (exit 137 / TERM_MEMLIMIT) found.")
        return

    print(f"Found {len(failed)} failed job(s). Resubmitting with {args.memory_multiplier}x memory.\n")
    run_dir_abs = str(run_dir)

    for p in failed:
        new_mem = int(p["mem"] * args.memory_multiplier)
        logdir = run_dir / "outputs" / p["year"]
        logdir_abs = str(logdir)
        output_root_arg = ""
        if p["output_root"]:
            output_root_arg = f" --output-root {p['output_root']}"

        if args.dry_run:
            print(
                f"  Would resubmit: {p['trigtype']}pe_{p['year']}_{p['idx']} "
                f"(mem {p['mem']} -> {new_mem} MB)"
            )
            continue

        script_name = f"run_pe_{p['trigtype']}_{p['year']}_{p['idx']}.sh"
        script_path = run_dir / script_name
        with open(script_path, "w") as handle:
            handle.write(
                SUBMISSION.format(
                    trigtype=p["trigtype"],
                    year=p["year"],
                    idx=p["idx"],
                    queue=p["queue"],
                    mem=new_mem,
                    n_jobs=p["n_jobs"],
                    output_root_arg=output_root_arg,
                    run_dir_abs=run_dir_abs,
                    logdir_abs=logdir_abs,
                )
            )
        os.system(f"bsub < {script_path}")
        script_path.unlink()
        print(f"  Resubmitted: {p['trigtype']}pe_{p['year']}_{p['idx']} (mem {p['mem']} -> {new_mem} MB)")
        sleep(0.01)


if __name__ == "__main__":
    main()
