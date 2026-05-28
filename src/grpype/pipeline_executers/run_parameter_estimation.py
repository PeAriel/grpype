from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from grpype.detection.global_params import DATAPATH
from grpype.followup.parameter_estimation import run_pe_chunk
from grpype.pipeline_executers.collectors import collect_pe_results


def _year_type(value: str) -> int | None:
    if value == "all":
        return None
    return int(value)


def run_parameter_estimation():
    parser = argparse.ArgumentParser()
    parser.add_argument("trigtype", choices=["p", "ts"], type=str)
    parser.add_argument("--year", type=_year_type, required=True, metavar="YEAR|all")
    parser.add_argument("--n_jobs", type=int, default=96)
    parser.add_argument("--save_every", type=int, default=100)
    parser.add_argument("--execution", choices=["server", "cluster"], type=str, default="server")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--rsp-root", type=str, default=None)
    parser.add_argument("--chunk-index", type=int, default=None)
    args = parser.parse_args()

    suffix = "_timeslides" if args.trigtype == "ts" else ""
    timeslides = 1.25 if args.trigtype == "ts" else None

    output_root = Path(args.output_root) if args.output_root is not None else DATAPATH
    rsp_root = output_root if args.rsp_root is None else Path(args.rsp_root)
    if args.year is None:
        trigfile = DATAPATH / f"results{suffix}/all_filtered_triggers.csv"
        output_dir = output_root / f"pe_results{suffix}"
    else:
        trigfile = DATAPATH / f"results{suffix}/{args.year}/filtered_triggers.csv"
        output_dir = output_root / f"pe_results{suffix}/{args.year}"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_dir / "samples", exist_ok=True)

    df = pd.read_csv(trigfile)
    chunks = np.array_split(df, args.n_jobs)

    if args.execution == "server":
        Parallel(n_jobs=args.n_jobs)(
            delayed(run_pe_chunk)(
                i,
                chunk,
                timeslides,
                output_dir,
                args.save_every,
                progress=True,
                rsp_root=rsp_root,
            )
            for i, chunk in enumerate(chunks)
        )
        
        collect_pe_results(args.trigtype, args.year, output_root=output_root, input_root=output_root)
        return

    elif args.execution == "cluster":
        if args.chunk_index is None:
            raise ValueError("Cluster execution requires --chunk-index.")

        save_every = 1
        chunk = chunks[args.chunk_index]
        run_pe_chunk(
            args.chunk_index,
            chunk,
            timeslides,
            output_dir,
            save_every,
            progress=False,
            rsp_root=output_root,
        )
        return


def main():
    run_parameter_estimation()


if __name__ == "__main__":
    main()
