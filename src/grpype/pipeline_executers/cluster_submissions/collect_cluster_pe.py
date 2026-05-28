"""
Collect PE chunk results from cluster runs into a single CSV for both p and ts.
Run after all cluster PE jobs have finished.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from grpype.detection.global_params import DATAPATH
from grpype.pipeline_executers.collectors import collect_pe_results


def _year_type(value: str) -> int | None:
    if value == "all":
        return None
    return int(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect PE chunk results from cluster (p and ts) into filtered_triggers_pe.csv."
    )
    parser.add_argument(
        "--year",
        type=_year_type,
        default=None,
        metavar="YEAR|all",
        help="Year subfolder (e.g. 2023) or 'all'. Default: all",
    )
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--input-root", type=str, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root) if args.output_root is not None else DATAPATH
    input_root = Path(args.input_root) if args.input_root is not None else output_root

    for trigtype in ("p", "ts"):
        collect_pe_results(
            trigtype,
            args.year,
            output_root=output_root,
            input_root=input_root,
        )


if __name__ == "__main__":
    main()
