from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

from grpype.detection.global_params import DATAPATH




def _collect_csvs(input_dir: Path, output_path: Path, pattern: str) -> None:
    csv_files = sorted(input_dir.glob(pattern))
    if not csv_files:
        print(f"No files found in {input_dir} matching {pattern}")
        return

    df_combined = pd.concat([pd.read_csv(path) for path in csv_files], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_combined.to_csv(output_path, index=False)
    print(f"Combined {len(csv_files)} files into {output_path}")


def collect_pe_results(trigtype: str, year: int | None = None, output_root: Path | str | None = None, input_root: Path | str | None = None) -> None:
    """
    Collect PE chunk results into a single CSV.
    """
    suffix = "_timeslides" if trigtype == "ts" else ""
    base_output = Path(output_root) if output_root is not None else DATAPATH
    base_input = Path(input_root) if input_root is not None else DATAPATH
    if year is None:
        input_dir = base_input / f"pe_results{suffix}"
        output_path = base_output / f"results{suffix}/filtered_triggers_pe.csv"
    else:
        input_dir = base_input / f"pe_results{suffix}/{year}"
        output_path = base_output / f"results{suffix}/{year}/filtered_triggers_pe.csv"

    _collect_csvs(input_dir, output_path, "pe_chunk_*.csv")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("trigtype", choices=["p", "ts"], type=str)
    # parser.add_argument("--year", type=int, default=None)
    # parser.add_argument("--output-root", type=str, default=None)
    # parser.add_argument("--input-root", type=str, default=None)
    # args = parser.parse_args()

    # collect_pe_results(args.trigtype, args.year, args.output_root, args.input_root)
    # collect_swift_followup(args.year, args.output_root, args.input_root)

    pass

if __name__ == "__main__":
    main()
