import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from grpype.detection.global_params import DATAPATH


# Files to concatenate for results/ (includes Swift followup)
RESULTS_FILES = [
    # (per-year filename, combined filename, has_trigger_index_column)
    ("filtered_triggers.csv", "combined_filtered_triggers.csv", False),
    ("filtered_triggers_with_pe_with_followup.csv", "combined_filtered_triggers_with_pe_with_followup.csv", False),
    ("filtered_triggers_pe.csv", "combined_filtered_triggers_pe.csv", True),
    ("swift_followup.csv", "combined_swift_followup.csv", True),
]

# Files to concatenate for results_timeslides/ (PE only, no Swift followup)
TIMESLIDES_FILES = [
    ("filtered_triggers.csv", "combined_filtered_triggers.csv", False),
    ("filtered_triggers_with_pe.csv", "combined_filtered_triggers_with_pe.csv", False),
    ("filtered_triggers_pe.csv", "combined_filtered_triggers_pe.csv", True),
]

YEARS = [str(y) for y in range(2013, 2026)]


def _concat_files(base_input, base_output, results_dir, files_config):
    """Concatenate per-year CSVs for a single results directory.

    Parameters
    ----------
    base_input : Path
        Root data path for reading.
    base_output : Path
        Root data path for writing.
    results_dir : str
        Either 'results' or 'results_timeslides'.
    files_config : list of tuples
        Each tuple is (per_year_filename, combined_filename, has_trigger_index).
    """
    all_frames = {combined_fname: [] for _, combined_fname, _ in files_config}

    for year in YEARS:
        paths = {combined_fname: base_input / results_dir / year / per_year_fname
                 for per_year_fname, combined_fname, _ in files_config}
        if not all(p.exists() for p in paths.values()):
            missing = [p.name for p in paths.values() if not p.exists()]
            print(f"Skipping {year}, not all required files exist")
            continue

        print(f'concatenating year: {year}')
        for per_year_fname, combined_fname, has_trigger_index in files_config:
            df = pd.read_csv(paths[combined_fname])
            df["year"] = year

            if not has_trigger_index:
                df["trigger_index"] = df.index

            all_frames[combined_fname].append(df)

    combined = {}
    for _, combined_fname, _ in files_config:
        frames = all_frames[combined_fname]
        if not frames:
            print(f"No files found for {combined_fname} in {results_dir}/")
            continue

        combined_df = pd.concat(frames, ignore_index=True)
        if "trigtime" in combined_df.columns:
            combined_df = combined_df.sort_values("trigtime").reset_index(drop=True)

        out_path = base_output / results_dir / combined_fname
        combined_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}  ({len(combined_df)} rows from {len(frames)} years)")
        combined[combined_fname] = combined_df

    return combined


def _verify_joins(combined):
    """Verify that auxiliary files are correctly joinable with combined triggers.

    Checks:
    - swift_followup.gbm_met == filtered_triggers.trigmet
    - filtered_triggers_pe.trigtime_ == filtered_triggers.trigtime
    """
    trigs_key = "combined_filtered_triggers.csv"
    if trigs_key not in combined:
        return

    trigs = combined[trigs_key]

    # Verify swift_followup join
    swift_key = "combined_swift_followup.csv"
    if swift_key in combined:
        swift = combined[swift_key]
        merged = trigs.merge(swift[["year", "trigger_index", "gbm_met"]],
                             on=["year", "trigger_index"], how="inner")
        mismatches = np.abs(merged["trigmet"] - merged["gbm_met"]) > 1e-2
        if mismatches.any():
            print(f"WARNING: {mismatches.sum()} trigmet/gbm_met mismatches in swift_followup join")
        else:
            print(f"OK: swift_followup join verified ({len(merged)} matched rows)")

    # Verify filtered_triggers_pe join
    pe_key = "combined_filtered_triggers_pe.csv"
    if pe_key in combined:
        pe = combined[pe_key]
        merged = trigs.merge(pe[["year", "trigger_index", "trigtime_"]],
                             on=["year", "trigger_index"], how="inner")
        merged["trigtime"] = pd.to_datetime(merged["trigtime"])
        merged["trigtime_"] = pd.to_datetime(merged["trigtime_"])
        mismatches = np.abs((merged["trigtime"] - merged["trigtime_"]).dt.total_seconds()) > 1e-2
        if mismatches.any():
            print(f"WARNING: {mismatches.sum()} trigtime/trigtime_ mismatches in PE join")
        else:
            print(f"OK: filtered_triggers_pe join verified ({len(merged)} matched rows)")


def concat_all(save=True):
    """Concatenate per-year trigger CSVs across all years.

    Processes both ``results/`` (full set including Swift followup) and
    ``results_timeslides/`` (triggers + PE only).

    Parameters
    ----------
    save : bool
        Whether to write concatenated CSVs to disk.
    input_root : str or None
        Override input base path (defaults to DATAPATH).
    output_root : str or None
        Override output base path (defaults to DATAPATH).
    """
    base_input = DATAPATH
    base_output = DATAPATH

    print("=== Processing results/ ===")
    results_combined = _concat_files(base_input, base_output, "results", RESULTS_FILES)
    _verify_joins(results_combined)
    
    print("\n=== Processing results_timeslides/ ===")
    ts_combined = _concat_files(base_input, base_output, "results_timeslides", TIMESLIDES_FILES)
    _verify_joins(ts_combined)

    return results_combined, ts_combined


def main():
    parser = argparse.ArgumentParser(
        description="Concatenate per-year filtered trigger CSVs across all years."
    )

    concat_all(save=True)


if __name__ == "__main__":
    main()
