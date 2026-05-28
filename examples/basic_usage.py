from datetime import datetime

from grpype.pipeline_executers import resolve_config, run_pipeline


def main() -> None:
    config = resolve_config()
    print(f"Using data path: {config.data_path}")

    start = datetime(2017, 8, 17, 12, 41)
    run_pipeline(start, delta=1, tte_npar=1)


if __name__ == "__main__":
    main()
