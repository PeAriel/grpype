"""Public API for the GBM detection pipeline executers."""

from grpype.detection.global_params import Config, resolve_config

def run_pipeline(*args, **kwargs):
    from grpype.pipeline_executers.run_pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)


def run_sliced_pipeline(*args, **kwargs):
    from grpype.pipeline_executers.run_sliced_pipeline import run_sliced_pipeline as _run_sliced_pipeline

    return _run_sliced_pipeline(*args, **kwargs)


def run_short_pipeline(*args, **kwargs):
    from grpype.pipeline_executers.run_short_pipeline import run_short_pipeline as _run_short_pipeline

    return _run_short_pipeline(*args, **kwargs)


__all__ = [
    "Config",
    "resolve_config",
    "run_pipeline",
    "run_sliced_pipeline",
    "run_short_pipeline",
]
