from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    data_path: Path
    detectors: list[str]
    ndetectors: int
    echans: int
    DATALEN: int
    EPS: float
    search_bank_folder: Path
    integration_sky_folder: Path
    integration_spec_folder: Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return data


def _require_key(data: dict[str, Any], key: str, path: Path) -> Any:
    if key not in data:
        raise KeyError(f"Missing '{key}' in config file: {path}")
    return data[key]


def _optional_key(data: dict[str, Any], key: str, default: Any) -> Any:
    return data.get(key, default)


def _normalize_bank_folder(folder: str | Path) -> Path:
    path = Path(folder)
    if not path.is_absolute() and path.parts and path.parts[0] == "templates" and len(path.parts) > 1:
        path = Path(*path.parts[1:])
    return path


def resolve_config(config_path: Path | None = None) -> Config:
    config_path = config_path or (_repo_root() / "config" / "config.yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data = _load_yaml(config_path)

    data_path = _require_key(data, "data_path", config_path)
    detectors = _require_key(data, "detectors", config_path)
    ndetectors = _require_key(data, "ndetectors", config_path)
    echans = _require_key(data, "echans", config_path)
    datalen = _require_key(data, "DATALEN", config_path)
    eps = _require_key(data, "EPS", config_path)
    search_bank_folder = _optional_key(data, "search_bank_folder", "search_bank")
    integration_sky_folder = _optional_key(data, "integration_sky_folder", "integration_sky")
    integration_spec_folder = _optional_key(data, "integration_spec_folder", "integration_spec")

    if not isinstance(data_path, str):
        raise TypeError("data_path must be a string")
    if not isinstance(detectors, list) or not all(isinstance(d, str) for d in detectors):
        raise TypeError("detectors must be a list of strings")
    if not isinstance(ndetectors, int):
        raise TypeError("ndetectors must be an int")
    if not isinstance(echans, int):
        raise TypeError("echans must be an int")
    if not isinstance(datalen, int):
        raise TypeError("DATALEN must be an int")
    if not isinstance(eps, float):
        raise TypeError("EPS must be a float")
    if not isinstance(search_bank_folder, (str, Path)):
        raise TypeError("search_bank_folder must be a string or Path")
    if not isinstance(integration_sky_folder, (str, Path)):
        raise TypeError("integration_sky_folder must be a string or Path")
    if not isinstance(integration_spec_folder, (str, Path)):
        raise TypeError("integration_spec_folder must be a string or Path")

    path_obj = Path(data_path)
    if not path_obj.is_absolute():
        path_obj = (config_path.parent / path_obj).resolve()
    if not path_obj.is_absolute():
        raise ValueError("data_path must be an absolute path")

    if ndetectors != len(detectors):
        raise ValueError(
            f"ndetectors ({ndetectors}) does not match len(detectors) ({len(detectors)})"
        )
    if datalen != ndetectors * echans:
        raise ValueError(
            f"DATALEN ({datalen}) does not match ndetectors * echans ({ndetectors * echans})"
        )

    return Config(
        data_path=path_obj,
        detectors=detectors,
        ndetectors=ndetectors,
        echans=echans,
        DATALEN=datalen,
        EPS=float(eps),
        search_bank_folder=_normalize_bank_folder(search_bank_folder),
        integration_sky_folder=_normalize_bank_folder(integration_sky_folder),
        integration_spec_folder=_normalize_bank_folder(integration_spec_folder),
    )


_config = resolve_config()

DATAPATH = _config.data_path
detectors = _config.detectors
ndetectors = _config.ndetectors
echans = _config.echans
DATALEN = _config.DATALEN
EPS = _config.EPS

# Template bank folder names.
SEARCH_BANK_FOLDER = _config.search_bank_folder
INTEGRATION_SKY_FOLDER = _config.integration_sky_folder
INTEGRATION_SPEC_FOLDER = _config.integration_spec_folder
