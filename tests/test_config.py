"""Tests for grpype.detection.global_params -- configuration loading."""
from pathlib import Path

import pytest
import yaml

from grpype.detection.global_params import Config, resolve_config


VALID_CONFIG = {
    "data_path": "/tmp/test_data",
    "detectors": ["n0", "n1"],
    "ndetectors": 2,
    "echans": 4,
    "DATALEN": 8,
    "EPS": 1e-9,
}


def _write_config(tmp_path, overrides=None):
    cfg = {**VALID_CONFIG, **(overrides or {})}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.dump(cfg))
    return p


class TestResolveConfig:
    def test_valid_config(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = resolve_config(path)
        assert isinstance(cfg, Config)
        assert cfg.ndetectors == 2
        assert cfg.echans == 4
        assert cfg.DATALEN == 8

    def test_missing_key_raises(self, tmp_path):
        bad = {k: v for k, v in VALID_CONFIG.items() if k != "echans"}
        p = tmp_path / "config.yaml"
        p.write_text(yaml.dump(bad))
        with pytest.raises(KeyError, match="echans"):
            resolve_config(p)

    def test_wrong_type_raises(self, tmp_path):
        path = _write_config(tmp_path, {"ndetectors": "two"})
        with pytest.raises(TypeError, match="ndetectors"):
            resolve_config(path)

    def test_ndetectors_mismatch_raises(self, tmp_path):
        path = _write_config(tmp_path, {"ndetectors": 5})
        with pytest.raises(ValueError, match="ndetectors"):
            resolve_config(path)

    def test_datalen_mismatch_raises(self, tmp_path):
        path = _write_config(tmp_path, {"DATALEN": 999})
        with pytest.raises(ValueError, match="DATALEN"):
            resolve_config(path)

    def test_relative_data_path(self, tmp_path):
        path = _write_config(tmp_path, {"data_path": "relative/data"})
        cfg = resolve_config(path)
        assert cfg.data_path.is_absolute()
        assert "relative" in str(cfg.data_path)

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            resolve_config(tmp_path / "nonexistent.yaml")

    def test_optional_bank_folders_defaults(self, tmp_path):
        path = _write_config(tmp_path)
        cfg = resolve_config(path)
        assert cfg.search_bank_folder == Path("search_bank")
        assert cfg.integration_sky_folder == Path("integration_sky")
        assert cfg.integration_spec_folder == Path("integration_spec")

    def test_custom_bank_folders(self, tmp_path):
        path = _write_config(tmp_path, {
            "search_bank_folder": "my_bank",
            "integration_sky_folder": "my_sky",
            "integration_spec_folder": "my_spec",
        })
        cfg = resolve_config(path)
        assert cfg.search_bank_folder == Path("my_bank")
