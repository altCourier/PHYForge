"""
Tests for physys/loader.py

Scope: loader.py's own contract only.

path -> dict -> Config

with FileNotFoundError / OSError / JSONDecodeError / validation errors
"""

import json

import pytest

from physys import loader
from physys.loader import load_config, ConfigLoadError


# ---------------------------------------------------------------------------
# Success path
# ---------------------------------------------------------------------------

def test_success_returns_parse_config_result(tmp_path, monkeypatch):
    """
    load_config should return exactly what parse_config returns.
    """

    config_path = tmp_path / "config.json"
    config_path.write_text('{"ankara": "luebeck"}')

    sentinel = object()
    captured = {}

    def fake_parse_config(raw):
        captured["raw"] = raw
        return sentinel

    monkeypatch.setattr(loader, "parse_config", fake_parse_config)

    result = load_config(config_path)

    assert result is sentinel
    assert captured["raw"] == {"ankara": "luebeck"}


def test_accepts_str_pathname(tmp_path, monkeypatch):
    """
    load_config should accept a str path, not just a Path object.
    """

    config_path = tmp_path / "config.json"
    config_path.write_text('{"ankara": "luebeck"}')

    monkeypatch.setattr(loader, "parse_config", lambda raw: raw)

    result = load_config(str(config_path))

    assert result == {"ankara": "luebeck"}


# ---------------------------------------------------------------------------
# File I/O errors -> ConfigLoadError
# ---------------------------------------------------------------------------

def test_missing_file_raises_config_load_error(tmp_path):
    missing = tmp_path / "does_not_exist.json"

    with pytest.raises(ConfigLoadError) as excinfo:
        load_config(missing)

    assert isinstance(excinfo.value.__cause__, FileNotFoundError)
    assert str(missing) in str(excinfo.value)


def test_os_error_on_read_raises_config_load_error(tmp_path):
    """
    Reading a directory as if it was a file raises IsADirectoryError,
    a subclass of OSError but NOT FileNotFoundError
    This exercises the loader's separate `except OSError` branch without needing to mock
    filesystem permissions (which isn't portable across platforms).
    """
    directory_path = tmp_path / "a_directory"
    directory_path.mkdir()

    with pytest.raises(ConfigLoadError) as excinfo:
        load_config(directory_path)

    assert isinstance(excinfo.value.__cause__, OSError)
    assert not isinstance(excinfo.value.__cause__, FileNotFoundError)


# ---------------------------------------------------------------------------
# JSON decode errors -> ConfigLoadError
# ---------------------------------------------------------------------------

def test_invalid_json_raises_config_load_error(tmp_path):
    bad_json_path = tmp_path / "config.json"
    bad_json_path.write_text("{not valid json")

    with pytest.raises(ConfigLoadError) as excinfo:
        load_config(bad_json_path)

    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)
    assert str(bad_json_path) in str(excinfo.value)


def test_empty_file_raises_config_load_error(tmp_path):
    """
    Testing against empty file.
    """
    empty_path = tmp_path / "config.json"
    empty_path.write_text("")

    with pytest.raises(ConfigLoadError) as excinfo:
        load_config(empty_path)

    assert isinstance(excinfo.value.__cause__, json.JSONDecodeError)


# ---------------------------------------------------------------------------
# Validation errors from parse_config -> ConfigLoadError
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("error_type", [ValueError, KeyError, TypeError])
def test_validation_errors_are_wrapped(tmp_path, monkeypatch, error_type):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"ankara": "luebeck"}')

    def fake_parse_config(raw):
        raise error_type("bad config")

    monkeypatch.setattr(loader, "parse_config", fake_parse_config)

    with pytest.raises(ConfigLoadError) as excinfo:
        load_config(config_path)

    assert isinstance(excinfo.value.__cause__, error_type)
    assert str(config_path) in str(excinfo.value)


def test_unrelated_exception_from_parse_config_is_not_wrapped(tmp_path, monkeypatch):
    """
    loader.py only catches (ValueError, KeyError, TypeError) from
    parse_config. Anything else (for example a bug that raises AttributeError)
    should propagate unwrapped, not be silently swallowed as a
    ConfigLoadError.
    """
    config_path = tmp_path / "config.json"
    config_path.write_text('{"ankara": "luebeck"}')

    def fake_parse_config(raw):
        raise AttributeError("unexpected bug in schema.py")

    monkeypatch.setattr(loader, "parse_config", fake_parse_config)

    with pytest.raises(AttributeError):
        load_config(config_path)


# ---------------------------------------------------------------------------
# Integration tripwire (real parse_config, not mocked)
# ---------------------------------------------------------------------------

def test_real_parse_config_success(tmp_path):
    """
    Exercises the real schema.parse_config to catch drift between it
    and the mocked behavior assumed
    """
    from physys.schema import Config

    minimal_valid_config = {
        "common": {},
        "modulation": {"type": "qam", "num_bits_per_symbol": 2},
        "channels": {
            "tdl": {"model": "A", "delay_spread": 100e-9, "carrier_frequency": 3.5e9},
            "system_level": {"variant": "umi", "carrier_frequency": 3.5e9},
        },
        "waveforms": {
            "time": {"bandwidth": 1e6, "num_time_samples": 128},
            "ofdm": {
                "resource_grid": {
                    "num_ofdm_symbols": 14,
                    "fft_size": 76,
                    "subcarrier_spacing": 30e3,
                },
            },
        },
        "sweep": {
            "ebno_db": {"start": -10, "stop": 10, "step": 2},
            "batch_size": 32,
            "num_symbols": 128,
        },
    }
    
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(minimal_valid_config))

    result = load_config(config_path)

    assert isinstance(result, Config)
