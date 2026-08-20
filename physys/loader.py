"""
loader.py

Reads a config.json file from disk and hands it off to schema.py's
parse_config for validation.

This file owns exactly one thing: path -> dict -> Config.
It does not re-validate anything schema.py already validates.
"""
import json

from pathlib import Path
from typing import Union

from .schema import parse_config, Config

class ConfigLoadError(Exception):
    """Raised when a config file can't be found, read, parsed as JSON,
    or validated against the schema. Wraps the underlying cause so
    callers only need to catch one exception type."""

def load_config(pathname: Union[str, Path]) -> Config:

    path = Path(pathname)

    try:
        raw_text = path.read_text()

    except FileNotFoundError as exc:
        raise ConfigLoadError(f"Config file not found: {path}") from exc
    
    except OSError as exc:
        raise ConfigLoadError(f"Could not read config file {path}: {exc}") from exc

    try:
        raw = json.loads(raw_text)

    except json.JSONDecodeError as exc:
        raise ConfigLoadError(f"Config file {path} is not valid JSON: {exc}") from exc

    try:
        return parse_config(raw)
    except (ValueError, KeyError, TypeError) as exc:
        raise ConfigLoadError(f"Config file {path} failed validation: {exc}") from exc
