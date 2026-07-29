"""
Tests for physys/schema.py
"""

import copy
import pytest

from physys.schema import (
    Config,
    CommonConfig,
    ModulationConfig,
    MapperConfig,
    DemapperConfig,
    ConstellationConfig,
    TDLChannelConfig,
    SystemLevelChannelConfig,
    ChannelsConfig,
    ResourceGridConfig,
    parse_config,
    parse_common,
    parse_modulation,
    parse_channels,
    parse_waveforms,
    parse_sweep,
    _strip_comments,
)


# ---------------------------------------------------------------------------
# A minimal but fully valid raw config dict.
# Every "missing field" test below starts from a deep copy of this and
# deletes exactly the one thing it's testing, so a failure always points
# at one specific field.
# ---------------------------------------------------------------------------

def base_config_dict() -> dict:
    return {
        "common": {},
        "modulation": {
            "type": "qam",
            "num_bits_per_symbol": 2,
        },
        "channels": {
            "tdl": {
                "model": "A",
                "delay_spread": 100e-9,
                "carrier_frequency": 3.5e9,
            },
            "system_level": {
                "variant": "umi",
                "carrier_frequency": 3.5e9,
            },
        },
        "waveforms": {
            "time": {
                "bandwidth": 1e6,
                "num_time_samples": 128,
            },
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


# ---------------------------------------------------------------------------
# 1. Valid config parses end-to-end
# ---------------------------------------------------------------------------

def test_valid_config_parses():

    config = parse_config(base_config_dict())

    assert isinstance(config, Config)
    assert config.modulation.type == "qam"
    assert config.modulation.num_bits_per_symbol == 2
    assert config.channels.tdl.model == "A"
    assert config.channels.system_level.variant == "umi"
    assert config.waveforms.time.num_time_samples == 128
    assert config.sweep.num_symbols == 128
    assert config.source is not None


def test_source_key_is_fully_optional():
    """source is missing entirely from base_config_dict() above -- this
    just makes that explicit as its own test rather than an implicit
    side effect of test_valid_config_parses."""
    raw = base_config_dict()
    assert "source" not in raw
    config = parse_config(raw)
    assert config.source is not None


# ---------------------------------------------------------------------------
# 2a. Missing required TOP-LEVEL sections -> KeyError (raw["x"] access)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", ["common", "modulation", "channels", "waveforms", "sweep"])
def test_missing_top_level_section_raises_keyerror(missing_key):
    raw = base_config_dict()
    del raw[missing_key]

    with pytest.raises(KeyError):
        parse_config(raw)


# ---------------------------------------------------------------------------
# 2b. Missing required NESTED keys -- exact exception type depends on
# whether schema.py uses raw["x"]/pop("x") (KeyError) or **kwargs into
# a dataclass with a no-default field (TypeError).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("missing_key", ["tdl", "system_level"])
def test_channels_missing_required_subsection_raises_keyerror(missing_key):
    raw = base_config_dict()["channels"]
    del raw[missing_key]

    with pytest.raises(KeyError):
        parse_channels(raw)


def test_tdl_missing_key_raises_typeerror():
    """model/delay_spread/carrier_frequency have no dataclass default,
    so an ABSENT key fails at TDLChannelConfig's own __init__, before
    __post_init__'s explicit check ever runs."""
    raw = base_config_dict()["channels"]
    del raw["tdl"]["model"]

    with pytest.raises(TypeError):
        parse_channels(raw)


def test_tdl_null_value_raises_valueerror():
    """Contrast with the above: a key that IS present but explicitly
    null (a real config.json mistake) passes construction and is
    instead caught by TDLChannelConfig.__post_init__'s required-field
    check, as ValueError -- not TypeError."""
    raw = base_config_dict()["channels"]
    raw["tdl"]["model"] = None

    with pytest.raises(ValueError, match="model"):
        parse_channels(raw)


def test_system_level_missing_key_raises_typeerror():
    raw = base_config_dict()["channels"]
    del raw["system_level"]["carrier_frequency"]

    with pytest.raises(TypeError):
        parse_channels(raw)


def test_waveforms_missing_resource_grid_raises_keyerror():
    raw = base_config_dict()["waveforms"]
    del raw["ofdm"]["resource_grid"]

    with pytest.raises(KeyError):
        parse_waveforms(raw)


def test_resource_grid_missing_key_raises_typeerror():
    raw = base_config_dict()["waveforms"]
    del raw["ofdm"]["resource_grid"]["fft_size"]

    with pytest.raises(TypeError):
        parse_waveforms(raw)


def test_time_waveform_missing_key_raises_typeerror():
    raw = base_config_dict()["waveforms"]
    del raw["time"]["num_time_samples"]

    with pytest.raises(TypeError):
        parse_waveforms(raw)


@pytest.mark.parametrize("missing_key", ["ebno_db", "batch_size", "num_symbols"])
def test_sweep_missing_required_key_raises_keyerror(missing_key):
    raw = base_config_dict()["sweep"]
    del raw[missing_key]

    with pytest.raises(KeyError):
        parse_sweep(raw)


def test_sweep_ebno_db_missing_key_raises_typeerror():
    raw = base_config_dict()["sweep"]
    del raw["ebno_db"]["step"]

    with pytest.raises(TypeError):
        parse_sweep(raw)


def test_modulation_missing_type_raises_valueerror():
    """type has no dataclass default in the raw dict sense -- it's
    fetched with raw.get("type") (returns None), so construction
    succeeds and __post_init__'s enum check is what actually catches
    it, as ValueError, not a missing-arg TypeError."""
    raw = {"num_bits_per_symbol": 2}

    with pytest.raises(ValueError):
        parse_modulation(raw)


def test_modulation_missing_num_bits_per_symbol_raises_valueerror():
    raw = {"type": "qam"}

    with pytest.raises(ValueError, match="integer"):
        parse_modulation(raw)


# ---------------------------------------------------------------------------
# 3. Bad enums raise (type, active_channel, active_waveform, variant)
# ---------------------------------------------------------------------------

def test_modulation_bad_type_raises_valueerror():
    raw = {"type": "psk", "num_bits_per_symbol": 2}  # "psk" is not qam/pam

    with pytest.raises(ValueError, match="qam.*pam|pam.*qam"):
        parse_modulation(raw)


def test_active_channel_bad_enum_raises_valueerror():
    raw = base_config_dict()
    raw["common"]["active_channel"] = "not_a_real_channel"

    with pytest.raises(ValueError, match="active_channel"):
        parse_config(raw)


def test_active_waveform_bad_enum_raises_valueerror():
    raw = base_config_dict()
    raw["common"]["active_waveform"] = "not_a_real_waveform"

    with pytest.raises(ValueError, match="active_waveform"):
        parse_config(raw)


def test_system_level_bad_variant_raises_valueerror():
    raw = base_config_dict()["channels"]
    raw["system_level"]["variant"] = "nyc"  # not umi/uma/rma

    with pytest.raises(ValueError, match="variant"):
        parse_channels(raw)


@pytest.mark.parametrize("channel_name", ["awgn", "tdl", "system_level", "rayleigh_block_fading"])
def test_active_channel_accepts_every_real_channel_name(channel_name):
    """Positive counterpart to the bad-enum test: valid_channels is
    derived from ChannelsConfig's own field names (not hand-typed), so
    every real channel name should be accepted -- this pins that
    derivation down rather than just trusting it."""
    raw = base_config_dict()
    raw["common"]["active_channel"] = channel_name

    config = parse_config(raw)
    assert config.common.active_channel == channel_name


# ---------------------------------------------------------------------------
# 4. _strip_comments drops "_"-prefixed keys and nothing else
# ---------------------------------------------------------------------------

def test_strip_comments_drops_underscore_keys():
    raw = {"_comment": "ignore me", "_description": "also ignore", "real_key": 1}

    result = _strip_comments(raw)

    assert result == {"real_key": 1}


def test_strip_comments_keeps_non_underscore_keys_unchanged():
    raw = {"a": 1, "b": [1, 2, 3], "c": {"nested": True}}

    result = _strip_comments(raw)

    assert result == raw


def test_strip_comments_does_not_mutate_input():
    raw = {"_comment": "x", "keep": 1}
    original = dict(raw)

    _strip_comments(raw)

    assert raw == original  # input dict untouched


def test_strip_comments_bare_underscore_key_is_dropped():
    """Edge case: a key that's literally just "_" still starts with
    "_", so per the documented convention it should be dropped too."""
    raw = {"_": "mystery", "keep": 1}

    result = _strip_comments(raw)

    assert result == {"keep": 1}


def test_config_parses_with_comment_keys_sprinkled_throughout():
    """End-to-end: config.json with inline '_comment' keys at several
    nesting levels should parse exactly as if they weren't there."""
    raw = base_config_dict()
    raw["_comment"] = "top level note"
    raw["modulation"]["_comment"] = "modulation note"
    raw["sweep"]["_comment"] = "sweep note"

    config = parse_config(raw)

    assert config.modulation.type == "qam"
    assert config.sweep.num_symbols == 128


# ---------------------------------------------------------------------------
# 5. num_symbols == num_time_samples cross-field check
#
# This validation does NOT exist yet -- Config.__post_init__ only checks
# active_channel/active_waveform. Per the "write the test first" plan,
# this is marked xfail(strict=True): it SHOULD fail right now. If you
# implement the check and this unexpectedly starts passing, pytest will
# flag it as XPASS (strict=True makes that a failure) -- that's your
# signal to delete the marker, not a bug in the test.
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="Config.__post_init__ doesn't yet enforce num_symbols == num_time_samples",
    strict=True,
)
def test_num_symbols_must_match_num_time_samples():
    raw = base_config_dict()
    raw["sweep"]["num_symbols"] = 128
    raw["waveforms"]["time"]["num_time_samples"] = 999  # deliberately mismatched

    with pytest.raises(ValueError, match="num_symbols|num_time_samples"):
        parse_config(raw)


# ---------------------------------------------------------------------------
# 6. ut_array default gotcha (found while writing these tests, not in
# the original plan -- flagging via a pinned test rather than silently
# "fixing" it, since it's genuinely ambiguous which behavior is intended)
# ---------------------------------------------------------------------------

def test_ut_array_omni_default_via_direct_construction():
    """SystemLevelChannelConfig's own default_factory gives ut_array
    antenna_pattern="omni" when constructed directly without ut_array."""
    cfg = SystemLevelChannelConfig(variant="umi", carrier_frequency=3.5e9)
    assert cfg.ut_array.antenna_pattern == "omni"


def test_ut_array_default_via_parse_channels_is_NOT_omni():
    """
    KNOWN DISCREPANCY: parse_channels always explicitly builds
    ut_array = PanelArrayConfig(**ut_array_raw), even when
    system_level.ut_array is entirely absent from config.json
    (ut_array_raw defaults to {} via .pop("ut_array", {})). That
    bypasses SystemLevelChannelConfig's field-level default_factory
    (which is what sets "omni"), so a real config.json that omits
    ut_array ends up with antenna_pattern="38.901" through
    parse_channels -- disagreeing with what direct construction gives.

    This test documents CURRENT behavior. If "omni" is the intended
    default for configs that omit ut_array, the fix belongs in
    parse_channels (e.g. only pass ut_array= when ut_array_raw is
    non-empty), not here.
    """
    raw = base_config_dict()["channels"]
    assert "ut_array" not in raw["system_level"]

    channels = parse_channels(raw)

    assert channels.system_level.ut_array.antenna_pattern == "38.901"