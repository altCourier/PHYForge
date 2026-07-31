"""
Tests for physys/builders.py

Two distinct testings are used:

1. DISPATCH / WIRING tests (build_channel, build_waveform, and the
   argument-wiring half of build_mapsys):
   the real Sionna classes that
   builders.py references (TDL, UMi, TimeChannel, Constellation, ...)
   are monkeypatched with MagicMock. 
   This isolates builders.py's logic

2. REAL-SIONNA INTEGRATION test (build_mapsys bit-width sweep):
   The whole point of that test is to confirm
   Sionna's actual Constellation can generate for 
   num_bits_per_symbol=10
"""
from unittest.mock import MagicMock

import pytest

from sionna.phy.channel import AWGN as RealAWGN
from sionna.phy.mapping import BinarySource

from physys import builders
from physys.builders import (
    build_channel,
    build_waveform,
    build_mapsys,
    build_source,
    _pd_kwargs,
)

from physys.schema import parse_config, AWGNChannelConfig, CommonConfig
from physys.runtime import PHYSys

# ---------------------------------------------------------------------------
# Shared config helpers
# ---------------------------------------------------------------------------

def _base_config_dict() -> dict:
    return {
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


def _config(active_channel="awgn", active_waveform="time", modulation=None):

    raw = _base_config_dict()

    raw["common"] = {"active_channel": active_channel, "active_waveform": active_waveform}

    if modulation is not None:
        raw["modulation"] = modulation

    return parse_config(raw)


# ---------------------------------------------------------------------------
# build_channel: dispatches to the right Sionna class for each of the 4
# channel types
# ---------------------------------------------------------------------------

def test_build_channel_awgn_dispatches_to_AWGN(monkeypatch):

    awgn_mock = MagicMock()
    monkeypatch.setattr(builders, "AWGN", awgn_mock)

    config = _config(active_channel="awgn")
    config.channels.awgn.precision = "single"

    result = build_channel(config)

    assert result is awgn_mock.return_value
    awgn_mock.assert_called_once()

    _, kwargs = awgn_mock.call_args
    assert kwargs["precision"] == "single"


def test_build_channel_tdl_dispatches_to_TDL(monkeypatch):

    tdl_mock = MagicMock()

    monkeypatch.setattr(builders, "TDL", tdl_mock)

    config = _config(active_channel="tdl")
    result = build_channel(config)

    assert result is tdl_mock.return_value

    _, kwargs = tdl_mock.call_args

    assert kwargs["model"] == config.channels.tdl.model
    assert kwargs["delay_spread"] == config.channels.tdl.delay_spread
    assert kwargs["carrier_frequency"] == config.channels.tdl.carrier_frequency


def test_build_channel_rayleigh_dispatches_to_RayleighBlockFading(monkeypatch):
    rayleigh_mock = MagicMock()
    monkeypatch.setattr(builders, "RayleighBlockFading", rayleigh_mock)

    config = _config(active_channel="rayleigh_block_fading")
    config.channels.rayleigh_block_fading.num_rx = 2
    config.channels.rayleigh_block_fading.num_tx_ant = 4

    result = build_channel(config)

    assert result is rayleigh_mock.return_value

    _, kwargs = rayleigh_mock.call_args
    assert kwargs["num_rx"] == 2
    assert kwargs["num_tx_ant"] == 4
    assert kwargs["num_rx_ant"] == config.channels.rayleigh_block_fading.num_rx_ant
    assert kwargs["num_tx"] == config.channels.rayleigh_block_fading.num_tx


@pytest.mark.parametrize("variant,class_attr,expects_o2i", [
    ("umi", "UMi", True),
    ("uma", "UMa", True),
    ("rma", "RMa", False),
])
def test_build_channel_system_level_dispatches_to_correct_variant(
    monkeypatch, variant, class_attr, expects_o2i
):
    variant_mock = MagicMock()
    panel_array_mock = MagicMock()

    monkeypatch.setattr(builders, class_attr, variant_mock)
    monkeypatch.setattr(builders, "PanelArray", panel_array_mock)

    raw = _base_config_dict()
    raw["common"] = {"active_channel": "system_level"}
    raw["channels"]["system_level"]["variant"] = variant

    config = parse_config(raw)

    result = build_channel(config)

    assert result is variant_mock.return_value
    variant_mock.assert_called_once()

    _, kwargs = variant_mock.call_args
    assert kwargs["bs_array"] is panel_array_mock.return_value
    assert kwargs["ut_array"] is panel_array_mock.return_value
    assert panel_array_mock.call_count == 2

    if expects_o2i:
        assert "o2i_model" in kwargs
        assert kwargs["o2i_model"] == config.channels.system_level.o2i_model

    else:
        assert "o2i_model" not in kwargs


def test_build_channel_unrecognized_config_type_raises_valueerror(monkeypatch):
    """
    Schema.py's Config.__post_init__ already guarantees active_channel
    is one of the 4 real field names, so this branch is normally
    unreachable -- but it's still a real safety net (e.g. against a
    future refactor that adds a 5th ChannelsConfig field without a
    matching isinstance branch here). Force it directly by faking what
    _active_channel_config returns.
    """
    monkeypatch.setattr(builders, "_active_channel_config", lambda config: object())

    with pytest.raises(ValueError):
        build_channel(_config(active_channel="awgn"))


# ---------------------------------------------------------------------------
# _pd_kwargs inheritance
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field", ["precision", "device"])
class TestPdKwargsInheritance:
    """
    AWGNChannelConfig/CommonConfig are used as is
    """

    def test_local_only_set_wins(self, field):

        local = AWGNChannelConfig(**{field: "local_val"})
        common = CommonConfig()

        result = _pd_kwargs(local, common)

        assert result[field] == "local_val"

    def test_common_fallback_used_when_local_unset(self, field):

        local = AWGNChannelConfig()
        common = CommonConfig(**{field: "common_val"})

        result = _pd_kwargs(local, common)

        assert result[field] == "common_val"

    def test_neither_set_omits_key_entirely(self, field):

        local = AWGNChannelConfig()
        common = CommonConfig()

        result = _pd_kwargs(local, common)

        assert field not in result

    def test_both_set_local_still_wins(self, field):

        """
        The case that's easy to get subtly wrong: local isn't just
        used 'when common is absent' -- it wins even when common has
        its own explicit, different value.
        """

        local = AWGNChannelConfig(**{field: "local_val"})
        common = CommonConfig(**{field: "common_val"})

        result = _pd_kwargs(local, common)

        assert result[field] == "local_val"


# ---------------------------------------------------------------------------
# build_waveform: AWGN bypasses both "time" and "ofdm"
# ---------------------------------------------------------------------------

def test_build_waveform_time_bypasses_awgn_with_no_wrapping():

    config = _config(active_waveform="time")
    channel_model = RealAWGN()

    result = build_waveform(config, channel_model)

    assert result is channel_model


def test_build_waveform_ofdm_bypasses_awgn_with_no_wrapping():

    config = _config(active_waveform="ofdm")
    channel_model = RealAWGN()

    result = build_waveform(config, channel_model)

    assert result is channel_model


# ---------------------------------------------------------------------------
# build_waveform: non-AWGN channels DO get wrapped 
# ---------------------------------------------------------------------------

def test_build_waveform_time_wraps_non_awgn_channel(monkeypatch):

    time_channel_mock = MagicMock()
    monkeypatch.setattr(builders, "TimeChannel", time_channel_mock)

    config = _config(active_waveform="time")
    fake_channel_model = object()  # stands in for a TDL/RayleighBlockFading instance

    result = build_waveform(config, fake_channel_model)

    assert result is time_channel_mock.return_value

    _, kwargs = time_channel_mock.call_args

    assert kwargs["channel_model"] is fake_channel_model
    assert kwargs["num_time_samples"] == config.waveforms.time.num_time_samples
    assert kwargs["bandwidth"] == config.waveforms.time.bandwidth


def test_build_waveform_ofdm_wraps_non_awgn_channel(monkeypatch):

    resource_grid_mock = MagicMock()
    ofdm_channel_mock = MagicMock()

    monkeypatch.setattr(builders, "ResourceGrid", resource_grid_mock)
    monkeypatch.setattr(builders, "OFDMChannel", ofdm_channel_mock)

    config = _config(active_waveform="ofdm")
    fake_channel_model = object()

    result = build_waveform(config, fake_channel_model)

    assert result is ofdm_channel_mock.return_value

    resource_grid_mock.assert_called_once()
    _, ofdm_kwargs = ofdm_channel_mock.call_args

    assert ofdm_kwargs["channel_model"] is fake_channel_model
    assert ofdm_kwargs["resource_grid"] is resource_grid_mock.return_value

    rg_cfg = config.waveforms.ofdm.resource_grid
    _, rg_kwargs = resource_grid_mock.call_args

    assert rg_kwargs["num_ofdm_symbols"] == rg_cfg.num_ofdm_symbols
    assert rg_kwargs["fft_size"] == rg_cfg.fft_size
    assert rg_kwargs["subcarrier_spacing"] == rg_cfg.subcarrier_spacing


def test_build_waveform_unknown_active_waveform_raises_valueerror():

    config = _config(active_waveform="time")
    
    config.common.active_waveform = "bogus"

    with pytest.raises(ValueError):
        build_waveform(config, object())


# ---------------------------------------------------------------------------
# build_source (trivial, but free to cover)
# ---------------------------------------------------------------------------

def test_build_source_returns_binary_source():

    result = build_source(_config())

    assert isinstance(result, BinarySource)


# ---------------------------------------------------------------------------
# build_mapsys: wiring
# ---------------------------------------------------------------------------

def test_build_mapsys_wires_config_fields_through(monkeypatch):

    fake_constellation = MagicMock(name="constellation_instance")
    constellation_mock = MagicMock(return_value=fake_constellation)

    mapper_mock = MagicMock()
    demapper_mock = MagicMock()

    monkeypatch.setattr(builders, "Constellation", constellation_mock)
    monkeypatch.setattr(builders, "Mapper", mapper_mock)
    monkeypatch.setattr(builders, "Demapper", demapper_mock)

    config = _config(modulation={
        "type": "qam",
        "num_bits_per_symbol": 4,
        "mapper": {"return_indices": True},
        "demapper": {"hard_out": True, "demapping_method": "maxlog"},
        "constellation": {"normalize": True, "center": True},
    })

    build_mapsys(config)

    constellation_mock.assert_called_once_with(
        "qam", 4, normalize=True, center=True, points=None
    )

    mapper_mock.assert_called_once_with(
        constellation=fake_constellation, return_indices=True
    )

    demapper_mock.assert_called_once_with(
        "maxlog", constellation=fake_constellation, hard_out=True
    )


# ---------------------------------------------------------------------------
# build_mapsys: real-Sionna integration sweep across every bit-width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mod_type,bits", [
    ("pam", 1),   # BPSK
    ("qam", 2),   # QPSK
    ("qam", 4),   # 16-QAM
    ("qam", 6),   # 64-QAM
    ("qam", 8),   # 256-QAM
    ("qam", 10),  # 1024-QAM
])

def test_build_mapsys_builds_at_every_required_bit_width(mod_type, bits):

    config = _config(modulation={"type": mod_type, "num_bits_per_symbol": bits})

    constellation, mapper, demapper = build_mapsys(config)

    assert constellation.num_bits_per_symbol == bits
    assert mapper is not None
    assert demapper is not None

def test_rma_ignores_o2i_model_without_error():
    """
    KNOWN DISCREPANCY: SystemLevelChannelConfig.o2i_model is a
    shared field across umi/uma/rma at the schema level, but RMa's
    Sionna constructor doesn't accept it -- build_channel silently
    drops it for rma rather than raising. This test pins that as
    current, intentional behavior. If RMa should instead reject a
    non-default o2i_model explicitly, the fix belongs in build_channel,
    not here.
    """

    raw = _base_config_dict()
    
    raw["common"]["active_channel"] = "system_level"
    raw["channels"]["system_level"]["variant"] = "rma"
    raw["channels"]["system_level"]["o2i_model"] = "high"  # should be silently ignored

    config = parse_config(raw)

    system = PHYSys(config)  # should not raise
    assert system._channel_model.__class__.__name__ == "RMa"
