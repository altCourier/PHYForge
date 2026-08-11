"""
builders.py

Consumes a validated Config (from schema.py, via loader.py) and
constructs live Sionna objects from it. This file is allowed to import
sionna/torch freely -- schema.py deliberately is not.

Split of responsibility:
- build_channel(config) returns the raw Sionna ChannelModel (AWGN, TDL,
  or a system-level UMi/UMa/RMa instance) -- NEVER wrapped in
  TimeChannel/OFDMChannel. That wrapping is build_waveform's job.
- build_waveform(config) takes that raw channel model and wraps it
  according to common.active_waveform.
"""

import dataclasses

import numpy as np

import sionna.phy

from sionna.phy.channel import AWGN, RayleighBlockFading, TimeChannel, OFDMChannel
from sionna.phy.channel.tr38901 import TDL, UMa, UMi, RMa, PanelArray

from sionna.phy.mapping import BinarySource, Constellation, Mapper, Demapper

from sionna.phy.ofdm import ResourceGrid

from .schema import (
    Config,
    CommonConfig,
    AWGNChannelConfig,
    TDLChannelConfig,
    SystemLevelChannelConfig,
    RayleighBlockFadingConfig,
)


def _pd_kwargs(local_config, common: CommonConfig) -> dict:
    """
    Resolve precision/device for a single block: the block's own value
    if it set one (non-null), otherwise common's global default.
    Returns a kwargs dict with only the keys that ended up non-None, so
    callers can blindly do kwargs.update(_pd_kwargs(...)) and let Sionna
    fall back to ITS OWN default when neither local nor common set one.

    This is the only place this inheritance rule is implemented -- see
    ARCHITECTURE.md's "code is the decider, not the config" and
    "no duplicated shapes" principles.
    """

    precision = local_config.precision if local_config.precision is not None else common.precision
    device = local_config.device if local_config.device is not None else common.device

    kwargs = {}
    if precision is not None:
        kwargs["precision"] = precision
    if device is not None:
        kwargs["device"] = device

    return kwargs


def _active_channel_config(config: Config):
    """
    Look up the one channels block that's actually live for this run.
    """
    return getattr(config.channels, config.common.active_channel)


def build_channel(config: Config):

    channel_config = _active_channel_config(config)
    common = config.common

    if isinstance(channel_config, AWGNChannelConfig):

        kwargs = _pd_kwargs(channel_config, common)

        return AWGN(**kwargs)

    if isinstance(channel_config, TDLChannelConfig):

        kwargs = dict(
            model = channel_config.model,
            delay_spread = channel_config.delay_spread,
            carrier_frequency = channel_config.carrier_frequency,
            num_sinusoids = channel_config.num_sinusoids,
            los_angle_of_arrival = channel_config.los_angle_of_arrival,
            min_speed = channel_config.min_speed,
            max_speed = channel_config.max_speed,
            num_rx_ant = channel_config.num_rx_ant,
            num_tx_ant = channel_config.num_tx_ant,
            spatial_corr_mat = channel_config.spatial_corr_mat,
            rx_corr_mat = channel_config.rx_corr_mat,
            tx_corr_mat = channel_config.tx_corr_mat,
        )

        kwargs.update(_pd_kwargs(channel_config, common))

        return TDL(**kwargs)

    if isinstance(channel_config, SystemLevelChannelConfig):

        variant_cls = {
            "umi": UMi,
            "uma": UMa,
            "rma": RMa,
        }[channel_config.variant]

        bs_array = PanelArray(
            num_rows_per_panel = channel_config.bs_array.num_rows_per_panel,
            num_cols_per_panel = channel_config.bs_array.num_cols_per_panel,
            polarization = channel_config.bs_array.polarization,
            polarization_type = channel_config.bs_array.polarization_type,
            antenna_pattern = channel_config.bs_array.antenna_pattern,
            carrier_frequency = channel_config.carrier_frequency,
        )

        ut_array = PanelArray(
            num_rows_per_panel = channel_config.ut_array.num_rows_per_panel,
            num_cols_per_panel = channel_config.ut_array.num_cols_per_panel,
            polarization = channel_config.ut_array.polarization,
            polarization_type = channel_config.ut_array.polarization_type,
            antenna_pattern = channel_config.ut_array.antenna_pattern,
            carrier_frequency = channel_config.carrier_frequency,
        )

        kwargs = dict(
            carrier_frequency = channel_config.carrier_frequency,
            ut_array = ut_array,
            bs_array = bs_array,
            direction = channel_config.direction,
            enable_pathloss = channel_config.enable_pathloss,
            enable_shadow_fading = channel_config.enable_shadow_fading,
        )

        if channel_config.variant in ("umi", "uma"):
            kwargs["o2i_model"] = channel_config.o2i_model

        kwargs.update(_pd_kwargs(channel_config, common))

        # TODO: gen_single_sector_topology (per ARCHITECTURE.md) still
        # needs to be called and its output fed to
        # variant_instance.set_topology(...) before this channel is
        # usable. That's a runtime concern (topology may change per
        # batch), so it likely belongs in runtime.py, not here.
        return variant_cls(**kwargs)

    if isinstance(channel_config, RayleighBlockFadingConfig):

        kwargs = dict(
            num_rx = channel_config.num_rx,
            num_tx = channel_config.num_tx,
            num_rx_ant = channel_config.num_rx_ant,
            num_tx_ant = channel_config.num_tx_ant,
        )

        kwargs.update(_pd_kwargs(channel_config, common))

        return RayleighBlockFading(**kwargs)

    raise ValueError(f"No builder for channel config type: {type(channel_config)!r}")


def build_source(config: Config):
    # Only "binary" exists right now
    return BinarySource()


def build_mapsys(config: Config):

    mod = config.modulation

    points = mod.constellation.points

    if points is not None:
        points = np.array([complex(re, im) for re, im in points], dtype=np.complex64)

    constellation = Constellation(
        mod.type,
        mod.num_bits_per_symbol,
        normalize = mod.constellation.normalize,
        center = mod.constellation.center,
        points = points,
    )

    mapper = Mapper(
        constellation = constellation,
        return_indices = mod.mapper.return_indices,
    )

    demapper = Demapper(
        mod.demapper.demapping_method,
        constellation = constellation,
        hard_out = mod.demapper.hard_out,
    )

    return constellation, mapper, demapper


def build_waveform(config: Config, channel_model):
    """
    Wraps a raw channel_model (from build_channel) according to
    common.active_waveform.
    This is the ONLY place TimeChannel/OFDMChannel wrapping happens --
    build_channel never wraps.
    """

    common = config.common

    if config.common.active_waveform == "time":

        wf = config.waveforms.time

        if isinstance(channel_model, AWGN):
            # AWGN has no notion of a time-domain filter tail --
            # it's applied directly, no TimeChannel wrapping at all.
            return channel_model

        kwargs = dict(
            channel_model = channel_model,
            bandwidth = wf.bandwidth,
            num_time_samples = wf.num_time_samples,
            maximum_delay_spread = wf.maximum_delay_spread,
            l_min = wf.l_min,
            l_max = wf.l_max,
            normalize_channel = wf.normalize_channel,
            return_channel = wf.return_channel,
        )
        kwargs.update(_pd_kwargs(wf, common))

        return TimeChannel(**kwargs)

    if config.common.active_waveform == "ofdm":

        # Check the early-exit case FIRST
        if isinstance(channel_model, AWGN):
            return channel_model

        rg_cfg = config.waveforms.ofdm.resource_grid

        rg_kwargs = dict(
            num_ofdm_symbols = rg_cfg.num_ofdm_symbols,
            fft_size = rg_cfg.fft_size,
            subcarrier_spacing = rg_cfg.subcarrier_spacing,
            num_tx = rg_cfg.num_tx,
            num_streams_per_tx = rg_cfg.num_streams_per_tx,
            cyclic_prefix_length = rg_cfg.cyclic_prefix_length,
            num_guard_carriers = rg_cfg.num_guard_carriers,
            dc_null = rg_cfg.dc_null,
            pilot_pattern = rg_cfg.pilot_pattern,
            pilot_ofdm_symbol_indices = rg_cfg.pilot_ofdm_symbol_indices,
        )

        rg_kwargs.update(_pd_kwargs(rg_cfg, common))

        resource_grid = ResourceGrid(**rg_kwargs)

        ofdm_cfg = config.waveforms.ofdm

        ofdm_kwargs = dict(
            channel_model = channel_model,
            resource_grid = resource_grid,
            normalize_channel = ofdm_cfg.normalize_channel,
            return_channel = ofdm_cfg.return_channel,
        )

        ofdm_kwargs.update(_pd_kwargs(ofdm_cfg, common))

        return OFDMChannel(**ofdm_kwargs)

    raise ValueError(f"Unknown active_waveform: {config.common.active_waveform!r}")
