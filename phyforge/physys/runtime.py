"""
runtime.py

Orchestrates the objects built.py hands back into the actual
Source -> Map -> Channel -> Demap pipeline from ARCHITECTURE.md's
pipeline diagram.

This file is where two of ARCHITECTURE.md's OPEN QUESTIONS stop being
hypothetical, since generate() can't run without *some* answer. Both
are given a provisional default here, flagged loudly, NOT quietly
resolved -- they should still be argued with:

1. TDL filter-tail handling (Open Question 1). Default here:
   TRUNCATE the TimeChannel output back to num_time_samples. This is
   the "simple but discards energy from the last few symbols" option
   the doc names, not a considered pick. See _apply_time_channel().

2. system_level topology (implicit open question -- not numbered in
   the doc, but required to make UMi/UMa/RMa usable at all). Default
   here: a FRESH topology is generated on every generate() call, never
   cached across calls, since topology tensors are batch_size-shaped
   and batch_size can differ between calls (e.g. sweep vs. one-off).
   See PHYSys._maybe_set_topology().

Also assumes Open Question 3 (FEC) is resolved as "not included yet".
coderate is hardcoded to 1.0 for the Eb/No -> No conversion below.

"""

import sionna.phy

from sionna.phy.utils import ebnodb2no
from sionna.phy.channel import AWGN
from sionna.phy.channel import gen_single_sector_topology

from .schema import Config, SystemLevelChannelConfig
from .builders import build_source, build_mapsys, build_channel, build_waveform


class PHYSys:
    """
    Same shape as the pre-refactor PHYSys class (per ARCHITECTURE.md):
    Source -> Map -> Channel -> Demap, exposed as generate() /
    generate_sweep().

    Built once per Config -- source/mapper/demapper/channel/waveform
    don't change between calls; only batch_size/num_symbols/ebno_db
    vary per generate() call.
    """

    def __init__(self, config: Config):
        self.config = config

        self.source = build_source(config)
        self.constellation, self.mapper, self.demapper = build_mapsys(config)

        # build_channel returns the RAW channel model (AWGN/TDL/
        # system-level/RayleighBlockFading); build_waveform wraps it
        # per common.active_waveform. Keep both: the raw model is
        # needed directly for set_topology() below, since that method
        # lives on the UMi/UMa/RMa instance, not on the TimeChannel/
        # OFDMChannel wrapper around it.
        self._channel_model = build_channel(config)
        self._handle = build_waveform(config, self._channel_model)

        self._coderate = 1.0  # no FEC yet

    # ---- topology (system_level only) ----

    def _active_channel_config(self):
        return getattr(self.config.channels, self.config.common.active_channel)

    def _maybe_set_topology(self, batch_size: int):
        """
        AWGN/TDL/RayleighBlockFading don't have set_topology() at all;
        UMi/UMa/RMa require it before first use. Regenerated every
        call rather than cached once in __init__, because topology
        tensors are shaped by batch_size and different generate()
        calls (or sweep steps) may use different batch sizes.
        """
        channel_config = self._active_channel_config()

        if not isinstance(channel_config, SystemLevelChannelConfig):
            return

        topology = gen_single_sector_topology(
            batch_size=batch_size,
            num_ut=1,
            scenario=channel_config.variant,
        )
        self._channel_model.set_topology(*topology)

    # ---- shape bookkeeping (the one place this project writes non-Sionna math) ----

    def _apply_time_channel(self, x, no, num_symbols: int):
        """
        Reshape flat [batch, num_symbols] mapper output into whatever
        the active waveform's handle() expects, call it, and reshape
        back down to [batch, num_symbols] for the demapper.

        AWGN's handle() (see build_waveform) is the identity passthrough
        for shape purposes -- it takes/returns [batch, num_symbols]
        directly, no reshape needed.
        """

        if isinstance(self._channel_model, AWGN):
            return self._handle(x, no)

        if self.config.common.active_waveform == "time":
            tdl_cfg = self._active_channel_config()
            num_tx_ant = getattr(tdl_cfg, "num_tx_ant", 1)

            # [batch, num_symbols] -> [batch, 1 (num_tx), num_tx_ant, num_time_samples]
            # ASSUMPTION: num_symbols == waveforms.time.num_time_samples.
            # Nothing currently enforces this at the schema level --
            # flagging as a gap, not a design decision.
            batch_size = x.shape[0]
            x_time = tf_reshape_for_time_channel(x, batch_size, num_tx_ant)

            y_time = self._handle(x_time, no)

            # TimeChannel's output is l_max - l_min samples LONGER than
            # its input (ARCHITECTURE.md Open Question 1). Default:
            # truncate back to num_symbols. This silently discards
            # energy belonging to the last few symbols -- the doc
            # flags this tradeoff explicitly, it isn't fixed here.
            y = y_time[..., :num_symbols]
            return y.reshape([batch_size, num_symbols])

        if self.config.common.active_waveform == "ofdm":
            # OFDMChannel operates on a full resource grid, not a flat
            # symbol stream -- mapping [batch, num_symbols] onto
            # [batch, num_tx, num_streams, num_ofdm_symbols, fft_size]
            # (data REs only, skipping pilots/guards/DC) is nontrivial
            # shape bookkeeping this draft does NOT yet implement.
            raise NotImplementedError(
                "OFDM resource-grid mapping is not yet implemented in "
                "runtime.py -- needs its own reshape helper analogous "
                "to the time-domain one above."
            )

        raise ValueError(
            f"Unknown active_waveform: {self.config.common.active_waveform!r}"
        )

    # ---- public API ----

    def generate(self, batch_size: int, num_symbols: int, ebno_db: float):
        """
        One Source -> Map -> Channel -> Demap pass.
        Returns (bits, llr) so callers (export.py / tests) can compute
        BER/BLER themselves rather than this file owning that metric.
        """

        self._maybe_set_topology(batch_size)

        num_bits_per_symbol = self.config.modulation.num_bits_per_symbol

        no = ebnodb2no(
            ebno_db,
            num_bits_per_symbol = num_bits_per_symbol,
            coderate = self._coderate,
        )

        bits = self.source([batch_size, num_symbols * num_bits_per_symbol])
        x = self.mapper(bits)  # [batch_size, num_symbols]

        y = self._apply_time_channel(x, no, num_symbols)

        llr = self.demapper(y, no)

        return bits, llr

    def generate_sweep(self):
        """
        Runs generate() once per Eb/No point in config.sweep.ebno_db,
        using config.sweep.batch_size / num_symbols for every point.
        Returns a dict keyed by ebno_db value -> (bits, llr), so
        export.py decides how to serialize it rather than this file
        picking a format.
        """

        sweep = self.config.sweep
        rng = sweep.ebno_db

        results = {}

        ebno_db = rng.start
        while ebno_db <= rng.stop:
            results[ebno_db] = self.generate(
                batch_size=sweep.batch_size,
                num_symbols=sweep.num_symbols,
                ebno_db=ebno_db,
            )
            ebno_db += rng.step

        return results


def tf_reshape_for_time_channel(x, batch_size: int, num_tx_ant: int):
    """
    [batch, num_symbols] -> [batch, 1, num_tx_ant, num_symbols]
    (num_tx is always 1 -- PHYSys doesn't model multi-user uplink/
    downlink multiplexing, per ARCHITECTURE.md's MIMO non-goal).

    Split out as its own function because it's the one piece of this
    branch that's PHYSys's own code rather than a Sionna call --
    ARCHITECTURE.md calls this out explicitly as acceptable ("shape
    bookkeeping, not signal processing").
    """
    return x.reshape([batch_size, 1, num_tx_ant, -1])
