"""
diagnostics/topology_check.py

Pre-export.py visual/diagnostic sanity check #3:
channel_model.show_topology() for each system-level variant (UMi/UMa/RMa).

Confirms:
  - UE drops actually land in a sane sector around the BS
  - no degenerate BS-at-origin-with-everything-on-top-of-it bugs

NOT a pytest test -- eyeball it. Sionna only exposes show_topology()
on system-level channel models (UMi/UMa/RMa), so this script is the
only one of the four diagnostics that touches that channel type.

Per the runtime docs: topology tensors are only populated inside
generate(), not in __init__ -- so this script calls sim.generate()
once (with a throwaway batch) purely to trigger _maybe_set_topology()
before calling show_topology().
"""

import matplotlib.pyplot as plt

from physys.schema import (
    Config, CommonConfig, BinarySourceConfig,
    ModulationConfig, MapperConfig, DemapperConfig, ConstellationConfig,
    ChannelsConfig, AWGNChannelConfig, TDLChannelConfig,
    SystemLevelChannelConfig, RayleighBlockFadingConfig, PanelArrayConfig,
    WaveformsConfig, TimeWaveformConfig, OFDMWaveformConfig, ResourceGridConfig,
    SweepConfig, EbNoRangeConfig,
)

from physys.runtime import PHYSys


# ---------------------------------------------------------------------
# Variants to check. carrier_frequency chosen per 3GPP-typical bands
# for each scenario; adjust if your actual deployment differs.
# ---------------------------------------------------------------------
VARIANTS = [
    ("umi", 3.5e9),
    ("uma", 3.5e9),
    ("rma", 700e6),
]

BATCH_SIZE = 16   # number of independent topology drops to plot
NUM_SYMBOLS = 64  # only needs to be non-degenerate for generate()'s
                   # throwaway call -- topology itself doesn't depend
                   # on this

EBNO_DB = 10.0    # arbitrary -- topology doesn't depend on Eb/No


def build_system_level_config(variant: str, carrier_frequency: float) -> Config:
    modulation = ModulationConfig(
        type="qam",
        num_bits_per_symbol=2,  # QPSK -- irrelevant to topology, just
                                 # needs to be valid so generate() runs
        mapper=MapperConfig(),
        demapper=DemapperConfig(),
        constellation=ConstellationConfig(),
    )

    system_level = SystemLevelChannelConfig(
        variant=variant,
        carrier_frequency=carrier_frequency,
        bs_array=PanelArrayConfig(),
        ut_array=PanelArrayConfig(antenna_pattern="omni"),
    )

    channels = ChannelsConfig(
        awgn=AWGNChannelConfig(),
        tdl=TDLChannelConfig(model="A", delay_spread=100e-9, carrier_frequency=carrier_frequency),
        system_level=system_level,
        rayleigh_block_fading=RayleighBlockFadingConfig(),
    )

    # active_waveform must name a WaveformsConfig field -- "time" is
    # the simplest valid choice here, and it's never actually
    # exercised for the topology check itself, only for generate()'s
    # one throwaway call.
    waveforms = WaveformsConfig(
        time=TimeWaveformConfig(bandwidth=1e6, num_time_samples=NUM_SYMBOLS),
        ofdm=OFDMWaveformConfig(
            resource_grid=ResourceGridConfig(
                num_ofdm_symbols=14, fft_size=64, subcarrier_spacing=15e3,
            )
        ),
    )

    sweep = SweepConfig(
        ebno_db=EbNoRangeConfig(start=EBNO_DB, stop=EBNO_DB, step=2.0),
        batch_size=BATCH_SIZE,
        num_symbols=NUM_SYMBOLS,
    )

    return Config(
        common=CommonConfig(active_channel="system_level", active_waveform="time"),
        source=BinarySourceConfig(),
        modulation=modulation,
        channels=channels,
        waveforms=waveforms,
        sweep=sweep,
    )


def main():
    for variant, fc in VARIANTS:
        print(f"=== {variant.upper()} @ {fc/1e9:.2f} GHz ===")

        config = build_system_level_config(variant, fc)
        sys = PHYSys(config)

        # Populate topology -- generate() calls _maybe_set_topology()
        # internally before doing anything else. We don't care about
        # the (bits, llr) it returns here.
        sys.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=EBNO_DB)

        # Sionna's show_topology() draws to the current matplotlib
        # figure/axes -- capture whatever it produces and save it.
        sys._channel_model.show_topology()

        out_path = f"topology_{variant}.png"
        plt.gcf().savefig(out_path, dpi=150)
        plt.close("all")
        print(f"  saved: {out_path}")

        # Sanity flag: with num_ut hardcoded to 1 in
        # _maybe_set_topology, every batch element is a single-UT
        # drop -- if you see UEs clustered exactly at the BS location,
        # or identical positions repeated across the batch, that's
        # the degenerate bug this script exists to catch, not
        # expected behavior.


if __name__ == "__main__":
    main()