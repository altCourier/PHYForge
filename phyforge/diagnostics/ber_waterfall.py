"""
diagnostics/ber_waterfall.py

Pre-export.py visual/diagnostic sanity check #1:
BER vs Eb/No waterfall curves, per modulation, AWGN only, overlaid
against theoretical closed-form BER for QAM/PAM.

Revised to use sionna.phy.utils.sim_ber for the Monte-Carlo accumulation
loop instead of a hand-rolled while-loop. sim_ber batches internally and
stops per-SNR-point once num_target_bit_errors is reached (or max_mc_iter
is hit), which is exactly what the old empirical_ber() did by hand.

The "low confidence" reliability flag (was the old code's own bookkeeping)
is reconstructed here via sim_ber's `callback` hook, since sim_ber/PlotBER
only return the final (ber, bler) tensors -- they don't expose per-point
"did we actually hit the target" status on their own.
"""

import math

import numpy as np
import matplotlib.pyplot as plt
import torch

from sionna.phy.utils import sim_ber

from physys.schema import (
    Config, CommonConfig, BinarySourceConfig,
    ModulationConfig, MapperConfig, DemapperConfig, ConstellationConfig,
    ChannelsConfig, AWGNChannelConfig, TDLChannelConfig,
    SystemLevelChannelConfig, RayleighBlockFadingConfig,
    WaveformsConfig, TimeWaveformConfig, OFDMWaveformConfig, ResourceGridConfig,
    SweepConfig, EbNoRangeConfig,
)

from physys.runtime import PHYSys


# ---------------------------------------------------------------------
# Config to test: (type, num_bits_per_symbol) pairs
# ---------------------------------------------------------------------
MODULATIONS = [
    ("pam", 1),   # BPSK
    ("qam", 2),   # QPSK
    ("qam", 4),   # 16-QAM
    ("qam", 6),   # 64-QAM
    ("qam", 8),   # 256-QAM
    ("qam", 10),  # 1024-QAM
]

EBNO_DB_RANGE = list(range(-10, 11, 2))  # -10 dB to 10 dB, step 2

# Adaptive stopping: accumulate batches until we've SEEN this many bit
# errors (so low-SNR points finish fast and high-SNR points don't run
# on a statistically useless handful of errors), or give up.
TARGET_ERRORS = 200
BATCH_SIZE = 50
NUM_SYMBOLS = 500
MAX_ITERATIONS = 200  # hard ceiling per Eb/No point, per modulation

OUTPUT_PATH = "ber_waterfall_awgn.png"


# ---------------------------------------------------------------------
# Theoretical closed-form BER, Gray-coded, AWGN.
# Q(x) = 0.5 * erfc(x / sqrt(2))
# ---------------------------------------------------------------------

def _q(x):
    return 0.5 * math.erfc(x / math.sqrt(2))


def theoretical_ber(mod_type: str, k: int, ebno_db: float) -> float:
    ebno_lin = 10 ** (ebno_db / 10.0)

    if k == 1:
        # BPSK, same formula whether you call it 1-bit "pam" or "qam"
        return _q(math.sqrt(2 * ebno_lin))

    m = 2 ** k

    if mod_type == "qam":
        # Square M-QAM, Gray-coded
        arg = math.sqrt(3 * k * ebno_lin / (m - 1))
        return (4.0 / k) * (1 - 1 / math.sqrt(m)) * _q(arg)

    if mod_type == "pam":
        arg = math.sqrt(6 * k * ebno_lin / (m ** 2 - 1))
        return (2.0 * (m - 1) / (m * k)) * _q(arg)

    raise ValueError(f"No theoretical BER formula for mod_type={mod_type!r}")


# ---------------------------------------------------------------------
# Config construction (mirrors runtime.py's own __main__ smoke test --
# TDL/system_level/ofdm fields below are dummy placeholders required
# by the dataclasses but never exercised, since active_channel=awgn
# short-circuits to the AWGN identity passthrough in build_waveform).
# ---------------------------------------------------------------------

def build_awgn_config(mod_type: str, k: int) -> Config:
    modulation = ModulationConfig(
        type=mod_type,
        num_bits_per_symbol=k,
        mapper=MapperConfig(),
        demapper=DemapperConfig(),
        constellation=ConstellationConfig(),
    )

    channels = ChannelsConfig(
        awgn=AWGNChannelConfig(),
        tdl=TDLChannelConfig(model="A", delay_spread=100e-9, carrier_frequency=3.5e9),
        system_level=SystemLevelChannelConfig(variant="umi", carrier_frequency=3.5e9),
        rayleigh_block_fading=RayleighBlockFadingConfig(),
    )

    waveforms = WaveformsConfig(
        time=TimeWaveformConfig(bandwidth=1e6, num_time_samples=NUM_SYMBOLS),
        ofdm=OFDMWaveformConfig(
            resource_grid=ResourceGridConfig(
                num_ofdm_symbols=14, fft_size=64, subcarrier_spacing=15e3,
            )
        ),
    )

    sweep = SweepConfig(
        ebno_db=EbNoRangeConfig(start=EBNO_DB_RANGE[0], stop=EBNO_DB_RANGE[-1], step=2.0),
        batch_size=BATCH_SIZE,
        num_symbols=NUM_SYMBOLS,
    )

    return Config(
        common=CommonConfig(active_channel="awgn", active_waveform="time"),
        source=BinarySourceConfig(),
        modulation=modulation,
        channels=channels,
        waveforms=waveforms,
        sweep=sweep,
    )


# ---------------------------------------------------------------------
# Empirical BER via sim_ber, with a reliability mask reconstructed
# through the callback hook (sim_ber itself doesn't return per-point
# "did we hit the target" status -- only the final ber/bler tensors).
# ---------------------------------------------------------------------

def empirical_ber_sweep(sys: PHYSys, ebno_db_range):
    """
    Runs sim_ber once across the whole Eb/No range for a given built
    PHYSys instance. Returns (ber: np.ndarray, reliable: np.ndarray[bool]).

    reliable[i] is False if that SNR point hit MAX_ITERATIONS without
    ever accumulating TARGET_ERRORS bit errors -- i.e. the same
    "low confidence" condition the old hand-rolled loop tracked itself.
    """

    def mc_fun(batch_size, ebno_db):
        bits, llr = sys.generate(
            batch_size=batch_size, num_symbols=NUM_SYMBOLS, ebno_db=ebno_db
        )
        return bits, llr

    # Track, per SNR index, whether TARGET_ERRORS was reached by the
    # time the last allowed Monte-Carlo iteration ran.
    n_points = len(ebno_db_range)
    reliable = [False] * n_points

    def callback(mc_iter, snr_idx, ebno_dbs, bit_errors, block_errors, nb_bits, nb_blocks):
        idx = int(snr_idx)
        reached = bit_errors[idx].item() >= TARGET_ERRORS
        hit_ceiling = mc_iter >= MAX_ITERATIONS
        if reached or hit_ceiling:
            reliable[idx] = reached
        return None  # let sim_ber's own early-stop/target logic keep driving

    ebno_tensor = torch.tensor(ebno_db_range, dtype=torch.float32)

    ber, _bler = sim_ber(
        mc_fun,
        ebno_tensor,
        batch_size=BATCH_SIZE,
        max_mc_iter=MAX_ITERATIONS,
        soft_estimates=True,          # llr is a logit; sim_ber hard-decides internally
        num_target_bit_errors=TARGET_ERRORS,
        early_stop=False,             # we want every point in the fixed range, not just until error-free
        callback=callback,
        verbose=False,
    )

    ber = ber.numpy() if hasattr(ber, "numpy") else np.asarray(ber)
    return ber, np.array(reliable, dtype=bool)


# ---------------------------------------------------------------------
# Main sweep + plot
# ---------------------------------------------------------------------

def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()

    for ax, (mod_type, k) in zip(axes, MODULATIONS):
        print(f"\n=== {mod_type}, k={k} ({2**k}-{mod_type.upper()}) ===")

        config = build_awgn_config(mod_type, k)
        sys = PHYSys(config)

        emp_ber, reliable_mask = empirical_ber_sweep(sys, EBNO_DB_RANGE)
        theo_ber = np.array([theoretical_ber(mod_type, k, db) for db in EBNO_DB_RANGE])

        for db, ber, t_ber, rel in zip(EBNO_DB_RANGE, emp_ber, theo_ber, reliable_mask):
            flag = "" if rel else "  (LOW CONFIDENCE -- too few errors)"
            print(f"  EbNo={db:+5.1f} dB  empirical={ber:.3e}  theory={t_ber:.3e}{flag}")

        ax.semilogy(EBNO_DB_RANGE, theo_ber, "-", label="theory", color="black")
        ax.semilogy(
            np.array(EBNO_DB_RANGE)[reliable_mask], emp_ber[reliable_mask],
            "o", label="empirical", color="tab:blue",
        )
        if (~reliable_mask).any():
            ax.semilogy(
                np.array(EBNO_DB_RANGE)[~reliable_mask], emp_ber[~reliable_mask],
                "x", label="empirical (low conf.)", color="tab:red",
            )

        ax.set_title(f"{2**k}-{mod_type.upper()} (k={k})")
        ax.set_xlabel("Eb/No (dB)")
        ax.set_ylabel("BER")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()