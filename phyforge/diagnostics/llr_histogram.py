"""
diagnostics/llr_histograms.py

Pre-export.py visual/diagnostic sanity check #4:
LLR distribution histograms per modulation / SNR.

Confirms:
  - LLR distributions actually separate 
  - sign/magnitude look sane
"""

import numpy as np
import matplotlib.pyplot as plt

from .ber_waterfall import build_awgn_config, MODULATIONS
from physys.runtime import PHYSys


SNR_POINTS_DB = [-10, 0, 10]

BATCH_SIZE = 20
NUM_SYMBOLS = 2000   # per (modulation, SNR) point -- LLR is one value
                     # per coded bit, so this gives
                     # batch_size * num_symbols * k LLRs per subplot

NUM_BINS = 80

OUTPUT_PATH = "llr_histograms.png"


def get_llrs(sys: PHYSys, ebno_db: float) -> np.ndarray:

    _, llr = sys.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=ebno_db)

    return llr.detach().cpu().numpy().flatten()


def main():
    fig, axes = plt.subplots(
        len(MODULATIONS), len(SNR_POINTS_DB),
        figsize=(5 * len(SNR_POINTS_DB), 4 * len(MODULATIONS)),
        squeeze=False,
    )

    for row, (mod_type, k) in enumerate(MODULATIONS):
        print(f"=== {mod_type}, k={k} ({2**k}-{mod_type.upper()}) ===")

        config = build_awgn_config(mod_type, k)
        sys = PHYSys(config)

        for col, ebno_db in enumerate(SNR_POINTS_DB):
            llrs = get_llrs(sys, ebno_db)

            n_nan = np.isnan(llrs).sum()
            n_inf = np.isinf(llrs).sum()
            if n_nan or n_inf:
                print(f"  EbNo={ebno_db:+d} dB  WARNING: {n_nan} NaN, {n_inf} Inf in LLRs")

            finite = llrs[np.isfinite(llrs)]

            ax = axes[row][col]
            ax.hist(finite, bins=NUM_BINS, color="tab:blue", alpha=0.75)
            ax.axvline(0, color="gray", linewidth=0.8)
            ax.set_title(f"{2**k}-{mod_type.upper()}, Eb/No={ebno_db:+d} dB", fontsize=9)
            ax.set_xlabel("LLR")
            ax.set_ylabel("count")
            ax.grid(True, alpha=0.3)

            print(
                f"  EbNo={ebno_db:+5.1f} dB  "
                f"mean|LLR|={np.abs(finite).mean():.2f}  "
                f"std={finite.std():.2f}"
            )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()