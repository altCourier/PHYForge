"""
diagnostics/constellation_scatter.py

Pre-export.py visual/diagnostic sanity check #2:
Constellation scatter plots of y at a few SNR points, one plot per
modulation order.

Confirms:
  - the noise cloud actually shrinks as SNR increases
  - cluster geometry matches the modulation
"""

import numpy as np
import matplotlib.pyplot as plt

from sionna.phy.utils import ebnodb2no

from .ber_waterfall import build_awgn_config, MODULATIONS
from physys.runtime import PHYSys


SNR_POINTS_DB = [-10, 0, 10]  # low / mid / high end of your actual sweep

NUM_POINTS = 3000  # scatter points per SNR point -- plenty for cluster
                    # shape, cheap enough not to need batching

OUTPUT_PATH = "constellation_scatter.png"


def get_xy(sys: PHYSys, ebno_db: float, num_points: int):
    """
    Runs Source -> Map -> Channel (AWGN only, no demap) and returns
    (x, y) as flat numpy complex arrays.
    """
    k = sys.config.modulation.num_bits_per_symbol

    bits = sys.source([1, num_points * k])
    x = sys.mapper(bits)  # [1, num_points], complex

    no = ebnodb2no(ebno_db, num_bits_per_symbol=k, coderate=sys._coderate)
    y = sys._handle(x, no)  # AWGN identity passthrough -- see build_waveform

    x_np = x.detach().cpu().numpy().flatten()
    y_np = y.detach().cpu().numpy().flatten()
    return x_np, y_np


def get_ideal_points(sys: PHYSys):
    """
    Reference constellation points, for overlay. Attribute name
    assumed to be `.points` per Sionna's Constellation -- if this
    doesn't match your installed version, this will raise and you'll
    need to adjust the attribute name below.
    """
    points = sys.constellation.points
    return points.detach().cpu().numpy().flatten()


def main():
    for mod_type, k in MODULATIONS:
        print(f"=== {mod_type}, k={k} ({2**k}-{mod_type.upper()}) ===")

        config = build_awgn_config(mod_type, k)
        sys = PHYSys(config)

        ideal = get_ideal_points(sys)

        fig, axes = plt.subplots(1, len(SNR_POINTS_DB), figsize=(5 * len(SNR_POINTS_DB), 5))
        if len(SNR_POINTS_DB) == 1:
            axes = [axes]

        for ax, ebno_db in zip(axes, SNR_POINTS_DB):
            _, y = get_xy(sys, ebno_db, NUM_POINTS)

            ax.scatter(y.real, y.imag, s=4, alpha=0.25, color="tab:blue", label="received (y)")
            ax.scatter(
                ideal.real, ideal.imag,
                marker="x", s=80, color="black", linewidths=1.5,
                label="ideal constellation",
            )

            ax.set_title(f"Eb/No = {ebno_db:+d} dB")
            ax.set_xlabel("I")
            ax.set_ylabel("Q")
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.axvline(0, color="gray", linewidth=0.5)
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")

        fig.suptitle(f"{2**k}-{mod_type.upper()} (k={k}) -- received constellation vs SNR")
        fig.tight_layout()

        out_path = f"{mod_type}_{2**k}_constellation.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()