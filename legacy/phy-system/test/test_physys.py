from physys import PHYSys

import matplotlib.pyplot as plt

if __name__ == "__main__":

    physys = PHYSys(
        source_type="binary",
        modulation="16qam",
        channel_type="awgn",
    )

    out = physys(
        batch_size=8,
        num_symbols=100,
        ebno_db=10.0,
    )

    print("=== PHYSys single-shot test ===")
    for name, tensor in out.items():
        print(f"{name:9s} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    bits = out["bits"]
    bits_hat = out["bits_hat"]
    ber = (bits != bits_hat).float().mean().item()
    print(f"\nBER @ 10 dB EbNo: {ber:.4f}")

# ---------------------------------------------------------------------------
# 1) Constellation scatter: ideal points vs. transmitted vs. received (noisy)
# ---------------------------------------------------------------------------

physys = PHYSys(
    source_type="binary",
    modulation="16qam",
    channel_type="awgn",
)

out = physys(batch_size=1, num_symbols=2000, ebno_db=12.0)

ideal_points = physys.constellation.points.detach().cpu().numpy().flatten()
x = out["x"].detach().cpu().numpy().flatten()
y = out["y"].detach().cpu().numpy().flatten()

fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y.real, y.imag, s=6, alpha=0.3, label="received (y, noisy)")
ax.scatter(x.real, x.imag, s=10, alpha=0.6, label="transmitted (x)")
ax.scatter(ideal_points.real, ideal_points.imag, s=120, marker="x",
           color="red", linewidths=2, label="ideal constellation points")
ax.set_xlabel("In-phase")
ax.set_ylabel("Quadrature")
ax.set_title("16-QAM constellation @ 12 dB EbNo")
ax.axhline(0, color="gray", linewidth=0.5)
ax.axvline(0, color="gray", linewidth=0.5)
ax.legend()
ax.set_aspect("equal")
fig.tight_layout()
fig.savefig("constellation_scatter.png", dpi=150)
plt.show()

# ---------------------------------------------------------------------------
# 2) BER vs EbNo curve
# ---------------------------------------------------------------------------

ebno_db_range = [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]
ber_values = []

for ebno_db in ebno_db_range:
    res = physys(batch_size=64, num_symbols=500, ebno_db=ebno_db)
    ber = (res["bits"] != res["bits_hat"]).float().mean().item()
    ber_values.append(ber)
    print(f"EbNo={ebno_db:5.1f} dB -> BER={ber:.5f}")

fig2, ax2 = plt.subplots(figsize=(6, 4.5))
ax2.semilogy(ebno_db_range, ber_values, marker="o")
ax2.set_xlabel("Eb/N0 (dB)")
ax2.set_ylabel("BER (log scale)")
ax2.set_title("16-QAM over AWGN: empirical BER curve")
ax2.grid(True, which="both", linestyle="--", alpha=0.5)
fig2.tight_layout()
fig2.savefig("ber_curve.png", dpi=150)
plt.show()