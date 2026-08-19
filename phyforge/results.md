1. Purpose

This document describes a synthetic Automatic Modulation Recognition (AMR) benchmark dataset generated using NVIDIA Sionna, covering three 3GPP TR 38.901 system-level channel environments (Rural Macro, Urban Macro, Urban Micro) across six modulation schemes and sixteen SNR levels. It documents the generation pipeline, the exact configuration used, the resulting file structure, and the verification checks performed on the output.

2. Generation Pipeline Overview

The dataset is produced by a custom pipeline (physys) built on top of Sionna, structured as: Source → Map → Channel → Demap, with the raw transmitted (x) and received (y) I/Q symbols captured directly from the channel stage rather than only the downstream bit/LLR outputs. Only the noisy received symbols (y) are exported as the dataset's feature tensor.

For each combination of (channel variant, modulation, SNR), the pipeline:

Builds a fresh PHYSys instance from a per-combination configuration (see Section 3).
Generates a fresh single-sector channel topology for every call (batch-size dependent, one user terminal, one base station).
Draws random bits, maps them to complex symbols (x), passes them through the channel to obtain noisy received symbols (y), and computes LLRs (bits/LLR are also produced but not used in this dataset).
Writes y into the HDF5 output file, batched to respect a maximum GPU batch size, alongside a one-hot modulation label and the SNR value used for that call.

This is looped over all 6 modulations × 16 SNR points × 1024 vectors, for each of the 3 channel variants, producing one HDF5 file per variant (Umi.h5, Uma.h5, Rma.h5).

3. Configuration Used

Channel model: 3GPP TR 38.901 system-level channel (Sionna's UMi/UMa/RMa classes), single-link (1 base-station antenna, 1 user-terminal antenna, single polarization, vertical polarization, downlink direction), carrier frequency 3.5 GHz.

Large-scale effects: Path loss and shadow fading are explicitly disabled for this dataset (disable_pathloss_and_shadowing: true). Only small-scale (multipath) fading from the channel model is present in the data. This isolates the fading/multipath character of each environment rather than mixing it with link-budget effects, and explains why the received signal energy stays in a narrow, comparable range (~1.5–2.0) across all three environments rather than differing by orders of magnitude.

Waveform domain: Time-domain (TimeChannel), not OFDM. Each vector has 1024 time samples, bandwidth 1 MHz, maximum delay spread 3 μs. The Sionna TimeChannel output is slightly longer than its input due to the filter tail; this implementation truncates the output back down to the requested vector length (1024), which discards a small amount of energy from the tail end of the last few symbols — a known, documented simplification rather than an oversight.

Modulations (6 total):

BPSK and QPSK: custom, manually specified unit-energy constellation points (not Sionna's built-in PSK).
16-QAM, 64-QAM, 256-QAM, 1024-QAM: Sionna's native QAM constellations (Gray-coded, non-normalized, non-centered).

SNR / noise: The dataset's "SNR" axis is actually Eb/N0 in dB, swept from −10 to +20 dB in steps of 2 dB (16 points), converted to noise power via Sionna's ebnodb2no() using each modulation's bits-per-symbol and a code rate of 1.0 (no forward error correction is modeled). Because this conversion depends on bits-per-symbol, the same nominal "SNR" label corresponds to a slightly different actual symbol-level SNR for different modulations — this is expected and standard for Eb/N0-referenced sweeps, but worth noting explicitly in case your teacher asks why energy levels aren't perfectly identical across modulations at a fixed SNR bin.

Per-modulation, per-SNR sampling: 1024 independent vectors of length 1024 complex samples each, generated in chunks of at most 64 vectors per forward pass (16 chunks) to keep memory usage bounded for high-order modulations (the "app" demapper materializes a tensor proportional to constellation size, which becomes large for 1024-QAM).

Topology: A new random single-sector topology (base station + 1 user terminal placement) is generated for every batch call rather than reused, consistent with treating each generated batch as an independent draw from the channel model's statistics for that environment.

4. Output File Structure

Each of the three files (Umi.h5, Uma.h5, Rma.h5) contains:

Data: complex64 array, shape (98304, 1024) — the noisy received I/Q symbol vectors.
Mods: float32 array, shape (98304, 6) — one-hot modulation label, order [BPSK, QPSK, 16QAM, 64QAM, 256QAM, 1024QAM].
SNRs: float32 array, shape (98304,) — the Eb/N0 value (dB) used to generate that row.

98304 = 6 modulations × 16 SNR levels × 1024 vectors. File size is approximately 712–713 MB per variant.

5. Verification Summary

The generated files were checked programmatically across five phases:

File Integrity: All three files match the expected size (~712.5–712.9 MB) and array shapes exactly. No NaN or Inf values were found in Data, Mods, or SNRs in any file.

Label Balance: All 98,304 Mods rows are strictly one-hot (values only 0/1, exactly one 1 per row). All 16 expected SNR values (−10 to +20 dB, step 2) are present, each with exactly 6,144 samples, and the SNR × modulation cross-tabulation confirms exactly 1,024 samples in every one of the 96 (16 SNR × 6 modulation) cells, for all three files. No dropped or duplicated batches.

Physical Plausibility (at 20 dB): Mean signal energy across modulations stays within a tight band per file (ratio of max to min energy: Rma 1.11×, Uma 1.12×, Umi 1.06×), consistent with unit-ish-energy constellations passed through comparable channel gain, as expected given path loss/shadowing are disabled.

Peak-to-Average Power Ratio (PAPR): PAPR increases with modulation order as theoretically expected (BPSK/QPSK lowest, 1024-QAM highest) for Rma and Uma. For Umi, 256-QAM and 1024-QAM PAPR values are within 0.02 dB of each other and reversed in order (6.19 dB vs 6.17 dB) — a negligible discrepancy attributable to finite-sample averaging (1024 vectors) rather than a generation defect, since the same reversal does not appear in the other two files and the magnitude is well within expected statistical noise for this sample size.

Constellation and Fading Checks: At high SNR (20 dB), constellation scatter plots show the expected clustered/rotated point structure (rotation from channel phase, not corruption); at low SNR (−10 dB), constellations correctly collapse into noise, as expected. Fading volatility, quantified via the coefficient of variation (CV) of per-vector signal amplitude, is consistently lower for Rma than for Uma/Umi across SNR levels, correctly reflecting Rma's calmer, more line-of-sight-dominated rural propagation versus the more volatile urban multipath environments — the channel models are behaving as physically distinct from one another, not producing near-identical statistics under different labels.

6. Overall Conclusion

Across all five verification phases, the three dataset files show no structural defects: correct sizes, correct shapes, zero corrupted values, perfectly balanced labels, and channel behavior that is physically plausible and correctly differentiated between environments. The one minor anomaly noted (Umi's 256-QAM/1024-QAM PAPR ordering) is a small-sample statistical artifact, not a data integrity issue.