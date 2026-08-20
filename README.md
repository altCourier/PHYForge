# PHYForge

**One config file drives an entire Sionna physical-layer pipeline — source, mapping, channel, waveform, and sweep — with no code changes between experiments.**

## Why

[Sionna](https://github.com/NVlabs/sionna) gives you the building blocks for PHY-layer simulation — sources, mappers, channel models, FEC — but every project ends up writing the same kind of glue code to wire them together: instantiate a source, instantiate a mapper, instantiate a channel, run a sweep. That glue code tends to be rewritten per-project, and switching something fundamental (e.g. an AWGN link to a 3GPP multipath model) usually means editing several files, not one.

`physys` (the package in this repo) is that glue code, written once, driven entirely by a single config file. Every channel and waveform it supports is fully parameterized in the config at all times — switching experiments means changing two fields, `active_channel` and `active_waveform`, not editing or deleting config blocks:

```json
{
  "common": {
    "active_channel": "tdl",
    "active_waveform": "time"
  },
  "channels": {
    "awgn": {},
    "tdl": {
      "model": "A",
      "delay_spread": 100e-9,
      "carrier_frequency": 140e9
    },
    "system_level": { "variant": "umi", "carrier_frequency": 3.5e9 },
    "rayleigh_block_fading": {}
  }
}
```

Change `active_channel` from `"tdl"` to `"system_level"` (with `variant: "uma"` or `"rma"`), and `physys` builds the right Sionna object graph for you — no code edits. Note that `channels.tdl` and `channels.system_level` (and both `waveforms.time`/`waveforms.ofdm`) are required keys in the config regardless of which one is active — see the API docs below for why.

## How it works (short version)

`config.json`, at the repo root, holds every parameter for every supported channel and waveform at once. Two fields under `common` — `active_channel` and `active_waveform` — name which blocks are actually used for a given run; everything else sits inert but ready. A builder layer reads the resolved config and constructs the matching Sionna object graph (e.g. `TDL` + `TimeChannel` for `active_channel: "tdl"`, vs. two `PanelArray`s + topology + `UMi`/`UMa`/`RMa` for `active_channel: "system_level"`).

- See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.
- See [physys/README.md](./physys/README.md) for the complete API reference — every module (`loader`, `schema`, `builders`, `runtime`, `export`, `amr_dataset`, `cli`), every config field with its default, and the known edge cases. Start there if you're wiring up a new config or calling `PHYSys` directly from Python.

## Quick usage

The CLI (`physys.cli`) has three subcommands — `run`, `sweep`, and `dataset`. Full flags and behavior are documented in [physys/README.md](./physys/README.md#7-physyscli); the short version:

```bash
# One run at a fixed Eb/N0, bits+LLR only
python -m physys.cli run -c config.json -o run.h5 --ebno-db 5.0

# Sweep config.sweep.ebno_db, bits+LLR only
python -m physys.cli sweep -c config.json -o sweep.h5

# Same sweep, also capturing raw tx/rx symbols (x/y) for AMR-style work
python -m physys.cli sweep -c config.json -o sweep.h5 --iq

# Drive the built-in AMR benchmark generator (config.amr_dataset block required)
python -m physys.cli dataset -c config.json -o out_dir/
```

The `dataset` subcommand and the `sweep --iq` command produce different file formats for different purposes — `dataset` writes the purpose-built, incrementally-growable `Umi.h5`/`Uma.h5`/`Rma.h5` benchmark format keyed by `config.amr_dataset` (multiple modulations and SNR points in one file, one-hot labeled); `sweep --iq` writes one `sweep.h5` per config, covering a single modulation's Eb/N0 sweep. The `make dataset` workflow below uses the latter, one config per modulation. See [physys/README.md](./physys/README.md#5-physysexport) if you want to switch to the former.

## Configuring a run

`config.json` carries parameters for *every* channel, waveform, and (optionally) the AMR benchmark generator at once — see the field-by-field reference in [physys/README.md](./physys/README.md#2-physysschema). A few things worth knowing before editing it:

- **Keys starting with `_` are comments, not config.** `_comment`/`_description` (or any `_`-prefixed key) at any level is stripped before validation, so you can document a value inline without it becoming a stray/unexpected field. The sample `config.json` at the repo root uses this throughout — feel free to leave those comments in place, or add your own.
- **All blocks are required, active or not.** `channels.tdl`, `channels.system_level`, `waveforms.time`, and `waveforms.ofdm` must all be present in the file even if `common.active_channel`/`common.active_waveform` only point at one of them. `active_channel`/`active_waveform` are pure selectors, dispatched on at build time — not inherited defaults, and not validated against real channel/waveform names until the whole config tree is assembled.
- **`sweep.num_symbols` must equal `waveforms.time.num_time_samples` when `active_waveform` is `"time"`.** Enforced at parse time (`ValueError` if it doesn't hold) rather than surfacing later as a shape error. In the sample config both are `1024` — keep them in sync if you change one.
- **Runtime-only values aren't in the file at all.** The noise power `no` (derived from `sweep.ebno_db` / `amr_dataset.snr_db` via Sionna's `ebnodb2no`), the source/batch shape (derived from `sweep.batch_size` × `sweep.num_symbols` × `modulation.num_bits_per_symbol`), and the channel object handed to `TimeChannel`/`OFDMChannel` are all computed per-call — you won't find them inside `channels.*`/`waveforms.*` because they can't be static per-config.
- **`system_level.o2i_model` vs. `average_street_width`/`average_building_height`.** Both are always present in the block for a uniform shape across variants, but only one pair is actually used, depending on `variant`: `o2i_model` for `"umi"`/`"uma"`, the street-width/building-height pair for `"rma"`. Setting the "wrong" one for a given variant validates fine but is silently ignored by the builder.

### `sweep` vs. `amr_dataset` — which fields actually get used

The top-level `modulation`/`channels`/`waveforms`/`sweep` blocks and the `amr_dataset` block aren't independent — **when `amr_dataset` is present and you run the `dataset` command, it overrides several of the top-level fields rather than sitting next to them:**

| Command | What it reads |
|---|---|
| `run` / `sweep` (with or without `--iq`) | `common.active_channel`/`active_waveform`, `modulation`, the matching `channels.*` block, the matching `waveforms.*` block, `sweep` — exactly as written. |
| `dataset` (CLI) / `make amr-dataset` | `amr_dataset.*` for the sweep axes (`variants`, `snr_db`, `num_vectors`, `vector_len`, `modulations`, `mapper`, `demapper`, `max_batch_size`, `disable_pathloss_and_shadowing`). For every `(variant, modulation)` cell it generates, it **overrides** `common.active_channel` to `"system_level"`, `channels.system_level.variant`, `modulation` (built from the matching `amr_dataset.modulations` entry, *not* the top-level `modulation` block), `waveforms.time.num_time_samples`, and `sweep.batch_size`/`sweep.num_symbols`. It also only ever *narrows* `channels.system_level.enable_pathloss`/`enable_shadow_fading` — with `disable_pathloss_and_shadowing: true` (the default), both are forced off regardless of what's set under `channels.system_level`. |

Practically: if you're running `make amr-dataset`, editing the top-level `modulation` block, `sweep.batch_size`/`num_symbols`, or `channels.system_level.variant` has **no effect** on the output — all of those are overridden per-cell from `amr_dataset`. Edit `amr_dataset.modulations`, `amr_dataset.variants`, and `amr_dataset.num_vectors`/`vector_len` instead. The top-level blocks still have to be present and well-formed (they're required keys, and `run`/`sweep`/`sweep --iq` do use them as-is), but for the AMR path they're "must validate," not "will be used."

## Generating data (Makefile)

The root `Makefile` wraps two independent generation paths. Don't confuse the Makefile target named `dataset` with the CLI subcommand named `dataset` — they read different config paths and write different files:

| Makefile target | Wraps | Reads | Produces |
|---|---|---|---|
| `make dataset` | `physys.cli sweep --iq`, once per modulation folder | `modulations/<mod>/config.json` (top-level `sweep`/`modulation`/etc.) | `modulations/<mod>/sweep.h5` |
| `make amr-dataset` | `physys.cli dataset`, once | root `config.json`'s `amr_dataset` block | `data/Umi.h5`, `data/Uma.h5`, `data/Rma.h5` |

### Per-modulation sweeps (`make dataset`)

Each modulation lives in its own folder under `modulations/`, each with its own `config.json`:

```
modulations/
├── BPSK/config.json
├── QPSK/config.json
├── 16-QAM/config.json
├── 64-QAM/config.json
├── 256-QAM/config.json
└── 1024-QAM/config.json
```

**List available targets:**

```bash
make help
```

**Generate `sweep.h5` for every modulation:**

```bash
make dataset
```

This first runs `check-configs` to confirm every modulation folder actually has a `config.json`, then builds any `sweep.h5` that's missing or older than its config (or older than `physys/cli.py`, which is also a prerequisite). Configs that haven't changed are skipped — only stale or missing outputs get regenerated.

**Generate a single modulation:**

```bash
make BPSK
```

or equivalently:

```bash
make modulations/BPSK/sweep.h5
```

**Generate a specific subset without editing the Makefile:**

```bash
make dataset MODULATIONS="BPSK QPSK"
```

**Remove all generated `sweep.h5` files:**

```bash
make clean
```

You'll be shown which files would be deleted and asked to confirm before anything is removed.

### AMR benchmark (`make amr-dataset`)

```bash
make amr-dataset
```

Runs `python -m physys.cli dataset -c config.json -o data/` once, driven entirely by the root `config.json`'s `amr_dataset` block (see [Configuring a run](#configuring-a-run) above for exactly which fields that reads and overrides). This is a single process that writes all three files (`Umi.h5`, `Uma.h5`, `Rma.h5`) together — there's no way to regenerate just one of them without rerunning the whole thing, which is why the Makefile tracks it with a single `data/.stamp` file rather than three separate output rules. Rebuilds when the root `config.json` or `physys/cli.py` changes.

```bash
make clean-amr-dataset
```

Removes `data/Umi.h5`, `data/Uma.h5`, `data/Rma.h5`, and `data/.stamp` — same confirm-before-delete behavior as `make clean`.

**Notes (both paths):**
- Rebuilds are triggered by either the relevant `config.json` changing or `physys/cli.py` changing — not just by re-running `make`.
- Generation runs **serially** by default (`.NOTPARALLEL:` in the Makefile), since it's assumed to share a single GPU. If your setup doesn't have that constraint, remove that line and use `make -j4 dataset` to parallelize across modulations (`amr-dataset` is a single target either way, so `-j` doesn't split it further).
- Every `export_*`/`open_amr_dataset` write in `physys.export` opens its target file in `"w"` mode — re-running a target overwrites its output rather than appending to it, which is exactly what the staleness checks above are built around.

## Diagnostics & verification

- `diagnostics/` holds standalone plotting scripts (BER waterfall, constellation scatter, LLR histogram, topology check) that read exported `.h5` files and drop figures into `imgs/`.
- `notebooks/` has notebooks for spot-checking a single exported file (`verify-h5.ipynb`, `verify-IQ-h5.ipynb`) and cross-checking the UMi/UMa/RMa AMR outputs (`verify-umi-uma-rma.ipynb`).
- `docs/amr-summary.pdf` is a written summary of the AMR dataset generation results.
- `data/` is where `make amr-dataset` writes `Umi.h5`/`Uma.h5`/`Rma.h5` plus its `.stamp` file. If you're tidying the repo, worth double-checking any dated subfolders under `data/` from earlier runs against the current Makefile (which writes a flat `data/.stamp`, not one per date) — decide whether those are snapshots you want to keep or leftovers from an older workflow before removing anything.

## Tests

```bash
pytest tests/
```

`tests/` mirrors the package layout (`test_loader.py`, `test_schema.py`, `test_builders.py`, `test_runtime.py`, `test_export.py`), plus `test_integration.py` for end-to-end config-to-HDF5 runs.