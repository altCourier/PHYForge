# PHYSys API Documentation

`physys` is a configuration-driven orchestrator built on top of **Sionna-PHY**. It
turns a JSON configuration file into a runnable `Source -> Map -> Channel -> Demap`
link-level simulation, with no Python code required to change channel models,
waveforms, or modulation schemes. It also drives the same pipeline across a grid of
`(system-level channel variant x modulation x SNR)` to produce AMR (Automatic
Modulation Recognition) training datasets.

```
Read JSON config -> Validate against schema -> Build Sionna objects -> Run simulation
```

```
Source -> Map -> Channel -> Demap
```

This document covers all seven modules (`loader`, `schema`, `builders`, `runtime`,
`export`, `amr_dataset`, `cli`), every configuration field with its default, and the
behavioral edge cases that show up in the current implementation.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Package Layout](#package-layout)
3. [physys.loader](#1-physysloader)
4. [physys.schema](#2-physysschema)
5. [physys.builders](#3-physysbuilders)
6. [physys.runtime](#4-physysruntime)
7. [physys.export](#5-physysexport)
8. [physys.amr_dataset](#6-physysamr_dataset)
9. [physys.cli](#7-physyscli)
10. [Configuration Reference (JSON)](#configuration-reference-json)
11. [End-to-End Execution Traces](#end-to-end-execution-traces)
12. [Known Limitations & Gotchas](#known-limitations--gotchas)
13. [Full config.json Examples](#full-configjson-examples)

---

## Quickstart

```python
from physys.loader import load_config
from physys.runtime import PHYSys
from physys import export

config = load_config("path/to/config.json")

sim = PHYSys(config)

# Single run (bits/llr only)
bits, llr = sim.generate(batch_size=32, num_symbols=64, ebno_db=5.0)
export.export_run(bits, llr, "run.h5", ebno_db=5.0, config=config)

# Sweep over config.sweep.ebno_db
results = sim.generate_sweep()   # {ebno_db: (bits, llr), ...}
export.export_sweep(results, "sweep.h5", config=config)

# Same, but also capturing raw tx/rx symbols (x/y) for AMR-style work
bits, llr, x, y = sim.generate_iq(batch_size=32, num_symbols=64, ebno_db=5.0)
export.export_run_iq(bits, llr, x, y, "run_iq.h5", ebno_db=5.0, config=config)

results_iq = sim.generate_sweep_iq()  # {ebno_db: (bits, llr, x, y), ...}
export.export_sweep_iq(results_iq, "sweep_iq.h5", config=config)
```

`PHYSys` builds every Sionna component exactly once, in `__init__`. Only
`batch_size`, `num_symbols`, and `ebno_db` vary between calls to `generate()` /
`generate_iq()`. `physys.export` then takes whatever `generate()` /
`generate_sweep()` / `generate_iq()` / `generate_sweep_iq()` produced and writes it
to disk as HDF5, independent of how the tensors were produced.

For the AMR benchmark generator (`Umi.h5`/`Uma.h5`/`Rma.h5`), use `physys.amr_dataset`
via the CLI's `dataset` subcommand rather than calling `PHYSys` directly — see
[physys.amr_dataset](#6-physysamr_dataset).

---

## Package Layout

| Module | Responsibility | Imports Sionna/PyTorch? |
|---|---|---|
| `physys.loader` | File I/O: path -> dict -> `Config` | No |
| `physys.schema` | Dataclasses + validation for the config tree | **No** (by design) |
| `physys.builders` | Validated `Config` -> live Sionna objects | Yes |
| `physys.runtime` | Orchestrates built objects into `generate()` / `generate_sweep()` / `generate_iq()` / `generate_sweep_iq()` | Yes |
| `physys.export` | Serializes `generate()`/`generate_sweep()`/`generate_iq()`/`generate_sweep_iq()` output to/from HDF5; also owns the low-level AMR benchmark file format (`open_amr_dataset`/`append_amr_batch`) | No (only `h5py`/`numpy`) |
| `physys.amr_dataset` | Drives `PHYSys` across `(channel_variant x modulation x snr_db)` per `config.amr_dataset`, producing `Umi.h5`/`Uma.h5`/`Rma.h5` | Indirectly (via `runtime`) |
| `physys.cli` | Thin `argparse` shell over the four modules above (`run`/`sweep`/`sweep --iq`/`dataset`) | Indirectly (via the above) |

`schema.py` is intentionally free of heavy imports. `export.py` follows the same
philosophy — it depends on `h5py`/`numpy` only, not Sionna or TensorFlow/PyTorch
directly, aside from duck-typing tensors via `.numpy()`/`.detach()`/`.cpu()`.
`amr_dataset.py` contains no modulation specs, SNR grids, or variant lists of its
own — those all come from `config.amr_dataset`.

---

## 1. physys.loader

### `class ConfigLoadError(Exception)`
The single exception type raised by `load_config`. Wraps every possible underlying
failure: missing file, unreadable file, invalid JSON, or schema validation failure.
So callers only need one `except` clause.

### `load_config(pathname: Union[str, Path]) -> Config`

Reads a JSON file from disk and returns a validated `Config` instance.

**Parameters**
- `pathname` — path to the `config.json` file (`str` or `Path`).

**Returns:** `schema.Config`

**Raises:** `ConfigLoadError` in every failure case:

| Underlying cause | Wrapped as |
|---|---|
| File does not exist | `ConfigLoadError("Config file not found: ...")` |
| File exists but unreadable (`OSError`) | `ConfigLoadError("Could not read config file ...")` |
| File is not valid JSON | `ConfigLoadError("... is not valid JSON: ...")` |
| JSON is valid but fails schema validation (`ValueError`/`KeyError`/`TypeError` from `parse_config`) | `ConfigLoadError("... failed validation: ...")` |

`load_config` does no validation itself beyond catching these three exception
categories — all real validation lives in `schema.parse_config`.

---

## 2. physys.schema

Defines the entire configuration tree as plain `dataclasses`. Two kinds of parsing
happen here:

- **Constructing** the dataclasses directly (e.g. in tests) — Python's normal
  dataclass rules apply (fields with no default are required).
- **Parsing raw JSON** via `parse_config` and its `parse_*` helpers — these apply
  their own defaulting rules on top of the dataclasses, described below.

### Comment stripping

Any JSON key starting with `_` (e.g. `"_comment"`, `"_note"`) is dropped before it
reaches a dataclass constructor, via `_strip_comments()`. This lets a `config.json`
carry inline documentation without failing validation. This applies at every level
that goes through a `parse_*` function, including `parse_amr_modulation` and
`parse_amr_dataset`.

### `CommonConfig`

| Field | Type | Default | Notes |
|---|---|---|---|
| `precision` | `Optional[str]` | `None` | Global default; any block can override locally. |
| `device` | `Optional[str]` | `None` | Same inheritance rule as `precision`. |
| `active_channel` | `str` | `"awgn"` | Selector only, never inherited. Must name a field of `ChannelsConfig`. |
| `active_waveform` | `str` | `"time"` | Selector only. Must name a field of `WaveformsConfig`. |

`active_channel`/`active_waveform` are **not** validated against real channel/waveform
names inside `CommonConfig.__post_init__` — `CommonConfig` doesn't know about
`ChannelsConfig`/`WaveformsConfig` yet. That check happens once, in
`Config.__post_init__`, after the whole tree is assembled.

### `BinarySourceConfig`
No fields. A placeholder. Seeding could be added later but is "postponed" for now.

### Modulation block

---

**`MapperConfig`**
| Field | Type | Default |
|---|---|---|
| `return_indices` | `bool` | `False` |

---

**`DemapperConfig`**
| Field | Type | Default |
|---|---|---|
| `hard_out` | `bool` | `False` |
| `demapping_method` | `Literal["app","maxlog"]` | `"app"` |

---

**`ConstellationConfig`** — only meaningful when `modulation.type == "custom"`, but
Sionna's `Constellation` always accepts these kwargs so they're always present:
| Field | Type | Default |
|---|---|---|
| `normalize` | `bool` | `False` |
| `center` | `bool` | `False` |
| `points` | `Optional[list]` | `None` |

---

**`ModulationConfig`** — `type`/`num_bits_per_symbol`/`mapper`/`demapper`/`constellation`
all required, no defaults:
| Field | Type |
|---|---|
| `type` | `Literal["qam","pam","custom"]` |
| `num_bits_per_symbol` | `int` |
| `mapper` | `MapperConfig` |
| `demapper` | `DemapperConfig` |
| `constellation` | `ConstellationConfig` |

`__post_init__` re-checks:
- `type ∈ {"qam","pam","custom"}` (belt-and-braces on top of the `Literal` type hint,
  which isn't enforced at runtime by plain dataclasses).
- `num_bits_per_symbol` is an `int`.
- **If `type == "qam"`: `num_bits_per_symbol` must be even.** QAM is built from two
  independent PAM arms on I/Q, so odd orders (e.g. BPSK) aren't representable as
  `"qam"` — use `type="custom"` with explicit `constellation.points` instead.
- **If `type == "custom"`: `constellation.points` must be set** (non-`None`).

`"pam"` has no additional parity constraint beyond being an `int`.

---

### Channel blocks (`ChannelsConfig`)

**`AWGNChannelConfig`**
| Field | Type | Default |
|---|---|---|
| `precision` | `Optional[str]` | `None` |
| `device` | `Optional[str]` | `None` |

---

**`TDLChannelConfig`**
| Field | Type | Default |
|---|---|---|
| `model` | `str` | *required* |
| `delay_spread` | `float` | *required* |
| `carrier_frequency` | `float` | *required* |
| `num_sinusoids` | `int` | `20` |
| `los_angle_of_arrival` | `float` | `0.7853981633974483` (π/4) |
| `min_speed` | `float` | `0.0` |
| `max_speed` | `Optional[float]` | `None` |
| `num_rx_ant` | `int` | `1` |
| `num_tx_ant` | `int` | `1` |
| `spatial_corr_mat` / `rx_corr_mat` / `tx_corr_mat` | `Optional[list]` | `None` |
| `precision` / `device` | `Optional[str]` | `None` |

`__post_init__` re-verifies `model`, `delay_spread`, `carrier_frequency` are non-`None`.

---

**`PanelArrayConfig`** (used for `bs_array`/`ut_array` in system-level channels)
| Field | Type | Default |
|---|---|---|
| `num_rows_per_panel` | `int` | `1` |
| `num_cols_per_panel` | `int` | `1` |
| `polarization` | `Literal["single","dual"]` | `"single"` |
| `polarization_type` | `Literal["V","H","cross"]` | `"V"` |
| `antenna_pattern` | `str` | `"38.901"` |

Note: `PanelArrayConfig` has no `carrier_frequency` field of its own — the builder
supplies the parent `SystemLevelChannelConfig.carrier_frequency` to both arrays (see
`build_channel`).

---

**`SystemLevelChannelConfig`** (shared shape for UMi/UMa/RMa)
| Field | Type | Default |
|---|---|---|
| `variant` | `Literal["umi","uma","rma"]` | *required* |
| `carrier_frequency` | `float` | *required* |
| `o2i_model` | `Literal["low","high"]` | `"low"` |
| `direction` | `Literal["uplink","downlink"]` | `"downlink"` |
| `enable_pathloss` | `bool` | `True` |
| `enable_shadow_fading` | `bool` | `True` |
| `average_street_width` | `float` | `20.0` |
| `average_building_height` | `float` | `5.0` |
| `bs_array` | `PanelArrayConfig` | `PanelArrayConfig()` |
| `ut_array` | `PanelArrayConfig` | `PanelArrayConfig(antenna_pattern="omni")` |
| `precision` / `device` | `Optional[str]` | `None` |

`__post_init__` re-checks:
- `variant ∈ {"umi","uma","rma"}`.
- `average_street_width > 0`.
- `average_building_height > 0`.

`average_street_width`/`average_building_height` are RMa-specific physical
parameters. They're present on every `SystemLevelChannelConfig` (not split into a
separate RMa-only dataclass) so the shape stays uniform across variants, but the
builder only ever passes them through to the Sionna constructor when
`variant == "rma"` — see `build_channel` below. Setting them under a `"umi"`/`"uma"`
config is accepted (validated, even) but has no observable effect, since `UMi`/`UMa`
don't take those constructor arguments.

---

**`RayleighBlockFadingConfig`**
| Field | Type | Default |
|---|---|---|
| `num_rx` | `int` | `1` |
| `num_rx_ant` | `int` | `1` |
| `num_tx` | `int` | `1` |
| `num_tx_ant` | `int` | `1` |
| `precision` / `device` | `Optional[str]` | `None` |

**`ChannelsConfig`** — all four blocks are required fields on the dataclass itself
(no defaults):
```python
ChannelsConfig(awgn, tdl, system_level, rayleigh_block_fading)
```
> When parsed from JSON via `parse_channels`, `awgn` and `rayleigh_block_fading` fall
> back to `{}` if absent from the file, but `tdl` and `system_level` are read with
> `raw["tdl"]` / `raw["system_level"]` — **a `KeyError` is raised if either key is
> missing**, even if that channel isn't the active one.

### Waveform blocks (`WaveformsConfig`)

---

**`ResourceGridConfig`**
| Field | Type | Default |
|---|---|---|
| `num_ofdm_symbols` | `int` | *required* |
| `fft_size` | `int` | *required* |
| `subcarrier_spacing` | `float` | *required* |
| `num_tx` | `int` | `1` |
| `num_streams_per_tx` | `int` | `1` |
| `cyclic_prefix_length` | `int` | `16` |
| `num_guard_carriers` | `List[int]` | `[0, 0]` |
| `dc_null` | `bool` | `True` |
| `pilot_pattern` | `Literal["kronecker","empty"]` | `"empty"` |
| `pilot_ofdm_symbol_indices` | `List[int]` | `[]` |
| `precision` / `device` | `Optional[str]` | `None` |

---

**`TimeWaveformConfig`**
| Field | Type | Default |
|---|---|---|
| `bandwidth` | `float` | *required* |
| `num_time_samples` | `int` | *required* |
| `maximum_delay_spread` | `float` | `3e-6` |
| `l_min` / `l_max` | `Optional[int]` | `None` |
| `normalize_channel` | `bool` | `False` |
| `return_channel` | `bool` | `False` |
| `precision` / `device` | `Optional[str]` | `None` |

---

**`OFDMWaveformConfig`**
| Field | Type | Default |
|---|---|---|
| `resource_grid` | `ResourceGridConfig` | *required* |
| `normalize_channel` | `bool` | `False` |
| `return_channel` | `bool` | `False` |
| `precision` / `device` | `Optional[str]` | `None` |

**`WaveformsConfig`** — both `time` and `ofdm` are required, no defaults.
`parse_waveforms` reads `raw["time"]` and `raw["ofdm"]` unconditionally — **both
blocks must be present in JSON even if only one waveform is active.**

### Sweep block

**`EbNoRangeConfig`** — `start: float`, `stop: float`, `step: float`, all required.

`__post_init__` requires **`step > 0`**, raising `ValueError` otherwise — a
non-positive step would make `generate_sweep()`'s (and `generate_sweep_iq()`'s /
`amr_dataset`'s) `while` loop never terminate. This same dataclass is reused as-is
for `amr_dataset.snr_db`.

**`SweepConfig`** — all required: `ebno_db: EbNoRangeConfig`, `batch_size: int`,
`num_symbols: int`.

### AMR-specific blocks

---

**`AmrModulationSpec`** — one entry per modulation the AMR dataset should cover.
Distinct from `ModulationConfig`: it carries a human-readable `name` (used as the
one-hot label and as the HDF5 group/attr identifier) alongside the same
type/order/constellation fields `ModulationConfig` needs, but *not* its own
`mapper`/`demapper` — those are shared across all modulations in a dataset run (see
`AmrDatasetConfig.mapper`/`demapper` below).

| Field | Type |
|---|---|
| `name` | `str` — unique per dataset run; becomes the one-hot label |
| `type` | `Literal["qam","pam","custom"]` |
| `num_bits_per_symbol` | `int` |
| `constellation` | `ConstellationConfig` |

**`AmrModulationSpec.to_modulation_config(mapper, demapper) -> ModulationConfig`**
Builds a full `ModulationConfig` by combining this spec's `type`/
`num_bits_per_symbol`/`constellation` with the caller-supplied shared
`mapper`/`demapper`. Deliberately reuses `ModulationConfig.__post_init__` for
validation (even-bits-for-qam, points-required-for-custom) rather than duplicating
those rules here — an `AmrModulationSpec` with `type="qam"` and an odd
`num_bits_per_symbol` fails at the point `to_modulation_config` is called (inside
`amr_dataset._build_config`, once per modulation per variant), with the same
`ValueError` message `ModulationConfig` would raise anywhere else.

---

**`AmrDatasetConfig`**
| Field | Type | Default | Notes |
|---|---|---|---|
| `variants` | `List[Literal["umi","uma","rma"]]` | *required* | Which system-level scenarios to generate; one output file per entry. |
| `snr_db` | `EbNoRangeConfig` | *required* | SNR (Eb/N0, dB) grid swept per modulation per variant. |
| `num_vectors` | `int` | *required* | Vectors per (modulation, SNR) cell. |
| `vector_len` | `int` | *required* | Complex samples per vector. Stamped onto `waveforms.time.num_time_samples` and `sweep.num_symbols` for every generated sub-config (see `amr_dataset._build_config`). |
| `mapper` | `MapperConfig` | *required* | Shared across every modulation in this dataset run. |
| `demapper` | `DemapperConfig` | *required* | Shared across every modulation in this dataset run. |
| `modulations` | `List[AmrModulationSpec]` | *required* | Must be non-empty; `name`s must be unique. |
| `max_batch_size` | `Optional[int]` | `None` | Upper bound on `batch_size` per `generate_iq()` call — see below. `None` means no chunking (one call of `batch_size=num_vectors`). |
| `disable_pathloss_and_shadowing` | `bool` | `True` | See below. |

`__post_init__` requires:
- `modulations` non-empty.
- `modulations[*].name` unique.
- every entry of `variants ∈ {"umi","uma","rma"}`.
- `max_batch_size` is either `None` or `> 0`.

**Why `max_batch_size` exists:** `num_vectors` worth of samples for one
(modulation, SNR) cell aren't necessarily generated in a single forward pass — the
demapper's `"app"` method materializes a `[batch, vector_len, num_points]` tensor,
which for high-order modulations (e.g. 1024QAM: `num_points=1024`) can exceed
available GPU memory well before `batch == num_vectors` is reached.
`amr_dataset._batch_chunks(total, max_chunk)` splits `num_vectors` into
`max_chunk`-sized pieces (plus a remainder chunk) so the AMR generator loops smaller
sub-batches and concatenates them via `export.append_amr_batch`, rather than the
demapper ever materializing the full-`num_vectors` intermediate tensor at once.

**Why `disable_pathloss_and_shadowing` exists:** the point of the AMR benchmark is
typically to isolate small-scale fading + AWGN effects on the received
constellation, independent of large-scale path loss / shadow fading, which would
otherwise dominate the received power at a fixed target SNR. When `True` (the
default), `amr_dataset._build_config` forces both
`system_level.enable_pathloss` and `system_level.enable_shadow_fading` to `False`
for the duration of dataset generation, regardless of what the base config's
`channels.system_level` block says (see `_build_config` below) — but only ever
*narrows*, never *widens*, those flags: `sl.enable_pathloss and not
ds.disable_pathloss_and_shadowing` means if the base config already has
`enable_pathloss=False`, the dataset flag can't turn it back on.

### Top-level `Config`

```python
Config(common, source, modulation, channels, waveforms, sweep, amr_dataset=None)
```
`common`, `source`, `modulation`, `channels`, `waveforms`, `sweep` are all required,
no defaults. `amr_dataset` is optional (`Optional[AmrDatasetConfig] = None`) — a
config with no `amr_dataset` block behaves exactly as before this feature was added;
`amr_dataset` is only consulted by `cli.run_dataset` / `amr_dataset.generate_all`.

`__post_init__` performs three checks:

1. Derives valid channel/waveform names directly from
   `dataclasses.fields(ChannelsConfig)` / `dataclasses.fields(WaveformsConfig)`
   (rather than a hand-maintained list, to avoid drift) and raises `ValueError` if
   `common.active_channel` or `common.active_waveform` doesn't match one of those
   field names.
2. **If `common.active_waveform == "time"`: `sweep.num_symbols` must equal
   `waveforms.time.num_time_samples`.** This closes what was previously an
   unenforced assumption in the time-domain reshape path (`runtime.
   _apply_time_channel`) — passing mismatched values now fails fast at config-parse
   time with a clear `ValueError`, instead of surfacing later as a shape error or
   silently wrong output inside Sionna. Only checked when `active_waveform=="time"`;
   `"ofdm"` configs aren't constrained by this rule.
3. Because `__post_init__` reruns on every `dataclasses.replace(config, ...)` call
   (not just on first construction via `parse_config`), check #2 also re-validates
   every per-modulation/per-variant sub-config `amr_dataset._build_config` produces —
   there's no way for the AMR generator to accidentally end up with a
   `num_symbols`/`num_time_samples` mismatch that schema validation would have
   caught on a hand-written config.

### Parser functions

| Function | Behavior |
|---|---|
| `parse_common(raw)` | Strips comments, `CommonConfig(**raw)`. |
| `parse_source(raw)` | Always returns `BinarySourceConfig()` (ignores `raw`). |
| `parse_modulation(raw)` | Builds nested `mapper`/`demapper`/`constellation` from sub-dicts (each defaulting to `{}` if absent). |
| `parse_channels(raw)` | See `ChannelsConfig` note above — `tdl`/`system_level` are mandatory keys; `bs_array`/`ut_array` are popped out of `system_level` before constructing it. |
| `parse_waveforms(raw)` | `time`/`ofdm` are mandatory keys; `resource_grid` is popped out of `ofdm` before constructing it. |
| `parse_sweep(raw)` | `ebno_db`, `batch_size`, `num_symbols` are mandatory keys. |
| `parse_amr_modulation(raw)` | Strips comments; `name`/`type`/`num_bits_per_symbol` are mandatory keys; `constellation` defaults to `{}`. |
| `parse_amr_dataset(raw)` | Returns `None` if `raw is None` (i.e. `amr_dataset` key absent from JSON) — this is how a plain non-AMR config skips the block entirely. Otherwise: `variants`, `snr_db`, `num_vectors`, `vector_len`, `modulations` are mandatory keys; `max_batch_size` defaults to `None`; `disable_pathloss_and_shadowing` defaults to `True`; `mapper`/`demapper` default to `{}`; `modulations` is parsed via `parse_amr_modulation` per entry. |
| `parse_config(raw)` | Top-level entry point; calls all of the above (including `parse_amr_dataset(raw.get("amr_dataset"))`) and assembles `Config`. |

---

## 3. physys.builders

Internal factory functions. Consumes a validated `Config` and returns live
Sionna/PyTorch objects. Normally called only from `PHYSys.__init__` — not part of the
day-to-day public API, but useful to call directly for debugging or building a custom
pipeline.

### `_pd_kwargs(local_config, common) -> dict`

The single implementation of the precision/device inheritance rule used everywhere
else in this file:

1. Use the block's own `precision`/`device` if it's non-`None`.
2. Otherwise fall back to `common.precision`/`common.device`.
3. If **both** are `None`, omit the key entirely from the returned dict — the caller
   then lets the underlying Sionna class use its own internal default rather than
   forcing one.

### `_active_channel_config(config)`
`getattr(config.channels, config.common.active_channel)` — looks up whichever
channel block is actually selected for this run.

### `build_channel(config: Config)`

Returns the **raw** Sionna channel model — never wrapped in `TimeChannel` /
`OFDMChannel`. Dispatches on the type of the active channel config:

| Config type | Returns |
|---|---|
| `AWGNChannelConfig` | `AWGN(**pd_kwargs)` |
| `TDLChannelConfig` | `TDL(model=, delay_spread=, carrier_frequency=, num_sinusoids=, los_angle_of_arrival=, min_speed=, max_speed=, num_rx_ant=, num_tx_ant=, spatial_corr_mat=, rx_corr_mat=, tx_corr_mat=, **pd_kwargs)` |
| `SystemLevelChannelConfig` | Builds `bs_array`/`ut_array` as `PanelArray(...)` (each stamped with the channel's `carrier_frequency`), picks `UMi`/`UMa`/`RMa` from `variant`, and constructs it with `carrier_frequency`, `ut_array`, `bs_array`, `direction`, `enable_pathloss`, `enable_shadow_fading`, plus **`o2i_model` only when `variant` is `"umi"` or `"uma"`**, plus **`average_street_width`/`average_building_height` only when `variant == "rma"`** (`RMa`'s constructor takes those two instead of `o2i_model`; `UMi`/`UMa`'s constructors take `o2i_model` instead of those two — each variant gets exactly the kwargs its own constructor accepts), plus `pd_kwargs`. |
| `RayleighBlockFadingConfig` | `RayleighBlockFading(num_rx=, num_tx=, num_rx_ant=, num_tx_ant=, **pd_kwargs)` |

Any other config type raises `ValueError`.

> **Topology is not set here.** The returned UMi/UMa/RMa instance still needs
> `set_topology(...)` called on it before it's usable — that's a runtime concern
> (topology depends on `batch_size`, which can vary between calls) and is handled by
> `PHYSys._maybe_set_topology` in `runtime.py`.

### `build_source(config)`
Always returns `BinarySource()`. `"binary"` is the only source type that currently
exists; the config's `source` block is accepted but ignored.

### `build_mapsys(config) -> tuple[Constellation, Mapper, Demapper]`

```python
constellation = Constellation(mod.type, mod.num_bits_per_symbol,
                               normalize=..., center=..., points=...)
mapper = Mapper(constellation=constellation, return_indices=...)
demapper = Demapper(mod.demapper.demapping_method, constellation=constellation,
                     hard_out=...)
```

`mod.constellation.points`, when set, is converted from a list of `[re, im]` pairs to
a `numpy.complex64` array before being handed to `Constellation` — required for
`type="custom"` configs (including every `AmrModulationSpec` whose
`to_modulation_config` produces a `"custom"` type, e.g. BPSK).

### `build_waveform(config, channel_model)`

Wraps the raw `channel_model` from `build_channel` according to
`common.active_waveform`. **This is the only place `TimeChannel`/`OFDMChannel`
wrapping happens.**

- **`active_waveform == "time"`**
  - If `channel_model` is an `AWGN` instance -> returned **unchanged**, no
    `TimeChannel` wrapping at all (AWGN has no time-domain filter tail).
  - Otherwise -> `TimeChannel(channel_model=, bandwidth=, num_time_samples=, maximum_delay_spread=, l_min=, l_max=, normalize_channel=, return_channel=, **pd_kwargs)`.
- **`active_waveform == "ofdm"`**
  - The `AWGN` early-exit is checked **first**, before the `ResourceGrid` is even
    built -> `channel_model` returned unchanged.
  - Otherwise -> builds a `ResourceGrid` from `waveforms.ofdm.resource_grid`, then
    `OFDMChannel(channel_model=, resource_grid=, normalize_channel=, return_channel=, **pd_kwargs)`.
- Any other value of `active_waveform` raises `ValueError`.

---

## 4. physys.runtime

### `class PHYSys`

The main simulation orchestrator. Built once per `Config`; only the per-call
arguments (`batch_size`, `num_symbols`, `ebno_db`) vary between calls.

**Constructor**

```python
PHYSys(config: Config)
```

Builds, in order: `self.source`, `(self.constellation, self.mapper, self.demapper)`,
`self._channel_model` (raw, via `build_channel`), `self._handle` (wrapped, via
`build_waveform`), and hardcodes `self._coderate = 1.0` (no FEC support yet).

**Key attributes**

| Attribute | What it is |
|---|---|
| `self.source` | Instantiated `BinarySource`. |
| `self.mapper` / `self.demapper` | Instantiated modulation components. |
| `self._channel_model` | The **raw** Sionna channel model (`AWGN`/`TDL`/`UMi`/`UMa`/`RMa`/`RayleighBlockFading`). Access directly for Sionna-specific methods like `show_topology()`. |
| `self._handle` | The waveform wrapper applied to the raw channel (may be identical to `self._channel_model` when AWGN is active — see `build_waveform`). |
| `self._coderate` | Always `1.0`. |

**`_active_channel_config(self)`**
Same lookup as `builders._active_channel_config`, duplicated here for local use.

**`_maybe_set_topology(self, batch_size)`**
No-op unless the active channel is a `SystemLevelChannelConfig`. Otherwise:

```python
topology = gen_single_sector_topology(batch_size=batch_size, num_ut=1,
                                       scenario=channel_config.variant)
self._channel_model.set_topology(*topology)
```

`num_ut` is **hardcoded to `1`** — a single user terminal per sector; not
configurable. A fresh topology is generated on **every** call (never cached), since
topology tensors are shaped by `batch_size`, which can differ between a one-off call
and a sweep step (and, in AMR dataset generation, between successive `max_batch_size`
chunks).

> Because topology is only generated inside `generate()`/`generate_iq()`, you must
> call one of them at least once before calling
> `sim._channel_model.show_topology()`.

**`_apply_time_channel(self, x, no, num_symbols)`**

The one place in the project that does shape bookkeeping rather than pure Sionna
calls:

- If `self._channel_model` is `AWGN` -> returns `self._handle(x, no)` directly, no
  reshape (identity passthrough — applies for both `"time"` and `"ofdm"` waveform
  settings, since both early-exit to the raw AWGN model in `build_waveform`).
- If `active_waveform == "time"` (non-AWGN):
  1. Reshapes `x` from `[batch, num_symbols]` to
     `[batch, 1, num_tx_ant, -1]` via `tf_reshape_for_time_channel` (`num_tx_ant`
     read off the active channel config, default `1` if the config type doesn't
     define it).
  2. Calls `self._handle(x_time, no)`.
  3. `TimeChannel`'s output is `l_max - l_min` samples **longer** than its input.
     The result is truncated back down to `num_symbols` (`y_time[..., :num_symbols]`)
     and reshaped to `[batch_size, num_symbols]`. **This discards energy belonging to
     the last few symbols** — a known, flagged tradeoff, not a bug fix candidate.
  - `num_symbols == waveforms.time.num_time_samples` is now enforced at config-parse
    time (see `Config.__post_init__` in [physys.schema](#2-physysschema)), so this
    path can assume it rather than merely hope for it.
- If `active_waveform == "ofdm"` (non-AWGN) -> **always raises `NotImplementedError`**.
  Mapping a flat `[batch, num_symbols]` stream onto a full OFDM resource grid (data
  REs only, skipping pilots/guards/DC) isn't implemented here yet, even though
  `build_waveform` happily constructs the `OFDMChannel` object itself.
- Any other `active_waveform` value raises `ValueError`.

**`_generate_core(self, batch_size, num_symbols, ebno_db) -> (bits, llr, x, y)`**

The single shared implementation behind both public entry points below — runs the
pipeline exactly once and returns every intermediate tensor a caller might want:

1. `self._maybe_set_topology(batch_size)`
2. `no = ebnodb2no(ebno_db, num_bits_per_symbol=config.modulation.num_bits_per_symbol, coderate=self._coderate)`
3. `bits = self.source([batch_size, num_symbols * num_bits_per_symbol])`
4. `x = self.mapper(bits)` -> `[batch_size, num_symbols]` (clean transmitted symbols)
5. `y = self._apply_time_channel(x, no, num_symbols)` -> `[batch_size, num_symbols]` (noisy received symbols / demapper input)
6. `llr = self.demapper(y, no)`
7. Returns `(bits, llr, x, y)`.

Because `generate()` and `generate_iq()` both delegate to this one method, a single
call produces internally-consistent bits/llr **and** x/y from the *same* random
draw — there's no risk of the two views of a run drifting apart from running the
pipeline twice.

**`generate(self, batch_size: int, num_symbols: int, ebno_db: float) -> (bits, llr)`**

```python
bits, llr, _x, _y = self._generate_core(batch_size, num_symbols, ebno_db)
return bits, llr
```

Contract unchanged from the pre-AMR version on purpose — existing callers
(`export.export_run`/`export_sweep`, tests) keep working as-is.

| Parameter | Meaning |
|---|---|
| `batch_size` | Number of independent batches to simulate. |
| `num_symbols` | Number of symbols per batch. |
| `ebno_db` | Eb/N0 in dB. |

**`generate_sweep(self) -> dict`**

```python
sweep = self.config.sweep
rng = sweep.ebno_db
results = {}
ebno_db = rng.start
while ebno_db <= rng.stop:
    results[ebno_db] = self.generate(batch_size=sweep.batch_size,
                                      num_symbols=sweep.num_symbols,
                                      ebno_db=ebno_db)
    ebno_db += rng.step
return results
```

Uses `sweep.batch_size`/`sweep.num_symbols` for **every** point (not configurable per
point). Returns a `dict` keyed by the `ebno_db` float value -> `(bits, llr)`, so
`export.py`/callers decide serialization format. Plain floating-point accumulation
loop — see Known Limitations #10.

**`generate_iq(self, batch_size: int, num_symbols: int, ebno_db: float) -> (bits, llr, x, y)`**

```python
return self._generate_core(batch_size, num_symbols, ebno_db)
```

Same pipeline as `generate()`, additionally exposing the raw I/Q symbol tensors:
- `x` — clean transmitted symbols (mapper output). Useful for a clean-vs-noisy
  constellation scatter plot.
- `y` — noisy received symbols (channel output / demapper input). This is the actual
  feature tensor an AMR model trains on, and what `export.append_amr_batch` stores
  as the `"Data"` dataset.

`bits`/`llr` are still returned alongside `x`/`y` (not dropped), so a single call can
feed both the existing BER/BLER path and the AMR dataset export path without running
the pipeline twice.

**`generate_sweep_iq(self) -> dict`**

AMR counterpart to `generate_sweep()`: same per-point looping behavior (including its
float-accumulation caveat), but calls `generate_iq()` instead of `generate()` at each
point. Returns `{ebno_db: (bits, llr, x, y), ...}`.

### `tf_reshape_for_time_channel(x, batch_size, num_tx_ant)`

```python
return x.reshape([batch_size, 1, num_tx_ant, -1])
```
`num_tx` is always `1` — PHYSys does not model multi-user uplink/downlink
multiplexing.

---

## 5. physys.export

Serialization module: takes tensors already produced by `PHYSys.generate()` /
`generate_sweep()` / `generate_iq()` / `generate_sweep_iq()` and writes them to disk
as HDF5, and reads them back. It also owns the separate, lower-level AMR benchmark
file format used by `amr_dataset.py` (`open_amr_dataset`/`append_amr_batch`), which
has its own layout (see below) rather than reusing `export_run_iq`/`export_sweep_iq`.

Imports: `h5py`, `numpy` (plus stdlib `dataclasses`, `json`, `time`, `pathlib`). No
Sionna/TensorFlow/PyTorch import — the only tensor-backend-specific logic in the
whole module is confined to `_to_numpy`.

### Internal helpers

**`_to_numpy(x)`**
Converts a Sionna/TensorFlow tensor, a torch tensor, or an already-`ndarray` value to
`numpy.ndarray`. If `x` has a `.detach()` method it's called first (torch tensors
with `requires_grad`); if it then has a `.cpu()` method that's called next
(GPU-resident torch tensors); finally, if the result has a `.numpy()` method that's
called, otherwise it falls back to `np.asarray(x)`. This is the single place a
different tensor backend would need to be accommodated.

**`_config_json(config) -> Optional[str]`**
Returns `None` if `config is None`, otherwise `json.dumps(dataclasses.asdict(config))`.
Used to embed the `schema.Config` that produced a run as a self-describing JSON
attribute on the HDF5 file — note this calls `dataclasses.asdict`, so `config` must be
an actual dataclass instance (e.g. `schema.Config`), not an arbitrary object. Since
`dataclasses.asdict` recurses through nested dataclasses, this also serializes
`config.amr_dataset` (including every `AmrModulationSpec`) whenever it's set.

**`_utc_timestamp() -> str`**
Returns the current UTC time as `"%Y-%m-%dT%H:%M:%SZ"` (e.g.
`"2026-08-03T14:22:01Z"`), used to stamp every file written with a `created_utc`
attribute.

**`_modulation_label(config) -> Optional[str]`**
Derives a human-readable modulation label (e.g. `"BPSK"`, `"16QAM"`) from
`config.modulation`, for stamping onto exported IQ files as `f.attrs["modulation"]` —
cheap to read back (e.g. in a PyTorch `DataLoader`) without parsing the full
`config_json`.

- Returns `None` if `config is None`, or if `config` has no `.modulation` attribute
  (i.e. doesn't look like a `schema.Config`) — callers should treat that the same as
  `_config_json`'s "attribute omitted" convention.
- 1-bit and 2-bit orders get their conventional names (`"BPSK"`, `"QPSK"`) regardless
  of `type`, via `_QAM_SPECIAL_NAMES`/`_PAM_SPECIAL_NAMES`; everything else is
  `f"{order}{TYPE}"` (e.g. `"16QAM"`, `"64QAM"`, `"8PAM"`) where `order = 2 **
  num_bits_per_symbol`.
- Only called from `export_run_iq`/`export_sweep_iq` — the `dataset` command's own
  file format (`open_amr_dataset`) derives its modulation labels directly from
  `AmrModulationSpec.name` instead, bypassing this function entirely.

### `export_run(bits, llr, path, *, ebno_db=None, config=None, compression="gzip") -> None`

Writes one `PHYSys.generate()` result to a **new** HDF5 file at `path`.

**Parameters**
| Parameter | Type | Notes |
|---|---|---|
| `bits`, `llr` | whatever `PHYSys.generate()` returned | Converted via `_to_numpy` before writing. |
| `path` | `str` or `Path` | Destination file. |
| `ebno_db` | `Optional[float]` | If given, recorded as a top-level file attr for convenience. |
| `config` | `Optional[schema.Config]` | If given, serialized via `_config_json` and stored as a `config_json` attr so a loaded file is self-describing. |
| `compression` | `str` | Passed straight to `h5py.create_dataset`; `"gzip"` by default. Pass `None` to disable compression. |

**Behavior**
- Creates `path.parent` if it doesn't exist (`mkdir(parents=True, exist_ok=True)`).
- Opens the file in `"w"` mode — **this creates or overwrites `path`**. It writes
  exactly one run per call, not an appending log.
- Datasets written at the file root: `"bits"`, `"llr"`.
- Attributes written at the file root: `created_utc` (always), `ebno_db` (only if
  passed), `config_json` (only if `config` is not `None`).

**Returns:** `None`.

### `export_sweep(results, path, *, config=None, compression="gzip") -> None`

Writes a `PHYSys.generate_sweep()` result (`{ebno_db: (bits, llr)}`) to **one** HDF5
file, as contiguous multi-dimensional arrays stacked along a new leading axis.

**Behavior**
- `ebno_values = sorted(results.keys())` — points are stacked in ascending Eb/N0
  order regardless of the dict's insertion order.
- File-root datasets: `"snr_labels"` (`float64`, one entry per point, in the same
  ascending order used to build the stacked arrays), `"bits"`, `"llr"` (each stacked
  along a new leading axis, so every point must share the same `bits`/`llr` shape —
  true today since `generate_sweep()` reuses `sweep.batch_size`/`sweep.num_symbols`
  for every point).
- File-root attributes: `created_utc`, `config_json` (if `config` given).

**Returns:** `None`.

### `load_run(path) -> dict`

Reads back a file written by `export_run`.

**Returns**
```python
{
    "bits": ndarray,
    "llr": ndarray,
    "ebno_db": float | None,   # None if the file has no ebno_db attr
    "config": dict | None,     # json.loads(config_json) if present, else None
}
```

Note `"config"` comes back as a plain `dict` (the result of `json.loads`), **not** a
reconstructed `schema.Config` dataclass instance.

### `load_sweep(path) -> dict`

Reads back a file written by `export_sweep`.

```python
results = {}
with h5py.File(path, "r") as f:
    for i, snr in enumerate(f["snr_labels"][...]):
        results[float(snr)] = (f["bits"][...][i], f["llr"][...][i])
```

**Returns:** `{ebno_db: (bits_ndarray, llr_ndarray), ...}`, keyed by the `snr_labels`
value at each stacked index.

---

### AMR extension: `*_iq` variants (bits/llr/x/y)

Mirror `export_run`/`export_sweep`/`load_run`/`load_sweep` exactly (same layout
style, same attrs, same compression handling), sourced from
`runtime.PHYSys.generate_iq()`/`generate_sweep_iq()`. The only differences: two extra
datasets (`"x"`, `"y"`) per file/group, and a `"modulation"` attr derived via
`_modulation_label`. `bits`/`llr` and the original four functions above are
untouched by this extension.

> **Complex data note:** `x`/`y` are complex64 tensors. `h5py` stores complex
> `ndarray`s natively via a compound `(r, i)` dtype and round-trips them
> transparently through `h5py`/`numpy` — no special handling needed here. That said,
> this is an `h5py`/`numpy` convention, not a universal HDF5 one: non-Python readers
> (MATLAB, some C++ HDF5 tools) may not recognize the compound type.

### `export_run_iq(bits, llr, x, y, path, *, ebno_db=None, config=None, compression="gzip") -> None`

Writes one `PHYSys.generate_iq()` result to a new HDF5 file at `path`. Same
creates/overwrites-in-`"w"`-mode semantics as `export_run`.

**Parameters** — same as `export_run`, plus:
| Parameter | Notes |
|---|---|
| `x` | Clean transmitted symbols (mapper output), complex. |
| `y` | Noisy received symbols (channel output / demapper input — the AMR feature tensor), complex, same shape as `x`. |

**Behavior** — datasets `"bits"`, `"llr"`, `"x"`, `"y"`; attrs `created_utc`,
`ebno_db` (if given), `config_json` (if `config` given), `modulation` (if
`_modulation_label(config)` returns non-`None`).

### `export_sweep_iq(results, path, *, config=None, compression="gzip") -> None`

Writes a `PHYSys.generate_sweep_iq()` result (`{ebno_db: (bits, llr, x, y)}`) to one
HDF5 file, same stacked-array layout style as `export_sweep`, with `"x"`/`"y"`
datasets added alongside `"bits"`/`"llr"`. File-root attrs also include `modulation`
(if derivable from `config`).

### `load_run_iq(path) -> dict`

Returns `{"bits", "llr", "x", "y", "ebno_db", "config", "modulation"}` — same
conventions as `load_run`, plus `"x"`/`"y"` ndarrays and `"modulation"` (`str | None`).

### `load_sweep_iq(path) -> dict`

Returns `{ebno_db: (bits_ndarray, llr_ndarray, x_ndarray, y_ndarray), ...}`, keyed by
`"snr_labels"` the same way `load_sweep` is.

> **Note:** matches `load_sweep`'s existing behavior of *not* surfacing file-level
> attrs (`created_utc`/`config_json`/`modulation`) in the returned dict — that gap
> predates this function and is inherited rather than fixed here, since widening the
> return shape is a decision that affects every caller. To read those attrs from a
> swept file directly:
> ```python
> with h5py.File(path, "r") as f:
>     modulation = f.attrs.get("modulation")
> ```

---

### AMR benchmark file format: `open_amr_dataset` / `append_amr_batch`

A **separate, lower-level format** from `export_run_iq`/`export_sweep_iq` above —
purpose-built for `amr_dataset.py`'s incremental, resizable, many-small-batches
write pattern (one modulation x one SNR x one chunk per call), rather than the
"whole result already in memory" shape `export_sweep_iq` assumes.

**`open_amr_dataset(path, mod_order: list, vector_len=1024, compression="gzip") -> h5py.File`**

Creates a new **resizable** HDF5 file with three top-level datasets, each starting
at length 0 and growable along axis 0:

| Dataset | Shape | dtype | Chunking |
|---|---|---|---|
| `"Data"` | `(0, vector_len)` growing to `(N, vector_len)` | `complex64` | `(1024, vector_len)` |
| `"Mods"` | `(0, len(mod_order))` growing to `(N, len(mod_order))` | `float32`, one-hot | `(1024, len(mod_order))` |
| `"SNRs"` | `(0,)` growing to `(N,)` | `float32` | `(1024,)` |

File-root attrs: `created_utc`, `mod_order` (`json.dumps(mod_order)` — the ordered
list of modulation names, used to interpret which one-hot column in `"Mods"` means
which modulation when reading the file back).

Opens the file in `"w"` mode — **creates or overwrites `path`**, same as every other
`export_*` writer in this module. Caller (`amr_dataset.generate_variant`) is
responsible for calling `.close()` when done (typically via `try`/`finally`).

**`append_amr_batch(f, y, mod_index: int, snr_db: float) -> None`**

Appends one `(modulation, snr)` batch to an already-open file from
`open_amr_dataset`:

1. `y` — `[batch, vector_len]` complex tensor (typically the `y` returned by
   `PHYSys.generate_iq`) — converted via `_to_numpy` and cast to `complex64`.
2. Builds a one-hot row per sample: `mods_onehot[:, mod_index] = 1.0`, width taken
   from `f["Mods"].shape[1]` (i.e. however many columns the file was opened with).
3. Builds an `snrs` row per sample, all equal to `snr_db`.
4. Resizes and appends all three datasets (`"Data"`, `"Mods"`, `"SNRs"`) along axis 0
   by `n = y.shape[0]` rows.

`mod_index` must be `< f["Mods"].shape[1]` (i.e. `< len(mod_order)` as passed to
`open_amr_dataset`) — see Known Limitations for what happens if it isn't.

---

## 6. physys.amr_dataset

Drives `PHYSys` across `(channel_variant x modulation x snr_db)`, entirely as
specified by `config.amr_dataset`, to produce `Umi.h5`/`Uma.h5`/`Rma.h5`. This module
contains **no modulation specs, SNR grids, or variant lists of its own** — every
axis of the sweep comes from the config.

```python
_VARIANT_FILENAMES = {"umi": "Umi.h5", "uma": "Uma.h5", "rma": "Rma.h5"}
```

### `_batch_chunks(total: int, max_chunk: Optional[int]) -> list[int]`

Splits `total` (i.e. `ds.num_vectors`) into a list of chunk sizes no larger than
`max_chunk`:
- `max_chunk is None` or `max_chunk >= total` -> `[total]` (one chunk, the whole
  thing).
- Otherwise -> `[max_chunk] * (total // max_chunk)`, plus a final remainder chunk if
  `total % max_chunk != 0`.

### `_build_config(base_config: Config, variant: str, mod_spec: AmrModulationSpec) -> Config`

Produces one fully-formed `Config` for a single `(variant, modulation)` cell, via
`dataclasses.replace` on `base_config` (never mutating it):

- `common.active_channel` -> `"system_level"`.
- `channels.system_level` -> a copy of `base_config.channels.system_level` with:
  - `variant` set to the requested variant.
  - `enable_pathloss` = `sl.enable_pathloss and not ds.disable_pathloss_and_shadowing`.
  - `enable_shadow_fading` = `sl.enable_shadow_fading and not ds.disable_pathloss_and_shadowing`.
- `modulation` -> `mod_spec.to_modulation_config(ds.mapper, ds.demapper)`.
- `waveforms.time.num_time_samples` -> `ds.vector_len`.
- `sweep.batch_size` -> `ds.num_vectors`, `sweep.num_symbols` -> `ds.vector_len`
  (kept in sync for `Config.__post_init__`'s `num_symbols == num_time_samples` check,
  though this flow calls `generate_iq()` directly rather than `generate_sweep()`, so
  `sweep.batch_size` itself isn't read by anything downstream in this path).

Because this goes through `dataclasses.replace(base_config, ...)`, `Config.
__post_init__` reruns and re-validates the result — including the
`active_channel`/`active_waveform` membership checks and the `time`-waveform
`num_symbols`/`num_time_samples` equality check described in
[physys.schema](#2-physysschema).

### `generate_variant(base_config: Config, variant: str, out_path: Path) -> Path`

Generates one full `(variant)` file (all modulations x all SNR points):

1. `mod_order = [m.name for m in base_config.amr_dataset.modulations]`.
2. `f = open_amr_dataset(out_path, vector_len=ds.vector_len, mod_order=mod_order)`.
3. For each `(mod_index, mod_spec)` in `ds.modulations`:
   - Builds a fresh `PHYSys` from `_build_config(base_config, variant, mod_spec)`
     (a new channel model/mapper/demapper per modulation, since each modulation
     needs its own `Mapper`/`Demapper`/`Constellation`).
   - Sweeps `snr_db` from `ds.snr_db.start` to `ds.snr_db.stop` in steps of
     `ds.snr_db.step` (float-accumulating `while` loop — same caveat as
     `generate_sweep`'s Known Limitation #10).
   - At each `snr_db`, loops `_batch_chunks(ds.num_vectors, ds.max_batch_size)`,
     calling `sim.generate_iq(batch_size=chunk_size, num_symbols=ds.vector_len,
     ebno_db=float(snr_db))` per chunk and appending only `y` (the noisy received
     symbols) via `append_amr_batch(f, y, mod_index, float(snr_db))` — `bits`/`llr`/
     `x` from each `generate_iq` call are discarded in this path (this file's job is
     the AMR feature tensor, not BER/BLER or a clean-symbol reference).
4. `f.close()` in a `finally` block, so a partially-written file is still closed
   (though not removed) if generation raises partway through.

**Returns:** `out_path`.

### `generate_all(base_config: Config, out_dir: Union[str, Path]) -> dict`

Top-level entry point, called by `cli.run_dataset`:

1. Raises `ValueError` if `base_config.amr_dataset is None`.
2. Creates `out_dir` if it doesn't exist.
3. For each `variant` in `base_config.amr_dataset.variants`, calls
   `generate_variant(base_config, variant, out_dir / _VARIANT_FILENAMES[variant])`.
4. Returns `{variant: out_path, ...}` for every variant generated.

---

## 7. physys.cli

Thin `argparse` shell over `loader`/`runtime`/`export`/`amr_dataset`. The reusable
units of work live as plain functions (`run_single`, `run_sweep`, `run_sweep_iq`,
`run_dataset`); `main()`/the parser is just glue around them, so the same functions
are importable and testable without going through `sys.argv`.

```
config.json -> load_config -> PHYSys -> generate()/generate_sweep() -> export
config.json -> load_config -> PHYSys -> generate_sweep_iq() -> export (--iq)
config.json -> load_config -> amr_dataset.generate_all() -> Umi.h5/Uma.h5/Rma.h5
```

### `run_single(config_path: Path, out_path: Path, ebno_db: float) -> Path`

Loads the config, runs one `generate()` call using
`config.sweep.batch_size`/`config.sweep.num_symbols` (the same values a sweep would
use at each point, so a single `run` and a `sweep` point are directly comparable),
exports via `export.export_run`.

### `run_sweep(config_path: Path, out_path: Path) -> Path`

Loads the config, runs `generate_sweep()` over `config.sweep`'s Eb/N0 range, exports
via `export.export_sweep`.

### `run_sweep_iq(config_path: Path, out_path: Path) -> Path`

AMR-flavored counterpart to `run_sweep`: runs `generate_sweep_iq()` (bits/llr/x/y per
point) instead of `generate_sweep()`, exports via `export.export_sweep_iq` instead of
`export.export_sweep` — same file also gets the `"modulation"` attr derived from
`config.modulation`.

### `run_dataset(config_path: Path, out_dir: Path) -> dict`

Loads the config; if `config.amr_dataset is None`, raises `ConfigLoadError`.
Otherwise delegates straight to `amr_dataset.generate_all(config, out_dir)`.

### Subcommands (`argparse`)

| Subcommand | Required args | Optional args | Calls |
|---|---|---|---|
| `run` | `-o/--output` | `-c/--config` (default `./config.json`), `--ebno-db` (required) | `run_single` |
| `sweep` | `-o/--output` | `-c/--config` (default `./config.json`), `--iq` (flag) | `run_sweep` or `run_sweep_iq` (if `--iq`) |
| `dataset` | `-o/--output` | `-c/--config` (default `./config.json`) | `run_dataset` |

`sweep --iq`'s help text: *"Also capture/export raw tx symbols (x) and rx symbols
(y) for AMR dataset generation, via `generate_sweep_iq()`/`export_sweep_iq()` instead
of the bits/llr-only path. Output file also gets a `'modulation'` attr."*

`dataset`'s help text: *"Generate the AMR benchmark (`Umi.h5`/`Uma.h5`/`Rma.h5`): 6
modulations x 16 SNR levels x 1024 vectors x 1024 complex samples each."* — this
describes the specific benchmark config the project ships with, not a hardcoded
constraint of `amr_dataset.py` itself, which reads all of those numbers from
`config.amr_dataset` (see Known Limitations regarding what happens if a config
deviates from "6 modulations").

### `main(argv=None) -> int`

Dispatches on `args.command`:
- `run` -> prints `"Wrote single run (ebno_db = {ebno_db}) -> {out_path}"`.
- `sweep` (no `--iq`) -> runs `run_sweep`, reloads via `export.load_sweep`, prints the
  count and list of Eb/N0 points written.
- `sweep --iq` -> runs `run_sweep_iq`, reloads via `export.load_sweep_iq`, prints the
  same, from the `--iq` path.
- `dataset` -> runs `run_dataset`, prints one `"Wrote {variant} -> {path}"` line per
  file written.

**Error handling:**
- `ConfigLoadError` -> prints `"Config error: {e}"` to stderr, returns `1`.
- `NotImplementedError` -> prints `"Not implemented for this config: {e}"` to stderr,
  returns `1` (this is what surfaces if you point `sweep`/`run`/`dataset` at an
  `active_waveform="ofdm"` config with a non-AWGN channel — see
  [Known Limitations #1](#known-limitations--gotchas)).
- Any other exception is **not** caught by `main()` and propagates as a normal
  Python traceback.

---

## Configuration Reference (JSON)

Top-level keys required by `parse_config(raw)`:

```json
{
  "common":     { ... },
  "modulation": { ... },
  "channels":   { "awgn": {}, "tdl": {...}, "system_level": {...}, "rayleigh_block_fading": {} },
  "waveforms":  { "time": {...}, "ofdm": {...} },
  "sweep":      { ... },
  "amr_dataset": { ... }
}
```

- `source` is optional (`parse_source` ignores its content).
- `channels.tdl` and `channels.system_level` are **mandatory keys**, even if inactive.
- `waveforms.time` and `waveforms.ofdm` are **both mandatory keys**, even if only one
  is active.
- `amr_dataset` is **optional** — omit it entirely for a plain simulation config; it
  is only required by the CLI's `dataset` subcommand.
- Keys starting with `_` anywhere in these blocks are stripped before validation —
  use them freely for inline comments.

See field-by-field tables in the [physys.schema](#2-physysschema) section above for
every key, type, and default. `physys.export` does not read `config.json` itself — it
only stores whatever `schema.Config` object it's handed as a JSON attr for
provenance.

---

## End-to-End Execution Traces

### Plain run / sweep

```
load_config(path)
  └─ json.loads
  └─ parse_config(raw)
       └─ Config(common, source, modulation, channels, waveforms, sweep, amr_dataset)
            └─ __post_init__: validates active_channel/active_waveform,
                               and (if active_waveform=="time")
                               sweep.num_symbols == waveforms.time.num_time_samples

PHYSys(config)
  ├─ build_source(config)                -> BinarySource
  ├─ build_mapsys(config)                -> (Constellation, Mapper, Demapper)
  ├─ build_channel(config)               -> raw ChannelModel (AWGN/TDL/UMi/UMa/RMa/Rayleigh)
  └─ build_waveform(config, channel)     -> wrapped handle (may == raw channel if AWGN)

sim.generate(batch_size, num_symbols, ebno_db)
  └─ _generate_core(...) -> (bits, llr, x, y); generate() returns (bits, llr) only
       ├─ _maybe_set_topology(batch_size)     [system_level only]
       ├─ ebnodb2no(..., coderate=1.0)
       ├─ source(...) -> bits
       ├─ mapper(bits) -> x
       ├─ _apply_time_channel(x, no, num_symbols) -> y
       │     ├─ AWGN            -> handle(x, no)                [no reshape]
       │     ├─ "time" (non-AWGN) -> reshape -> handle -> truncate -> reshape back
       │     └─ "ofdm" (non-AWGN) -> NotImplementedError
       └─ demapper(y, no) -> llr

export.export_run(bits, llr, path, ebno_db=..., config=config)
  └─ h5py.File(path, "w") -> datasets "bits"/"llr" + attrs (created_utc, ebno_db, config_json)
```

### AMR dataset generation (`cli.py dataset`)

```
load_config(path) -> config    [config.amr_dataset must be set]

amr_dataset.generate_all(config, out_dir)
  └─ for variant in config.amr_dataset.variants:
       generate_variant(config, variant, out_dir / f"{Variant}.h5")
         ├─ open_amr_dataset(out_path, vector_len, mod_order) -> h5py.File (resizable)
         └─ for mod_index, mod_spec in enumerate(config.amr_dataset.modulations):
              cfg = _build_config(config, variant, mod_spec)
                 -> active_channel="system_level", modulation from mod_spec,
                    waveforms.time.num_time_samples = vector_len,
                    system_level.variant = variant,
                    enable_pathloss/enable_shadow_fading narrowed by
                    disable_pathloss_and_shadowing
              sim = PHYSys(cfg)
              for snr_db in [snr_db.start, snr_db.start+step, ..., <= snr_db.stop]:
                for chunk_size in _batch_chunks(num_vectors, max_batch_size):
                  _bits, _llr, _x, y = sim.generate_iq(chunk_size, vector_len, snr_db)
                  append_amr_batch(f, y, mod_index, snr_db)
         └─ f.close()
```

---

## Known Limitations & Gotchas

1. **OFDM is not runnable end-to-end.** `build_waveform` will happily construct an
   `OFDMChannel` for you, but `PHYSys.generate()`/`generate_iq()` raises
   `NotImplementedError` the moment it reaches `_apply_time_channel` for any
   non-AWGN OFDM config. The resource-grid reshape logic simply hasn't been written
   yet.

2. **TDL/time-waveform tail truncation loses energy.** `TimeChannel`'s output is
   longer than its input by `l_max - l_min` samples; the extra samples (belonging to
   the tail of the channel's impulse response) are truncated away rather than folded
   back in. Applies equally to `generate()` and `generate_iq()`.

3. **No FEC.** `self._coderate` is hardcoded to `1.0` everywhere `ebnodb2no` is
   called, in both `generate()`/`generate_iq()`.

4. **System-level topology only supports a single UT.** `num_ut=1` is hardcoded in
   `_maybe_set_topology`; multi-UT scenarios aren't wired up. Applies to AMR dataset
   generation too, since it always uses a `system_level` channel.

5. **`show_topology()` needs a prior `generate()`/`generate_iq()` call.** Topology
   tensors are only populated inside `_generate_core`, not in `__init__`.

6. **`o2i_model`/`average_street_width`/`average_building_height` are
   variant-specific.** `o2i_model` only applies to `"umi"`/`"uma"`;
   `average_street_width`/`average_building_height` only apply to `"rma"`. Setting
   the "wrong" one for a given `variant` is accepted by schema validation but has no
   effect on the built Sionna object.

7. **Precision/device resolution can end up fully implicit.** If neither a block nor
   `common` sets `precision`/`device`, `_pd_kwargs` omits both keys entirely and the
   underlying Sionna class's own default takes over — worth knowing if you're
   debugging a dtype/device mismatch that doesn't show up anywhere in the config.

8. **`ChannelsConfig`/`WaveformsConfig` require all sub-blocks in JSON**, active or
   not — `tdl`, `system_level`, `time`, and `ofdm` keys must all exist even if you're
   only running AWGN + time waveform.

9. **`generate_sweep()`/`generate_sweep_iq()`/`amr_dataset`'s SNR loop all use float
   accumulation** (`ebno_db += step` / `snr_db += step` in a `while` loop), so
   results-dict keys (or, for `amr_dataset`, the exact number of SNR points actually
   generated) can be sensitive to floating-point rounding when `step` doesn't evenly
   divide `stop - start`.

10. **`export_run`/`export_run_iq` overwrite, they don't append; `export_sweep`/
    `export_sweep_iq` do too.** All four open their target file in `"w"` mode, so
    calling any of them in a loop against the same `path` destroys the previous
    write rather than accumulating results. `open_amr_dataset` also opens in `"w"`
    mode — but `amr_dataset.py` is designed around this (one file per variant,
    opened once, then incrementally grown via `append_amr_batch` within that single
    open handle), so this isn't a gotcha for that path specifically.

11. **`load_run`'s `"config"` is a `dict`, not a `Config`.** `export_run`/
    `export_run_iq`/`export_sweep`/`export_sweep_iq` serialize `config` via
    `dataclasses.asdict`, and the corresponding `load_*` functions hand it back via
    plain `json.loads` — there's no automatic reconstruction into a `schema.Config`
    instance (including its `amr_dataset` sub-tree, when present), so code that needs
    a live `Config` object back has to rebuild it itself.

12. **`load_sweep`/`load_sweep_iq` don't surface file-level attrs.**
    `created_utc`/`config_json`/`modulation` are written at the file root by
    `export_sweep`/`export_sweep_iq` but aren't included in what `load_sweep`/
    `load_sweep_iq` return — read them directly via `h5py.File(path).attrs` if
    needed.

13. **`generate_variant` opens its output file once per variant and closes it in a
    `finally` block.** If generation is interrupted partway through (e.g. by an
    exception raised inside the modulation/SNR loop), the file is still closed but
    is left on disk exactly as far as generation got — there's no atomic
    write-then-rename and no partial-file marker.

---

## Full config.json Examples

### Plain simulation (TDL channel, time waveform, no FEC)

```json
{
  "_comment": "Example: TDL channel over a time-domain waveform, no FEC.",
  "common": {
    "active_channel": "tdl",
    "active_waveform": "time",
    "precision": "single",
    "device": "cpu"
  },
  "modulation": {
    "type": "qam",
    "num_bits_per_symbol": 4,
    "mapper": { "return_indices": false },
    "demapper": { "demapping_method": "app", "hard_out": false },
    "constellation": { "normalize": true, "center": false, "points": null }
  },
  "channels": {
    "awgn": {},
    "tdl": {
      "model": "A",
      "delay_spread": 100e-9,
      "carrier_frequency": 3.5e9,
      "num_rx_ant": 1,
      "num_tx_ant": 1
    },
    "system_level": {
      "variant": "umi",
      "carrier_frequency": 3.5e9,
      "o2i_model": "low",
      "direction": "downlink"
    },
    "rayleigh_block_fading": {}
  },
  "waveforms": {
    "time": {
      "bandwidth": 1e6,
      "num_time_samples": 64
    },
    "ofdm": {
      "resource_grid": {
        "num_ofdm_symbols": 14,
        "fft_size": 64,
        "subcarrier_spacing": 15e3
      }
    }
  },
  "sweep": {
    "ebno_db": { "start": 0.0, "stop": 10.0, "step": 2.0 },
    "batch_size": 32,
    "num_symbols": 64
  }
}
```

> Note `num_symbols` (64) matches `waveforms.time.num_time_samples` (64) — enforced
> by `Config.__post_init__` since `active_waveform` is `"time"`.

### AMR dataset generation (adds `amr_dataset`)

The `common`/`modulation`/`channels`/`waveforms`/`sweep` blocks above are still
required (and still validated) even for a `dataset` run — most of their content is
overridden per-cell by `amr_dataset._build_config`, but the file must still parse.
Add:

```json
{
  "amr_dataset": {
    "_comment": "6 modulations x 16 SNR levels x 1024 vectors x 1024 samples/vector.",
    "variants": ["umi", "uma", "rma"],
    "snr_db": { "start": -10.0, "stop": 20.0, "step": 2.0 },
    "num_vectors": 1024,
    "vector_len": 1024,
    "max_batch_size": 128,
    "disable_pathloss_and_shadowing": true,
    "mapper": { "return_indices": false },
    "demapper": { "demapping_method": "app", "hard_out": false },
    "modulations": [
      { "name": "BPSK",   "type": "custom", "num_bits_per_symbol": 1,
        "constellation": { "points": [[-1.0, 0.0], [1.0, 0.0]] } },
      { "name": "QPSK",   "type": "qam", "num_bits_per_symbol": 2, "constellation": {} },
      { "name": "16QAM",  "type": "qam", "num_bits_per_symbol": 4, "constellation": {} },
      { "name": "64QAM",  "type": "qam", "num_bits_per_symbol": 6, "constellation": {} },
      { "name": "256QAM", "type": "qam", "num_bits_per_symbol": 8, "constellation": {} },
      { "name": "1024QAM","type": "qam", "num_bits_per_symbol": 10, "constellation": {} }
    ]
  }
}
```

> `snr_db.start=-10.0`, `stop=20.0`, `step=2.0` -> 16 points (`(20 - (-10)) / 2 + 1`),
> matching the `dataset` subcommand's advertised "16 SNR levels" — subject to
> Known Limitation #9's float-accumulation caveat.