# PHYSys API Documentation

`physys` is a configuration-driven orchestrator built on top of **Sionna-PHY**. It turns a
JSON configuration file into a runnable `Source -> Map -> Channel -> Demap` link-level
simulation, with no Python code required to change channel models, waveforms, or
modulation schemes.

```
Read JSON config -> Validate against schema -> Build Sionna objects -> Run simulation
```

```
Source -> Map -> Channel -> Demap
```

This document covers all four modules (`loader`, `schema`, `builders`, `runtime`),
every configuration field with its default, and the behavioral edge cases that show up
in the current implementation.

---

## Table of Contents

1. [Quickstart](#quickstart)
2. [Package Layout](#package-layout)
3. [physys.loader](#1-physysloader)
4. [physys.schema](#2-physysschema)
5. [physys.builders](#3-physysbuilders)
6. [physys.runtime](#4-physysruntime)
7. [Configuration Reference (JSON)](#configuration-reference-json)
8. [End-to-End Execution Trace](#end-to-end-execution-trace)
9. [Known Limitations & Gotchas](#known-limitations--gotchas)
10. [Full config.json Example](#full-configjson-example)

---

## Quickstart

```python
from physys.loader import load_config
from physys.runtime import PHYSys

config = load_config("path/to/config.json")

sim = PHYSys(config)

# Single run
bits, llr = sim.generate(batch_size=32, num_symbols=64, ebno_db=5.0)

# Sweep over config.sweep.ebno_db
results = sim.generate_sweep()   # {ebno_db: (bits, llr), ...}
```

`PHYSys` builds every Sionna component exactly once, in `__init__`. Only
`batch_size`, `num_symbols`, and `ebno_db` vary between calls to `generate()`.

---

## Package Layout

| Module | Responsibility | Imports Sionna/PyTorch? |
|---|---|---|
| `physys.loader` | File I/O: path -> dict -> `Config` | No |
| `physys.schema` | Dataclasses + validation for the config tree | **No** (by design) |
| `physys.builders` | Validated `Config` -> live Sionna objects | Yes |
| `physys.runtime` | Orchestrates built objects into `generate()` / `generate_sweep()` | Yes |

`schema.py` is intentionally free of heavy imports.

---

## 1. physys.loader

### `class ConfigLoadError(Exception)`
The single exception type raised by `load_config`. Wraps every possible underlying
failure: missing file, unreadable file, invalid JSON, or schema validation failure. So callers only need one `except` clause.

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
that goes through a `parse_*` function.

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
No fields. A placeholder. seeding could be added later but is
"postponed" for now.

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

**`ModulationConfig`** — all fields required, no defaults:
| Field | Type |
|---|---|
| `type` | `Literal["qam","pam"]` |
| `num_bits_per_symbol` | `int` |
| `mapper` | `MapperConfig` |
| `demapper` | `DemapperConfig` |
| `constellation` | `ConstellationConfig` |

`__post_init__` re-checks that `type` is `"qam"`/`"pam"` and that
`num_bits_per_symbol` is an `int`, raising `ValueError` otherwise (belt-and-braces on
top of the `Literal` type hint, which isn't enforced at runtime by plain dataclasses).

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
| `bs_array` | `PanelArrayConfig` | `PanelArrayConfig()` |
| `ut_array` | `PanelArrayConfig` | `PanelArrayConfig(antenna_pattern="omni")` |
| `precision` / `device` | `Optional[str]` | `None` |

`__post_init__` re-checks `variant ∈ {"umi","uma","rma"}`.

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

**`WaveformsConfig`** — both `time` and `ofdm` are required, no defaults. `parse_waveforms`
reads `raw["time"]` and `raw["ofdm"]` unconditionally — **both blocks must be present
in JSON even if only one waveform is active.**

### Sweep block

**`EbNoRangeConfig`** — all required: `start: float`, `stop: float`, `step: float`.

**`SweepConfig`** — all required: `ebno_db: EbNoRangeConfig`, `batch_size: int`,
`num_symbols: int`.

### Top-level `Config`

```python
Config(common, source, modulation, channels, waveforms, sweep)
```
All six fields are required, no defaults.

`__post_init__` derives valid channel/waveform names directly from
`dataclasses.fields(ChannelsConfig)` / `dataclasses.fields(WaveformsConfig)` (rather
than a hand-maintained list, to avoid drift) and raises `ValueError` if
`common.active_channel` or `common.active_waveform` doesn't match one of those field
names.

### Parser functions

| Function | Behavior |
|---|---|
| `parse_common(raw)` | Strips comments, `CommonConfig(**raw)`. |
| `parse_source(raw)` | Always returns `BinarySourceConfig()` (ignores `raw`). |
| `parse_modulation(raw)` | Builds nested `mapper`/`demapper`/`constellation` from sub-dicts (each defaulting to `{}` if absent). |
| `parse_channels(raw)` | See `ChannelsConfig` note above — `tdl`/`system_level` are mandatory keys; `bs_array`/`ut_array` are popped out of `system_level` before constructing it. |
| `parse_waveforms(raw)` | `time`/`ofdm` are mandatory keys; `resource_grid` is popped out of `ofdm` before constructing it. |
| `parse_sweep(raw)` | `ebno_db`, `batch_size`, `num_symbols` are mandatory keys. |
| `parse_config(raw)` | Top-level entry point; calls all of the above and assembles `Config`. |

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
| `SystemLevelChannelConfig` | Builds `bs_array`/`ut_array` as `PanelArray(...)` (each stamped with the channel's `carrier_frequency`), picks `UMi`/`UMa`/`RMa` from `variant`, and constructs it with `carrier_frequency`, `ut_array`, `bs_array`, `direction`, `enable_pathloss`, `enable_shadow_fading`, plus `o2i_model` **only when `variant` is `"umi"` or `"uma"`** (RMa's constructor doesn't take it — if you set `o2i_model` on an `rma` config it is silently ignored), plus `pd_kwargs`. |
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

The main simulation orchestrator. Built once per `Config`; only `generate()`'s
arguments vary between calls.

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
and a sweep step.

> Because topology is only generated inside `generate()`, you must call `generate()`
> at least once before calling `sim._channel_model.show_topology()`.

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
  - **Unenforced assumption:** `num_symbols` must equal
    `waveforms.time.num_time_samples`. Nothing in the schema checks this.
- If `active_waveform == "ofdm"` (non-AWGN) -> **always raises `NotImplementedError`**.
  Mapping a flat `[batch, num_symbols]` stream onto a full OFDM resource grid (data
  REs only, skipping pilots/guards/DC) isn't implemented here yet, even though
  `build_waveform` happily constructs the `OFDMChannel` object itself.
- Any other `active_waveform` value raises `ValueError`.

**`generate(self, batch_size: int, num_symbols: int, ebno_db: float) -> tuple[Tensor, Tensor]`**

Executes one full pass:

1. `self._maybe_set_topology(batch_size)`
2. `no = ebnodb2no(ebno_db, num_bits_per_symbol=config.modulation.num_bits_per_symbol, coderate=self._coderate)`
3. `bits = self.source([batch_size, num_symbols * num_bits_per_symbol])`
4. `x = self.mapper(bits)` -> `[batch_size, num_symbols]`
5. `y = self._apply_time_channel(x, no, num_symbols)`
6. `llr = self.demapper(y, no)`
7. Returns `(bits, llr)` — metric computation (BER/BLER) is left to the caller.

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
`export.py`/callers decide serialization format. Note: this is a plain floating-point
accumulation loop — if `step` doesn't divide `(stop - start)` evenly, floating-point
drift can cause the last point to be included or excluded unexpectedly.

### `tf_reshape_for_time_channel(x, batch_size, num_tx_ant)`

```python
return x.reshape([batch_size, 1, num_tx_ant, -1])
```
`num_tx` is always `1` — PHYSys does not model multi-user uplink/downlink
multiplexing.

---

## Configuration Reference (JSON)

Top-level keys required by `parse_config(raw)`:

```json
{
  "common":     { ... },
  "modulation": { ... },
  "channels":   { "awgn": {}, "tdl": {...}, "system_level": {...}, "rayleigh_block_fading": {} },
  "waveforms":  { "time": {...}, "ofdm": {...} },
  "sweep":      { ... }
}
```

- `source` is optional (`parse_source` ignores its content).
- `channels.tdl` and `channels.system_level` are **mandatory keys**, even if inactive.
- `waveforms.time` and `waveforms.ofdm` are **both mandatory keys**, even if only one
  is active.
- Keys starting with `_` anywhere in these blocks are stripped before validation —
  use them freely for inline comments.

See field-by-field tables in the [physys.schema](#2-physysschema) section above for
every key, type, and default.

---

## End-to-End Execution Trace

```
load_config(path)
  └─ json.loads
  └─ parse_config(raw)
       └─ Config(common, source, modulation, channels, waveforms, sweep)
            └─ __post_init__: validates active_channel / active_waveform

PHYSys(config)
  ├─ build_source(config)                -> BinarySource
  ├─ build_mapsys(config)                -> (Constellation, Mapper, Demapper)
  ├─ build_channel(config)               -> raw ChannelModel (AWGN/TDL/UMi/UMa/RMa/Rayleigh)
  └─ build_waveform(config, channel)     -> wrapped handle (may == raw channel if AWGN)

sim.generate(batch_size, num_symbols, ebno_db)
  ├─ _maybe_set_topology(batch_size)     [system_level only]
  ├─ ebnodb2no(..., coderate=1.0)
  ├─ source(...) -> bits
  ├─ mapper(bits) -> x
  ├─ _apply_time_channel(x, no, num_symbols) -> y
  │     ├─ AWGN            -> handle(x, no)                [no reshape]
  │     ├─ "time" (non-AWGN) -> reshape -> handle -> truncate -> reshape back
  │     └─ "ofdm" (non-AWGN) -> NotImplementedError
  └─ demapper(y, no) -> llr
```

---

## Known Limitations & Gotchas

1. **OFDM is not runnable end-to-end.** `build_waveform` will happily construct an
   `OFDMChannel` for you, but `PHYSys.generate()` raises `NotImplementedError` the
   moment it reaches `_apply_time_channel` for any non-AWGN OFDM config. The
   resource-grid reshape logic simply hasn't been written yet.
2. **TDL/time-waveform tail truncation loses energy.** `TimeChannel`'s output is
   longer than its input by `l_max - l_min` samples; the extra samples (belonging to
   the tail of the channel's impulse response) are truncated away rather than folded
   back in.
3. **No FEC.** `self._coderate` is hardcoded to `1.0` everywhere `ebnodb2no` is
   called.
4. **`num_symbols` must equal `waveforms.time.num_time_samples`.** This is assumed by
   the time-domain reshape path but never checked or enforced by the schema — passing
   mismatched values will produce a shape error or silently wrong output depending on
   how Sionna's `TimeChannel` reacts.
5. **System-level topology only supports a single UT.** `num_ut=1` is hardcoded in
   `_maybe_set_topology`; multi-UT scenarios aren't wired up.
6. **`show_topology()` needs a prior `generate()` call.** Topology tensors are only
   populated inside `generate()`, not in `__init__`.
7. **`o2i_model` is silently dropped for RMa.** It's only passed through to the
   constructor for `"umi"`/`"uma"` variants; setting it under an `"rma"`
   `system_level` config has no effect.
8. **Precision/device resolution can end up fully implicit.** If neither a block nor
   `common` sets `precision`/`device`, `_pd_kwargs` omits both keys entirely and the
   underlying Sionna class's own default takes over — worth knowing if you're
   debugging a dtype/device mismatch that doesn't show up anywhere in the config.
9. **`ChannelsConfig`/`WaveformsConfig` require all sub-blocks in JSON**, active or
   not — `tdl`, `system_level`, `time`, and `ofdm` keys must all exist even if you're
   only running AWGN + time waveform.
10. **`generate_sweep()`'s Eb/N0 loop uses float accumulation** (`ebno_db += step` in
    a `while` loop), so results dict keys and the exact number of points can be
    sensitive to floating-point rounding when `step` doesn't evenly divide
    `stop - start`.

---

## Full config.json Example

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

> Note `num_symbols` (64) matches `waveforms.time.num_time_samples` (64) — required
> per gotcha #4 above, since this example's `active_waveform` is `"time"`.