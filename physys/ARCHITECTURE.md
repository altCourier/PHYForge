# PHYSys Architecture

Status: design draft. This document describes the intended architecture before implementation, so decisions can be reviewed and argued with before code is written. Sections marked **OPEN QUESTION** are unresolved and should not be treated as settled.

## Core philosophy

1. **Pure composition — never reimplement, only compose.** PHYSys does not reimplement any signal-processing logic, ever. Every operation is a direct call into a Sionna class or function. This is a hard rule, not a preference: before writing any new function in `builders.py` or `runtime.py`, check whether Sionna already exposes it. Given the breadth of Sionna's own API — sources, mappers, every channel model family, FEC, OFDM resource grids/modulators/estimators/equalizers, filters — the honest expectation is that PHYSys should almost never need to write actual DSP code. If a builder function is doing math instead of instantiating and wiring Sionna objects, that's a signal something's wrong. PHYSys's own code is limited to: parsing config, validating it, and constructing/wiring the corresponding Sionna objects together.

2. **One file, nested sections, decider fields for choice.** All experiment configuration lives in a single `config.json` at the repo root. Every possible variant of a section (every channel model, every waveform) is present in the file at all times, fully parameterized — nothing is deleted or commented out to switch experiments. A top-level `active_channel` field and a top-level `active_waveform` field are the only things that change between runs: they name which block under `channels` / `waveforms` is actually used. Everything else in those objects sits inert. This is deliberately different from a `type`-discriminator-per-section pattern (Kubernetes/Terraform-style) — it's closer to keeping every environment's config in one `docker-compose.yml` and picking which service actually runs.
3. **The code is the decider, not the config.** The config only ever supplies parameter values. It never encodes branching logic itself — `active_channel: "tdl"` is a plain string the loader reads and looks up; nothing about *how* that dispatch happens lives in the JSON. All "given this name, build that Sionna object" logic lives inside PHYSys's builder layer, in exactly one place per section.
4. **No duplicated shapes.** UMi, UMa, and RMa are not three separate config blocks — they share one `system_level` block with a `variant` field, because their parameters (antenna arrays, topology, pathloss/shadow-fading flags) are identical; only which Sionna class gets constructed differs. `precision` and `device` are declared once at the top level and inherited by every block; a block only needs its own `precision`/`device` entry if it's overriding the global value, not repeating it.
5. **Adding a new variant should not require touching existing variants.** Supporting a new channel model or waveform means adding a new block under `channels`/`waveforms` + a new builder branch — not modifying how existing blocks are parsed or built.

## High-level pipeline

```
config.json
     |
     v
[ schema.py + loader.py ]  -- reject invalid/incomplete configs early, with a clear error
     |
     v
[ builders.py ]  -- config -> live Sionna objects (Source, Mapper/Demapper, Channel, ...)
     |
     v
[ runtime.py ]  -- generate() / generate_sweep(), same shape as the current
                    PHYSys class: Source -> Map -> Channel -> Demap
     |
     v
[ export.py ]  -- write generated samples to disk
```

## File structure

```
physys/
├── physys/
│   ├── __init__.py
│   ├── schema.py       # all section dataclasses + registries (block name -> dataclass)
│   │                    # -- pure data + validation, NO Sionna imports at all
│   ├── loader.py         # config.json -> validated Config object; resolves active_channel/active_waveform
│   ├── builders.py        # schema -> live Sionna objects, one function per logical section
│   ├── runtime.py          # PHYSys.generate() / generate_sweep()
│   ├── export.py            # write generated samples to disk
│   └── cli.py                # e.g. `physys run` (reads config.json from repo root)
├── config.json           # the one config, at repo root — every module feeds from this
├── tests/
├── README.md
├── ARCHITECTURE.md
├── LICENSE
└── pyproject.toml
```

**Deliberately flat, not pre-split into subpackages.** An earlier draft split `builders/channel/` into its own subpackage (`awgn.py`, `tdl.py`, `system_level.py`) before any code existed to justify it — designing for hypothetical future complexity instead of what's actually there. The rule going forward: start with one file per layer (`schema.py`, `builders.py`, etc.); a file only gets split once something inside it actually earns the split by growing unwieldy. No speculative subpackages ahead of real code.

**No duplication between `schema.py` and `builders.py`.** These stay strictly separate concerns:
- `schema.py` only describes shape and validity — "is this JSON well-formed for a `tdl` block." It has zero Sionna imports, so it can be unit-tested without Sionna installed at all.
- `builders.py` only consumes an already-validated schema object and constructs Sionna objects from it — it never re-validates or re-interprets what a field means, it trusts the schema layer already did that.

If a rule about a field's meaning is ever duplicated in both files, that's a bug in the design — it means the two layers' responsibilities have blurred together.

## Config schema (current shape)

Top level:

- `precision`, `device` — global defaults, inherited by every block unless a block sets its own non-null value
- `source` — currently `binary` only
- `modulation` — `type`, `num_bits_per_symbol`, `demapping_method`, plus nested `mapper`, `demapper`, `constellation` blocks (these consume `type`/`num_bits_per_symbol`/`demapping_method` from the parent rather than repeating them)
- `active_channel` — string naming which block under `channels` is used this run
- `channels` — always contains all known channel blocks, fully parameterized; only `active_channel` decides which one runs
- `active_waveform` — string naming which block under `waveforms` is used this run
- `waveforms` — `time` (drives `TimeChannel`) and `ofdm` (drives `OFDMChannel` + a nested `resource_grid` block, since `ResourceGrid` is a distinct Sionna object from `OFDMChannel` itself)
- `sweep` — SNR range, batch size, num_symbols

### `channels` blocks

**`awgn`** — no channel-model-specific fields (still takes global `precision`/`device`):
```json
"awgn": { "precision": null, "device": null }
```

**`tdl`** — 3GPP TDL multipath. Note `bandwidth` is *not* a TDL parameter — it belongs to `TimeChannel` (see `waveforms.time`), a separate wrapping block:
```json
"tdl": {
  "model": "A",
  "delay_spread": 1e-7,
  "carrier_frequency": 1.4e11,
  "num_sinusoids": 20,
  "los_angle_of_arrival": 0.7853981633974483,
  "min_speed": 0.0,
  "max_speed": null,
  "num_rx_ant": 1,
  "num_tx_ant": 1,
  "spatial_corr_mat": null,
  "rx_corr_mat": null,
  "tx_corr_mat": null,
  "precision": null,
  "device": null
}
```

**`system_level`** — shared shape for UMi/UMa/RMa, `variant` picks the concrete Sionna class:
```json
"system_level": {
  "variant": "umi",
  "carrier_frequency": 1.4e11,
  "o2i_model": "low",
  "direction": "downlink",
  "enable_pathloss": true,
  "enable_shadow_fading": true,
  "bs_array": { "num_rows_per_panel": 1, "num_cols_per_panel": 1, "polarization": "single", "polarization_type": "V", "antenna_pattern": "38.901" },
  "ut_array": { "num_rows_per_panel": 1, "num_cols_per_panel": 1, "polarization": "single", "polarization_type": "V", "antenna_pattern": "omni" }
}
```

**`rayleigh_block_fading` — OPEN QUESTION.** Mechanically a clean fit (same `ChannelModel` interface as `TDL`: `(batch_size, num_time_steps, sampling_frequency) -> (a, tau)`), but nothing in the project's actual dataset spec (3GPP TS 38.211, UMi/UMa/RMa) calls for it. Not yet decided whether this is included as a pipeline smoke-test channel or left out entirely.

### `waveforms` blocks

**`time`** — drives `TimeChannel`:
```json
"time": {
  "bandwidth": 1e6,
  "num_time_samples": 100,
  "maximum_delay_spread": 3e-6,
  "l_min": null,
  "l_max": null,
  "normalize_channel": false,
  "return_channel": false,
  "precision": null,
  "device": null
}
```

**`ofdm`** — drives `OFDMChannel`, with a nested `resource_grid` block for the separate `ResourceGrid` object it requires:
```json
"ofdm": {
  "resource_grid": {
    "num_ofdm_symbols": 14,
    "fft_size": 128,
    "subcarrier_spacing": 30e3,
    "num_tx": 1,
    "num_streams_per_tx": 1,
    "cyclic_prefix_length": 16,
    "num_guard_carriers": [5, 6],
    "dc_null": true,
    "pilot_pattern": "kronecker",
    "pilot_ofdm_symbol_indices": [2, 11],
    "precision": null,
    "device": null
  },
  "normalize_channel": false,
  "return_channel": false,
  "precision": null,
  "device": null
}
```

## Builder layer responsibilities

`builders.py` holds one function per logical section. Every function's job is to look up Sionna classes and wire them — nothing computed by hand:

- `build_channel(config)` reads `active_channel`, looks up the matching block under `channels`, and dispatches:
  - `awgn` → `sionna.phy.channel.AWGN`, used directly
  - `tdl` → `sionna.phy.channel.tr38901.TDL` + `sionna.phy.channel.TimeChannel` (when `active_waveform == "time"`), handling the reshape from flat symbol sequences to `TimeChannel`'s expected `[batch, num_tx, num_tx_ant, num_time_samples]`, and the filter-tail policy (see Open Questions) — this reshape/tail glue is the one piece of this branch that is PHYSys's own code, since it's shape bookkeeping, not signal processing
  - `system_level` → dispatches again on `variant` (`umi`/`uma`/`rma`) to `sionna.phy.channel.tr38901.PanelArray` (x2, for `bs_array`/`ut_array`) + the matching `{UMi,UMa,RMa}` class, topology from `gen_single_sector_topology` — one function, one `if/elif` on `variant`, not three near-identical builder branches
- `build_source(config)` → `sionna.phy.mapping.BinarySource`
- `build_mapsys(config)` → `sionna.phy.mapping.Constellation`, then `Mapper` and `Demapper` both built from that same `Constellation` instance plus the parent `modulation` block's `type`/`num_bits_per_symbol`/`demapping_method` — the config states these once, the builder wires them into both
- `build_waveform(config)` reads `active_waveform` and dispatches:
  - `time` → wraps the channel in `sionna.phy.channel.TimeChannel`
  - `ofdm` → builds `sionna.phy.ofdm.ResourceGrid` from `waveforms.ofdm.resource_grid`, then `sionna.phy.channel.OFDMChannel`

Each branch returns a common interface (e.g. a callable `handle(x, no) -> y`) so `runtime.py` doesn't need to know which variant it's holding.

## Extensibility model

Adding a new channel or waveform means:
1. Add one new block under `channels` (or `waveforms`) in `config.json`, fully parameterized like the others.
2. Add one new branch to the relevant builder function, wiring an existing Sionna class — not new DSP code.
3. Nothing else changes — existing blocks, and the `active_channel`/`active_waveform` mechanism itself, are untouched.

## Open questions

### 1. TDL filter-tail handling
`TimeChannel`'s output is `l_max - l_min` samples longer than its input. Two options:
- Truncate the output back to the input length (simple, but silently discards energy belonging to the last few symbols).
- Zero-pad the input with `l_max - l_min` flush symbols before the channel, then decide how to treat those extra positions downstream.

Not yet decided which should be the default, or whether this should be a config field itself.

### 2. Sionna `Block` integration
Sionna's own components (e.g. `TimeChannel`) subclass `sionna.phy.Block`. Whether PHYSys's own wrapper classes should also subclass `Block` (for consistency with Sionna's build/call/forward lifecycle) or remain plain Python classes is undecided — depends on fully reviewing `Block`'s contract first.

### 3. FEC
Whether the pipeline includes forward error correction at all is unresolved upstream of this project. If added, it's presumably another block under a new top-level section, sitting between mapping and the channel.

### 4. RayleighBlockFading
See `channels.rayleigh_block_fading` above — mechanically supportable, not yet justified by the actual project scope.

## Non-goals (for now)

- Ray tracing (`sionna.rt`) — explicitly out of scope for the originating project.
- CDL — considered and explicitly excluded; TDL and system-level (UMi/UMa/RMa) cover the project's actual 3GPP scope.
- MIMO beyond what a channel model natively supports via antenna array config.
- A GUI or notebook widget — config.json is the only intended interface for v1.