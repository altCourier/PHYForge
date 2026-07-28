# PHYSys Architecture

Status: design draft. This document describes the intended architecture before implementation, so decisions can be reviewed and argued with before code is written. Sections marked **OPEN QUESTION** are unresolved and should not be treated as settled.

## Core philosophy

1. **Pure composition — never reimplement, only compose.** PHYSys does not reimplement any signal-processing logic, ever. Every operation is a direct call into a Sionna class or function. This is a hard rule, not a preference: before writing any new function in `builders.py` or `runtime.py`, check whether Sionna already exposes it. Given the breadth of Sionna's own API — sources, mappers, every channel model family, FEC, OFDM resource grids/modulators/estimators/equalizers, filters — the honest expectation is that PHYSys should almost never need to write actual DSP code. If a builder function is doing math instead of instantiating and wiring Sionna objects, that's a signal something's wrong. PHYSys's own code is limited to: parsing config, validating it, and constructing/wiring the corresponding Sionna objects together.

2. **One file, discriminated schema.** All experiment configuration lives in a single `config.json`. Sections that have fundamentally different shapes depending on a choice (e.g. channel model) use a `type` discriminator field, and the rest of that section's fields are interpreted according to `type`. This is the same pattern used by e.g. Kubernetes manifests or Terraform resource blocks — one file, but not a flat schema.
3. **The builder owns branching, not the caller.** Code that consumes PHYSys never branches on config values itself. All "if type is X, construct Y" logic lives inside PHYSys's builder layer, in exactly one place per section.
4. **Adding a new variant should not require touching existing variants.** Supporting a new channel model, source type, or modulation should mean adding a new schema variant + a new builder branch — not modifying how existing variants are parsed or built.

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
│   ├── schema.py       # all section schemas (source, modulation, channel, sweep, export)
│   │                    # -- pure data + validation, NO Sionna imports at all
│   ├── loader.py         # config.json -> validated schema objects
│   ├── builders.py        # schema -> live Sionna objects, one function per channel type
│   ├── runtime.py          # PHYSys.generate() / generate_sweep()
│   ├── export.py            # write generated samples to disk
│   └── cli.py                # e.g. `physys run config.json`
├── examples/configs/
├── tests/
├── README.md
├── ARCHITECTURE.md
├── LICENSE
└── pyproject.toml
```

**Deliberately flat, not pre-split into subpackages.** An earlier draft of this structure split `builders/channel/` into its own subpackage (`awgn.py`, `tdl.py`, `system_level.py`) before any code existed to justify it — designing for hypothetical future complexity instead of what's actually there. The rule going forward: start with one file per layer (`schema.py`, `builders.py`, etc.); a file only gets split once something inside it actually earns the split by growing unwieldy (the way `fading.py` was split out of `channel.py` in the originating project, only after the TDL-specific logic got big enough to deserve its own file). No speculative subpackages ahead of real code.

**No duplication between `schema.py` and `builders.py`.** These stay strictly separate concerns and should never both contain logic for the same thing:
- `schema.py` only describes shape and validity — "is this JSON well-formed for a `tdl` channel." It has zero Sionna imports, so it can be unit-tested without Sionna installed at all.
- `builders.py` only consumes an already-validated schema object and constructs Sionna objects from it — it never re-validates or re-interprets what a field means, it trusts the schema layer already did that.

If a rule about a field's meaning is ever duplicated in both files (e.g. "delay_spread must be positive" checked in both schema validation and again inside the builder), that's a bug in the design — it means the two layers' responsibilities have blurred together.

## Config schema (draft shape)

Top-level sections:

- `source` — discriminated on `type` (`binary`, later `text`/`audio`)
- `modulation` — constellation config (`type`, `num_bits_per_symbol`, or a custom constellation reference)
- `channel` — discriminated on `type`; this is the section with the most divergent shapes between variants (see below)
- `sweep` — what varies across the dataset generation run (SNR range, batch size, num_symbols, repetitions)
- `export` — output format/location for generated samples

### Channel section: the two known branches

**`type: "awgn"`** — minimal, no fading:
```json
{ "type": "awgn" }
```

**`type: "tdl"`** — 3GPP TDL multipath:
```json
{
  "type": "tdl",
  "model": "A",
  "delay_spread": 100e-9,
  "carrier_frequency": 140e9,
  "min_speed": 0.0,
  "max_speed": null,
  "bandwidth": 1e6,
  "l_min": null,
  "l_max": null
}
```

**`type: "umi" | "uma" | "rma"`** — 3GPP system-level models. Structurally different from `tdl`, not a superset of it: requires antenna array definitions and a topology, neither of which `tdl` has any concept of.
```json
{
  "type": "umi",
  "carrier_frequency": 140e9,
  "o2i_model": "low",
  "direction": "downlink",
  "enable_pathloss": true,
  "enable_shadow_fading": true,
  "bs_array": {
    "num_rows_per_panel": 1,
    "num_cols_per_panel": 1,
    "polarization": "single",
    "polarization_type": "V",
    "antenna_pattern": "38.901"
  },
  "ut_array": {
    "num_rows_per_panel": 1,
    "num_cols_per_panel": 1,
    "polarization": "single",
    "polarization_type": "V",
    "antenna_pattern": "omni"
  }
}
```

This is why the schema is discriminated rather than flat: a flat schema with all possible fields present (many left null depending on type) would be misleading about which fields actually apply, and wouldn't scale cleanly to a fourth or fifth channel type.

## Builder layer responsibilities

`builders.py` holds one function per section type. Every function's job is to look up Sionna classes and wire them — nothing computed by hand:

- `build_channel(config) -> ChannelHandle` dispatches on `config.type`:
  - `awgn` → `sionna.phy.channel.AWGN`, used directly
  - `tdl` → `sionna.phy.channel.tr38901.TDL` + `sionna.phy.channel.TimeChannel`, handles the reshape from flat symbol sequences to `TimeChannel`'s expected `[batch, num_tx, num_tx_ant, num_time_samples]`, and the filter-tail policy (see Open Questions) — the reshape/tail glue is the one piece of this branch that is PHYSys's own code, since it's just shape bookkeeping, not signal processing
  - `umi`/`uma`/`rma` → `sionna.phy.channel.tr38901.PanelArray` (x2, for `bs_array`/`ut_array`) + `sionna.phy.channel.tr38901.{UMi,UMa,RMa}`, topology from `gen_single_sector_topology`
  - future `cdl` → `sionna.phy.channel.tr38901.CDL`, same pattern
- `build_source(config)` → `sionna.phy.mapping.BinarySource` (and later `SymbolSource`/`PAMSource`/`QAMSource` if those cover `text`/`audio`-style sources better than hand-rolling one)
- `build_mapsys(config)` → `sionna.phy.mapping.Mapper` + `Demapper` + `Constellation`
- future OFDM builder → `sionna.phy.ofdm.ResourceGrid` + `ResourceGridMapper` + `OFDMModulator`/`OFDMDemodulator` + `sionna.phy.channel.OFDMChannel` — again, entirely existing Sionna classes, just wired differently than the single-carrier path

Each branch returns a common interface (e.g. a callable `handle(x, no) -> y`) so `runtime.py` doesn't need to know which variant it's holding.

## Extensibility model

Adding CDL support, for example, means:
1. Add a `cdl` schema variant to `schema.py` (new field shapes as needed).
2. Add one new branch to `build_channel` in `builders.py`, wiring `sionna.phy.channel.tr38901.CDL` — an existing Sionna class, not new DSP code.
3. Nothing else changes — existing `awgn`/`tdl`/`umi` configs and their builder branches are untouched.

This is the property that makes "only touch config.json" true for *users* of PHYSys, while being honest that a new variant is real (if isolated) implementation work for whoever's extending PHYSys itself.

## Open questions

These are unresolved and materially affect the schema/architecture. Listed here so they're visible rather than silently assumed.

### 1. Waveform: single-carrier vs. OFDM
The current pipeline (Source → Mapper → Channel → Demapper) is single-carrier — there is no resource grid, IFFT, or cyclic prefix anywhere in it. `TimeChannel` fits this. If OFDM support is added later, it is **not** a new `channel.type` variant — it changes the shape of the entire pipeline (a `resource_grid` config section, a mapper stage, `OFDMChannel` instead of `TimeChannel`). This likely needs a top-level `waveform: "single_carrier" | "ofdm"` discriminator that changes which pipeline stages exist at all, not just how the channel section is parsed.

### 2. TDL filter-tail handling
`TimeChannel`'s output is `l_max - l_min` samples longer than its input. Two options:
- Truncate the output back to the input length (simple, but silently discards energy belonging to the last few symbols).
- Zero-pad the input with `l_max - l_min` flush symbols before the channel, then decide how to treat those extra positions downstream.

Not yet decided which should be the default, or whether this should be a config field itself (`channel.tail_handling: "truncate" | "pad"`).

### 3. Sionna `Block` integration
Sionna's own components (e.g. `TimeChannel`) subclass `sionna.phy.Block`. Whether PHYSys's own wrapper classes should also subclass `Block` (for consistency with Sionna's build/call/forward lifecycle) or remain plain Python classes is undecided — depends on fully reviewing `Block`'s contract first.

### 4. FEC
Whether the pipeline includes forward error correction at all is unresolved upstream of this project (see project notes). If added, it's presumably another discriminated section (`fec: { type: "none" | "ldpc" | ... }`) sitting between mapping and the channel.

## Non-goals (for now)

- Ray tracing (`sionna.rt`) — explicitly out of scope for the originating project; not planned for an initial version.
- MIMO beyond what a channel model natively supports via antenna array config.
- A GUI or notebook widget — config.json is the only intended interface for v1.