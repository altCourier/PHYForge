# PHYSys

**A single config file drives an entire Sionna physical-layer pipeline — source, mapping, channel, and sweep — with no code changes between experiments.**

> Status: early design / planning stage. Schema and architecture are being drafted before implementation begins. Nothing in this README describes shipped functionality yet.

## Why

[Sionna](https://github.com/NVlabs/sionna) gives you the building blocks for PHY-layer simulation — sources, mappers, channel models, FEC — but every project ends up writing the same kind of glue code to wire them together: instantiate a source, instantiate a mapper, instantiate a channel, run a sweep. That glue code tends to be rewritten per-project, and switching something fundamental (e.g. an AWGN link to a 3GPP multipath model) usually means editing several files, not one.

PHYSys is that glue code, written once, driven entirely by a config file:

```json
{
  "source": { "type": "binary" },
  "modulation": { "type": "qam", "num_bits_per_symbol": 4 },
  "channel": {
    "type": "tdl",
    "model": "A",
    "delay_spread": 100e-9,
    "carrier_frequency": 140e9
  },
  "sweep": {
    "ebno_db": { "start": -10, "stop": 10, "step": 2 },
    "batch_size": 8,
    "num_symbols": 100
  }
}
```

Change `channel.type` from `"tdl"` to `"umi"` (or `"uma"`, `"rma"`), adjust that section's fields, and PHYSys builds the right object graph for you — no code edits.

## Who this is for

Anyone using Sionna to run parameter sweeps for dataset generation, benchmarking, or reproducible experiment configs — particularly useful when you want to hand off or version-control *what a run was* without shipping a script.

## Quick start

> Installation and usage instructions will be added once the first implementation lands. Sionna PHY itself requires Python 3.11+ and PyTorch 2.9+.

## How it works (short version)

A `config.json` is parsed against a **discriminated schema** — one field (`type`) selects which variant of a section applies, and the rest of that section's shape depends on it. A builder layer reads the resolved config and constructs the matching Sionna object graph (e.g. `TDL` + `TimeChannel` for a `tdl` channel section, vs. two `PanelArray`s + topology + `UMi` for a `umi` section). See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Project origin

PHYSys grew out of wrapper classes (`Source`, `MapSys`, `Channel`, `PHYSys`) originally written for a THz-band AMC dataset-generation project. This repository generalizes that pattern into a standalone, config-first tool.

## Roadmap

- [ ] Finalize discriminated config schema (single-carrier channels: AWGN, TDL)
- [ ] Builder layer for schema → Sionna object graph
- [ ] System-level channel support (UMi/UMa/RMa: antenna arrays, topology)
- [ ] OFDM waveform support (resource grid, modulator/demodulator) — separate pipeline shape from single-carrier, see ARCHITECTURE.md open questions
- [ ] Sweep execution + dataset export
- [ ] Documentation + examples

## Contributing

Not yet open for contributions — architecture is still being drafted. Issues/discussion welcome once the schema is finalized.

## License

TBD (likely Apache-2.0, matching Sionna's own license, pending confirmation).