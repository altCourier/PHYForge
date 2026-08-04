# PHYSys

**One config file drives an entire Sionna physical-layer pipeline — source, mapping, channel, waveform, and sweep — with no code changes between experiments.**

> Status: early design / planning stage. Schema and architecture are being drafted before implementation begins. Nothing in this README describes shipped functionality yet.

## Why

[Sionna](https://github.com/NVlabs/sionna) gives you the building blocks for PHY-layer simulation — sources, mappers, channel models, FEC — but every project ends up writing the same kind of glue code to wire them together: instantiate a source, instantiate a mapper, instantiate a channel, run a sweep. That glue code tends to be rewritten per-project, and switching something fundamental (e.g. an AWGN link to a 3GPP multipath model) usually means editing several files, not one.

PHYSys is that glue code, written once, driven entirely by a single config file. Every channel and waveform PHYSys supports is fully parameterized in the file at all times — switching experiments means changing two fields, `active_channel` and `active_waveform`, not editing or deleting config blocks:

```json
{
  "active_channel": "tdl",
  "active_waveform": "time",
  "channels": {
    "awgn": {},
    "tdl": {
      "model": "A",
      "delay_spread": 100e-9,
      "carrier_frequency": 140e9
    },
    "system_level": { "variant": "umi", "...": "..." }
  }
}
```

Change `active_channel` from `"tdl"` to `"system_level"` (with `variant: "uma"` or `"rma"`), and PHYSys builds the right object graph for you — no code edits.

## Who this is for

Anyone using Sionna to run parameter sweeps for dataset generation, benchmarking, or reproducible experiment configs — particularly useful when you want to hand off or version-control *what a run was* without shipping a script.

## Quick start

> Installation and usage instructions will be added once the first implementation lands. Sionna PHY itself requires Python 3.11+ and PyTorch 2.9+.

## How it works (short version)

`config.json`, at the repo root, holds every parameter for every supported channel and waveform at once. Two top-level fields — `active_channel` and `active_waveform` — name which blocks are actually used for a given run; everything else sits inert but ready. A builder layer reads the resolved config and constructs the matching Sionna object graph (e.g. `TDL` + `TimeChannel` for `active_channel: "tdl"`, vs. two `PanelArray`s + topology + `UMi`/`UMa`/`RMa` for `active_channel: "system_level"`). See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design.

## Project origin

PHYSys grew out of wrapper classes (`Source`, `MapSys`, `Channel`, `PHYSys`) originally written for a THz-band AMC dataset-generation project. This repository generalizes that pattern into a standalone, config-first tool.

## Roadmap

- [x] Finalize config schema (channels: AWGN, TDL, system-level UMi/UMa/RMa; waveforms: time, OFDM)
- [x] Builder layer for schema → Sionna object graph
- [ ] Sweep execution + dataset export
- [ ] Documentation + examples
