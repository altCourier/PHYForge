# PHYSys

**One config file drives an entire Sionna physical-layer pipeline — source, mapping, channel, waveform, and sweep — with no code changes between experiments.**

> Status: Confirming dataset outputs

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

## How it works (short version)

`config.json`, at the repo root, holds every parameter for every supported channel and waveform at once. Two top-level fields — `active_channel` and `active_waveform` — name which blocks are actually used for a given run; everything else sits inert but ready. A builder layer reads the resolved config and constructs the matching Sionna object graph (e.g. `TDL` + `TimeChannel` for `active_channel: "tdl"`, vs. two `PanelArray`s + topology + `UMi`/`UMa`/`RMa` for `active_channel: "system_level"`). 
 * See [ARCHITECTURE.md](./ARCHITECTURE.md) for the full design. 
 * Read API documentation from [README.md](/sionna-practice/phyforge/physys/README.md) 

## Generating a dataset (Makefile)

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

A `Makefile` at the repo root wraps the underlying CLI call (`python -m physys.cli sweep -c config.json -o sweep.h5 --iq`) so you don't have to run it once per modulation by hand.

**List available targets:**

```bash
make help
```

**Generate `sweep.h5` for every modulation:**

```bash
make dataset
```

This first runs `check-configs` to confirm every modulation folder actually has a `config.json`, then builds any `sweep.h5` that's missing or older than its config. Configs that haven't changed are skipped — only stale or missing outputs get regenerated.

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

**Notes:**
- Rebuilds are triggered by either the modulation's `config.json` changing or the CLI module (`physys/cli.py`) changing — not just by re-running `make`.
- Sweeps run **serially** by default (`.NOTPARALLEL:` in the Makefile), since they're assumed to share a single GPU. If your setup doesn't have that constraint, remove that line and use `make -j4 dataset` to parallelize across modulations.

## Project origin

PHYSys grew out of wrapper classes (`Source`, `MapSys`, `Channel`, `PHYSys`) originally written for a THz-band AMC dataset-generation project. This repository generalizes that pattern into a standalone, config-first tool.

## Roadmap

- [x] Finalize config schema (channels: AWGN, TDL, system-level UMi/UMa/RMa; waveforms: time, OFDM)
- [x] Builder layer for schema → Sionna object graph
- [x] Sweep execution + dataset export
- [x] Documentation + examples

## Revised Roadmap

### Phase 0 

- [ ] Draft the OFDM-scope "question" issue
- [ ] Resolve #28
- [ ] Resolve #39

### Phase 1  blockers, layout-independent
- [ ] #32
- [ ] #30
- [ ] #25
- [ ] #19

### Phase 2  combination infrastructure 
- [ ] Decide: append-mode in `export.py`, or a separate merge script that reads N per-modulation `sweep.h5` files and writes one scenario file?
- [ ] #29
- [ ] #31

### Phase 3  scenario-first restructuring
- [ ] Redesign `modulations/<MOD>/config.json` into a modulation x scenario matrix, update the Makefile accordingly, wire in the Phase 2 merge step per scenario

### Phase 4  deferred, tracked via the question issue
- [ ] #16 OFDM, and whatever #17/#19 turn into once OFDM is live