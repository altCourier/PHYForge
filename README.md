# sionna-practice

Practice code and findings from my summer internship at Universität zu Lübeck,
working with [Sionna](https://nvlabs.github.io/sionna/) for physical-layer (PHY)
link-level simulation.

This is a personal working repository, not a general-purpose library — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Layout

```
.
.
├── docs                # Sionna and other libraries learning material (exercises, notebooks, notes)
│   ├── exercises
│   ├── notebooks
│   └── notes
├── legacy              # earlier prototype(s), kept for reference only
│   └── phy-system
└── phyforge            # active project: config-driven PHY simulation + AMR dataset generation
    ├── diagnostics
    ├── imgs
    ├── modulations
    │   ├── 1024-QAM
    │   ├── 16-QAM
    │   ├── 256-QAM
    │   ├── 64-QAM
    │   ├── BPSK
    │   └── QPSK
    ├── notebooks
    ├── physys
    └── tests
```

### `phyforge/` — the active project

`phyforge` turns a JSON config into a runnable `Source -> Map -> Channel -> Demap`
link-level simulation on top of Sionna-PHY, and can be used as an AMR
(Automatic Modulation Recognition) dataset generator.

- [`phyforge/ARCHITECTURE.md`](phyforge/ARCHITECTURE.md) — design, open questions,
  and the reasoning behind them
- [`phyforge/physys/README.md`](phyforge/physys/README.md) — package-level docs for
  the `physys` module (`schema` / `builders` / `runtime` / `export` / `cli`)
- `phyforge/modulations/` — per-modulation configs (BPSK through 1024-QAM) and
  generated sweep outputs
- `phyforge/diagnostics/` — BER waterfall, constellation scatter, LLR histogram, and
  topology-check plotting utilities
- `phyforge/tests/` — pytest suite

### `legacy/phy-system/`

An earlier, monolithic prototype of the same Source -> Map → Channel → Demap
pipeline. Predates the `schema` / `builders` / `runtime` / `export` split now used
in `phyforge`. Kept for reference; not actively developed.

### `exercises/`, `notebooks/`, `notes/`

Sionna learning material from earlier in the internship — tutorials, beginner
exercises, standards-comparison notes. Not part of the `phyforge` project itself.

## Status

Actively in progress as part of an ongoing internship. Expect rough edges,
half-finished config paths, and notebooks that read
more like diagnostic logs than polished writeups because that's what they are.
