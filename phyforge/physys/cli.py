"""
physys.cli
==========

Thin command-line entry point over the real pipeline:

    config.json -> load_config -> PHYSys -> generate()/generate_sweep() -> export
    config.json -> load_config -> PHYSys -> generate_sweep_iq() -> export (--iq)

Usage
-----
    python -m physys.cli run    -c config.json -o run.h5   --ebno-db 5.0
    python -m physys.cli sweep  -c config.json -o sweep.h5
    python -m physys.cli sweep  -c config.json -o sweep.h5 --iq
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from physys.loader import ConfigLoadError, load_config
from physys.runtime import PHYSys
from physys.schema import Config
from physys import export


# --------------------------------------------------------------------------- #
# Reusable library functions -- these are the actual units of work.
# main()/argparse below is just a thin shell around them.
# --------------------------------------------------------------------------- #

def run_single(config_path: Path, out_path: Path, ebno_db: float) -> Path:
    """
    Load config, run one generate() call, export to `out_path`.

    Uses config.sweep.batch_size / config.sweep.num_symbols for the run
    (same values the sweep would use at each point), so a single `run`
    and a `sweep` point are directly comparable.
    """

    config: Config = load_config(config_path)
    sim = PHYSys(config)

    bits, llr = sim.generate(
        batch_size=config.sweep.batch_size,
        num_symbols=config.sweep.num_symbols,
        ebno_db=ebno_db,
    )

    export.export_run(bits, llr, out_path, ebno_db=ebno_db, config=config)

    return out_path


def run_sweep(config_path: Path, out_path: Path) -> Path:
    """
    Load config, run generate_sweep() over config.sweep's Eb/N0 range,
    export all points to `out_path`.
    """

    config: Config = load_config(config_path)

    sim = PHYSys(config)

    results = sim.generate_sweep()

    export.export_sweep(results, out_path, config=config)

    return out_path


def run_sweep_iq(config_path: Path, out_path: Path) -> Path:
    """
    AMR counterpart to run_sweep(): load config, run
    generate_sweep_iq() over config.sweep's Eb/N0 range (bits/llr/x/y
    per point), export all points to `out_path` via export_sweep_iq()
    -- same file also gets the "modulation" attr derived from
    config.modulation.
    """

    config: Config = load_config(config_path)

    sim = PHYSys(config)

    results = sim.generate_sweep_iq()

    export.export_sweep_iq(results, out_path, config=config)

    return out_path


# --------------------------------------------------------------------------- #
# argparse shell
# --------------------------------------------------------------------------- #

def _build_parser() -> argparse.ArgumentParser:

    parser = argparse.ArgumentParser(
        prog = "physys",
        description = "Run the PHYSys pipeline (config.json -> generate -> export).",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    run_p = subparsers.add_parser("run", help="Single generate() call, exported to one .h5 file.")

    run_p.add_argument("-c", "--config", type=Path, default=Path("config.json"),
                        help="Path to config.json (default: ./config.json)")

    run_p.add_argument("-o", "--output", type=Path, required=True,
                        help="Output .h5 path")

    run_p.add_argument("--ebno-db", type=float, required=True,
                        help="Eb/N0 in dB for this run")

    sweep_p = subparsers.add_parser("sweep", help="generate_sweep() over config.sweep, exported to one .h5 file.")

    sweep_p.add_argument("-c", "--config", type=Path, default=Path("config.json"),
                          help="Path to config.json (default: ./config.json)")

    sweep_p.add_argument("-o", "--output", type=Path, required=True,
                          help="Output .h5 path")

    sweep_p.add_argument("--iq", action="store_true",
                          help="Also capture/export raw tx symbols (x) and rx symbols (y) "
                               "for AMR dataset generation, via generate_sweep_iq()/"
                               "export_sweep_iq() instead of the bits/llr-only path. "
                               "Output file also gets a 'modulation' attr.")

    return parser


def main(argv: list[str] | None = None) -> int:

    parser = _build_parser()

    args = parser.parse_args(argv)

    try:

        if args.command == "run":

            out_path = run_single(args.config, args.output, args.ebno_db)
            print(f"Wrote single run (ebno_db = {args.ebno_db}) -> {out_path}")

        elif args.command == "sweep":

            if args.iq:

                out_path = run_sweep_iq(args.config, args.output)

                reloaded = export.load_sweep_iq(out_path)

                ebno_points = sorted(reloaded.keys())

                print(f"Wrote IQ sweep ({len(ebno_points)} Eb/N0 points: {ebno_points}) "
                      f"-> {out_path}")

            else:

                out_path = run_sweep(args.config, args.output)

                reloaded = export.load_sweep(out_path)

                ebno_points = sorted(reloaded.keys())

                print(f"Wrote sweep ({len(ebno_points)} Eb/N0 points: {ebno_points}) -> {out_path}")

    except ConfigLoadError as e:

        print(f"Config error: {e}", file = sys.stderr)
        return 1

    except NotImplementedError as e:

        print(f"Not implemented for this config: {e}", file = sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())