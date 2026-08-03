"""Integration test for the full physys pipeline.

Exercises, end to end, exactly the flow described in the API docs:

    load_config(config.json) -> Config
        -> PHYSys(config)             (loader/schema/builders/runtime)
            -> sim.generate(...)       -> (bits, llr)
            -> sim.generate_sweep()    -> {ebno_db: (bits, llr)}
        -> export.export_run / load_run
        -> export.export_sweep / load_sweep

This is a real integration test: nothing here is mocked or stubbed. It
imports the actual `physys` package and builds real Sionna/PyTorch objects,
so it requires whatever `physys` itself requires to run (Sionna, TF/torch,
etc.) to be installed in the environment you run pytest in.

Usage
-----
Run from anywhere; the test will look for `config.json` at the repository
root (see `_find_config_path` below). If your layout differs, either:

    pytest --physys-config=/path/to/config.json
    PHYSYS_CONFIG_PATH=/path/to/config.json pytest

Batch sizes / symbol counts used here are intentionally small (not the
values in config.json's `sweep` block) purely to keep the test fast --
the sweep's Eb/N0 range/step is also shrunk at test time for the same
reason. The pipeline exercised is identical either way.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pytest

from physys.loader import ConfigLoadError, load_config
from physys.runtime import PHYSys
from physys.schema import AWGNChannelConfig, Config
from physys import export


# --------------------------------------------------------------------------- #
# pytest CLI / config plumbing
#
# NOTE: the --physys-config option itself is registered in conftest.py, next
# to this file -- pytest_addoption is only honored by pytest when it lives in
# a conftest.py (or a plugin), not in a regular test module. Make sure
# conftest.py sits alongside this file (or anywhere upward in the rootdir).
# --------------------------------------------------------------------------- #

def _find_config_path(request) -> Path:
    """Locate config.json: CLI flag > env var > repo-root search."""
    cli_value = request.config.getoption("--physys-config")
    if cli_value:
        path = Path(cli_value)
        if not path.is_file():
            pytest.fail(f"--physys-config path does not exist: {path}")
        return path

    env_value = os.environ.get("PHYSYS_CONFIG_PATH")
    if env_value:
        path = Path(env_value)
        if not path.is_file():
            pytest.fail(f"PHYSYS_CONFIG_PATH does not exist: {path}")
        return path

    # Walk upward from this test file looking for config.json, which covers
    # both "tests/ dir with config.json one level up" and "config.json next
    # to the test file" layouts.
    here = Path(__file__).resolve().parent
    for candidate_dir in [here, *here.parents]:
        candidate = candidate_dir / "config.json"
        if candidate.is_file():
            return candidate

    # Last resort: current working directory (e.g. `pytest` run from repo root).
    cwd_candidate = Path.cwd() / "config.json"
    if cwd_candidate.is_file():
        return cwd_candidate

    pytest.fail(
        "Could not locate config.json. Pass --physys-config=<path>, set "
        "PHYSYS_CONFIG_PATH, or place config.json at the repo root next to "
        "this test."
    )


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def config_path(request) -> Path:
    return _find_config_path(request)


@pytest.fixture(scope="session")
def config(config_path) -> Config:
    """The real, validated Config loaded from disk (loader + schema)."""
    return load_config(config_path)


@pytest.fixture(scope="session")
def sim(config) -> PHYSys:
    """A real PHYSys built from the loaded config (builders + runtime)."""
    return PHYSys(config)


def _is_awgn_active(config: Config) -> bool:
    active_channel_config = getattr(config.channels, config.common.active_channel)
    return isinstance(active_channel_config, AWGNChannelConfig)


def _safe_num_symbols(config: Config, default: int = 64) -> int:
    """Pick a num_symbols value that respects gotcha #4 in the docs:

    for a non-AWGN channel on the "time" waveform, num_symbols must equal
    waveforms.time.num_time_samples or the reshape/truncate path in
    _apply_time_channel will error or silently misbehave.
    """
    if config.common.active_waveform == "time" and not _is_awgn_active(config):
        return config.waveforms.time.num_time_samples
    return default

def _reset_topology(sim) -> None:
    """System-level channels (unlike AWGN) freeze their topology buffer
    shapes after the first set_topology() call -- a real constraint of the
    installed Sionna version, not a physys bug. Since `sim` is reused across
    tests/calls with different batch_size, reset before each generate call
    so a new batch_size doesn't hit the frozen-shape RuntimeError. No-op for
    channels (e.g. AWGN) that don't expose reset_topology.
    """
    reset = getattr(sim._channel_model, "reset_topology", None)
    if callable(reset):
        reset()


# --------------------------------------------------------------------------- #
# loader / schema
# --------------------------------------------------------------------------- #

def test_load_config_returns_validated_config(config):
    assert isinstance(config, Config)
    # active_channel/active_waveform must have been validated against the
    # real field names of ChannelsConfig/WaveformsConfig in Config.__post_init__.
    assert hasattr(config.channels, config.common.active_channel)
    assert hasattr(config.waveforms, config.common.active_waveform)


def test_load_config_missing_file_raises(tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(ConfigLoadError):
        load_config(missing)


def test_load_config_invalid_json_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{not valid json")
    with pytest.raises(ConfigLoadError):
        load_config(bad)


# --------------------------------------------------------------------------- #
# builders / runtime: construction
# --------------------------------------------------------------------------- #

def test_physys_builds_all_components(sim):
    assert sim.source is not None
    assert sim.mapper is not None
    assert sim.demapper is not None
    assert sim._channel_model is not None
    assert sim._handle is not None
    # No FEC support yet -- hardcoded per the docs.
    assert sim._coderate == 1.0


# --------------------------------------------------------------------------- #
# runtime: generate()
# --------------------------------------------------------------------------- #

def test_generate_single_run(config, sim):
    active_waveform = config.common.active_waveform
    awgn_active = _is_awgn_active(config)

    if active_waveform == "ofdm" and not awgn_active:
        # Documented as always unimplemented: OFDMChannel gets built, but
        # _apply_time_channel raises before doing any real work.
        with pytest.raises(NotImplementedError):
            sim.generate(batch_size=4, num_symbols=64, ebno_db=5.0)
        return

    num_symbols = _safe_num_symbols(config)
    bits, llr = sim.generate(batch_size=4, num_symbols=num_symbols, ebno_db=5.0)

    bits_np = export._to_numpy(bits)
    llr_np = export._to_numpy(llr)

    assert bits_np.shape[0] == 4
    assert llr_np.shape[0] == 4
    assert np.isfinite(np.asarray(llr_np)).all()
    # bits should be binary (0/1) regardless of channel/waveform.
    unique_vals = set(np.unique(np.round(bits_np)).tolist())
    assert unique_vals <= {0.0, 1.0}


# --------------------------------------------------------------------------- #
# export: single-run round trip
# --------------------------------------------------------------------------- #

def test_export_and_load_run_roundtrip(config, sim, tmp_path):
    if config.common.active_waveform == "ofdm" and not _is_awgn_active(config):
        pytest.skip("OFDM + non-AWGN generate() is not implemented yet.")

    num_symbols = _safe_num_symbols(config)
    bits, llr = sim.generate(batch_size=4, num_symbols=num_symbols, ebno_db=3.0)

    out_path = tmp_path / "run.h5"
    export.export_run(bits, llr, out_path, ebno_db=3.0, config=config)

    assert out_path.is_file()

    reloaded = export.load_run(out_path)

    np.testing.assert_array_equal(reloaded["bits"], export._to_numpy(bits))
    np.testing.assert_array_equal(reloaded["llr"], export._to_numpy(llr))
    assert reloaded["ebno_db"] == pytest.approx(3.0)

    # config comes back as a plain dict (json.loads), not a Config instance,
    # per the documented behavior -- compare against dataclasses.asdict.
    assert reloaded["config"] == json.loads(json.dumps(dataclasses.asdict(config)))


def test_export_run_overwrites_not_appends(config, sim, tmp_path):
    """Gotcha #11: export_run opens in 'w' mode -- a second call to the same
    path overwrites the first run rather than accumulating it."""
    if config.common.active_waveform == "ofdm" and not _is_awgn_active(config):
        pytest.skip("OFDM + non-AWGN generate() is not implemented yet.")

    num_symbols = _safe_num_symbols(config)
    out_path = tmp_path / "run.h5"

    bits1, llr1 = sim.generate(batch_size=4, num_symbols=num_symbols, ebno_db=0.0)
    export.export_run(bits1, llr1, out_path, ebno_db=0.0, config=config)

    bits2, llr2 = sim.generate(batch_size=4, num_symbols=num_symbols, ebno_db=8.0)
    export.export_run(bits2, llr2, out_path, ebno_db=8.0, config=config)

    reloaded = export.load_run(out_path)
    assert reloaded["ebno_db"] == pytest.approx(8.0)
    np.testing.assert_array_equal(reloaded["bits"], export._to_numpy(bits2))


def test_export_run_without_optional_fields(sim, config, tmp_path):
    """ebno_db/config are optional -- file must still load with None back."""
    if config.common.active_waveform == "ofdm" and not _is_awgn_active(config):
        pytest.skip("OFDM + non-AWGN generate() is not implemented yet.")

    num_symbols = _safe_num_symbols(config)
    _reset_topology(sim)
    bits, llr = sim.generate(batch_size=2, num_symbols=num_symbols, ebno_db=1.0)

    out_path = tmp_path / "run_minimal.h5"
    export.export_run(bits, llr, out_path)  # no ebno_db, no config

    reloaded = export.load_run(out_path)
    assert reloaded["ebno_db"] is None
    assert reloaded["config"] is None


# --------------------------------------------------------------------------- #
# runtime + export: sweep round trip
# --------------------------------------------------------------------------- #

def test_generate_sweep_and_export_roundtrip(config, sim, tmp_path):
    if config.common.active_waveform == "ofdm" and not _is_awgn_active(config):
        pytest.skip("OFDM + non-AWGN generate() is not implemented yet.")

    num_symbols = _safe_num_symbols(config)

    # Shrink the sweep for test speed. PHYSys reads sweep params from
    # self.config.sweep at call time (not at __init__ time), so mutating it
    # on the already-built sim before calling generate_sweep() exercises the
    # exact same code path generate_sweep() documents, just fewer points.
    original_sweep = sim.config.sweep
    small_sweep = dataclasses.replace(
        original_sweep,
        ebno_db=dataclasses.replace(
            original_sweep.ebno_db, start=0.0, stop=2.0, step=2.0
        ),
        batch_size=2,
        num_symbols=num_symbols,
    )
    sim.config.sweep = small_sweep
    try:
        _reset_topology(sim)
        results = sim.generate_sweep()
    finally:
        sim.config.sweep = original_sweep

    # start=0, stop=2, step=2 -> exactly {0.0, 2.0} baring float drift.
    assert set(round(k, 4) for k in results.keys()) == {0.0, 2.0}
    for bits, llr in results.values():
        assert export._to_numpy(bits).shape[0] == 2

    out_path = tmp_path / "sweep.h5"
    export.export_sweep(results, out_path, config=config)
    assert out_path.is_file()

    reloaded = export.load_sweep(out_path)

    assert set(reloaded.keys()) == set(results.keys())
    for ebno_db, (bits, llr) in results.items():
        r_bits, r_llr = reloaded[ebno_db]
        np.testing.assert_array_equal(r_bits, export._to_numpy(bits))
        np.testing.assert_array_equal(r_llr, export._to_numpy(llr))


def test_export_sweep_overwrites_not_appends(config, sim, tmp_path):
    if config.common.active_waveform == "ofdm" and not _is_awgn_active(config):
        pytest.skip("OFDM + non-AWGN generate() is not implemented yet.")

    num_symbols = _safe_num_symbols(config)
    original_sweep = sim.config.sweep

    def _tiny_sweep(stop):
        return dataclasses.replace(
            original_sweep,
            ebno_db=dataclasses.replace(
                original_sweep.ebno_db, start=0.0, stop=stop, step=stop or 1.0
            ),
            batch_size=2,
            num_symbols=num_symbols,
        )

    out_path = tmp_path / "sweep.h5"

    sim.config.sweep = _tiny_sweep(0.0)  # just {0.0}
    try:
        _reset_topology(sim)
        first_results = sim.generate_sweep()
    finally:
        sim.config.sweep = original_sweep
    export.export_sweep(first_results, out_path, config=config)

    sim.config.sweep = _tiny_sweep(4.0)  # {0.0, 4.0}
    try:
        _reset_topology(sim)
        second_results = sim.generate_sweep()
    finally:
        sim.config.sweep = original_sweep
    export.export_sweep(second_results, out_path, config=config)

    reloaded = export.load_sweep(out_path)
    assert set(round(k, 4) for k in reloaded.keys()) == {0.0, 4.0}


# --------------------------------------------------------------------------- #
# end-to-end smoke test mirroring the docs' Quickstart section verbatim
# --------------------------------------------------------------------------- #

def test_quickstart_end_to_end(config, tmp_path):
    """Mirrors the Quickstart section of the docs as closely as possible."""
    if config.common.active_waveform == "ofdm" and not _is_awgn_active(config):
        pytest.skip("OFDM + non-AWGN generate() is not implemented yet.")

    sim = PHYSys(config)
    num_symbols = _safe_num_symbols(config)

    _reset_topology(sim)
    bits, llr = sim.generate(batch_size=8, num_symbols=num_symbols, ebno_db=5.0)
    export.export_run(bits, llr, tmp_path / "run.h5", ebno_db=5.0, config=config)
    assert (tmp_path / "run.h5").is_file()

    original_sweep = sim.config.sweep
    sim.config.sweep = dataclasses.replace(
        original_sweep,
        ebno_db=dataclasses.replace(original_sweep.ebno_db, start=0.0, stop=2.0, step=2.0),
        batch_size=2,
        num_symbols=num_symbols,
    )
    try:
        _reset_topology(sim)
        results = sim.generate_sweep()
    finally:
        sim.config.sweep = original_sweep

    export.export_sweep(results, tmp_path / "sweep.h5", config=config)
    assert (tmp_path / "sweep.h5").is_file()

    reloaded_sweep = export.load_sweep(tmp_path / "sweep.h5")
    assert 0.0 in reloaded_sweep