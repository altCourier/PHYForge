"""
Tests for physys.export.
"""

import dataclasses
import json

import h5py
import numpy as np
import pytest

from physys.export import export_run, export_sweep, load_run, load_sweep


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

class FakeTensor:
    """Stands in for a Sionna/TF eager tensor: anything with a `.numpy()` method."""

    def __init__(self, array):
        self._array = np.asarray(array)

    def numpy(self):
        return self._array


@dataclasses.dataclass
class FakeSubConfig:
    demapping_method: str = "app"


@dataclasses.dataclass
class FakeConfig:
    """Stand-in for schema.Config: a plain (possibly nested) dataclass."""

    active_channel: str = "awgn"
    num_bits_per_symbol: int = 4
    sub: FakeSubConfig = dataclasses.field(default_factory=FakeSubConfig)


@pytest.fixture
def bits_llr():
    rng = np.random.default_rng(0)
    bits = rng.integers(0, 2, size=(8, 32)).astype(np.float32)
    llr = rng.standard_normal((8, 32)).astype(np.float32)
    return bits, llr


# ---------------------------------------------------------------------------
# export_run / load_run
# ---------------------------------------------------------------------------

def test_export_run_roundtrip_minimal(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "run.h5"

    export_run(bits, llr, path)
    out = load_run(path)

    assert np.array_equal(out["bits"], bits)
    assert np.array_equal(out["llr"], llr)
    assert out["ebno_db"] is None
    assert out["config"] is None


def test_export_run_roundtrip_with_ebno_and_config(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "run.h5"
    config = FakeConfig(active_channel="tdl", num_bits_per_symbol=6)

    export_run(bits, llr, path, ebno_db=5.0, config=config)
    out = load_run(path)

    assert out["ebno_db"] == pytest.approx(5.0)
    assert out["config"] == dataclasses.asdict(config)
    # nested dataclass survives the JSON round trip too
    assert out["config"]["sub"]["demapping_method"] == "app"


def test_export_run_accepts_tensor_like_objects(tmp_path):
    """Anything with `.numpy()` (Sionna/TF tensors) should convert transparently."""
    bits = FakeTensor(np.ones((4, 16), dtype=np.float32))
    llr = FakeTensor(np.zeros((4, 16), dtype=np.float32))
    path = tmp_path / "run.h5"

    export_run(bits, llr, path)
    out = load_run(path)

    assert np.array_equal(out["bits"], np.ones((4, 16), dtype=np.float32))
    assert np.array_equal(out["llr"], np.zeros((4, 16), dtype=np.float32))


def test_export_run_creates_parent_directories(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "nested" / "dirs" / "run.h5"

    export_run(bits, llr, path)

    assert path.exists()
    out = load_run(path)
    assert np.array_equal(out["bits"], bits)


def test_export_run_overwrites_rather_than_appends(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "run.h5"

    export_run(bits, llr, path, ebno_db=1.0)
    new_bits = bits * 0  # distinguishable second payload
    export_run(new_bits, llr, path, ebno_db=2.0)

    out = load_run(path)
    assert np.array_equal(out["bits"], new_bits)
    assert out["ebno_db"] == pytest.approx(2.0)


def test_export_run_compression_none(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "run.h5"

    export_run(bits, llr, path, compression=None)
    out = load_run(path)

    assert np.array_equal(out["bits"], bits)
    with h5py.File(path, "r") as f:
        assert f["bits"].compression is None


def test_export_run_records_created_utc(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "run.h5"

    export_run(bits, llr, path)

    with h5py.File(path, "r") as f:
        # ISO-ish "YYYY-MM-DDTHH:MM:SSZ" -- just check it parses as that shape
        ts = f.attrs["created_utc"]
        assert len(ts) == 20
        assert ts[4] == "-" and ts[10] == "T" and ts[-1] == "Z"


# ---------------------------------------------------------------------------
# export_sweep / load_sweep
# ---------------------------------------------------------------------------

def test_export_sweep_roundtrip(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "sweep.h5"
    results = {
        4.0: (bits, llr),
        0.0: (bits + 1, llr + 1),
        2.0: (bits + 2, llr + 2),
    }

    export_sweep(results, path)
    out = load_sweep(path)

    assert set(out.keys()) == {0.0, 2.0, 4.0}
    for ebno_db, (b, l) in results.items():
        loaded_b, loaded_l = out[ebno_db]
        assert np.array_equal(loaded_b, b)
        assert np.array_equal(loaded_l, l)


def test_export_sweep_matches_by_attr_not_group_name(tmp_path, bits_llr):
    """load_sweep must key off each group's `ebno_db` attr, not by parsing names."""
    bits, llr = bits_llr
    path = tmp_path / "sweep.h5"
    export_sweep({1.5: (bits, llr)}, path)

    with h5py.File(path, "r+") as f:
        # rename the group to something that wouldn't parse back to 1.5
        f.move("ebno_db=1.5000", "point_A")
        assert f["point_A"].attrs["ebno_db"] == 1.5

    out = load_sweep(path)
    assert set(out.keys()) == {1.5}


def test_export_sweep_records_sorted_ebno_points_attr(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "sweep.h5"
    results = {4.0: (bits, llr), 0.0: (bits, llr), 2.0: (bits, llr)}

    export_sweep(results, path)

    with h5py.File(path, "r") as f:
        assert list(f.attrs["ebno_db_points"]) == [0.0, 2.0, 4.0]


def test_export_sweep_allows_per_point_shapes(tmp_path):
    """generate_sweep() reuses one batch_size/num_symbols today, but export_sweep
    itself shouldn't assume every point shares a shape."""
    path = tmp_path / "sweep.h5"
    results = {
        0.0: (np.zeros((4, 8)), np.zeros((4, 8))),
        2.0: (np.zeros((16, 32)), np.zeros((16, 32))),
    }

    export_sweep(results, path)
    out = load_sweep(path)

    assert out[0.0][0].shape == (4, 8)
    assert out[2.0][0].shape == (16, 32)


def test_export_sweep_with_config(tmp_path, bits_llr):
    bits, llr = bits_llr
    path = tmp_path / "sweep.h5"
    config = FakeConfig(active_channel="system_level")

    export_sweep({0.0: (bits, llr)}, path, config=config)

    with h5py.File(path, "r") as f:
        assert json.loads(f.attrs["config_json"]) == dataclasses.asdict(config)


def test_export_sweep_empty_results(tmp_path):
    """Edge case: an empty sweep dict shouldn't error, just produce an empty file."""
    path = tmp_path / "sweep.h5"

    export_sweep({}, path)
    out = load_sweep(path)

    assert out == {}
    with h5py.File(path, "r") as f:
        assert len(f.attrs["ebno_db_points"]) == 0

