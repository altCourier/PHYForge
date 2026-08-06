"""physys.export

Writes the tensors produced by ``runtime.PHYSys.generate()`` /
``generate_sweep()`` (bits/llr) and ``generate_iq()`` /
``generate_sweep_iq()`` (bits/llr/x/y) to disk as HDF5.

This module does exactly one thing: serialize
already-computed tensors to a file, and read
them back.

AMR EXTENSION: export_run_iq / export_sweep_iq / load_run_iq /
load_sweep_iq mirror export_run / export_sweep / load_run / load_sweep
exactly (same stacked-array layout, same attrs, same compression
handling) but additionally carry the raw transmitted symbols (x) and
raw received noisy symbols (y) that runtime.generate_iq() now exposes.
bits/llr and the original four functions are left untouched.

NOTE on complex data: x/y are complex64 tensors. h5py stores complex
ndarrays natively via a compound (r, i) dtype and round-trips them
transparently through h5py/numpy -- no special handling needed here.
That said, this is an h5py/numpy convention, not a universal HDF5
one: non-Python readers (MATLAB, some C++ HDF5 tools) may not
recognize the compound type. Flagging since it isn't obvious from the
API surface.
"""

import dataclasses
import json
import time
from pathlib import Path
from typing import Optional, Union
 
import h5py
import numpy as np
 
 
def _to_numpy(x):
    """
    Sionna/TensorFlow tensors, torch tensors, or already-ndarray -> ndarray.
    """
    
    if hasattr(x, "detach"):
        x = x.detach()

    if hasattr(x, "cpu"):
        x = x.cpu()

    if hasattr(x, "numpy"):
        return x.numpy()

    return np.asarray(x)
 
 
def _config_json(config) -> Optional[str]:

    if config is None:

        return None

    return json.dumps(dataclasses.asdict(config))
 
 
def _utc_timestamp() -> str:

    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# Standard order-N naming per num_bits_per_symbol. schema.ModulationConfig
# only ever carries `type` ("qam"/"pam") + `num_bits_per_symbol` -- there's
# no human label like "BPSK"/"16-QAM" anywhere in the config itself, so it
# has to be derived. 1/2 bits get their conventional PSK-family names since
# that's how everyone actually refers to them; everything else falls back
# to "{order}{TYPE}" (e.g. "16QAM", "64QAM", "8PAM").
_QAM_SPECIAL_NAMES = {1: "BPSK", 2: "QPSK"}
_PAM_SPECIAL_NAMES = {1: "BPSK"}  # 1-bit PAM is a 2-level real signal, same as BPSK


def _modulation_label(config) -> Optional[str]:
    """
    Derive a human-readable modulation label (e.g. "BPSK", "16QAM") from
    config.modulation, for stamping onto exported AMR files as
    f.attrs["modulation"] -- cheap to read back in a PyTorch DataLoader
    without parsing the full config_json.

    Returns None if `config` is None or doesn't look like a
    schema.Config (no `.modulation` attr) -- callers should treat that
    as "attribute omitted", same convention as _config_json().
    """

    if config is None:
        return None

    modulation = getattr(config, "modulation", None)

    if modulation is None:
        return None

    mod_type = modulation.type
    n = modulation.num_bits_per_symbol
    order = 2 ** n

    if mod_type == "qam":
        return _QAM_SPECIAL_NAMES.get(n, f"{order}QAM")

    if mod_type == "pam":
        return _PAM_SPECIAL_NAMES.get(n, f"{order}PAM")

    # Schema only allows "qam"/"pam" today (ModulationConfig.__post_init__
    # enforces it), but don't silently mislabel if that ever changes.
    return f"{order}{mod_type.upper()}"
 

def export_run(
    bits,
    llr,
    path: Union[str, Path],
    *,
    ebno_db: Optional[float] = None,
    config=None,
    compression: str = "gzip",
) -> None:
    
    """
    Write one `PHYSys.generate()` result to a new HDF5 file at `path`.
 
    This creates/overwrites `path` — it writes exactly one run, not an appending
    log. This could be changed.
 
    Parameters
    ----------
    bits, llr : whatever PHYSys.generate() returned
    path : destination file
    ebno_db : optional, recorded as a file attr for convenience
    config : optional physys.schema.Config used to produce this run; stored as a
        JSON attr so a loaded file is self-describing
    compression : passed straight to h5py.create_dataset; "gzip" by default, pass
        None to disable
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
 
    bits_np = _to_numpy(bits)
    llr_np = _to_numpy(llr)
 
    with h5py.File(path, "w") as f:

        f.create_dataset("bits", data = bits_np, compression = compression)
        f.create_dataset("llr", data = llr_np, compression = compression)

        f.attrs["created_utc"] = _utc_timestamp()

        if ebno_db is not None:
            f.attrs["ebno_db"] = float(ebno_db)

        cfg_json = _config_json(config)

        if cfg_json is not None:
            f.attrs["config_json"] = cfg_json
 
 
def export_sweep(
    results: dict,
    path: Union[str, Path],
    *,
    config=None,
    compression: str = "gzip",
) -> None:
    """
    Write a `PHYSys.generate_sweep()` result to contiguous multi-dimensional arrays.
    """
    
    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    ebno_values = sorted(results.keys())

    stacked_bits = np.stack([_to_numpy(results[ebno][0]) for ebno in ebno_values])
    stacked_llr = np.stack([_to_numpy(results[ebno][1]) for ebno in ebno_values])
    snr_labels = np.array(ebno_values, dtype=np.float64)

    with h5py.File(path, "w") as f:

        f.attrs["created_utc"] = _utc_timestamp()
        
        cfg_json = _config_json(config)

        if cfg_json is not None:
            f.attrs["config_json"] = cfg_json

        f.create_dataset("snr_labels", data=snr_labels)
        f.create_dataset("bits", data=stacked_bits, compression=compression)
        f.create_dataset("llr", data=stacked_llr, compression=compression)
 
 
def load_run(path: Union[str, Path]) -> dict:

    """
    Read back a file written by export_run().
 
    Returns {"bits": ndarray, "llr": ndarray, "ebno_db": float | None,
    "config": dict | None}.
    
    """

    path = Path(path)

    with h5py.File(path, "r") as f:

        out = {
            "bits": f["bits"][...],
            "llr": f["llr"][...],
            "ebno_db": float(f.attrs["ebno_db"]) if "ebno_db" in f.attrs else None,
            "config": json.loads(f.attrs["config_json"]) if "config_json" in f.attrs else None,
        }
    return out
 
 
def load_sweep(path: Union[str, Path]) -> dict:
    """
    Read back a file written by export_sweep().
 
    Returns {ebno_db: (bits_ndarray, llr_ndarray)}, keyed by the float recorded in
    each group's `ebno_db` attr (not by re-parsing the group name).
    """

    path = Path(path)

    results = {}

    with h5py.File(path, "r") as f:

        snr_labels = f["snr_labels"][...]
        bits = f["bits"][...]
        llr = f["llr"][...]

        for i, snr in enumerate(snr_labels):

            results[float(snr)] = (bits[i], llr[i])

    return results


# ---------------------------------------------------------------------------
# AMR extension: bits/llr/x/y variants.
#
# Mirror export_run/export_sweep/load_run/load_sweep exactly (same file
# layout style, same attrs, same compression handling) -- the only
# difference is two extra datasets, "x" (clean tx symbols) and "y"
# (noisy rx symbols / demapper input), sourced from
# runtime.PHYSys.generate_iq() / generate_sweep_iq().
# ---------------------------------------------------------------------------


def export_run_iq(
    bits,
    llr,
    x,
    y,
    path: Union[str, Path],
    *,
    ebno_db: Optional[float] = None,
    config=None,
    compression: str = "gzip",
) -> None:
    """
    Write one `PHYSys.generate_iq()` result to a new HDF5 file at `path`.

    Same semantics as export_run (creates/overwrites `path`, one run
    per call, not an appending log), with two additional datasets:

    Parameters
    ----------
    bits, llr, x, y : whatever PHYSys.generate_iq() returned
        x -- clean transmitted symbols (mapper output), complex
        y -- noisy received symbols (channel output / demapper input,
             i.e. the AMR feature tensor), complex, same shape as x
    path : destination file
    ebno_db : optional, recorded as a file attr for convenience
    config : optional physys.schema.Config used to produce this run; stored as a
        JSON attr so a loaded file is self-describing
    compression : passed straight to h5py.create_dataset; "gzip" by default, pass
        None to disable
    """

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    bits_np = _to_numpy(bits)
    llr_np = _to_numpy(llr)
    x_np = _to_numpy(x)
    y_np = _to_numpy(y)

    with h5py.File(path, "w") as f:

        f.create_dataset("bits", data=bits_np, compression=compression)
        f.create_dataset("llr", data=llr_np, compression=compression)
        f.create_dataset("x", data=x_np, compression=compression)
        f.create_dataset("y", data=y_np, compression=compression)

        f.attrs["created_utc"] = _utc_timestamp()

        if ebno_db is not None:
            f.attrs["ebno_db"] = float(ebno_db)

        cfg_json = _config_json(config)

        if cfg_json is not None:
            f.attrs["config_json"] = cfg_json

        mod_label = _modulation_label(config)

        if mod_label is not None:
            f.attrs["modulation"] = mod_label


def export_sweep_iq(
    results: dict,
    path: Union[str, Path],
    *,
    config=None,
    compression: str = "gzip",
) -> None:
    """
    Write a `PHYSys.generate_sweep_iq()` result to contiguous
    multi-dimensional arrays -- same layout style as export_sweep,
    with "x" and "y" datasets added alongside "bits"/"llr".

    `results` is `{ebno_db: (bits, llr, x, y)}` as returned by
    generate_sweep_iq(); stacked along a new leading axis in
    ascending ebno_db order, same as export_sweep's bits/llr.
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)

    ebno_values = sorted(results.keys())

    stacked_bits = np.stack([_to_numpy(results[ebno][0]) for ebno in ebno_values])
    stacked_llr = np.stack([_to_numpy(results[ebno][1]) for ebno in ebno_values])
    stacked_x = np.stack([_to_numpy(results[ebno][2]) for ebno in ebno_values])
    stacked_y = np.stack([_to_numpy(results[ebno][3]) for ebno in ebno_values])
    snr_labels = np.array(ebno_values, dtype=np.float64)

    with h5py.File(path, "w") as f:

        f.attrs["created_utc"] = _utc_timestamp()

        cfg_json = _config_json(config)

        if cfg_json is not None:
            f.attrs["config_json"] = cfg_json

        mod_label = _modulation_label(config)

        if mod_label is not None:
            f.attrs["modulation"] = mod_label

        f.create_dataset("snr_labels", data=snr_labels)
        f.create_dataset("bits", data=stacked_bits, compression=compression)
        f.create_dataset("llr", data=stacked_llr, compression=compression)
        f.create_dataset("x", data=stacked_x, compression=compression)
        f.create_dataset("y", data=stacked_y, compression=compression)


def load_run_iq(path: Union[str, Path]) -> dict:
    """
    Read back a file written by export_run_iq().

    Returns {"bits": ndarray, "llr": ndarray, "x": ndarray, "y": ndarray,
    "ebno_db": float | None, "config": dict | None,
    "modulation": str | None}.
    """

    path = Path(path)

    with h5py.File(path, "r") as f:

        out = {
            "bits": f["bits"][...],
            "llr": f["llr"][...],
            "x": f["x"][...],
            "y": f["y"][...],
            "ebno_db": float(f.attrs["ebno_db"]) if "ebno_db" in f.attrs else None,
            "config": json.loads(f.attrs["config_json"]) if "config_json" in f.attrs else None,
            "modulation": f.attrs["modulation"] if "modulation" in f.attrs else None,
        }
    return out


def load_sweep_iq(path: Union[str, Path]) -> dict:
    """
    Read back a file written by export_sweep_iq().

    Returns {ebno_db: (bits_ndarray, llr_ndarray, x_ndarray, y_ndarray)},
    keyed by the value recorded in "snr_labels" (same convention as
    load_sweep).

    NOTE: matches load_sweep's existing behavior of NOT surfacing
    file-level attrs (created_utc/config_json/modulation) -- that gap
    predates this function and is inherited rather than fixed here,
    since widening the return shape is a decision that affects every
    caller. If you want the modulation/config attrs back from a swept
    file, read them directly, e.g.:
        with h5py.File(path, "r") as f:
            modulation = f.attrs.get("modulation")
    """

    path = Path(path)

    results = {}

    with h5py.File(path, "r") as f:

        snr_labels = f["snr_labels"][...]
        bits = f["bits"][...]
        llr = f["llr"][...]
        x = f["x"][...]
        y = f["y"][...]

        for i, snr in enumerate(snr_labels):

            results[float(snr)] = (bits[i], llr[i], x[i], y[i])

    return results