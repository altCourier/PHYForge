"""physys.export

Writes the tensors produced by ``runtime.PHYSys.generate()`` /
``generate_sweep()`` to disk as HDF5.

This module does exactly one thing: serialize
already-computed tensors to a file, and read
them back.

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
    Write a `PHYSys.generate_sweep()` result ({ebno_db: (bits, llr)}) to one file.
 
    One HDF5 group per Eb/N0 point.
    
    """

    path = Path(path)

    path.parent.mkdir(parents=True, exist_ok=True)
 
    ebno_values = sorted(results.keys())
 
    with h5py.File(path, "w") as f:

        f.attrs["created_utc"] = _utc_timestamp()
        f.attrs["ebno_db_points"] = np.array(ebno_values, dtype=np.float64)

        cfg_json = _config_json(config)

        if cfg_json is not None:
            f.attrs["config_json"] = cfg_json
 
        for ebno_db in ebno_values:
            
            bits, llr = results[ebno_db]

            grp = f.create_group(f"ebno_db={ebno_db:.4f}")
            grp.attrs["ebno_db"] = float(ebno_db)
            grp.create_dataset("bits", data=_to_numpy(bits), compression=compression)
            grp.create_dataset("llr", data=_to_numpy(llr), compression=compression)
 
 
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
        for grp in f.values():

            ebno_db = float(grp.attrs["ebno_db"])
            results[ebno_db] = (grp["bits"][...], grp["llr"][...])

    return results
