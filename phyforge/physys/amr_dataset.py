"""
physys.amr_dataset

Drives PHYSys across (channel_variant x modulation x snr_db), as
specified entirely by config.amr_dataset, to produce Umi.h5/Uma.h5/
Rma.h5. This file contains no modulation specs, SNR grids, or variant
lists of its own -- those all come from config.json.
"""

import dataclasses
from pathlib import Path
from typing import Union

from .schema import Config, AmrModulationSpec
from .runtime import PHYSys
from .export import open_amr_dataset, append_amr_batch

_VARIANT_FILENAMES = {"umi": "Umi.h5", "uma": "Uma.h5", "rma": "Rma.h5"}

def _batch_chunks(total: int, max_chunk):

    if max_chunk is None or max_chunk >= total:
        return [total]
    sizes = [max_chunk] * (total // max_chunk)
    remainder = total % max_chunk
    if remainder:
        sizes.append(remainder)
    return sizes

def _build_config(base_config: Config, variant: str, mod_spec: AmrModulationSpec) -> Config:
    ds = base_config.amr_dataset
    sl = base_config.channels.system_level

    system_level = dataclasses.replace(
        sl,
        variant = variant,
        enable_pathloss = sl.enable_pathloss and not ds.disable_pathloss_and_shadowing,
        enable_shadow_fading = sl.enable_shadow_fading and not ds.disable_pathloss_and_shadowing,
    )

    return dataclasses.replace(
        base_config,
        common = dataclasses.replace(base_config.common, active_channel="system_level"),
        modulation = mod_spec.to_modulation_config(ds.mapper, ds.demapper),
        channels = dataclasses.replace(base_config.channels, system_level=system_level),
        waveforms = dataclasses.replace(
            base_config.waveforms,
            time = dataclasses.replace(base_config.waveforms.time, num_time_samples=ds.vector_len),
        ),
        sweep = dataclasses.replace(
            base_config.sweep,
            batch_size = ds.num_vectors,
            num_symbols = ds.vector_len,
        ),
    )


def generate_variant(base_config: Config, variant: str, out_path: Path) -> Path:


    ds = base_config.amr_dataset
    mod_order = [m.name for m in ds.modulations]

    print(f"DEBUG max_batch_size = {ds.max_batch_size!r}")

    f = open_amr_dataset(out_path, vector_len=ds.vector_len, mod_order=mod_order)
    try:
        for mod_index, mod_spec in enumerate(ds.modulations):
            config = _build_config(base_config, variant, mod_spec)
            sim = PHYSys(config)

            snr_db = ds.snr_db.start
            while snr_db <= ds.snr_db.stop:
                for chunk_size in _batch_chunks(ds.num_vectors, ds.max_batch_size):
                    _bits, _llr, _x, y = sim.generate_iq(
                        batch_size = chunk_size,
                        num_symbols = ds.vector_len,
                        ebno_db = float(snr_db),
                    )
                    append_amr_batch(f, y, mod_index, float(snr_db))
                snr_db += ds.snr_db.step
    finally:
        f.close()

    return out_path


def generate_all(base_config: Config, out_dir: Union[str, Path]) -> dict:
    ds = base_config.amr_dataset
    if ds is None:
        raise ValueError(
            "config.amr_dataset is not set -- add an 'amr_dataset' block to config.json"
        )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    written = {}
    for variant in ds.variants:
        out_path = out_dir / _VARIANT_FILENAMES[variant]
        written[variant] = generate_variant(base_config, variant, out_path)

    return written
