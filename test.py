import numpy as np
from physys.loader import load_config
from physys.runtime import PHYSys
import dataclasses

config = load_config("config.json")  # adjust path
ds = config.amr_dataset
mod_spec = ds.modulations[0]  # e.g. BPSK -- adjust index if needed
print("Testing modulation:", mod_spec.name)

from physys.amr_dataset import _build_config

for ebno_db in [-10.0, 0.0, 10.0, 20.0]:
    cfg = _build_config(config, "rma", mod_spec)  # match whichever variant you used
    sim = PHYSys(cfg)
    bits, llr, x, y = sim.generate_iq(batch_size=512, num_symbols=ds.vector_len, ebno_db=ebno_db)

    x_np = x.cpu().numpy() if hasattr(x, "cpu") else np.asarray(x)
    y_np = y.cpu().numpy() if hasattr(y, "cpu") else np.asarray(y)

    sig_power = np.mean(np.abs(x_np) ** 2)
    err = y_np - x_np
    err_power = np.mean(np.abs(err) ** 2)
    achieved_snr_db = 10 * np.log10(sig_power / err_power)

    print(f"ebno_db={ebno_db:6.1f}  |x|^2={sig_power:.4f}  |y-x|^2={err_power:.4f}  "
          f"achieved_SNR={achieved_snr_db:7.2f} dB")