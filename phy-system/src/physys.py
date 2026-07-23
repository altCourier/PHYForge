from typing import Optional, Dict
import torch

from sionna.phy.config import Precision
from sionna.phy.utils import ebnodb2no

from source import Source
from constellation import Constellation
from mapsys import MapSys
from channel import Channel


class PHYSys:
    """
    Top-level wrapper: Source -> MapSys.map -> Channel -> MapSys.demap
    (same MapSys instance, used first as mapper then as demapper).

    Pure composition -- every underlying operation is a Sionna class/method.
    This class only owns the wiring and the THz constellation lookup.
    """

    def __init__(
        self,
        source_type: str,
        modulation: str,
        channel_type: str,
        demapping_method: str = "app",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
    ) -> None:

        self.precision = precision
        self.device = device

        self.constellation = Constellation(
            modulation,
            precision=precision,
            device=device,
        )

        self.num_bits_per_symbol = self.constellation.num_bits_per_symbol

        self.source = Source(
            source_type,
            precision=precision,
            device=device,
        )

        self.mapsys = MapSys(
            constellation=self.constellation.sionna_object,
            demapping_method=demapping_method,
            precision=precision,
            device=device,
        )

        self.channel = Channel(
            channel_type,
            precision=precision,
            device=device,
        )

    def generate(
        self,
        batch_size: int,
        num_symbols: int,
        ebno_db: float,
        coderate: float = 1.0,
    ) -> Dict[str, torch.Tensor]:

        num_bits = num_symbols * self.num_bits_per_symbol

        bits = self.source(batch_size=batch_size, num_bits=num_bits)
        x = self.mapsys.map(bits)

        no = ebnodb2no(
            ebno_db=ebno_db,
            num_bits_per_symbol=self.num_bits_per_symbol,
            coderate=coderate,
        )

        y = self.channel(x, no)
        llr, bits_hat = self.mapsys.demap(y, no)

        return {
            "bits": bits,
            "x": x,
            "y": y,
            "no": no,
            "llr": llr,
            "bits_hat": bits_hat,
        }

    def generate_sweep(
        self,
        batch_size: int,
        num_symbols: int,
        ebno_db_range: list[float],
        coderate: float = 1.0,
    ) -> dict[float, dict[str, torch.Tensor]]:

        return {
            ebno_db: self.generate(
                batch_size=batch_size,
                num_symbols=num_symbols,
                ebno_db=ebno_db,
                coderate=coderate,
            )
            for ebno_db in ebno_db_range
        }

    def __call__(self, *args, **kwargs) -> Dict[str, torch.Tensor]:
        return self.generate(*args, **kwargs)


if __name__ == "__main__":

    physys = PHYSys(
        source_type="binary",
        modulation="16qam",
        channel_type="awgn",
    )

    out = physys(batch_size=4, num_symbols=10, ebno_db=10.0)

    for name, tensor in out.items():
        print(f"{name}: {tuple(tensor.shape)}")