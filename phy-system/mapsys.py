from sionna.phy.mapping import Mapper, Demapper
from sionna.phy.config import Precision
from typing import Optional
import torch

# For testing
from source import Source
from sionna.phy.utils import ebnodb2no

class MapSys:

    def __init__(
        self,
        constellation_type: str,
        num_bits_per_symbol: int,
        demapping_method: str = "app",
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        ) -> None:

        self.mapper = Mapper(
            constellation_type,
            num_bits_per_symbol,
            precision=precision,
            device=device,
        )

        self.demapper = Demapper(
            demapping_method,
            constellation_type,
            num_bits_per_symbol,
            hard_out=False,
            precision=precision,
            device=device,
        )

        self.num_bits_per_symbol = num_bits_per_symbol

    def map(self, bits: torch.Tensor) -> torch.Tensor:
        
        assert bits.shape[-1] % self.num_bits_per_symbol == 0, (
            f"Last dim of bits ({bits.shape[-1]}) must be a multiple of "
            f"num_bits_per_symbol ({self.num_bits_per_symbol})"
        )

        return self.mapper(bits)

    def demap(self, y, no) -> tuple[torch.Tensor, torch.Tensor]:
        
        llr = self.demapper(y, no)
        bits_hat = (llr < 0).to(llr.dtype)
        return llr, bits_hat

if __name__ == "__main__":
    
    BATCH_SIZE = 4
    NUM_BITS_PER_SYMBOL = 4
    NUM_SYMBOLS = 10
    NUM_BITS = NUM_SYMBOLS * NUM_BITS_PER_SYMBOL

    source = Source("binary")
    bits = source(
        batch_size = BATCH_SIZE,
        num_bits = NUM_BITS,
        )

    print("bits:", bits.shape)

    mapsys = MapSys("qam", NUM_BITS_PER_SYMBOL)

    x = mapsys.map(bits)
    print("symbols:", x.shape)

    no = ebnodb2no( ebno_db = 10.0, num_bits_per_symbol = NUM_BITS_PER_SYMBOL , coderate = 0.5 )
    
    y = x # No Channel for now 
    
    llr, bits_hat = mapsys.demap(y, no)
    print("llr:", llr.shape, "bits_hat:", bits_hat.shape)

