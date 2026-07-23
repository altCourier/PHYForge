from sionna.phy.channel import AWGN
from sionna.phy.config import Precision

from enum import Enum
from typing import Optional
import torch

# Unit testing

from source import Source
from mapsys import MapSys
from sionna.phy.utils import ebnodb2no

class ChannelType(Enum):
    AWGN = "awgn"


CHANNEL_REGISTERY = {
    ChannelType.AWGN : AWGN,
}

class Channel:

    def __init__(

        self,
        channel_type: str,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,

    ) -> None:

        ctype = ChannelType(channel_type)
        channel_cls = CHANNEL_REGISTERY[ctype]
        self._channel = channel_cls(
            precision = precision,
            device = device
        )

    def add_noise(self, x: torch.Tensor, no) -> torch.Tensor:
        return self._channel(x, no)

    def __call__(self, x: torch.Tensor, no) -> torch.Tensor:
        return self.add_noise(x, no)

if __name__ == "__main__":
    
    batch_size = 4
    num_bits_per_symbol = 4
    num_symbols = 10
    num_bits = num_symbols * num_bits_per_symbol
 
    # Source declaration
    source = Source("binary")
    bits = source(batch_size = batch_size, num_bits = num_bits)

    # MapSYS declaration
    mapsys = MapSys("qam", num_bits_per_symbol)
    x = mapsys.map(bits)

    # Using the new channel
    channel = Channel("awgn")
    
    no = ebnodb2no(ebno_db=10.0, num_bits_per_symbol=num_bits_per_symbol, coderate=1.0)
    y = channel.add_noise(x, no)

    llr, bits_hat = mapsys.demap(y, no)
    
    print("x:", x.shape, "y:", y.shape, "llr:", llr.shape, "bits_hat:", bits_hat.shape)

