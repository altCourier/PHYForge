from sionna.phy.mapping import BinarySource
from sionna.phy.config import Precision

from enum import Enum
from typing import Optional
import torch

class SourceType(Enum):
    BINARY = "binary"
    TEXT   = "text"
    AUDIO  = "audio"

SOURCE_REGISTERY = {

    SourceType.BINARY : BinarySource,
    # SourceType.TEXT : ...,
    # SourceType.ADUIO : ...,

}

class Source:

    def __init__(

        self,
        source_type: str,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,

    ) -> None:

        stype = SourceType(source_type)
        source_cls = SOURCE_REGISTERY[stype]
        self._source = source_cls(precision = precision, device = device)

    def __call__(self, batch_size: int, num_bits: int) -> torch.Tensor:
        
        shape = [batch_size, num_bits]
        return self._source(shape)

if __name__ == "__main__":

    source = Source("binary")
    bits = source(batch_size = 4, num_bits = 10)
    
    print(bits.shape)