from sionna.phy.mapping import Constellation as SionnaConstellation
from sionna.phy.config import Precision

from enum import Enum
from typing import Optional, Callable
import torch


class ThzModulation(Enum):
    PI2_BPSK  = "pi2_bpsk"
    PI2_QPSK  = "pi2_qpsk"
    PI2_PSK8  = "pi2_8psk"
    QAM16     = "16qam"
    QAM64     = "64qam"
    PI2_APSK8 = "pi2_8apsk"
    APSK16    = "16apsk"
    APSK32    = "32apsk"
    OOK       = "ook"

def _build_qam(bits: int) -> dict:
    return {"constellation_type": "qam", "num_bits_per_symbol": bits}


def _build_custom(bits: int, points: torch.Tensor) -> dict:
    return {
        "constellation_type": "custom",
        "num_bits_per_symbol": bits,
        "initial_value": points,
    }

THZ_MODULATION_REGISTRY: dict[ThzModulation, Callable[[], dict]] = {
    ThzModulation.QAM16:     lambda: _build_qam(4),
    ThzModulation.QAM64:     lambda: _build_qam(6),

    ThzModulation.PI2_BPSK:  lambda: _build_custom(1, None),  # TODO
    ThzModulation.PI2_QPSK:  lambda: _build_custom(2, None),  # TODO
    ThzModulation.PI2_PSK8:  lambda: _build_custom(3, None),  # TODO

    ThzModulation.PI2_APSK8: lambda: _build_custom(3, None),  # TODO
    ThzModulation.APSK16:    lambda: _build_custom(4, None),  # TODO
    ThzModulation.APSK32:    lambda: _build_custom(5, None),  # TODO

    ThzModulation.OOK:       lambda: _build_custom(1, None),  # TODO
}


class Constellation:
    """
    Wraps sionna.phy.mapping.Constellation. Given a ThzModulation, looks up
    how to construct it via THZ_MODULATION_REGISTRY, then instantiates
    Sionna's own Constellation class with those args. No custom math lives
    here only dispatch + the point tensors registered above.
    """

    def __init__(
            
        self,
        modulation: str,
        precision: Optional[Precision] = None,
        device: Optional[str] = None,
        
    ) -> None:

        mtype = ThzModulation(modulation)
        builder = THZ_MODULATION_REGISTRY[mtype]
        cfg = builder()

        if cfg["constellation_type"] == "custom" and cfg["initial_value"] is None:
            raise NotImplementedError(
                f"{mtype.value}: point tensor not filled in yet -- "
                f"pull exact constellation from the 802.15.3d figure first."
            )

        self.modulation = mtype
        self.num_bits_per_symbol = cfg["num_bits_per_symbol"]

        if cfg["constellation_type"] == "qam":
            self._constellation = SionnaConstellation(
                "qam",
                cfg["num_bits_per_symbol"],
                precision=precision,
                device=device,
            )
        else:
            self._constellation = SionnaConstellation(
                "custom",
                cfg["num_bits_per_symbol"],
                initial_value=cfg["initial_value"],
                precision=precision,
                device=device,
            )

    @property
    def points(self) -> torch.Tensor:
        return self._constellation.points

    @property
    def sionna_object(self):
        """Raw Sionna Constellation, to pass straight into Mapper/Demapper."""
        return self._constellation

    def show(self, **kwargs):
        return self._constellation.show(**kwargs)


if __name__ == "__main__":
    qam16 = Constellation("16qam")
    print("16-QAM points:", qam16.points.shape)

    try:
        bpsk = Constellation("pi2_bpsk")
    except NotImplementedError as e:
        print("Expected gap:", e)