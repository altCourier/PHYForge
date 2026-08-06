from sionna.phy.channel import AWGN, TimeChannel
from sionna.phy.channel.tr38901 import TDL
from sionna.phy.channel.utils import time_lag_discrete_time_channel
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
    TDL = "tdl"


class Channel:
    """
    Wraps Sionna channel models behind a single interface.

    - AWGN: stateless noise injection. call(x, no) -> y
    - TDL: 3GPP multipath fading model. Internally builds a `TimeChannel`
      (which itself handles CIR generation + time-domain filtering) and
      layers AWGN on top via the `no` parameter, same as AWGN.
    """

    def __init__(
        self,
        channel_type: str,

        # --- TDL-specific config (ignored for AWGN) ---
        tdl_model: str = "A",
        delay_spread: float = 100e-9,
        carrier_frequency: Optional[float] = None,
        min_speed: float = 0.0,
        max_speed: Optional[float] = None,
        bandwidth: Optional[float] = None,
        num_time_samples: Optional[int] = None,
        l_min: Optional[int] = None,
        l_max: Optional[int] = None,

        # --- shared ---
        precision: Optional[Precision] = None,
        device: Optional[str] = None,

    ) -> None:

        ctype = ChannelType(channel_type)
        self.channel_type = ctype
        self.precision = precision
        self.device = device

        if ctype is ChannelType.AWGN:
            self._channel = AWGN(precision=precision, device=device)

        elif ctype is ChannelType.TDL:
            if bandwidth is None or num_time_samples is None:
                raise ValueError(
                    "TDL requires `bandwidth` and `num_time_samples` to be "
                    "set (num_time_samples should match the number of "
                    "symbols you'll pass through the channel per call)."
                )

            self._tdl = TDL(
                model=tdl_model,
                delay_spread=delay_spread,
                carrier_frequency=carrier_frequency,
                min_speed=min_speed,
                max_speed=max_speed,
                precision=precision,
                device=device,
            )

            self._time_channel = TimeChannel(
                channel_model=self._tdl,
                bandwidth=bandwidth,
                num_time_samples=num_time_samples,
                l_min=l_min,
                l_max=l_max,
                precision=precision,
                device=device,
            )

            self.l_min = self._time_channel.l_min
            self.l_max = self._time_channel.l_max

    def add_noise(self, x: torch.Tensor, no) -> torch.Tensor:
        """
        AWGN-only path: x -> y, no fading applied.
        """

        if self.channel_type is not ChannelType.AWGN:
            raise RuntimeError("add_noise() is only valid for ChannelType.AWGN")

        return self._channel(x, no)

    def apply_fading(self, x: torch.Tensor, no) -> torch.Tensor:
        """
        TDL path: apply multipath fading + noise to x.
 
        x comes in as [batch, num_symbols] (MapSys.map()'s output shape).
        TimeChannel needs [batch, num_tx=1, num_tx_ant=1, num_time_samples],
        and returns y one filter-tail longer than the input -- see the class
        docstring above for why the tail is truncated here.
        """
        if self.channel_type is not ChannelType.TDL:
            raise RuntimeError("apply_fading() is only valid for ChannelType.TDL")
 
        batch_size, num_symbols = x.shape
        x_4d = x.reshape(batch_size, 1, 1, num_symbols)
 
        y_4d = self._time_channel(x_4d, no)
 
        y_4d = y_4d[..., :num_symbols]
 
        return y_4d.reshape(batch_size, num_symbols)
 
    def __call__(self, x: torch.Tensor, no) -> torch.Tensor:

        if self.channel_type is ChannelType.AWGN:
            return self.add_noise(x, no)
        
        elif self.channel_type is ChannelType.TDL:
            return self.apply_fading(x, no)


if __name__ == "__main__":

    batch_size = 4
    num_bits_per_symbol = 4
    num_symbols = 10
    num_bits = num_symbols * num_bits_per_symbol

    # Source declaration
    source = Source("binary")
    bits = source(batch_size=batch_size, num_bits=num_bits)

    # MapSYS declaration
    mapsys = MapSys("qam", num_bits_per_symbol)
    x = mapsys.map(bits)

    no = ebnodb2no(ebno_db=10.0, num_bits_per_symbol=num_bits_per_symbol, coderate=1.0)

    # --- AWGN path (unchanged, known-good) ---
    awgn_channel = Channel("awgn")
    y_awgn = awgn_channel.add_noise(x, no)
    llr, bits_hat = mapsys.demap(y_awgn, no)
    print("[AWGN] x:", x.shape, "y:", y_awgn.shape, "llr:", llr.shape, "bits_hat:", bits_hat.shape)

    # --- TDL path ---
    tdl_channel = Channel(
        "tdl",
        tdl_model="A",
        delay_spread=100e-9,
        carrier_frequency=3.5e9,
        bandwidth=1e6,          # TODO: set to your actual system bandwidth
        num_time_samples=num_symbols,
    )
    y_tdl = tdl_channel.apply_fading(x, no)
    llr_tdl, bits_hat_tdl = mapsys.demap(y_tdl, no)
    print("[TDL] x:", x.shape, "y:", y_tdl.shape, "llr:", llr_tdl.shape, "bits_hat:", bits_hat_tdl.shape)