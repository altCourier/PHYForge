"""
schema.py

This file exists solely for config validation.
There is deliberately no heavy imports such as sionna or torch.
It checks whether the config.json is well-formed.

"""

from dataclasses import dataclass
from typing import Literal, Optional, Union

# ---- source ----

@dataclass
class BinarySourceConfig:

    type: Literal["binary"] = "binary"
    
    # TODO: This module can also take seeding. Without proper need
    # it is postponed.
    
    def __post_init__(self):

        if self.type != "binary":
            raise ValueError(f"BinarySourceConfig.type must be 'binary', got {self.type!r}")

# ---- modulation ----

@dataclass
class ModulationConfig:

    type: Literal["qam", "pam"]
    num_bits_per_symbol: int

    def __post_init__(self):

        if self.type not in ["qam", "pam"]:
            raise ValueError(f"ModulationConfig.type must be 'qam' or 'pam' got {self.type!r}")

        if not isinstance(self.num_bits_per_symbol, int):
            raise ValueError("num_bits_per_symbol must be an integer")

# ---- channels ----

@dataclass
class AWGNChannelConfig:

    type: Literal["awgn"] = "awgn"

    def __post_init__(self):

        if self.type != "awgn":
            raise ValueError(f"AWGNChannelConfig.type must be 'awgn' got {self.type!r}")

@dataclass
class TDLChannelConfig:

    model: str
    delay_spread: float
    carrier_frequency: float
    bandwidth: float

    type: Literal["tdl"] = "tdl"
    
    min_speed: float = 0.0
    max_speed: Optional[float] = None

    l_min: Optional[int] = None
    l_max: Optional[int] = None


    def __post_init__(self):

        if self.type != "tdl":
            raise ValueError(f"TdlChannelConfig.type must be 'tdl', got {self.type!r}")

# --- initiating ---

SourceConfig = Union[BinarySourceConfig]
ChannelConfig = Union[AWGNChannelConfig, TDLChannelConfig]


# --- parser functions ---

def parse_source(raw: dict) -> SourceConfig:

    t = raw.get("type")

    if t == "binary":
        return BinarySourceConfig(type=t)

    raise ValueError(f"Unknown source type: {t!r}")

def parse_modulation(raw: dict) -> ModulationConfig:

    t = raw.get("type")

    if t not in ("qam", "pam"):
        raise ValueError(f"Unknown modulation type: {t!r}")

    return ModulationConfig(type=t, num_bits_per_symbol=raw.get("num_bits_per_symbol"))

def parse_channel(raw: dict) -> ChannelConfig:

    t = raw.get("type")

    if t == "awgn":
        return AWGNChannelConfig(type=t)

    elif t == "tdl":
        return TDLChannelConfig(
            type=t,
            model=raw.get("model"),
            delay_spread=raw.get("delay_spread"),
            carrier_frequency=raw.get("carrier_frequency"),
            bandwidth=raw.get("bandwidth"),
            min_speed=raw.get("min_speed", 0.0),
            max_speed=raw.get("max_speed"),
            l_min=raw.get("l_min"),
            l_max=raw.get("l_max"),
        )

    raise ValueError(f"Unknown channel type: {t!r}")

# --- top level config ---

@dataclass
class Config:
    source: SourceConfig
    modulation: ModulationConfig
    channel: ChannelConfig

# --- top level parser ---

def parse_config(raw: dict) -> Config:
    return Config(
        source = parse_source(raw["source"]),
        modulation = parse_modulation(raw["modulation"]),
        channel = parse_channel(raw["channel"]),
    )