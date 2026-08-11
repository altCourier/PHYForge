"""
schema.py

This file exists solely for config validation.
There is deliberately no heavy imports such as sionna or torch.
It checks whether the config.json is well-formed.

"""

import dataclasses
from dataclasses import dataclass, field
from typing import Literal, Optional, List


# ---- common ----

@dataclass
class CommonConfig:

    # precision/device are inherited DEFAULTS -- any block below can
    # still override them locally with its own non-null value.
    precision: Optional[str] = None
    device: Optional[str] = None

    # NOTE: whether these NAME a real channel/waveform is validated in
    # Config.__post_init__, not here -- CommonConfig on its own doesn't
    # know what channels/waveforms exist yet (ChannelsConfig/WaveformsConfig
    # are defined later in this file), so that check has to wait until
    # the full Config is assembled.

    # active_channel/active_waveform are pure SELECTORS -- they are
    # never inherited, only looked up once (by builders.py) to decide
    active_channel: str = "awgn"
    active_waveform: str = "time"

# ---- source ----

@dataclass
class BinarySourceConfig:
    # TODO: this module can also take seeding. Without proper need
    # it is postponed. (carried over from the old draft)
    pass

# ---- modulation ----

@dataclass
class MapperConfig:
    return_indices: bool = False

@dataclass
class DemapperConfig:
    hard_out: bool = False
    demapping_method: Literal["app", "maxlog"] = "app"

@dataclass
class ConstellationConfig:
    # normalize/center/points only apply when modulation.type == "custom".
    # Unused for qam/pam, kept here since Sionna's Constellation always
    # accepts them.
    normalize: bool = False
    center: bool = False
    points: Optional[list] = None

@dataclass
class ModulationConfig:

    type: Literal["qam", "pam", "custom"]

    num_bits_per_symbol: int

    mapper: MapperConfig
    demapper: DemapperConfig
    constellation: ConstellationConfig

    def __post_init__(self):
        if self.type not in ("qam", "pam", "custom"):
            raise ValueError(f"ModulationConfig.type must be 'qam'/'pam'/'custom', got {self.type!r}")
        
        if not isinstance(self.num_bits_per_symbol, int):
            raise ValueError("num_bits_per_symbol must be an integer")

        if self.type == "qam" and self.num_bits_per_symbol % 2 != 0:
            raise ValueError(
                f"type='qam' requires an even num_bits_per_symbol "
                f"(QAM is built from two independent PAM arms on I/Q), got {self.num_bits_per_symbol}. "
                f"Use type='custom' for odd orders like BPSK."
            )

        if self.type == "custom":

            pts = self.constellation.points

            if pts is None:
                raise ValueError("type='custom' requires constellation.points to be set")

# ---- channels ----

@dataclass
class AWGNChannelConfig:

    precision: Optional[str] = None
    device: Optional[str] = None


@dataclass
class TDLChannelConfig:

    model: str
    delay_spread: float
    carrier_frequency: float

    num_sinusoids: int = 20
    los_angle_of_arrival: float = 0.7853981633974483
    min_speed: float = 0.0
    max_speed: Optional[float] = None

    num_rx_ant: int = 1
    num_tx_ant: int = 1
    spatial_corr_mat: Optional[list] = None
    rx_corr_mat: Optional[list] = None
    tx_corr_mat: Optional[list] = None

    precision: Optional[str] = None
    device: Optional[str] = None

    def __post_init__(self):

        required = ("model", "delay_spread", "carrier_frequency")
        missing = [name for name in required if getattr(self, name) is None]

        if missing:

            raise ValueError(
                f"TDLChannelConfig missing required field(s): {', '.join(missing)}"
            )


@dataclass
class PanelArrayConfig:

    num_rows_per_panel: int = 1
    num_cols_per_panel: int = 1

    polarization: Literal["single", "dual"] = "single"
    polarization_type: Literal["V", "H", "cross"] = "V"

    antenna_pattern: str = "38.901"


@dataclass
class SystemLevelChannelConfig:

    # Shared shape for UMi/UMa/RMa -- `variant` picks the concrete
    # Sionna class in builders.py; the fields below don't change
    # per variant.
    variant: Literal["umi", "uma", "rma"]

    carrier_frequency: float

    o2i_model: Literal["low", "high"] = "low"
    direction: Literal["uplink", "downlink"] = "downlink"

    enable_pathloss: bool = True
    enable_shadow_fading: bool = True

    bs_array: PanelArrayConfig = field(default_factory=PanelArrayConfig)
    ut_array: PanelArrayConfig = field(
        default_factory=lambda: PanelArrayConfig(antenna_pattern="omni")
    )

    precision: Optional[str] = None
    device: Optional[str] = None

    def __post_init__(self):
        if self.variant not in ("umi", "uma", "rma"):
            raise ValueError(
                f"SystemLevelChannelConfig.variant must be 'umi'/'uma'/'rma', got {self.variant!r}"
            )


@dataclass
class RayleighBlockFadingConfig:

    num_rx: int = 1
    num_rx_ant: int = 1
    num_tx: int = 1
    num_tx_ant: int = 1

    precision: Optional[str] = None
    device: Optional[str] = None


@dataclass
class ChannelsConfig:

    awgn: AWGNChannelConfig
    tdl: TDLChannelConfig
    system_level: SystemLevelChannelConfig
    rayleigh_block_fading: RayleighBlockFadingConfig


# ---- waveforms ----

@dataclass
class ResourceGridConfig:
    num_ofdm_symbols: int
    fft_size: int
    subcarrier_spacing: float

    num_tx: int = 1
    num_streams_per_tx: int = 1
    cyclic_prefix_length: int = 16
    num_guard_carriers: List[int] = field(default_factory=lambda: [0, 0])
    dc_null: bool = True
    pilot_pattern: Literal["kronecker", "empty"] = "empty"
    pilot_ofdm_symbol_indices: List[int] = field(default_factory=list)

    precision: Optional[str] = None
    device: Optional[str] = None


@dataclass
class TimeWaveformConfig:
    bandwidth: float
    num_time_samples: int
    maximum_delay_spread: float = 3e-6

    l_min: Optional[int] = None
    l_max: Optional[int] = None

    normalize_channel: bool = False
    return_channel: bool = False

    precision: Optional[str] = None
    device: Optional[str] = None


@dataclass
class OFDMWaveformConfig:
    resource_grid: ResourceGridConfig

    normalize_channel: bool = False
    return_channel: bool = False

    precision: Optional[str] = None
    device: Optional[str] = None


@dataclass
class WaveformsConfig:
    time: TimeWaveformConfig
    ofdm: OFDMWaveformConfig


# ---- sweep ----

@dataclass
class EbNoRangeConfig:
    start: float
    stop: float
    step: float

    def __post_init__(self):

        if self.step <= 0:

            raise ValueError(
                f"EbNoRangeConfig.step must be > 0, got {self.step!r} "
                f"(a non-positive step makes generate_sweep()'s loop never terminate)"
            )


@dataclass
class SweepConfig:
    ebno_db: EbNoRangeConfig
    batch_size: int
    num_symbols: int

# ---- AMR specific ----

@dataclass
class AmrModulationSpec:
    name: str
    type: Literal["qam", "pam", "custom"]
    num_bits_per_symbol: int
    constellation: ConstellationConfig

    def to_modulation_config(self, mapper: MapperConfig, demapper: DemapperConfig) -> ModulationConfig:
        # Reuses ModulationConfig's own __post_init__ validation (even-bits-for-qam,
        # points-required-for-custom) -- no validation logic duplicated here.
        return ModulationConfig(
            type = self.type,
            num_bits_per_symbol = self.num_bits_per_symbol,
            mapper = mapper,
            demapper = demapper,
            constellation = self.constellation,
        )


@dataclass
class AmrDatasetConfig:
    variants: List[Literal["umi", "uma", "rma"]]
    snr_db: EbNoRangeConfig
    num_vectors: int
    vector_len: int
    mapper: MapperConfig
    demapper: DemapperConfig
    modulations: List[AmrModulationSpec]

    # Upper bound on batch_size per generate_iq() call. num_vectors is
    # produced by looping smaller sub-batches and concatenating, not
    # necessarily in one forward pass -- the demapper's "app" method
    # materializes a [batch, vector_len, num_points] tensor, which for
    # high-order modulations (e.g. 1024QAM: num_points=1024) can exceed
    # available GPU memory well before num_vectors=1024 is reached.
    # None -> no chunking, one call of batch_size=num_vectors (previous behavior).
    max_batch_size: Optional[int] = None

    disable_pathloss_and_shadowing: bool = True

    def __post_init__(self):
        if not self.modulations:
            raise ValueError("amr_dataset.modulations must not be empty")
        names = [m.name for m in self.modulations]
        if len(names) != len(set(names)):
            raise ValueError(f"amr_dataset.modulations names must be unique, got {names}")
        for v in self.variants:
            if v not in ("umi", "uma", "rma"):
                raise ValueError(f"amr_dataset.variants entries must be 'umi'/'uma'/'rma', got {v!r}")
        if self.max_batch_size is not None and self.max_batch_size <= 0:
            raise ValueError(f"amr_dataset.max_batch_size must be > 0, got {self.max_batch_size!r}")

# ---- top level ----

@dataclass
class Config:
    common: CommonConfig

    source: BinarySourceConfig
    modulation: ModulationConfig

    channels: ChannelsConfig
    waveforms: WaveformsConfig

    sweep: SweepConfig

    amr_dataset: Optional[AmrDatasetConfig] = None

    def __post_init__(self):

        valid_channels = tuple(f.name for f in dataclasses.fields(ChannelsConfig))
        valid_waveforms = tuple(f.name for f in dataclasses.fields(WaveformsConfig))

        if self.common.active_channel not in valid_channels:
            raise ValueError(
                f"active_channel must be one of {valid_channels}, "
                f"got {self.common.active_channel!r}"
            )

        if self.common.active_waveform not in valid_waveforms:
            raise ValueError(
                f"active_waveform must be one of {valid_waveforms}, "
                f"got {self.common.active_waveform!r}"
            )

        if (self.common.active_waveform == "time"
            and self.sweep.num_symbols != self.waveforms.time.num_time_samples):
            raise ValueError(
                f"sweep.num_symbols ({self.sweep.num_symbols}) must equal "
                f"waveforms.time.num_time_samples ({self.waveforms.time.num_time_samples}) "
                f"when active_waveform='time'"
            )

# ---- parser functions ----

def _strip_comments(raw: dict) -> dict:
    """Drop documentation-only keys (e.g. "_comment", "_description")
    so config.json can carry inline notes without schema.py treating
    them as unknown fields. Convention: any key starting with "_" is
    a comment, never real data.
    """

    return {k: v for k, v in raw.items() if not k.startswith("_")}


def parse_source(raw: dict) -> BinarySourceConfig:

    return BinarySourceConfig()


def parse_modulation(raw: dict) -> ModulationConfig:

    raw = _strip_comments(raw)

    return ModulationConfig(
        
        type = raw.get("type"),

        num_bits_per_symbol = raw.get("num_bits_per_symbol"),

        mapper = MapperConfig(**_strip_comments(raw.get("mapper", {}))),
        demapper = DemapperConfig(**_strip_comments(raw.get("demapper", {}))),

        constellation = ConstellationConfig(**_strip_comments(raw.get("constellation", {}))),
    )


def parse_channels(raw: dict) -> ChannelsConfig:

    system_level_raw = _strip_comments(raw["system_level"])

    bs_array_raw = system_level_raw.pop("bs_array", {})
    ut_array_raw = system_level_raw.pop("ut_array", {})

    return ChannelsConfig(

        awgn = AWGNChannelConfig(**_strip_comments(raw.get("awgn", {}))),

        tdl = TDLChannelConfig(**_strip_comments(raw["tdl"])),

        system_level = SystemLevelChannelConfig(
            **system_level_raw,
            bs_array = PanelArrayConfig(**_strip_comments(bs_array_raw)),
            ut_array = PanelArrayConfig(**_strip_comments(ut_array_raw)),
        ),

        rayleigh_block_fading=RayleighBlockFadingConfig(
            **_strip_comments(raw.get("rayleigh_block_fading", {}))
        ),
    )


def parse_waveforms(raw: dict) -> WaveformsConfig:
    ofdm_raw = _strip_comments(raw["ofdm"])
    resource_grid_raw = _strip_comments(ofdm_raw.pop("resource_grid"))

    return WaveformsConfig(
        time = TimeWaveformConfig(**_strip_comments(raw["time"])),
        ofdm = OFDMWaveformConfig(
            **ofdm_raw,
            resource_grid=ResourceGridConfig(**resource_grid_raw),
        ),
    )


def parse_sweep(raw: dict) -> SweepConfig:
    raw = _strip_comments(raw)
    return SweepConfig(
        ebno_db = EbNoRangeConfig(**raw["ebno_db"]),
        batch_size = raw["batch_size"],
        num_symbols = raw["num_symbols"],
    )


# ---- top level parser ----

def parse_common(raw: dict) -> CommonConfig:
    raw = _strip_comments(raw)
    return CommonConfig(**raw)


def parse_config(raw: dict) -> Config:
    return Config(
        common = parse_common(raw["common"]),
        source = parse_source(raw.get("source", {})),
        modulation = parse_modulation(raw["modulation"]),
        channels = parse_channels(raw["channels"]),
        waveforms = parse_waveforms(raw["waveforms"]),
        sweep = parse_sweep(raw["sweep"]),
        amr_dataset = parse_amr_dataset(raw.get("amr_dataset")),
    )

def parse_amr_modulation(raw: dict) -> AmrModulationSpec:
    raw = _strip_comments(raw)
    return AmrModulationSpec(
        name = raw["name"],
        type = raw["type"],
        num_bits_per_symbol = raw["num_bits_per_symbol"],
        constellation = ConstellationConfig(**_strip_comments(raw.get("constellation", {}))),
    )


def parse_amr_dataset(raw: Optional[dict]) -> Optional[AmrDatasetConfig]:

    if raw is None:
        return None
    
    raw = _strip_comments(raw)

    return AmrDatasetConfig(
        variants = raw["variants"],
        snr_db = EbNoRangeConfig(**raw["snr_db"]),
        num_vectors = raw["num_vectors"],
        vector_len = raw["vector_len"],
        max_batch_size = raw.get("max_batch_size"),
        disable_pathloss_and_shadowing = raw.get("disable_pathloss_and_shadowing", True),
        mapper = MapperConfig(**_strip_comments(raw.get("mapper", {}))),
        demapper = DemapperConfig(**_strip_comments(raw.get("demapper", {}))),
        modulations = [parse_amr_modulation(m) for m in raw["modulations"]],
    )
