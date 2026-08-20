"""
Tests for physys/runtime.py

Mostly real-Sionna integration tests (this file's whole job is
orchestrating the real pipeline, so mocking it away would test nothing
useful). The one exception is the truncation "front not back" test,
which monkeypatches PHYSys._handle on a single instance with a
position-marker tensor -- explained on that test.
"""
import torch
import pytest

from sionna.phy.utils import ebnodb2no
import sionna.phy

from physys.runtime import PHYSys, tf_reshape_for_time_channel
from physys.schema import (
    Config, CommonConfig, BinarySourceConfig,
    ModulationConfig, MapperConfig, DemapperConfig, ConstellationConfig,
    ChannelsConfig, AWGNChannelConfig, TDLChannelConfig,
    SystemLevelChannelConfig, RayleighBlockFadingConfig,
    WaveformsConfig, TimeWaveformConfig, OFDMWaveformConfig, ResourceGridConfig,
    SweepConfig, EbNoRangeConfig,
)


# ---------------------------------------------------------------------------
# Shared config helper. Built directly via dataclasses (mirrors the
# original __main__ smoke-test block) rather than parse_config + dict,
# since num_symbols/variant/etc need to vary per test.
# ---------------------------------------------------------------------------

def _config(active_channel = "awgn", active_waveform = "time", num_symbols = 64,
            variant = "umi", num_bits_per_symbol = 2):
    
    return Config(

        common = CommonConfig(active_channel = active_channel, active_waveform = active_waveform),

        source = BinarySourceConfig(),

        modulation = ModulationConfig(
            type = "qam",
            num_bits_per_symbol = num_bits_per_symbol,
            mapper = MapperConfig(),
            demapper = DemapperConfig(),
            constellation = ConstellationConfig(),
        ),

        channels = ChannelsConfig(
            awgn = AWGNChannelConfig(),
            tdl = TDLChannelConfig(model = "A", delay_spread = 100e-9, carrier_frequency = 3.5e9),
            system_level = SystemLevelChannelConfig(variant = variant, carrier_frequency = 3.5e9),
            rayleigh_block_fading = RayleighBlockFadingConfig(),
        ),

        waveforms = WaveformsConfig(

            time = TimeWaveformConfig(bandwidth = 1e6, num_time_samples = num_symbols),

            ofdm = OFDMWaveformConfig(
                resource_grid = ResourceGridConfig(
                    num_ofdm_symbols = 14, fft_size = 76, subcarrier_spacing = 30e3,
                    pilot_ofdm_symbol_indices=[2, 11],
                )
            ),
        ),

        sweep = SweepConfig(
            ebno_db = EbNoRangeConfig(start = 0.0, stop = 0.0, step = 2.0),
            batch_size = 4,
            num_symbols = num_symbols,
        ),
    )


BATCH_SIZE = 4
NUM_SYMBOLS = 64


# ---------------------------------------------------------------------------
# Shapes/dtypes: AWGN, TDL, and all three system-level variants
# ---------------------------------------------------------------------------

def test_generate_awgn_shapes_and_dtypes():
    config = _config(active_channel="awgn", active_waveform="time", num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    bits, llr = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)

    expected_shape = (BATCH_SIZE, NUM_SYMBOLS * config.modulation.num_bits_per_symbol)
    assert bits.shape == expected_shape
    assert llr.shape == bits.shape  # one LLR per transmitted bit
    assert torch.is_floating_point(bits)
    assert torch.is_floating_point(llr)
    assert torch.isfinite(llr).all()


def test_generate_tdl_shapes_and_dtypes():
    config = _config(active_channel="tdl", active_waveform="time", num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    bits, llr = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)

    assert llr.shape == bits.shape
    assert torch.isfinite(llr).all()


def test_generate_rayleigh_shapes_and_dtypes():
    config = _config(active_channel="rayleigh_block_fading", active_waveform="time",
                      num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    bits, llr = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)

    assert llr.shape == bits.shape
    assert torch.isfinite(llr).all()


@pytest.mark.parametrize("variant", ["umi", "uma", "rma"])
def test_generate_system_level_shapes_and_dtypes(variant):
    """UMi was already smoke-tested manually per the plan -- this
    parametrization is what actually gets UMa/RMa covered too."""
    config = _config(
        active_channel="system_level", active_waveform="time",
        num_symbols=NUM_SYMBOLS, variant=variant,
    )
    system = PHYSys(config)

    bits, llr = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)

    assert llr.shape == bits.shape
    assert torch.isfinite(llr).all()


# ---------------------------------------------------------------------------
# generate_sweep()
# ---------------------------------------------------------------------------

def test_generate_sweep_keys_match_expected_ebno_points():
    config = _config(active_channel="awgn", active_waveform="time", num_symbols=NUM_SYMBOLS)
    config.sweep.ebno_db = EbNoRangeConfig(start=0.0, stop=6.0, step=2.0)
    config.sweep.batch_size = BATCH_SIZE
    config.sweep.num_symbols = NUM_SYMBOLS

    system = PHYSys(config)
    results = system.generate_sweep()

    assert set(results.keys()) == {0.0, 2.0, 4.0, 6.0}

    for ebno_db, (bits, llr) in results.items():
        assert llr.shape == bits.shape
        assert torch.isfinite(llr).all()


def test_generate_sweep_uses_sweep_batch_size_and_num_symbols():
    """generate_sweep() should ignore any batch_size/num_symbols passed
    elsewhere and always use config.sweep's values for every point."""
    config = _config(active_channel="awgn", active_waveform="time", num_symbols=NUM_SYMBOLS)
    config.sweep.ebno_db = EbNoRangeConfig(start=0.0, stop=0.0, step=2.0)
    config.sweep.batch_size = 7
    config.sweep.num_symbols = NUM_SYMBOLS

    system = PHYSys(config)
    results = system.generate_sweep()

    bits, llr = results[0.0]
    assert bits.shape[0] == 7


def test_generate_sweep_float_step_drift_current_behavior():
    """
    PINS current behavior, doesn't endorse it: step=0.3 doesn't evenly
    divide (stop - start), so float accumulation in the while loop can
    include/exclude the last point unpredictably. If this test starts
    failing after a future change to generate_sweep's loop, that's a
    deliberate signal to re-check this test, not silently update it.
    """
    config = _config(active_channel="awgn", active_waveform="time", num_symbols=NUM_SYMBOLS)
    config.sweep.ebno_db = EbNoRangeConfig(start=0.0, stop=1.0, step=0.3)
    config.sweep.batch_size = BATCH_SIZE
    config.sweep.num_symbols = NUM_SYMBOLS

    system = PHYSys(config)
    results = system.generate_sweep()

    # 0.0, 0.3, 0.6, 0.9 all <= 1.0; 1.2 is not. Document exactly this.
    assert len(results) == 4


def test_generate_sweep_rejects_nonpositive_step_at_construction():
    with pytest.raises(ValueError, match="step"):
        EbNoRangeConfig(start=0.0, stop=10.0, step=0.0)

# ---------------------------------------------------------------------------
# No NaN/Inf in llr across a full SNR sweep, not just one point.
# Focused on AWGN/TDL (system-level already gets one point checked above --
# repeating the full sweep for all 3 variants would be expensive for
# limited extra signal, since ebnodb2no's edge-case behavior doesn't
# depend on which channel type is active).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("ebno_db", [-10.0, -5.0, 0.0, 10.0, 20.0])
def test_generate_awgn_llr_finite_across_snr_sweep(ebno_db):
    config = _config(active_channel="awgn", active_waveform="time", num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    _, llr = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=ebno_db)

    assert torch.isfinite(llr).all(), f"non-finite LLR at ebno_db={ebno_db}"


@pytest.mark.parametrize("ebno_db", [-10.0, -5.0, 0.0, 10.0, 20.0])
def test_generate_tdl_llr_finite_across_snr_sweep(ebno_db):
    config = _config(active_channel="tdl", active_waveform="time", num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    _, llr = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=ebno_db)

    assert torch.isfinite(llr).all(), f"non-finite LLR at ebno_db={ebno_db}"


# ---------------------------------------------------------------------------
# Truncation math (TDL + time waveform)
# ---------------------------------------------------------------------------

def test_tdl_time_channel_output_length_before_truncation():
    """
    Real Sionna, no mocking: confirm the untruncated TimeChannel output
    really is num_symbols + (l_max - l_min) long, per Open Question 1
    in the module docstring.

    NOTE: assumes TimeChannel exposes l_min/l_max as public attributes
    after construction. If this errors with AttributeError rather than
    an assertion failure, that's a naming mismatch to fix, not a bug
    in runtime.py.
    """
    config = _config(active_channel="tdl", active_waveform="time", num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    l_min = system._handle.l_min
    l_max = system._handle.l_max

    num_bits_per_symbol = config.modulation.num_bits_per_symbol
    no = ebnodb2no(5.0, num_bits_per_symbol=num_bits_per_symbol, coderate=1.0)

    bits = system.source([BATCH_SIZE, NUM_SYMBOLS * num_bits_per_symbol])
    x = system.mapper(bits)
    x_time = tf_reshape_for_time_channel(x, BATCH_SIZE, 1)

    y_time = system._handle(x_time, no)  # untruncated

    assert y_time.shape[-1] == NUM_SYMBOLS + (l_max - l_min)


def test_truncation_keeps_first_num_symbols_not_last(monkeypatch):
    """
    Isolates JUST the truncation slicing logic from real (noisy,
    random-fading) TDL output, by replacing PHYSys._handle on this one
    instance with a fake that returns a position-marker tensor
    (0, 1, 2, ...) of exactly the length a real TimeChannel would
    produce. If _apply_time_channel kept the LAST num_symbols instead
    of the first, y[0] would be arange(length-num_symbols, length)
    instead of arange(0, num_symbols) -- this test would catch that.
    """
    config = _config(active_channel="tdl", active_waveform="time", num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    l_min = system._handle.l_min
    l_max = system._handle.l_max
    full_length = NUM_SYMBOLS + (l_max - l_min)

    def fake_handle(x_time, no):
        marker = torch.arange(full_length, dtype=torch.float32)
        return marker.reshape([1, 1, 1, full_length]).expand(BATCH_SIZE, 1, 1, full_length).clone()

    monkeypatch.setattr(system, "_handle", fake_handle)

    num_bits_per_symbol = config.modulation.num_bits_per_symbol
    no = ebnodb2no(5.0, num_bits_per_symbol=num_bits_per_symbol, coderate=1.0)
    bits = system.source([BATCH_SIZE, NUM_SYMBOLS * num_bits_per_symbol])
    x = system.mapper(bits)

    y = system._apply_time_channel(x, no, NUM_SYMBOLS)

    assert y.shape == (BATCH_SIZE, NUM_SYMBOLS)
    assert torch.equal(y[0], torch.arange(NUM_SYMBOLS, dtype=torch.float32))


# ---------------------------------------------------------------------------
# OFDM path: contract test -- must still raise NotImplementedError
# ---------------------------------------------------------------------------

def test_ofdm_active_waveform_raises_not_implemented():
    config = _config(active_channel="tdl", active_waveform="ofdm", num_symbols=NUM_SYMBOLS)
    system = PHYSys(config)

    with pytest.raises(NotImplementedError):
        system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)


# ---------------------------------------------------------------------------
# Topology
# ---------------------------------------------------------------------------

def test_system_level_topology_varies_across_calls():
    config = _config(
        active_channel="system_level", active_waveform="time",
        num_symbols=NUM_SYMBOLS, variant="umi",
    )
    system = PHYSys(config)

    _, llr1 = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)
    _, llr2 = system.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)

    assert not torch.equal(llr1, llr2)


@pytest.fixture
def restore_sionna_seed():
    original = sionna.phy.config.seed
    yield
    sionna.phy.config.seed = original

def test_system_level_topology_reproducible_with_seed():
    """
    Inverse of the above: if you seed torch before building AND running
    PHYSys, you should get identical output back -- useful for
    reproducing/debugging a specific bad sample later.

    Reseeds before *constructing* PHYSys, not just before generate(),
    since topology generation happens inside generate() but bit
    sourcing and any other RNG draws could plausibly happen at other
    points in the pipeline too.

    If this fails, it likely means some part of the pipeline (topology
    generation, fading, etc.) draws from NumPy's global RNG rather than
    torch's -- worth knowing either way, so don't just delete the test.
    """
    config = _config(
        active_channel="system_level", active_waveform="time",
        num_symbols=NUM_SYMBOLS, variant="umi",
    )

    sionna.phy.config.seed = 1234
    system_a = PHYSys(config)
    _, llr_a = system_a.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)

    sionna.phy.config.seed = 1234
    system_b = PHYSys(config)
    _, llr_b = system_b.generate(batch_size=BATCH_SIZE, num_symbols=NUM_SYMBOLS, ebno_db=5.0)

    assert torch.equal(llr_a, llr_b)
