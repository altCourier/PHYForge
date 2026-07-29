"""
builders.py

Consumes validated schema objects (from schema.py) and constructs
live Sionna objects from them. This file is allowed to import
sionna/torch freely — schema.py deliberately is not.

Every build_* function for a given section returns objects with a
common interface, so runtime.py never branches on config.type itself.
"""

import sionna.phy

from schema import SourceConfig, ModulationConfig, ChannelConfig, TDLChannelConfig, AWGNChannelConfig


class ChannelHandle:
    """
    Common interface every channel branch returns, so runtime.py
    can call any channel the same way regardless of type.
    """

    def __call__(self, x, no):
        raise NotImplementedError


class AWGNChannelHandle(ChannelHandle):
    def __init__(self):
        # TODO: instantiate sionna.phy.channel.AWGN

        

        pass

    def __call__(self, x, no):
        # TODO: call the underlying AWGN object with (x, no)
        # check the AWGN docs for the exact expected call signature
        pass


class TDLChannelHandle(ChannelHandle):
    def __init__(self, config: TDLChannelConfig):
        # TODO: 1. instantiate sionna.phy.channel.tr38901.TDL from config fields
        #       2. instantiate sionna.phy.channel.TimeChannel, wrapping the TDL
        #          object — check what args TimeChannel needs beyond the model
        #          itself (bandwidth, num_time_samples, l_min/l_max...)
        #       3. if config.l_min/l_max are None, where do you get real values?
        #          (hint: the TDL object itself exposes these after construction)
        pass

    def __call__(self, x, no):
        # TODO: reshape x from flat symbols into TimeChannel's expected
        #       [batch, num_tx, num_tx_ant, num_time_samples] shape,
        #       call the wrapped TimeChannel, then decide: truncate or
        #       pad the output back to the input length? (your own
        #       open question #2 — pick a default, note it in a comment)
        pass


def build_channel(config: ChannelConfig) -> ChannelHandle:

    if isinstance(config, AWGNChannelConfig):

        return AWGNChannelHandle()

    elif isinstance(config, TDLChannelConfig):
        return TDLChannelHandle(config)

    raise ValueError(f"No builder for channel config type: {type(config)!r}")


def build_source(config: SourceConfig):
    # TODO: dispatch on config.type once more than "binary" exists;
    # for now this can just always return sionna.phy.mapping.BinarySource()
    pass


def build_mapsys(config: ModulationConfig):
    # TODO: construct Constellation(config.type, config.num_bits_per_symbol),
    # then Mapper(constellation) and Demapper(...) — check Demapper's
    # required args, it needs to know the demapping method ("app" vs "maxlog")
    # which your schema doesn't currently capture — you may need to add
    # a field for that
    pass