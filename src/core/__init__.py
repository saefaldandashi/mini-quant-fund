"""
Core Module - Foundation types for the trading system.

Provides:
- Instrument types (Spot, Futures)
- Position management (Long/Short)
- Trade recording
"""

from .instruments import (
    Instrument,
    SpotInstrument,
    FutureInstrument,
    AssetClass,
    SettlementType,
    InstrumentRegistry,
    get_instrument_registry,
)

from .positions import (
    Position,
    PositionSide,
    Trade,
    PositionBook,
)

__all__ = [
    # Instruments
    'Instrument',
    'SpotInstrument',
    'FutureInstrument',
    'AssetClass',
    'SettlementType',
    'InstrumentRegistry',
    'get_instrument_registry',
    
    # Positions
    'Position',
    'PositionSide',
    'Trade',
    'PositionBook',
]
