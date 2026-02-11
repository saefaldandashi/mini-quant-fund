"""
Symbol Validator - Filters delisted, invalid, and problematic symbols.

CRITICAL FIX #7: Data Quality - Filter Delisted/Missing Symbols
Prevents errors from delisted symbols, invalid tickers, and missing data.
"""

import logging
import re
from typing import Set, List, Dict, Optional
from datetime import datetime, timedelta
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class SymbolValidator:
    """
    Validates and filters stock symbols to prevent errors from:
    - Delisted symbols
    - Invalid ticker formats
    - Symbols with missing data
    - International/exchange-specific symbols that cause issues
    """
    
    def __init__(self, cache_file: Optional[str] = None):
        """
        Initialize symbol validator.
        
        Args:
            cache_file: Path to cache file for bad symbols (persists across runs)
        """
        self.cache_file = cache_file or "outputs/bad_symbols_cache.json"
        self.bad_symbols: Set[str] = set()
        self.bad_symbols_timestamp: Dict[str, datetime] = {}
        self.cache_duration = timedelta(days=30)  # Re-check after 30 days
        
        # Known problematic patterns
        self.invalid_patterns = [
            r'^\$',  # Symbols starting with $ (e.g., $TASI.SR)
            r'\.SR$',  # Saudi Riyadh exchange
            r'\.AD$',  # Australian exchange
            r'\.L$',  # London exchange (if not handled)
            r'\.TO$',  # Toronto exchange (if not handled)
            r'^[0-9]',  # Symbols starting with numbers
        ]
        
        # Known delisted/problematic symbols from logs
        self.known_bad_symbols = {
            'ANSS',  # Delisted or data unavailable
            '$TASI.SR',  # Invalid format
            '$ADI.AD',  # Invalid format
        }
        
        # Load cached bad symbols
        self._load_cache()
    
    def _load_cache(self):
        """Load cached bad symbols from file."""
        try:
            cache_path = Path(self.cache_file)
            if cache_path.exists():
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    self.bad_symbols = set(data.get('bad_symbols', []))
                    # Convert timestamps
                    for symbol, ts_str in data.get('timestamps', {}).items():
                        try:
                            self.bad_symbols_timestamp[symbol] = datetime.fromisoformat(ts_str)
                        except:
                            pass
                logger.info(f"Loaded {len(self.bad_symbols)} bad symbols from cache")
        except Exception as e:
            logger.warning(f"Could not load symbol cache: {e}")
    
    def _save_cache(self):
        """Save bad symbols to cache file."""
        try:
            cache_path = Path(self.cache_file)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'bad_symbols': list(self.bad_symbols),
                'timestamps': {
                    symbol: ts.isoformat() 
                    for symbol, ts in self.bad_symbols_timestamp.items()
                }
            }
            
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not save symbol cache: {e}")
    
    def is_valid_format(self, symbol: str) -> bool:
        """
        Check if symbol has valid format.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            True if format is valid, False otherwise
        """
        if not symbol or len(symbol) == 0:
            return False
        
        # Check against invalid patterns
        for pattern in self.invalid_patterns:
            if re.search(pattern, symbol):
                return False
        
        # Basic format check: alphanumeric, dots, hyphens allowed
        if not re.match(r'^[A-Z0-9][A-Z0-9\.\-]*$', symbol.upper()):
            return False
        
        return True
    
    def is_known_bad(self, symbol: str) -> bool:
        """
        Check if symbol is in known bad list.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            True if known to be bad, False otherwise
        """
        return symbol.upper() in self.known_bad_symbols or symbol.upper() in self.bad_symbols
    
    def mark_as_bad(self, symbol: str, reason: str = "Data unavailable"):
        """
        Mark a symbol as bad (delisted, missing data, etc.).
        
        Args:
            symbol: Stock symbol to mark
            reason: Reason for marking as bad
        """
        symbol_upper = symbol.upper()
        self.bad_symbols.add(symbol_upper)
        self.bad_symbols_timestamp[symbol_upper] = datetime.now()
        logger.debug(f"Marked {symbol} as bad: {reason}")
        self._save_cache()
    
    def should_retry(self, symbol: str) -> bool:
        """
        Check if we should retry fetching data for a symbol.
        Symbols are re-checked after cache_duration.
        
        Args:
            symbol: Stock symbol to check
            
        Returns:
            True if should retry, False if still bad
        """
        symbol_upper = symbol.upper()
        if symbol_upper not in self.bad_symbols:
            return True
        
        # Check if cache expired
        if symbol_upper in self.bad_symbols_timestamp:
            age = datetime.now() - self.bad_symbols_timestamp[symbol_upper]
            if age > self.cache_duration:
                # Remove from bad list to retry
                self.bad_symbols.discard(symbol_upper)
                del self.bad_symbols_timestamp[symbol_upper]
                self._save_cache()
                return True
        
        return False
    
    def validate_symbol(self, symbol: str) -> tuple[bool, Optional[str]]:
        """
        Validate a single symbol.
        
        Args:
            symbol: Stock symbol to validate
            
        Returns:
            Tuple of (is_valid, reason_if_invalid)
        """
        if not symbol:
            return False, "Empty symbol"
        
        symbol_upper = symbol.upper()
        
        # Check format
        if not self.is_valid_format(symbol_upper):
            return False, f"Invalid format: {symbol}"
        
        # Check known bad symbols
        if self.is_known_bad(symbol_upper):
            if not self.should_retry(symbol_upper):
                return False, f"Known bad symbol (delisted/missing data): {symbol}"
        
        return True, None
    
    def validate_symbols(self, symbols: List[str]) -> tuple[List[str], List[str]]:
        """
        Validate a list of symbols, returning valid and invalid ones.
        
        Args:
            symbols: List of symbols to validate
            
        Returns:
            Tuple of (valid_symbols, invalid_symbols_with_reasons)
        """
        valid = []
        invalid = []
        
        for symbol in symbols:
            is_valid, reason = self.validate_symbol(symbol)
            if is_valid:
                valid.append(symbol)
            else:
                invalid.append(f"{symbol}: {reason}")
        
        if invalid:
            logger.info(f"Filtered {len(invalid)} invalid symbols: {', '.join(invalid[:10])}")
            if len(invalid) > 10:
                logger.info(f"... and {len(invalid) - 10} more")
        
        return valid, invalid
    
    def filter_symbols(self, symbols: List[str]) -> List[str]:
        """
        Filter symbols, returning only valid ones.
        
        Args:
            symbols: List of symbols to filter
            
        Returns:
            List of valid symbols
        """
        valid, _ = self.validate_symbols(symbols)
        return valid


# Global instance
_symbol_validator: Optional[SymbolValidator] = None


def get_symbol_validator() -> SymbolValidator:
    """Get or create global symbol validator instance."""
    global _symbol_validator
    if _symbol_validator is None:
        _symbol_validator = SymbolValidator()
    return _symbol_validator
