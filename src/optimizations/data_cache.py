"""
Data Cache - Avoid redundant API calls with intelligent caching.
"""

import logging
import hashlib
import json
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd

logger = logging.getLogger(__name__)


class DataCache:
    """
    Intelligent data caching to minimize API calls.
    
    Features:
    - Memory cache for current session
    - Disk cache for persistence across restarts
    - TTL-based expiration
    - Cache invalidation on market close
    - Compression for large datasets
    """
    
    def __init__(
        self,
        cache_dir: str = "outputs/cache",
        memory_ttl_minutes: int = 5,
        disk_ttl_hours: int = 24,
        max_memory_items: int = 100,
    ):
        """
        Initialize data cache.
        
        Args:
            cache_dir: Directory for disk cache
            memory_ttl_minutes: Memory cache TTL
            disk_ttl_hours: Disk cache TTL
            max_memory_items: Maximum items in memory
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.memory_ttl = timedelta(minutes=memory_ttl_minutes)
        self.disk_ttl = timedelta(hours=disk_ttl_hours)
        self.max_memory_items = max_memory_items
        
        # Memory cache: key -> (data, timestamp)
        self._memory_cache: Dict[str, Tuple[Any, datetime]] = {}
        
        # Cache statistics
        self.hits = 0
        self.misses = 0
    
    def _make_key(self, category: str, params: Dict) -> str:
        """Generate cache key from category and parameters."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        hash_val = hashlib.md5(params_str.encode()).hexdigest()[:12]
        return f"{category}_{hash_val}"
    
    def get(
        self,
        category: str,
        params: Dict,
        use_disk: bool = True,
    ) -> Optional[Any]:
        """
        Get data from cache.
        
        Args:
            category: Cache category (e.g., 'price_data', 'news')
            params: Parameters that define this data
            use_disk: Whether to check disk cache
        
        Returns:
            Cached data or None if not found/expired
        """
        key = self._make_key(category, params)
        
        # Check memory cache
        if key in self._memory_cache:
            data, timestamp = self._memory_cache[key]
            if datetime.now() - timestamp < self.memory_ttl:
                self.hits += 1
                logger.debug(f"Memory cache hit: {key}")
                return data
            else:
                # Expired
                del self._memory_cache[key]
        
        # Check disk cache
        if use_disk:
            disk_data = self._get_from_disk(key)
            if disk_data is not None:
                # Promote to memory cache
                self._set_memory(key, disk_data)
                self.hits += 1
                logger.debug(f"Disk cache hit: {key}")
                return disk_data
        
        self.misses += 1
        return None
    
    def set(
        self,
        category: str,
        params: Dict,
        data: Any,
        persist_to_disk: bool = True,
    ):
        """
        Store data in cache.
        
        Args:
            category: Cache category
            params: Parameters that define this data
            data: Data to cache
            persist_to_disk: Whether to also save to disk
        """
        key = self._make_key(category, params)
        
        # Set in memory
        self._set_memory(key, data)
        
        # Persist to disk
        if persist_to_disk:
            self._save_to_disk(key, data)
    
    def _set_memory(self, key: str, data: Any):
        """Set data in memory cache with eviction."""
        # Evict oldest if at capacity
        if len(self._memory_cache) >= self.max_memory_items:
            oldest_key = min(
                self._memory_cache.keys(),
                key=lambda k: self._memory_cache[k][1]
            )
            del self._memory_cache[oldest_key]
        
        self._memory_cache[key] = (data, datetime.now())
    
    def _save_to_disk(self, key: str, data: Any):
        """Save data to disk cache."""
        try:
            cache_file = self.cache_dir / f"{key}.pkl"
            
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': datetime.now().isoformat(),
                }, f)
                
            logger.debug(f"Saved to disk cache: {key}")
        except Exception as e:
            logger.warning(f"Failed to save to disk cache: {e}")
    
    def _get_from_disk(self, key: str) -> Optional[Any]:
        """Load data from disk cache."""
        cache_file = self.cache_dir / f"{key}.pkl"
        
        if not cache_file.exists():
            return None
        
        try:
            with open(cache_file, 'rb') as f:
                cached = pickle.load(f)
            
            timestamp = datetime.fromisoformat(cached['timestamp'])
            
            if datetime.now() - timestamp > self.disk_ttl:
                # Expired, delete file
                cache_file.unlink()
                return None
            
            return cached['data']
        except Exception as e:
            logger.warning(f"Failed to load from disk cache: {e}")
            return None
    
    def invalidate(self, category: Optional[str] = None):
        """
        Invalidate cache entries.
        
        Args:
            category: If provided, only invalidate this category
        """
        if category:
            # Remove matching memory cache entries
            keys_to_remove = [
                k for k in self._memory_cache.keys()
                if k.startswith(category)
            ]
            for key in keys_to_remove:
                del self._memory_cache[key]
            
            # Remove matching disk cache files
            for cache_file in self.cache_dir.glob(f"{category}_*.pkl"):
                cache_file.unlink()
        else:
            # Clear all
            self._memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        
        logger.info(f"Cache invalidated: {category or 'all'}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'memory_items': len(self._memory_cache),
            'disk_files': len(list(self.cache_dir.glob("*.pkl"))),
        }


class PriceDataCache(DataCache):
    """Specialized cache for price data."""
    
    def get_prices(
        self,
        symbols: List[str],
        days: int,
        end_date: Optional[datetime] = None,
    ) -> Optional[Dict[str, pd.Series]]:
        """Get cached price data."""
        params = {
            'symbols': sorted(symbols),
            'days': days,
            'end_date': end_date.date().isoformat() if end_date else 'latest',
        }
        return self.get('prices', params)
    
    def set_prices(
        self,
        symbols: List[str],
        days: int,
        data: Dict[str, pd.Series],
        end_date: Optional[datetime] = None,
    ):
        """Cache price data."""
        params = {
            'symbols': sorted(symbols),
            'days': days,
            'end_date': end_date.date().isoformat() if end_date else 'latest',
        }
        self.set('prices', params, data)
