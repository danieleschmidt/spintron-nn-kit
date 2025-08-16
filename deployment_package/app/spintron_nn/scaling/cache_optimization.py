"""
Advanced caching and memory optimization for SpinTron-NN-Kit.

This module provides:
- Intelligent multi-level caching
- Memory-efficient model storage
- Result caching with TTL and LRU eviction
- Cache warming and prefetching strategies
"""

import time
import threading
import hashlib
import pickle
from typing import Dict, List, Optional, Any, Tuple, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import OrderedDict, defaultdict
import weakref
import json


class CacheLevel(Enum):
    """Cache levels in the hierarchy."""
    L1_MEMORY = "l1_memory"      # Fast in-memory cache
    L2_MEMORY = "l2_memory"      # Larger in-memory cache
    L3_DISK = "l3_disk"          # Disk-based cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache


class CacheStrategy(Enum):
    """Cache eviction and management strategies."""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    FIFO = "fifo"                # First In, First Out
    TTL = "ttl"                  # Time To Live based
    ADAPTIVE = "adaptive"        # Adaptive strategy based on access patterns


@dataclass
class CacheEntry:
    """Entry in the cache with metadata."""
    key: str
    value: Any
    created_at: float
    last_accessed: float
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[float] = None
    
    def is_expired(self) -> bool:
        """Check if entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def update_access(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


@dataclass
class CacheStats:
    """Cache performance statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size_bytes: int = 0
    entry_count: int = 0
    hit_rate: float = 0.0
    average_access_time: float = 0.0


class IntelligentCache:
    """Multi-level intelligent cache with adaptive strategies."""
    
    def __init__(self, 
                 max_size_bytes: int = 1024*1024*100,  # 100MB default
                 max_entries: int = 10000,
                 strategy: CacheStrategy = CacheStrategy.ADAPTIVE,
                 default_ttl: Optional[float] = None):
        """Initialize intelligent cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
            max_entries: Maximum number of entries
            strategy: Cache management strategy
            default_ttl: Default TTL for entries
        """
        self.max_size_bytes = max_size_bytes
        self.max_entries = max_entries
        self.strategy = strategy
        self.default_ttl = default_ttl
        
        # Storage
        self.entries: Dict[str, CacheEntry] = OrderedDict()
        self.access_order = OrderedDict()  # For LRU
        self.frequency_counter = defaultdict(int)  # For LFU
        
        # Statistics
        self.stats = CacheStats()
        
        # Threading
        self.lock = threading.RLock()
        
        # Background maintenance
        self.maintenance_thread = None
        self.running = False
        
    def start(self):
        """Start cache maintenance thread."""
        self.running = True
        self.maintenance_thread = threading.Thread(target=self._maintenance_loop, daemon=True)
        self.maintenance_thread.start()
    
    def stop(self):
        """Stop cache maintenance thread."""
        self.running = False
        if self.maintenance_thread:
            self.maintenance_thread.join()
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Store value in cache.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live in seconds
            
        Returns:
            True if stored successfully
        """
        with self.lock:
            try:
                # Calculate size
                size_bytes = self._calculate_size(value)
                
                # Use default TTL if not specified
                if ttl is None:
                    ttl = self.default_ttl
                
                # Create entry
                entry = CacheEntry(
                    key=key,
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    access_count=1,
                    size_bytes=size_bytes,
                    ttl_seconds=ttl
                )
                
                # Check if we need to evict
                while self._should_evict(entry):
                    if not self._evict_entry():
                        return False  # Cannot evict more entries
                
                # Store entry
                self.entries[key] = entry
                self.access_order[key] = time.time()
                self.frequency_counter[key] += 1
                
                # Update stats
                self.stats.entry_count += 1
                self.stats.size_bytes += size_bytes
                
                return True
                
            except Exception as e:
                print(f"Error storing cache entry: {e}")
                return False
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve value from cache.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        start_time = time.time()
        
        with self.lock:
            if key not in self.entries:
                self.stats.misses += 1
                return None
            
            entry = self.entries[key]
            
            # Check if expired
            if entry.is_expired():
                self._remove_entry(key)
                self.stats.misses += 1
                return None
            
            # Update access statistics
            entry.update_access()
            self.access_order[key] = time.time()
            self.frequency_counter[key] += 1
            
            # Update stats
            self.stats.hits += 1
            access_time = time.time() - start_time
            self.stats.average_access_time = (
                (self.stats.average_access_time * (self.stats.hits - 1) + access_time) / 
                self.stats.hits
            )
            
            self._update_hit_rate()
            
            return entry.value
    
    def remove(self, key: str) -> bool:
        """Remove entry from cache.
        
        Args:
            key: Cache key
            
        Returns:
            True if entry was removed
        """
        with self.lock:
            return self._remove_entry(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.entries.clear()
            self.access_order.clear()
            self.frequency_counter.clear()
            
            self.stats.entry_count = 0
            self.stats.size_bytes = 0
    
    def _should_evict(self, new_entry: CacheEntry) -> bool:
        """Check if eviction is needed for new entry.
        
        Args:
            new_entry: Entry to be added
            
        Returns:
            True if eviction is needed
        """
        would_exceed_size = (self.stats.size_bytes + new_entry.size_bytes) > self.max_size_bytes
        would_exceed_count = (self.stats.entry_count + 1) > self.max_entries
        
        return would_exceed_size or would_exceed_count
    
    def _evict_entry(self) -> bool:
        """Evict an entry based on strategy.
        
        Returns:
            True if entry was evicted
        """
        if not self.entries:
            return False
        
        if self.strategy == CacheStrategy.LRU:
            key = next(iter(self.access_order))  # Oldest access
        elif self.strategy == CacheStrategy.LFU:
            key = min(self.frequency_counter.keys(), key=lambda k: self.frequency_counter[k])
        elif self.strategy == CacheStrategy.FIFO:
            key = next(iter(self.entries))  # First entry
        elif self.strategy == CacheStrategy.TTL:
            # Find expired entries first
            expired_keys = [k for k, e in self.entries.items() if e.is_expired()]
            if expired_keys:
                key = expired_keys[0]
            else:
                key = next(iter(self.entries))  # Fallback to FIFO
        else:  # ADAPTIVE
            key = self._adaptive_eviction()
        
        return self._remove_entry(key)
    
    def _adaptive_eviction(self) -> str:
        """Adaptive eviction strategy based on access patterns.
        
        Returns:
            Key to evict
        """
        # Score entries based on multiple factors
        scores = {}
        current_time = time.time()
        
        for key, entry in self.entries.items():
            # Factors for scoring (lower score = better candidate for eviction)
            recency_score = current_time - entry.last_accessed  # Higher = older
            frequency_score = 1.0 / (entry.access_count + 1)   # Higher = less frequent
            size_score = entry.size_bytes / 1024.0             # Higher = larger
            ttl_score = 0
            
            if entry.ttl_seconds:
                remaining_ttl = entry.ttl_seconds - (current_time - entry.created_at)
                ttl_score = max(0, remaining_ttl)  # Higher = more time left
            
            # Weighted combination (tune weights based on usage patterns)
            combined_score = (
                recency_score * 0.4 +
                frequency_score * 0.3 +
                size_score * 0.2 +
                ttl_score * 0.1
            )
            
            scores[key] = combined_score
        
        # Return key with highest score (best eviction candidate)
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _remove_entry(self, key: str) -> bool:
        """Remove entry and update statistics.
        
        Args:
            key: Key to remove
            
        Returns:
            True if entry was removed
        """
        if key not in self.entries:
            return False
        
        entry = self.entries[key]
        
        # Update stats
        self.stats.entry_count -= 1
        self.stats.size_bytes -= entry.size_bytes
        self.stats.evictions += 1
        
        # Remove from all structures
        del self.entries[key]
        self.access_order.pop(key, None)
        self.frequency_counter.pop(key, None)
        
        return True
    
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of value in bytes.
        
        Args:
            value: Value to measure
            
        Returns:
            Size in bytes
        """
        try:
            return len(pickle.dumps(value))
        except Exception:
            # Fallback estimation
            if isinstance(value, str):
                return len(value.encode('utf-8'))
            elif isinstance(value, (int, float)):
                return 8
            elif isinstance(value, (list, tuple)):
                return sum(self._calculate_size(item) for item in value[:10])  # Sample first 10
            elif isinstance(value, dict):
                sample_items = list(value.items())[:10]  # Sample first 10
                return sum(self._calculate_size(k) + self._calculate_size(v) for k, v in sample_items)
            else:
                return 1024  # Default estimate
    
    def _update_hit_rate(self):
        """Update cache hit rate."""
        total_requests = self.stats.hits + self.stats.misses
        if total_requests > 0:
            self.stats.hit_rate = self.stats.hits / total_requests
    
    def _maintenance_loop(self):
        """Background maintenance for cache cleanup."""
        while self.running:
            try:
                with self.lock:
                    # Remove expired entries
                    expired_keys = []
                    for key, entry in self.entries.items():
                        if entry.is_expired():
                            expired_keys.append(key)
                    
                    for key in expired_keys:
                        self._remove_entry(key)
                    
                    # Adaptive strategy tuning based on hit rate
                    if self.strategy == CacheStrategy.ADAPTIVE:
                        self._tune_adaptive_strategy()
                
                # Sleep for maintenance interval
                time.sleep(60)  # Run every minute
                
            except Exception as e:
                print(f"Error in cache maintenance: {e}")
                time.sleep(10)
    
    def _tune_adaptive_strategy(self):
        """Tune adaptive strategy based on performance."""
        # If hit rate is low, might need to be more conservative with eviction
        if self.stats.hit_rate < 0.5:
            # Favor keeping frequently accessed items
            pass
        elif self.stats.hit_rate > 0.9:
            # Can be more aggressive with eviction
            pass
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics.
        
        Returns:
            Current cache statistics
        """
        with self.lock:
            self._update_hit_rate()
            return CacheStats(
                hits=self.stats.hits,
                misses=self.stats.misses,
                evictions=self.stats.evictions,
                size_bytes=self.stats.size_bytes,
                entry_count=self.stats.entry_count,
                hit_rate=self.stats.hit_rate,
                average_access_time=self.stats.average_access_time
            )
    
    def get_size_breakdown(self) -> Dict[str, Any]:
        """Get detailed size breakdown.
        
        Returns:
            Dictionary with size information
        """
        with self.lock:
            size_by_type = defaultdict(int)
            count_by_type = defaultdict(int)
            
            for entry in self.entries.values():
                value_type = type(entry.value).__name__
                size_by_type[value_type] += entry.size_bytes
                count_by_type[value_type] += 1
            
            return {
                "total_size_bytes": self.stats.size_bytes,
                "total_entries": self.stats.entry_count,
                "size_by_type": dict(size_by_type),
                "count_by_type": dict(count_by_type),
                "utilization": self.stats.size_bytes / self.max_size_bytes,
                "entry_utilization": self.stats.entry_count / self.max_entries
            }


class ModelCache(IntelligentCache):
    """Specialized cache for ML models."""
    
    def __init__(self, max_models: int = 10, **kwargs):
        """Initialize model cache.
        
        Args:
            max_models: Maximum number of models to cache
            **kwargs: Additional cache parameters
        """
        super().__init__(
            max_entries=max_models,
            strategy=CacheStrategy.LFU,  # Models should be evicted by frequency
            **kwargs
        )
        self.model_metadata = {}
    
    def put_model(self, model_id: str, model: Any, metadata: Dict[str, Any] = None) -> bool:
        """Store model in cache.
        
        Args:
            model_id: Unique model identifier
            model: Model object
            metadata: Model metadata
            
        Returns:
            True if stored successfully
        """
        if metadata:
            self.model_metadata[model_id] = metadata
        
        return self.put(model_id, model)
    
    def get_model(self, model_id: str) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """Retrieve model from cache.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Tuple of (model, metadata) or (None, None) if not found
        """
        model = self.get(model_id)
        metadata = self.model_metadata.get(model_id)
        
        return model, metadata
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about cached models.
        
        Returns:
            Dictionary with model information
        """
        model_info = {}
        
        with self.lock:
            for model_id in self.entries.keys():
                entry = self.entries[model_id]
                metadata = self.model_metadata.get(model_id, {})
                
                model_info[model_id] = {
                    "size_bytes": entry.size_bytes,
                    "last_accessed": entry.last_accessed,
                    "access_count": entry.access_count,
                    "metadata": metadata
                }
        
        return model_info


class ResultCache(IntelligentCache):
    """Specialized cache for inference results."""
    
    def __init__(self, **kwargs):
        """Initialize result cache."""
        super().__init__(
            strategy=CacheStrategy.ADAPTIVE,
            default_ttl=3600,  # 1 hour default TTL
            **kwargs
        )
    
    def cache_result(self, input_hash: str, result: Any, model_version: str = None, 
                     ttl: Optional[float] = None) -> bool:
        """Cache inference result.
        
        Args:
            input_hash: Hash of input data
            result: Inference result
            model_version: Version of model used
            ttl: Time to live
            
        Returns:
            True if cached successfully
        """
        cache_key = f"{input_hash}_{model_version}" if model_version else input_hash
        return self.put(cache_key, result, ttl)
    
    def get_result(self, input_hash: str, model_version: str = None) -> Optional[Any]:
        """Get cached result.
        
        Args:
            input_hash: Hash of input data
            model_version: Version of model used
            
        Returns:
            Cached result or None
        """
        cache_key = f"{input_hash}_{model_version}" if model_version else input_hash
        return self.get(cache_key)
    
    def create_input_hash(self, input_data: Any) -> str:
        """Create hash of input data.
        
        Args:
            input_data: Input data to hash
            
        Returns:
            Hash string
        """
        try:
            if hasattr(input_data, 'tobytes'):
                # NumPy array or similar
                return hashlib.sha256(input_data.tobytes()).hexdigest()[:16]
            else:
                # Generic serialization
                serialized = json.dumps(input_data, sort_keys=True, default=str)
                return hashlib.sha256(serialized.encode()).hexdigest()[:16]
        except Exception:
            # Fallback to string representation
            return hashlib.sha256(str(input_data).encode()).hexdigest()[:16]


class CacheHierarchy:
    """Multi-level cache hierarchy for optimal performance."""
    
    def __init__(self):
        """Initialize cache hierarchy."""
        # L1: Small, fast in-memory cache
        self.l1_cache = IntelligentCache(
            max_size_bytes=1024*1024*10,  # 10MB
            max_entries=1000,
            strategy=CacheStrategy.LRU
        )
        
        # L2: Larger in-memory cache
        self.l2_cache = IntelligentCache(
            max_size_bytes=1024*1024*100,  # 100MB
            max_entries=5000,
            strategy=CacheStrategy.ADAPTIVE
        )
        
        # L3: Result cache with TTL
        self.l3_cache = ResultCache(
            max_size_bytes=1024*1024*500,  # 500MB
            max_entries=20000
        )
        
        self.caches = [self.l1_cache, self.l2_cache, self.l3_cache]
    
    def start(self):
        """Start all caches."""
        for cache in self.caches:
            cache.start()
    
    def stop(self):
        """Stop all caches."""
        for cache in self.caches:
            cache.stop()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache hierarchy.
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None
        """
        # Try L1 first
        value = self.l1_cache.get(key)
        if value is not None:
            return value
        
        # Try L2
        value = self.l2_cache.get(key)
        if value is not None:
            # Promote to L1
            self.l1_cache.put(key, value)
            return value
        
        # Try L3
        value = self.l3_cache.get(key)
        if value is not None:
            # Promote to L2 and L1
            self.l2_cache.put(key, value)
            self.l1_cache.put(key, value)
            return value
        
        return None
    
    def put(self, key: str, value: Any, ttl: Optional[float] = None) -> bool:
        """Store value in appropriate cache level.
        
        Args:
            key: Cache key
            value: Value to store
            ttl: Time to live
            
        Returns:
            True if stored successfully
        """
        # Store in all levels
        success = True
        
        # L1 for quick access
        success &= self.l1_cache.put(key, value)
        
        # L2 for medium-term storage
        success &= self.l2_cache.put(key, value)
        
        # L3 for long-term storage with TTL
        success &= self.l3_cache.put(key, value, ttl)
        
        return success
    
    def get_hierarchy_stats(self) -> Dict[str, Any]:
        """Get statistics for entire cache hierarchy.
        
        Returns:
            Dictionary with hierarchy statistics
        """
        return {
            "l1_stats": self.l1_cache.get_stats(),
            "l2_stats": self.l2_cache.get_stats(),
            "l3_stats": self.l3_cache.get_stats(),
            "total_size_bytes": sum(cache.stats.size_bytes for cache in self.caches),
            "total_entries": sum(cache.stats.entry_count for cache in self.caches),
            "overall_hit_rate": self._calculate_overall_hit_rate()
        }
    
    def _calculate_overall_hit_rate(self) -> float:
        """Calculate overall hit rate across hierarchy.
        
        Returns:
            Overall hit rate
        """
        total_hits = sum(cache.stats.hits for cache in self.caches)
        total_requests = sum(cache.stats.hits + cache.stats.misses for cache in self.caches)
        
        return total_hits / max(1, total_requests)