"""
Intelligent caching system for spintronic neural networks.
Implements adaptive caching strategies with predictive prefetching.
"""

import time
import math
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import json


class CacheStrategy(Enum):
    """Cache replacement strategies."""
    LRU = "lru"              # Least Recently Used
    LFU = "lfu"              # Least Frequently Used
    ARC = "arc"              # Adaptive Replacement Cache
    PREDICTIVE = "predictive" # ML-based predictive caching


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    access_count: int = 0
    last_access_time: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    size_bytes: int = 0
    access_pattern: List[float] = field(default_factory=list)
    prediction_score: float = 0.0


@dataclass
class CacheStatistics:
    """Cache performance statistics."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    total_size_bytes: int = 0
    average_access_time_ms: float = 0.0
    hit_rate: float = 0.0
    

class IntelligentCache:
    """Intelligent cache with adaptive strategies and predictive prefetching."""
    
    def __init__(self, 
                 max_size_mb: float = 100.0,
                 strategy: CacheStrategy = CacheStrategy.PREDICTIVE,
                 ttl_seconds: Optional[float] = None):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.strategy = strategy
        self.ttl_seconds = ttl_seconds
        
        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.frequency_tracker: Dict[str, int] = defaultdict(int)
        
        # ARC-specific structures
        self.arc_t1: OrderedDict[str, CacheEntry] = OrderedDict()  # Recent cache
        self.arc_t2: OrderedDict[str, CacheEntry] = OrderedDict()  # Frequent cache
        self.arc_b1: Set[str] = set()  # Recent evicted
        self.arc_b2: Set[str] = set()  # Frequent evicted
        self.arc_p = 0  # Target size for T1
        
        # Predictive caching
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.sequence_predictions: Dict[Tuple[str, ...], str] = {}
        self.recent_accesses: List[str] = []
        self.pattern_window_size = 10
        
        # Statistics
        self.stats = CacheStatistics()
        self.performance_history: List[Dict[str, float]] = []
        
        # Adaptive parameters
        self.learning_rate = 0.1
        self.prediction_threshold = 0.7
        self.adaptation_interval = 100  # requests
        
    def _calculate_size(self, value: Any) -> int:
        """Estimate size of cached value in bytes."""
        if isinstance(value, (str, bytes)):
            return len(value)
        elif isinstance(value, (int, float)):
            return 8
        elif isinstance(value, (list, tuple)):
            return sum(self._calculate_size(item) for item in value)
        elif isinstance(value, dict):
            return sum(self._calculate_size(k) + self._calculate_size(v) 
                      for k, v in value.items())
        else:
            # Rough estimate for complex objects
            return len(str(value)) * 2
            
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry is expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - entry.creation_time > self.ttl_seconds
        
    def _update_access_pattern(self, key: str) -> None:
        """Update access pattern for predictive caching."""
        current_time = time.time()
        self.access_patterns[key].append(current_time)
        
        # Keep only recent access times
        cutoff_time = current_time - 3600  # 1 hour
        self.access_patterns[key] = [t for t in self.access_patterns[key] 
                                   if t > cutoff_time]
        
        # Update sequence prediction
        self.recent_accesses.append(key)
        if len(self.recent_accesses) > self.pattern_window_size:
            self.recent_accesses.pop(0)
            
        # Learn sequence patterns
        if len(self.recent_accesses) >= 3:
            sequence = tuple(self.recent_accesses[-3:-1])
            next_key = self.recent_accesses[-1]
            self.sequence_predictions[sequence] = next_key
            
    def _predict_next_access(self) -> Optional[str]:
        """Predict next likely access using learned patterns."""
        if len(self.recent_accesses) < 2:
            return None
            
        # Sequence-based prediction
        last_sequence = tuple(self.recent_accesses[-2:])
        if last_sequence in self.sequence_predictions:
            return self.sequence_predictions[last_sequence]
            
        # Frequency-based prediction with time decay
        current_time = time.time()
        scores = {}
        
        for key, access_times in self.access_patterns.items():
            if not access_times:
                continue
                
            # Calculate frequency score with time decay
            frequency_score = len(access_times)
            
            # Calculate recency score
            last_access = max(access_times)
            recency_score = 1.0 / (1.0 + (current_time - last_access) / 60.0)  # 1-minute decay
            
            # Calculate periodicity score
            if len(access_times) >= 3:
                intervals = [access_times[i] - access_times[i-1] 
                           for i in range(1, len(access_times))]
                avg_interval = sum(intervals) / len(intervals)
                time_since_last = current_time - last_access
                periodicity_score = 1.0 / (1.0 + abs(time_since_last - avg_interval) / avg_interval)
            else:
                periodicity_score = 0.0
                
            # Combined score
            scores[key] = frequency_score * 0.4 + recency_score * 0.4 + periodicity_score * 0.2
            
        if scores:
            best_key = max(scores, key=scores.get)
            if scores[best_key] > self.prediction_threshold:
                return best_key
                
        return None
        
    def _evict_lru(self) -> Optional[str]:
        """Evict least recently used entry."""
        if not self.cache:
            return None
            
        # Find LRU entry
        lru_key = min(self.cache.keys(), 
                     key=lambda k: self.cache[k].last_access_time)
        
        evicted_entry = self.cache.pop(lru_key)
        self.stats.evictions += 1
        self.stats.total_size_bytes -= evicted_entry.size_bytes
        
        return lru_key
        
    def _evict_lfu(self) -> Optional[str]:
        """Evict least frequently used entry."""
        if not self.cache:
            return None
            
        # Find LFU entry
        lfu_key = min(self.cache.keys(),
                     key=lambda k: self.cache[k].access_count)
        
        evicted_entry = self.cache.pop(lfu_key)
        self.stats.evictions += 1
        self.stats.total_size_bytes -= evicted_entry.size_bytes
        
        return lfu_key
        
    def _evict_arc(self) -> Optional[str]:
        """Evict using Adaptive Replacement Cache algorithm."""
        # Simplified ARC implementation
        if len(self.arc_t1) >= max(1, self.arc_p):
            # Evict from T1
            if self.arc_t1:
                lru_key = next(iter(self.arc_t1))
                evicted_entry = self.arc_t1.pop(lru_key)
                self.arc_b1.add(lru_key)
                self.stats.evictions += 1
                self.stats.total_size_bytes -= evicted_entry.size_bytes
                return lru_key
        else:
            # Evict from T2
            if self.arc_t2:
                lru_key = next(iter(self.arc_t2))
                evicted_entry = self.arc_t2.pop(lru_key)
                self.arc_b2.add(lru_key)
                self.stats.evictions += 1
                self.stats.total_size_bytes -= evicted_entry.size_bytes
                return lru_key
                
        return None
        
    def _evict_predictive(self) -> Optional[str]:
        """Evict using predictive scoring."""
        if not self.cache:
            return None
            
        # Calculate eviction scores (lower = more likely to evict)
        current_time = time.time()
        scores = {}
        
        for key, entry in self.cache.items():
            # Base score components
            recency_score = 1.0 / (1.0 + (current_time - entry.last_access_time) / 60.0)
            frequency_score = math.log(1 + entry.access_count)
            
            # Prediction score - how likely this will be accessed again soon
            prediction_score = 0.0
            if key in self.access_patterns:
                access_times = self.access_patterns[key]
                if len(access_times) >= 2:
                    # Calculate access frequency
                    if len(access_times) > 1:
                        intervals = [access_times[i] - access_times[i-1] 
                                   for i in range(1, len(access_times))]
                        avg_interval = sum(intervals) / len(intervals)
                        time_since_last = current_time - entry.last_access_time
                        prediction_score = 1.0 / (1.0 + abs(time_since_last / avg_interval))
                        
            # Combined eviction score (higher = keep, lower = evict)
            scores[key] = recency_score * 0.3 + frequency_score * 0.4 + prediction_score * 0.3
            
        # Evict entry with lowest score
        evict_key = min(scores, key=scores.get)
        evicted_entry = self.cache.pop(evict_key)
        self.stats.evictions += 1
        self.stats.total_size_bytes -= evicted_entry.size_bytes
        
        return evict_key
        
    def _make_space(self, required_bytes: int) -> None:
        """Make space in cache by evicting entries."""
        while self.stats.total_size_bytes + required_bytes > self.max_size_bytes:
            evicted = None
            
            if self.strategy == CacheStrategy.LRU:
                evicted = self._evict_lru()
            elif self.strategy == CacheStrategy.LFU:
                evicted = self._evict_lfu()
            elif self.strategy == CacheStrategy.ARC:
                evicted = self._evict_arc()
            elif self.strategy == CacheStrategy.PREDICTIVE:
                evicted = self._evict_predictive()
                
            if evicted is None:
                break  # No more entries to evict
                
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        start_time = time.time()
        self.stats.total_requests += 1
        
        # Check if key exists and is not expired
        if key in self.cache:
            entry = self.cache[key]
            if not self._is_expired(entry):
                # Cache hit
                entry.access_count += 1
                entry.last_access_time = time.time()
                self.frequency_tracker[key] += 1
                
                # Move to end for LRU
                if self.strategy == CacheStrategy.LRU:
                    self.cache.move_to_end(key)
                    
                self.stats.cache_hits += 1
                self._update_access_pattern(key)
                
                # Update access time statistics
                access_time = (time.time() - start_time) * 1000
                self.stats.average_access_time_ms = (
                    (self.stats.average_access_time_ms * (self.stats.total_requests - 1) + access_time) /
                    self.stats.total_requests
                )
                
                return entry.value
            else:
                # Expired entry
                self._remove(key)
                
        # Cache miss
        self.stats.cache_misses += 1
        self._update_access_pattern(key)
        
        # Update hit rate
        self.stats.hit_rate = self.stats.cache_hits / self.stats.total_requests
        
        return None
        
    def put(self, key: str, value: Any) -> None:
        """Put value into cache."""
        size_bytes = self._calculate_size(value)
        
        # Check if value is too large for cache
        if size_bytes > self.max_size_bytes:
            return
            
        # Remove existing entry if present
        if key in self.cache:
            self._remove(key)
            
        # Make space if needed
        self._make_space(size_bytes)
        
        # Create new entry
        entry = CacheEntry(
            key=key,
            value=value,
            size_bytes=size_bytes,
            access_count=1,
            last_access_time=time.time(),
            creation_time=time.time()
        )
        
        # Add to appropriate cache structure based on strategy
        if self.strategy == CacheStrategy.ARC:
            # Simplified ARC insertion
            if key in self.arc_b1:
                # Adapt: increase p
                self.arc_p = min(self.arc_p + max(1, len(self.arc_b2) // len(self.arc_b1)), 
                               self.max_size_bytes // 2)
                self.arc_b1.remove(key)
                self.arc_t2[key] = entry
            elif key in self.arc_b2:
                # Adapt: decrease p
                self.arc_p = max(self.arc_p - max(1, len(self.arc_b1) // len(self.arc_b2)), 0)
                self.arc_b2.remove(key)
                self.arc_t2[key] = entry
            else:
                # New key, add to T1
                self.arc_t1[key] = entry
        else:
            self.cache[key] = entry
            
        self.stats.total_size_bytes += size_bytes
        self._update_access_pattern(key)
        
    def _remove(self, key: str) -> None:
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.stats.total_size_bytes -= entry.size_bytes
            
    def prefetch(self) -> List[str]:
        """Perform predictive prefetching."""
        if self.strategy != CacheStrategy.PREDICTIVE:
            return []
            
        predicted_keys = []
        
        # Predict next access
        next_key = self._predict_next_access()
        if next_key and next_key not in self.cache:
            predicted_keys.append(next_key)
            
        # Prefetch based on access patterns
        current_time = time.time()
        for key, access_times in self.access_patterns.items():
            if key in self.cache or len(access_times) < 3:
                continue
                
            # Check if this key shows periodic access pattern
            intervals = [access_times[i] - access_times[i-1] 
                        for i in range(1, len(access_times))]
            
            if len(intervals) >= 2:
                avg_interval = sum(intervals) / len(intervals)
                time_since_last = current_time - access_times[-1]
                
                # Predict if access is due soon
                if abs(time_since_last - avg_interval) < avg_interval * 0.2:
                    predicted_keys.append(key)
                    
        return predicted_keys[:5]  # Limit prefetch count
        
    def adapt_strategy(self) -> None:
        """Adapt caching strategy based on performance."""
        if self.stats.total_requests % self.adaptation_interval != 0:
            return
            
        current_hit_rate = self.stats.hit_rate
        
        # Record performance
        self.performance_history.append({
            'timestamp': time.time(),
            'hit_rate': current_hit_rate,
            'avg_access_time_ms': self.stats.average_access_time_ms,
            'strategy': self.strategy.value
        })
        
        # Adaptive strategy switching (simplified)
        if len(self.performance_history) >= 3:
            recent_performance = self.performance_history[-3:]
            avg_hit_rate = sum(p['hit_rate'] for p in recent_performance) / len(recent_performance)
            
            if avg_hit_rate < 0.5 and self.strategy != CacheStrategy.PREDICTIVE:
                # Switch to predictive caching if performance is poor
                self.strategy = CacheStrategy.PREDICTIVE
                print(f"Switched to predictive caching strategy (hit rate: {avg_hit_rate:.2%})")
                
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        return {
            'strategy': self.strategy.value,
            'total_requests': self.stats.total_requests,
            'cache_hits': self.stats.cache_hits,
            'cache_misses': self.stats.cache_misses,
            'hit_rate': self.stats.hit_rate,
            'evictions': self.stats.evictions,
            'total_size_mb': self.stats.total_size_bytes / (1024 * 1024),
            'max_size_mb': self.max_size_bytes / (1024 * 1024),
            'utilization': self.stats.total_size_bytes / self.max_size_bytes,
            'average_access_time_ms': self.stats.average_access_time_ms,
            'cached_entries': len(self.cache),
            'patterns_tracked': len(self.access_patterns),
            'sequence_patterns': len(self.sequence_predictions),
            'performance_history_length': len(self.performance_history)
        }
        
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.arc_t1.clear()
        self.arc_t2.clear()
        self.arc_b1.clear()
        self.arc_b2.clear()
        self.stats.total_size_bytes = 0
        self.stats.evictions = 0
        
    def export_cache_data(self) -> Dict[str, Any]:
        """Export cache data for analysis."""
        return {
            'statistics': self.get_cache_statistics(),
            'access_patterns': {k: v[-10:] for k, v in self.access_patterns.items()},  # Last 10 accesses
            'sequence_predictions': {str(k): v for k, v in self.sequence_predictions.items()},
            'frequency_tracker': dict(self.frequency_tracker),
            'performance_history': self.performance_history[-50:],  # Last 50 records
            'recent_accesses': self.recent_accesses[-20:]  # Last 20 accesses
        }


class MultiLevelCache:
    """Multi-level caching system with different strategies per level."""
    
    def __init__(self):
        self.l1_cache = IntelligentCache(max_size_mb=10, strategy=CacheStrategy.LRU)    # Fast, small
        self.l2_cache = IntelligentCache(max_size_mb=50, strategy=CacheStrategy.ARC)    # Medium
        self.l3_cache = IntelligentCache(max_size_mb=200, strategy=CacheStrategy.PREDICTIVE)  # Large, smart
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from multi-level cache."""
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
        
    def put(self, key: str, value: Any) -> None:
        """Put value into multi-level cache."""
        # Put in all levels
        self.l1_cache.put(key, value)
        self.l2_cache.put(key, value)
        self.l3_cache.put(key, value)
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics for all cache levels."""
        return {
            'l1_cache': self.l1_cache.get_cache_statistics(),
            'l2_cache': self.l2_cache.get_cache_statistics(),
            'l3_cache': self.l3_cache.get_cache_statistics()
        }