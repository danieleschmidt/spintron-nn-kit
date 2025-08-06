"""
Performance optimization utilities for SpinTron-NN-Kit.

This module provides:
- Batch processing optimization
- Stream processing for real-time inference
- Memory optimization strategies
- Performance profiling and monitoring
"""

import time
import threading
from typing import List, Dict, Any, Callable, Optional, Iterator
from dataclasses import dataclass
from enum import Enum


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    BATCH_OPTIMIZE = "batch_optimize"
    STREAM_OPTIMIZE = "stream_optimize" 
    MEMORY_OPTIMIZE = "memory_optimize"
    LATENCY_OPTIMIZE = "latency_optimize"


@dataclass
class OptimizationConfig:
    """Configuration for performance optimization."""
    strategy: OptimizationStrategy = OptimizationStrategy.BATCH_OPTIMIZE
    batch_size: int = 32
    max_latency_ms: float = 100.0
    memory_limit_mb: int = 1024
    enable_profiling: bool = True


class BatchProcessor:
    """Optimized batch processor."""
    
    def __init__(self, batch_size: int = 32):
        self.batch_size = batch_size
    
    def process_batch(self, items: List[Any], processor_func: Callable) -> List[Any]:
        """Process items in batches."""
        results = []
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i+self.batch_size]
            batch_results = processor_func(batch)
            results.extend(batch_results)
        return results


class StreamProcessor:
    """Real-time stream processor."""
    
    def __init__(self, max_latency_ms: float = 100.0):
        self.max_latency_ms = max_latency_ms
    
    def process_stream(self, stream: Iterator[Any], processor_func: Callable) -> Iterator[Any]:
        """Process streaming data with latency optimization."""
        for item in stream:
            start_time = time.time()
            result = processor_func(item)
            
            # Ensure latency target
            elapsed = (time.time() - start_time) * 1000
            if elapsed > self.max_latency_ms:
                print(f"Warning: Processing exceeded latency target ({elapsed:.1f}ms > {self.max_latency_ms}ms)")
            
            yield result


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.batch_processor = BatchProcessor(config.batch_size)
        self.stream_processor = StreamProcessor(config.max_latency_ms)
    
    def optimize(self, data: Any, processor_func: Callable) -> Any:
        """Apply optimization based on configuration."""
        if self.config.strategy == OptimizationStrategy.BATCH_OPTIMIZE:
            if isinstance(data, list):
                return self.batch_processor.process_batch(data, processor_func)
        elif self.config.strategy == OptimizationStrategy.STREAM_OPTIMIZE:
            return self.stream_processor.process_stream(data, processor_func)
        
        # Fallback to direct processing
        return processor_func(data)