"""
Performance Optimization Utilities for SpinTron-NN-Kit.

This module provides performance optimization tools including caching,
parallel processing, memory management, and auto-scaling capabilities.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import logging
import time
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import lru_cache, wraps
import psutil
import gc
from pathlib import Path
import pickle

from ..converter.pytorch_parser import SpintronicModel
from ..core.crossbar import MTJCrossbar


logger = logging.getLogger(__name__)


@dataclass
class PerformanceConfig:
    """Configuration for performance optimizations."""
    
    # Caching settings
    enable_result_caching: bool = True
    cache_size_mb: int = 100
    cache_ttl_seconds: int = 3600  # 1 hour
    
    # Parallel processing
    max_workers: int = None  # Auto-detect
    use_process_pool: bool = False  # Thread pool by default
    chunk_size: int = 10
    
    # Memory management
    enable_memory_optimization: bool = True
    gc_threshold: int = 1000  # Trigger GC after N operations
    memory_limit_gb: float = 4.0
    
    # Auto-scaling
    enable_auto_scaling: bool = False
    target_cpu_percent: float = 70.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 50.0
    
    # Profiling
    enable_profiling: bool = False
    profile_output_dir: str = "performance_profiles"


class ModelCache:
    """Intelligent caching system for model results and computations."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.cache = {}
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.Lock()
        
        # Calculate max cache entries based on memory limit
        avg_result_size = 1024  # Assume 1KB per result
        max_entries = int((config.cache_size_mb * 1024 * 1024) / avg_result_size)
        self.max_entries = max_entries
        
        logger.info(f"Initialized ModelCache with {max_entries} max entries")
    
    def get(self, key: str) -> Optional[Any]:
        """Retrieve cached result."""
        if not self.config.enable_result_caching:
            return None
        
        with self.lock:
            if key in self.cache:
                # Check TTL
                current_time = time.time()
                if current_time - self.access_times[key] < self.config.cache_ttl_seconds:
                    self.access_times[key] = current_time
                    self.hit_count += 1
                    return self.cache[key]
                else:
                    # Expired
                    del self.cache[key]
                    del self.access_times[key]
            
            self.miss_count += 1
            return None
    
    def put(self, key: str, value: Any):
        """Store result in cache."""
        if not self.config.enable_result_caching:
            return
        
        with self.lock:
            current_time = time.time()
            
            # Evict if at capacity
            if len(self.cache) >= self.max_entries:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = current_time
    
    def _evict_lru(self):
        """Evict least recently used entry."""
        if not self.cache:
            return
        
        # Find least recently used
        oldest_key = min(self.access_times.keys(), key=lambda k: self.access_times[k])
        del self.cache[oldest_key]
        del self.access_times[oldest_key]
    
    def clear(self):
        """Clear all cached entries."""
        with self.lock:
            self.cache.clear()
            self.access_times.clear()
            self.hit_count = 0
            self.miss_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.hit_count + self.miss_count
        hit_rate = self.hit_count / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_count': self.hit_count,
            'miss_count': self.miss_count,
            'hit_rate': hit_rate,
            'cache_size': len(self.cache),
            'max_entries': self.max_entries
        }


class PerformanceOptimizer:
    """Main performance optimization coordinator."""
    
    def __init__(self, config: Optional[PerformanceConfig] = None):
        self.config = config or PerformanceConfig()
        
        # Initialize components
        self.cache = ModelCache(self.config)
        self.memory_monitor = MemoryManager(self.config)
        self.profiler = PerformanceProfiler(self.config)
        
        # Parallel processing setup
        self.executor = None
        self._setup_executor()
        
        # Performance metrics
        self.metrics = {
            'operations_count': 0,
            'total_time': 0.0,
            'cache_hits': 0,
            'memory_optimizations': 0
        }
        
        logger.info("Performance optimizer initialized")
    
    def _setup_executor(self):
        """Setup parallel execution environment."""
        max_workers = self.config.max_workers
        if max_workers is None:
            max_workers = min(4, mp.cpu_count())
        
        if self.config.use_process_pool:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
            logger.info(f"Using ProcessPoolExecutor with {max_workers} workers")
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Using ThreadPoolExecutor with {max_workers} workers")
    
    def optimize_inference(self, func: Callable) -> Callable:
        """Decorator to optimize inference functions."""
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = self._generate_cache_key(func.__name__, args, kwargs)
            
            # Try cache first
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.metrics['cache_hits'] += 1
                return cached_result
            
            # Performance monitoring
            start_time = time.time()
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                self.memory_monitor.optimize_before_operation()
            
            # Execute function
            try:
                if self.config.enable_profiling:
                    with self.profiler.profile_context(func.__name__):
                        result = func(*args, **kwargs)
                else:
                    result = func(*args, **kwargs)
                
                # Cache result
                self.cache.put(cache_key, result)
                
                # Update metrics
                execution_time = time.time() - start_time
                self.metrics['operations_count'] += 1
                self.metrics['total_time'] += execution_time
                
                return result
                
            finally:
                # Memory cleanup
                if self.config.enable_memory_optimization:
                    self.memory_monitor.cleanup_after_operation()
        
        return wrapper
    
    def parallel_inference(
        self,
        model: Union[torch.nn.Module, SpintronicModel],
        inputs: List[torch.Tensor],
        batch_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Perform parallel inference on multiple inputs."""
        if batch_size is None:
            batch_size = self.config.chunk_size
        
        # Split inputs into batches
        input_batches = [
            inputs[i:i + batch_size] 
            for i in range(0, len(inputs), batch_size)
        ]
        
        # Parallel execution
        futures = []
        for batch in input_batches:
            future = self.executor.submit(self._process_batch, model, batch)
            futures.append(future)
        
        # Collect results
        results = []
        for future in futures:
            batch_results = future.result()
            results.extend(batch_results)
        
        return results
    
    def _process_batch(
        self,
        model: Union[torch.nn.Module, SpintronicModel],
        batch: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Process a batch of inputs."""
        results = []
        
        for input_tensor in batch:
            if isinstance(model, SpintronicModel):
                # Use spintronic forward method
                output = model.forward(input_tensor)
            else:
                # Standard PyTorch model
                with torch.no_grad():
                    output = model(input_tensor)
            
            results.append(output)
        
        return results
    
    def optimize_crossbar_operations(
        self,
        crossbars: List[MTJCrossbar],
        operations: List[Dict[str, Any]]
    ) -> List[Any]:
        """Optimize operations across multiple crossbar arrays."""
        # Group operations by crossbar
        crossbar_ops = {}
        for i, op in enumerate(operations):
            crossbar_id = op.get('crossbar_id', i % len(crossbars))
            if crossbar_id not in crossbar_ops:
                crossbar_ops[crossbar_id] = []
            crossbar_ops[crossbar_id].append(op)
        
        # Parallel execution across crossbars
        futures = []
        for crossbar_id, ops in crossbar_ops.items():
            crossbar = crossbars[crossbar_id]
            future = self.executor.submit(self._execute_crossbar_ops, crossbar, ops)
            futures.append(future)
        
        # Collect results
        all_results = []
        for future in futures:
            results = future.result()
            all_results.extend(results)
        
        return all_results
    
    def _execute_crossbar_ops(
        self,
        crossbar: MTJCrossbar,
        operations: List[Dict[str, Any]]
    ) -> List[Any]:
        """Execute operations on a single crossbar."""
        results = []
        
        for op in operations:
            op_type = op.get('type', 'compute')
            
            if op_type == 'compute':
                input_voltages = op['input_voltages']
                result = crossbar.compute_vmm(input_voltages)
            elif op_type == 'program':
                row, col, state = op['row'], op['col'], op['state']
                result = crossbar.write_cell(row, col, state)
            elif op_type == 'read':
                rows, cols = op['rows'], op['cols']
                result = crossbar.analog_read(rows, cols)
            else:
                result = None
            
            results.append(result)
        
        return results
    
    def _generate_cache_key(self, func_name: str, args: Tuple, kwargs: Dict) -> str:
        """Generate cache key for function call."""
        # Create hashable representation
        key_parts = [func_name]
        
        # Hash arguments
        for arg in args:
            if isinstance(arg, torch.Tensor):
                # Use tensor shape and a sample of values
                key_parts.append(f"tensor_{arg.shape}_{arg.sum().item():.6f}")
            elif isinstance(arg, (int, float, str)):
                key_parts.append(str(arg))
            else:
                key_parts.append(str(type(arg)))
        
        # Hash keyword arguments
        for k, v in sorted(kwargs.items()):
            if isinstance(v, torch.Tensor):
                key_parts.append(f"{k}_tensor_{v.shape}_{v.sum().item():.6f}")
            else:
                key_parts.append(f"{k}_{v}")
        
        return "_".join(key_parts)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        cache_stats = self.cache.get_stats()
        memory_stats = self.memory_monitor.get_stats()
        
        # Calculate derived metrics
        avg_operation_time = (
            self.metrics['total_time'] / self.metrics['operations_count']
            if self.metrics['operations_count'] > 0 else 0.0
        )
        
        operations_per_second = (
            self.metrics['operations_count'] / self.metrics['total_time']
            if self.metrics['total_time'] > 0 else 0.0
        )
        
        report = {
            'operations_count': self.metrics['operations_count'],
            'total_time_seconds': self.metrics['total_time'],
            'avg_operation_time_ms': avg_operation_time * 1000,
            'operations_per_second': operations_per_second,
            'cache_performance': cache_stats,
            'memory_performance': memory_stats,
            'profiling_enabled': self.config.enable_profiling
        }
        
        if self.config.enable_profiling:
            report['profiling_results'] = self.profiler.get_results()
        
        return report
    
    def cleanup(self):
        """Cleanup resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        self.cache.clear()
        self.memory_monitor.cleanup()
        
        logger.info("Performance optimizer cleanup completed")


class MemoryManager:
    """Memory optimization and management."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.operation_count = 0
        self.memory_stats = {
            'gc_collections': 0,
            'peak_memory_mb': 0.0,
            'memory_optimizations': 0
        }
    
    def optimize_before_operation(self):
        """Optimize memory before expensive operation."""
        if not self.config.enable_memory_optimization:
            return
        
        current_memory = self._get_memory_usage_mb()
        
        # Check memory limit
        if current_memory > self.config.memory_limit_gb * 1024:
            self._force_garbage_collection()
            self.memory_stats['memory_optimizations'] += 1
        
        # Update peak memory
        if current_memory > self.memory_stats['peak_memory_mb']:
            self.memory_stats['peak_memory_mb'] = current_memory
    
    def cleanup_after_operation(self):
        """Cleanup memory after operation."""
        self.operation_count += 1
        
        # Periodic garbage collection
        if self.operation_count % self.config.gc_threshold == 0:
            self._force_garbage_collection()
    
    def _force_garbage_collection(self):
        """Force garbage collection."""
        collected = gc.collect()
        self.memory_stats['gc_collections'] += 1
        
        # Clear PyTorch cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.debug(f"Garbage collection freed {collected} objects")
    
    def _get_memory_usage_mb(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory management statistics."""
        current_memory = self._get_memory_usage_mb()
        
        return {
            'current_memory_mb': current_memory,
            'peak_memory_mb': self.memory_stats['peak_memory_mb'],
            'gc_collections': self.memory_stats['gc_collections'],
            'memory_optimizations': self.memory_stats['memory_optimizations'],
            'memory_limit_gb': self.config.memory_limit_gb
        }
    
    def cleanup(self):
        """Final cleanup."""
        self._force_garbage_collection()


class PerformanceProfiler:
    """Performance profiling and analysis."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.profiles = {}
        self.current_profiles = {}
        
        if config.enable_profiling:
            self.output_dir = Path(config.profile_output_dir)
            self.output_dir.mkdir(exist_ok=True)
    
    def profile_context(self, operation_name: str):
        """Context manager for profiling operations."""
        return ProfileContext(self, operation_name)
    
    def start_profile(self, operation_name: str):
        """Start profiling an operation."""
        if not self.config.enable_profiling:
            return
        
        self.current_profiles[operation_name] = {
            'start_time': time.time(),
            'start_memory': self._get_memory_usage()
        }
    
    def end_profile(self, operation_name: str):
        """End profiling an operation."""
        if not self.config.enable_profiling or operation_name not in self.current_profiles:
            return
        
        profile_data = self.current_profiles[operation_name]
        end_time = time.time()
        end_memory = self._get_memory_usage()
        
        # Calculate metrics
        duration = end_time - profile_data['start_time']
        memory_delta = end_memory - profile_data['start_memory']
        
        # Store results
        if operation_name not in self.profiles:
            self.profiles[operation_name] = []
        
        self.profiles[operation_name].append({
            'duration_ms': duration * 1000,
            'memory_delta_mb': memory_delta,
            'timestamp': end_time
        })
        
        # Cleanup
        del self.current_profiles[operation_name]
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    def get_results(self) -> Dict[str, Any]:
        """Get profiling results summary."""
        results = {}
        
        for operation, profiles in self.profiles.items():
            if not profiles:
                continue
            
            durations = [p['duration_ms'] for p in profiles]
            memory_deltas = [p['memory_delta_mb'] for p in profiles]
            
            results[operation] = {
                'call_count': len(profiles),
                'avg_duration_ms': np.mean(durations),
                'min_duration_ms': np.min(durations),
                'max_duration_ms': np.max(durations),
                'std_duration_ms': np.std(durations),
                'avg_memory_delta_mb': np.mean(memory_deltas),
                'total_time_ms': np.sum(durations)
            }
        
        return results
    
    def save_results(self, filename: Optional[str] = None):
        """Save profiling results to file."""
        if not self.config.enable_profiling:
            return
        
        if filename is None:
            filename = f"profile_results_{int(time.time())}.json"
        
        output_file = self.output_dir / filename
        
        results = self.get_results()
        
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Profiling results saved to {output_file}")


class ProfileContext:
    """Context manager for profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str):
        self.profiler = profiler
        self.operation_name = operation_name
    
    def __enter__(self):
        self.profiler.start_profile(self.operation_name)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.profiler.end_profile(self.operation_name)


class AutoScaler:
    """Automatic scaling based on system load."""
    
    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.current_workers = config.max_workers or mp.cpu_count()
        self.min_workers = 1
        self.max_workers = mp.cpu_count() * 2
        
        self.scaling_history = []
        self.last_scale_time = 0
        self.scale_cooldown = 30  # 30 seconds between scaling actions
        
        logger.info(f"AutoScaler initialized with {self.current_workers} workers")
    
    def should_scale(self) -> Tuple[bool, str, int]:
        """Check if scaling is needed.""" 
        if not self.config.enable_auto_scaling:
            return False, "disabled", self.current_workers
        
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return False, "cooldown", self.current_workers
        
        # Get system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        # Scaling decisions
        if cpu_percent > self.config.scale_up_threshold and self.current_workers < self.max_workers:
            new_workers = min(self.current_workers + 1, self.max_workers)
            return True, "scale_up", new_workers
        
        elif cpu_percent < self.config.scale_down_threshold and self.current_workers > self.min_workers:
            new_workers = max(self.current_workers - 1, self.min_workers)
            return True, "scale_down", new_workers
        
        return False, "stable", self.current_workers
    
    def apply_scaling(self, new_worker_count: int, reason: str):
        """Apply scaling decision."""
        old_count = self.current_workers
        self.current_workers = new_worker_count
        self.last_scale_time = time.time()
        
        # Record scaling event
        self.scaling_history.append({
            'timestamp': self.last_scale_time,
            'action': reason,
            'old_workers': old_count,
            'new_workers': new_worker_count,
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent
        })
        
        logger.info(f"Scaled {reason}: {old_count} -> {new_worker_count} workers")
    
    def get_scaling_stats(self) -> Dict[str, Any]:
        """Get scaling statistics."""
        return {
            'current_workers': self.current_workers,
            'min_workers': self.min_workers,
            'max_workers': self.max_workers,
            'scaling_events': len(self.scaling_history),
            'last_scaling_history': self.scaling_history[-10:] if self.scaling_history else []
        }


# Convenience functions and decorators

def cached_inference(cache_size: int = 100):
    """Decorator for caching inference results."""
    cache = {}
    access_order = []
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key
            key = str(args) + str(sorted(kwargs.items()))
            
            if key in cache:
                # Move to end (most recently used)
                access_order.remove(key)
                access_order.append(key)
                return cache[key]
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            if len(cache) >= cache_size:
                # Remove least recently used
                lru_key = access_order.pop(0)
                del cache[lru_key]
            
            cache[key] = result
            access_order.append(key)
            
            return result
        
        wrapper.cache_clear = lambda: (cache.clear(), access_order.clear())
        wrapper.cache_info = lambda: {
            'size': len(cache), 
            'maxsize': cache_size
        }
        
        return wrapper
    return decorator


def parallel_batch_process(batch_size: int = 10, max_workers: int = None):
    """Decorator for parallel batch processing."""
    def decorator(func):
        @wraps(func)
        def wrapper(inputs: List[Any], *args, **kwargs):
            if not isinstance(inputs, list):
                inputs = [inputs]
            
            # Split into batches
            batches = [
                inputs[i:i + batch_size] 
                for i in range(0, len(inputs), batch_size)
            ]
            
            # Process in parallel
            workers = max_workers or min(4, mp.cpu_count())
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [
                    executor.submit(func, batch, *args, **kwargs)
                    for batch in batches
                ]
                
                results = []
                for future in futures:
                    batch_results = future.result()
                    if isinstance(batch_results, list):
                        results.extend(batch_results)
                    else:
                        results.append(batch_results)
            
            return results
        
        return wrapper
    return decorator


def memory_efficient(gc_frequency: int = 100):
    """Decorator for memory-efficient operations."""
    call_count = [0]  # Mutable container for closure
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            call_count[0] += 1
            
            try:
                result = func(*args, **kwargs)
                
                # Periodic garbage collection
                if call_count[0] % gc_frequency == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                return result
            
            except MemoryError:
                # Emergency cleanup
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Retry once
                return func(*args, **kwargs)
        
        return wrapper
    return decorator