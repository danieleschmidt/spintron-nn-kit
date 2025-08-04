#!/usr/bin/env python3
"""
Performance Benchmark Suite for SpinTron-NN-Kit.

This script validates that the framework meets performance requirements
and provides benchmarking capabilities.
"""

import time
import sys
import os
import numpy as np
import json
from pathlib import Path

# Add spintron_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test if we can import without PyTorch (basic structural test)
try:
    # Test basic imports
    from spintron_nn.core.mtj_models import MTJConfig, MTJDevice
    from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig
    from spintron_nn.utils.performance import PerformanceOptimizer, PerformanceConfig
    print("✓ Core imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def benchmark_mtj_device():
    """Benchmark MTJ device operations."""
    print("\n=== MTJ Device Benchmark ===")
    
    config = MTJConfig()
    device = MTJDevice(config)
    
    # Benchmark resistance reading
    start_time = time.time()
    for _ in range(1000):
        resistance = device.resistance
    read_time = time.time() - start_time
    
    # Benchmark switching
    start_time = time.time()
    switch_count = 0
    for _ in range(100):
        if device.switch(0.5):  # 0.5V switching
            switch_count += 1
    switch_time = time.time() - start_time
    
    results = {
        'resistance_reads_per_sec': 1000 / read_time,
        'switching_time_ms': switch_time * 10,  # 100 switches
        'switch_success_rate': switch_count / 100,
        'read_latency_us': read_time / 1000 * 1e6
    }
    
    print(f"  Resistance reads: {results['resistance_reads_per_sec']:.0f}/sec")
    print(f"  Switch latency: {results['switching_time_ms']:.2f} ms")
    print(f"  Switch success rate: {results['switch_success_rate']:.1%}")
    
    return results


def benchmark_crossbar():
    """Benchmark crossbar array operations."""
    print("\n=== Crossbar Array Benchmark ===")
    
    config = CrossbarConfig(rows=64, cols=64)
    crossbar = MTJCrossbar(config)
    
    # Generate test weights
    weights = np.random.randn(64, 64) * 0.5
    
    # Benchmark weight programming
    start_time = time.time()
    crossbar.set_weights(weights)
    program_time = time.time() - start_time
    
    # Benchmark vector-matrix multiplication
    input_voltages = np.random.randn(64) * 0.1
    
    start_time = time.time()
    for _ in range(100):
        output = crossbar.compute_vmm(input_voltages)
    vmm_time = time.time() - start_time
    
    results = {
        'programming_time_ms': program_time * 1000,
        'vmm_operations_per_sec': 100 / vmm_time,
        'vmm_latency_us': vmm_time / 100 * 1e6,
        'crossbar_size': f"{config.rows}x{config.cols}",
        'total_cells': config.rows * config.cols
    }
    
    print(f"  Programming time: {results['programming_time_ms']:.2f} ms")
    print(f"  VMM operations: {results['vmm_operations_per_sec']:.0f}/sec")
    print(f"  VMM latency: {results['vmm_latency_us']:.2f} μs")
    
    return results


def benchmark_performance_optimizer():
    """Benchmark performance optimization utilities."""
    print("\n=== Performance Optimizer Benchmark ===")
    
    config = PerformanceConfig(
        enable_result_caching=True,
        cache_size_mb=10,
        max_workers=2
    )
    optimizer = PerformanceOptimizer(config)
    
    # Test caching performance
    @optimizer.optimize_inference
    def dummy_inference(x):
        time.sleep(0.001)  # Simulate 1ms inference
        return x * 2
    
    # First call (cache miss)
    start_time = time.time()
    result1 = dummy_inference(np.array([1, 2, 3]))
    first_call_time = time.time() - start_time
    
    # Second call (cache hit)
    start_time = time.time()
    result2 = dummy_inference(np.array([1, 2, 3]))
    second_call_time = time.time() - start_time
    
    cache_speedup = first_call_time / second_call_time if second_call_time > 0 else float('inf')
    
    results = {
        'first_call_time_ms': first_call_time * 1000,
        'cached_call_time_ms': second_call_time * 1000,
        'cache_speedup': cache_speedup,
        'cache_stats': optimizer.cache.get_stats()
    }
    
    print(f"  First call: {results['first_call_time_ms']:.2f} ms")
    print(f"  Cached call: {results['cached_call_time_ms']:.4f} ms")
    print(f"  Cache speedup: {cache_speedup:.1f}x")
    
    optimizer.cleanup()
    return results


def benchmark_memory_usage():
    """Benchmark memory usage."""
    print("\n=== Memory Usage Benchmark ===")
    
    import psutil
    process = psutil.Process()
    
    # Baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create large crossbar arrays
    crossbars = []
    for _ in range(10):
        config = CrossbarConfig(rows=128, cols=128)
        crossbar = MTJCrossbar(config)
        weights = np.random.randn(128, 128)
        crossbar.set_weights(weights)
        crossbars.append(crossbar)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Cleanup
    del crossbars
    import gc
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    results = {
        'baseline_memory_mb': baseline_memory,
        'peak_memory_mb': peak_memory,
        'final_memory_mb': final_memory,
        'memory_increase_mb': peak_memory - baseline_memory,
        'memory_recovered_mb': peak_memory - final_memory
    }
    
    print(f"  Baseline memory: {baseline_memory:.1f} MB")
    print(f"  Peak memory: {peak_memory:.1f} MB") 
    print(f"  Memory increase: {results['memory_increase_mb']:.1f} MB")
    print(f"  Memory recovered: {results['memory_recovered_mb']:.1f} MB")
    
    return results


def validate_performance_requirements():
    """Validate that performance meets requirements."""
    print("\n=== Performance Requirements Validation ===")
    
    requirements = {
        'mtj_read_latency_us': {'max': 10.0, 'description': 'MTJ read latency'},
        'vmm_latency_us': {'max': 100.0, 'description': 'Vector-matrix multiply latency'},
        'cache_speedup': {'min': 5.0, 'description': 'Cache speedup factor'},
        'memory_efficiency': {'max': 50.0, 'description': 'Memory usage per crossbar (MB)'}
    }
    
    # Run benchmarks
    mtj_results = benchmark_mtj_device()
    crossbar_results = benchmark_crossbar()
    optimizer_results = benchmark_performance_optimizer()
    memory_results = benchmark_memory_usage()
    
    # Validation
    validation_results = {}
    
    # MTJ read latency
    actual_latency = mtj_results['read_latency_us']
    req = requirements['mtj_read_latency_us']
    validation_results['mtj_read_latency'] = {
        'actual': actual_latency,
        'requirement': req['max'],
        'passed': actual_latency <= req['max'],
        'description': req['description']
    }
    
    # VMM latency
    actual_vmm_latency = crossbar_results['vmm_latency_us']
    req = requirements['vmm_latency_us']
    validation_results['vmm_latency'] = {
        'actual': actual_vmm_latency,
        'requirement': req['max'],
        'passed': actual_vmm_latency <= req['max'],
        'description': req['description']
    }
    
    # Cache speedup
    actual_speedup = optimizer_results['cache_speedup']
    req = requirements['cache_speedup']
    validation_results['cache_speedup'] = {
        'actual': actual_speedup,
        'requirement': req['min'],
        'passed': actual_speedup >= req['min'],
        'description': req['description']
    }
    
    # Memory efficiency
    memory_per_crossbar = memory_results['memory_increase_mb'] / 10  # 10 crossbars
    req = requirements['memory_efficiency']
    validation_results['memory_efficiency'] = {
        'actual': memory_per_crossbar,
        'requirement': req['max'],
        'passed': memory_per_crossbar <= req['max'],
        'description': req['description']
    }
    
    # Print validation results
    print("\nValidation Results:")
    all_passed = True
    for test_name, result in validation_results.items():
        status = "✓ PASS" if result['passed'] else "✗ FAIL"
        print(f"  {status} {result['description']}: {result['actual']:.2f} (req: {result['requirement']})")
        if not result['passed']:
            all_passed = False
    
    return validation_results, all_passed


def main():
    """Run complete performance benchmark suite."""
    print("SpinTron-NN-Kit Performance Benchmark Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run validation
        validation_results, all_passed = validate_performance_requirements()
        
        total_time = time.time() - start_time
        
        # Summary
        print(f"\n=== Benchmark Summary ===")
        print(f"Total benchmark time: {total_time:.2f} seconds")
        print(f"All requirements met: {'Yes' if all_passed else 'No'}")
        
        # Save results
        results = {
            'timestamp': time.time(),
            'total_time_seconds': total_time,
            'validation_results': validation_results,
            'all_requirements_met': all_passed
        }
        
        output_file = Path(__file__).parent / 'benchmark_results.json'
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to: {output_file}")
        
        # Exit code
        sys.exit(0 if all_passed else 1)
        
    except Exception as e:
        print(f"\n✗ Benchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()