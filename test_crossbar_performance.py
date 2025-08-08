#!/usr/bin/env python3
"""
Simple crossbar performance test to measure current ops/sec.
"""

import time
import sys
from pathlib import Path

# Add spintron_nn to path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from spintron_nn.core.mtj_models import MTJConfig, MTJDevice
    from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig
    print("✓ Imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Create mock numpy if not available
try:
    import numpy as np
except ImportError:
    class MockNumpyArray:
        def __init__(self, data):
            if isinstance(data, (list, tuple)):
                self.data = data
                self.shape = (len(data),)
            else:
                self.data = [data]
                self.shape = (1,)
        
        def __getitem__(self, idx):
            return self.data[idx]
        
        def __len__(self):
            return len(self.data)
        
        def min(self):
            return min(self.data)
        
        def max(self):
            return max(self.data)
        
        def reshape(self, *shape):
            # Simple reshape for testing
            return self
        
        def sum(self, axis=None):
            if axis is None:
                return sum(self.data)
            return MockNumpyArray([sum(self.data)])
        
        def mean(self, axis=None):
            if axis is None:
                return sum(self.data) / len(self.data)
            return MockNumpyArray([sum(self.data) / len(self.data)])
        
        def __mul__(self, other):
            if hasattr(other, 'data'):
                return MockNumpyArray([a * b for a, b in zip(self.data, other.data)])
            else:
                return MockNumpyArray([x * other for x in self.data])
        
        def __rmul__(self, other):
            return self.__mul__(other)
        
        def __add__(self, other):
            if hasattr(other, 'data'):
                return MockNumpyArray([a + b for a, b in zip(self.data, other.data)])
            else:
                return MockNumpyArray([x + other for x in self.data])
        
        def __sub__(self, other):
            if hasattr(other, 'data'):
                return MockNumpyArray([a - b for a, b in zip(self.data, other.data)])
            else:
                return MockNumpyArray([x - other for x in self.data])
    
    class MockNumpy:
        def __init__(self):
            pass
        
        def array(self, data):
            return MockNumpyArray(data)
        
        def zeros(self, shape):
            if isinstance(shape, tuple):
                size = shape[0] * shape[1]
                data = [0.0] * size
                result = MockNumpyArray(data)
                result.shape = shape
                return result
            else:
                return MockNumpyArray([0.0] * shape)
        
        def random(self):
            import random
            class MockRandom:
                def normal(self, mean, std, size=None):
                    if size is None:
                        return random.gauss(mean, std)
                    elif isinstance(size, tuple):
                        data = [random.gauss(mean, std) for _ in range(size[0] * size[1])]
                        result = MockNumpyArray(data)
                        result.shape = size
                        return result
                    else:
                        return MockNumpyArray([random.gauss(mean, std) for _ in range(size)])
                
                def randn(self, *shape):
                    if len(shape) == 1:
                        return MockNumpyArray([random.gauss(0, 1) for _ in range(shape[0])])
                    else:
                        size = shape[0] * shape[1]
                        data = [random.gauss(0, 1) for _ in range(size)]
                        result = MockNumpyArray(data)
                        result.shape = shape
                        return result
                
                def random(self):
                    return random.random()
            
            return MockRandom()
        
        def dot(self, a, b):
            # Simple dot product for 2D arrays
            if hasattr(a, 'shape') and len(a.shape) == 2:
                # Matrix-vector multiply
                result_data = []
                rows, cols = a.shape
                for i in range(cols):
                    sum_val = 0.0
                    for j in range(rows):
                        a_idx = i * rows + j
                        sum_val += a.data[a_idx] * b.data[j]
                    result_data.append(sum_val)
                return MockNumpyArray(result_data)
            else:
                # Vector dot product
                return sum(av * bv for av, bv in zip(a.data, b.data))
        
        def abs(self, arr):
            return MockNumpyArray([abs(x) for x in arr.data])
        
        def clip(self, arr, min_val, max_val):
            return MockNumpyArray([max(min_val, min(max_val, x)) for x in arr.data])
    
    np = MockNumpy()
    print("Using mock numpy implementation")


def benchmark_crossbar_performance():
    """Test current crossbar VMM performance."""
    print("\n=== Testing Current Crossbar Performance ===")
    
    # Test with different sizes
    sizes = [32, 64, 128]
    
    for size in sizes:
        print(f"\nTesting {size}x{size} crossbar:")
        
        config = CrossbarConfig(rows=size, cols=size)
        crossbar = MTJCrossbar(config)
        
        # Generate test weights - simple pattern
        weights_data = []
        for i in range(size):
            row = []
            for j in range(size):
                row.append(0.1 * (i + j))
            weights_data.append(row)
        
        weights = MockNumpyArray(weights_data) if 'Mock' in str(type(np)) else np.array(weights_data)
        weights.shape = (size, size)
        
        # Program weights
        start_time = time.time()
        crossbar.set_weights(weights)
        program_time = time.time() - start_time
        
        # Generate test input
        input_data = [0.1 * i for i in range(size)]
        input_voltages = MockNumpyArray(input_data) if 'Mock' in str(type(np)) else np.array(input_data)
        
        # Benchmark VMM operations
        num_ops = 100
        start_time = time.time()
        for _ in range(num_ops):
            output = crossbar.compute_vmm(input_voltages)
        vmm_time = time.time() - start_time
        
        ops_per_sec = num_ops / vmm_time
        
        print(f"  Programming time: {program_time * 1000:.2f} ms")
        print(f"  VMM operations: {ops_per_sec:.0f}/sec")
        print(f"  VMM latency: {vmm_time / num_ops * 1e6:.2f} μs")
        
        if ops_per_sec < 1000:
            print(f"  ⚠️  Performance below target (1000 ops/sec)")
        else:
            print(f"  ✓ Performance meets target")


if __name__ == '__main__':
    benchmark_crossbar_performance()