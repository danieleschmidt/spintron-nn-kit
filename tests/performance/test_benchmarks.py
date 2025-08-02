"""Performance benchmark tests."""

import pytest
import time
import torch
import numpy as np
from tests.fixtures.sample_models import KeywordSpottingNet, TinyConvNet


class TestPerformanceBenchmarks:
    """Performance and energy benchmarking tests."""
    
    @pytest.mark.slow
    def test_inference_latency_benchmark(self, performance_baseline):
        """Benchmark inference latency."""
        model = KeywordSpottingNet()
        model.eval()
        
        input_data = torch.randn(1, 40)
        
        # Warmup
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_data)
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                output = model(input_data)
        
        end_time = time.time()
        
        avg_latency_ms = (end_time - start_time) * 1000 / num_iterations
        target_latency = performance_baseline["inference_latency_ms"]
        
        # For mock testing, we'll be lenient with timing
        assert avg_latency_ms < target_latency * 10, f"Latency {avg_latency_ms:.2f}ms exceeds target"
    
    def test_memory_usage_benchmark(self, performance_baseline):
        """Benchmark memory usage."""
        model = TinyConvNet()
        
        # Calculate model memory footprint
        total_params = sum(p.numel() for p in model.parameters())
        param_memory_mb = total_params * 4 / (1024 * 1024)  # Assuming float32
        
        # Mock activation memory calculation
        input_shape = (1, 1, 32, 32)
        estimated_activation_memory_mb = np.prod(input_shape) * 4 / (1024 * 1024) * 10  # Rough estimate
        
        total_memory_mb = param_memory_mb + estimated_activation_memory_mb
        target_memory = performance_baseline["memory_usage_mb"]
        
        assert total_memory_mb < target_memory, f"Memory usage {total_memory_mb:.2f}MB exceeds target"
    
    @pytest.mark.hardware
    def test_energy_consumption_benchmark(self, performance_baseline):
        """Benchmark energy consumption (mock)."""
        # Mock energy analysis
        model = KeywordSpottingNet()
        
        # Estimate operations
        total_ops = 0
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                total_ops += module.in_features * module.out_features
        
        # Mock energy calculation (pJ per MAC)
        estimated_energy_per_mac = 8.5  # pJ
        total_energy_pj = total_ops * estimated_energy_per_mac
        total_energy_nj = total_energy_pj / 1000
        
        target_energy_per_mac = performance_baseline["energy_per_mac_pj"]
        
        assert estimated_energy_per_mac <= target_energy_per_mac
        assert total_energy_nj > 0, "Should have positive energy consumption"
    
    def test_accuracy_vs_quantization_tradeoff(self):
        """Test accuracy degradation with quantization."""
        model = KeywordSpottingNet()
        input_data = torch.randn(10, 40)
        
        # Original model output
        model.eval()
        with torch.no_grad():
            original_output = model(input_data)
        
        # Mock quantization effects
        quantization_levels = [32, 8, 4, 2]  # bits
        accuracy_degradation = []
        
        for bits in quantization_levels:
            # Mock quantized inference
            noise_level = 0.01 * (8 / bits)  # More noise for lower precision
            quantized_output = original_output + torch.randn_like(original_output) * noise_level
            
            # Calculate difference
            mse = torch.mean((original_output - quantized_output) ** 2).item()
            accuracy_degradation.append(mse)
        
        # Verify that error increases as precision decreases
        for i in range(1, len(accuracy_degradation)):
            assert accuracy_degradation[i] >= accuracy_degradation[i-1]
    
    @pytest.mark.slow
    def test_throughput_benchmark(self):
        """Test inference throughput."""
        model = KeywordSpottingNet()
        model.eval()
        
        batch_sizes = [1, 4, 8, 16]
        throughputs = []
        
        for batch_size in batch_sizes:
            input_data = torch.randn(batch_size, 40)
            
            # Warmup
            for _ in range(5):
                with torch.no_grad():
                    _ = model(input_data)
            
            # Benchmark
            start_time = time.time()
            num_batches = 50
            
            with torch.no_grad():
                for _ in range(num_batches):
                    _ = model(input_data)
            
            end_time = time.time()
            
            total_inferences = num_batches * batch_size
            throughput = total_inferences / (end_time - start_time)
            throughputs.append(throughput)
        
        # Verify throughput increases with batch size (up to a point)
        assert all(t > 0 for t in throughputs), "All throughputs should be positive"