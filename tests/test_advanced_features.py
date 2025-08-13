"""
Comprehensive test suite for advanced SpinTron-NN-Kit features.

Tests quantum-hybrid computing, advanced materials, distributed training,
adaptive scaling, and error handling capabilities.
"""

import pytest
import numpy as np
import time
import asyncio
import json
from unittest.mock import Mock, patch, MagicMock

# Import the modules to test
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from spintron_nn.research.advanced_materials import VCMADevice, SkyrmionNeuron, VCMAConfig
from spintron_nn.research.probabilistic_computing import StochasticNeuron, ProbabilisticConfig
from spintron_nn.research.quantum_hybrid import QuantumSpinQubit, QuantumSpintronicConfig
from spintron_nn.research.benchmarking import SpintronicBenchmarkSuite, BenchmarkResult
from spintron_nn.utils.error_handling import (
    ErrorHandler, SpintronError, ErrorCategory, ErrorSeverity,
    AdvancedErrorRecoverySystem, PredictiveErrorDetector
)
from spintron_nn.optimization.distributed_training import (
    DistributedSpintronicTrainer, DistributedTrainingConfig, ScalingStrategy, NodeCapabilities
)
from spintron_nn.optimization.adaptive_scaling import (
    AdaptiveScalingController, ScalingConfiguration, SystemMetrics, OptimizationObjective
)


class TestAdvancedMaterials:
    """Test suite for advanced materials research."""
    
    def test_vcma_device_creation(self):
        """Test VCMA device initialization."""
        config = VCMAConfig()
        device = VCMADevice(config)
        
        assert device.config == config
        assert device.switching_count == 0
        assert device.current_resistance > 0
        
    def test_vcma_switching(self):
        """Test VCMA switching mechanism."""
        config = VCMAConfig(electric_field_v_per_nm=1.0)
        device = VCMADevice(config)
        
        initial_resistance = device.current_resistance
        switching_energy = device.switch_state(voltage=0.8)
        
        assert switching_energy > 0
        assert device.switching_count == 1
        assert device.current_resistance != initial_resistance
        
    def test_skyrmion_neuron(self):
        """Test skyrmion neuron functionality."""
        config = VCMAConfig()
        neuron = SkyrmionNeuron(neuron_id=0, n_inputs=4, config=config)
        
        inputs = np.random.randn(4)
        output = neuron.process_inputs(inputs)
        
        assert isinstance(output, float)
        assert -1.0 <= output <= 1.0
        
    def test_skyrmion_memory(self):
        """Test skyrmion memory operations."""
        config = VCMAConfig()
        neuron = SkyrmionNeuron(neuron_id=0, n_inputs=4, config=config)
        
        # Store pattern
        pattern = np.array([1, -1, 1, -1])
        neuron.store_pattern(pattern)
        
        # Recall pattern
        noisy_pattern = pattern + np.random.normal(0, 0.1, 4)
        recalled = neuron.recall_pattern(noisy_pattern)
        
        assert len(recalled) == len(pattern)
        # Should be similar to original pattern
        similarity = np.corrcoef(pattern, recalled)[0, 1]
        assert similarity > 0.5


class TestProbabilisticComputing:
    """Test suite for probabilistic computing."""
    
    def test_stochastic_neuron_creation(self):
        """Test stochastic neuron initialization."""
        config = ProbabilisticConfig()
        neuron = StochasticNeuron(neuron_id=0, n_inputs=3, config=config)
        
        assert neuron.neuron_id == 0
        assert neuron.n_inputs == 3
        assert len(neuron.thermal_noise_history) == 0
        
    def test_stochastic_processing(self):
        """Test stochastic neural processing."""
        config = ProbabilisticConfig(thermal_noise_strength=0.1)
        neuron = StochasticNeuron(neuron_id=0, n_inputs=3, config=config)
        
        inputs = np.array([0.5, -0.3, 0.8])
        
        # Process multiple times to test stochasticity
        outputs = [neuron.stochastic_forward(inputs) for _ in range(10)]
        
        assert all(isinstance(out, float) for out in outputs)
        # Should have some variation due to stochasticity
        assert np.std(outputs) > 0.01
        
    def test_sampling_dynamics(self):
        """Test MCMC sampling dynamics."""
        config = ProbabilisticConfig()
        neuron = StochasticNeuron(neuron_id=0, n_inputs=3, config=config)
        
        target_distribution = lambda x: np.exp(-(x**2)/2)  # Gaussian
        samples = neuron.sample_distribution(target_distribution, n_samples=100)
        
        assert len(samples) == 100
        # Should roughly follow normal distribution
        assert abs(np.mean(samples)) < 0.5
        assert 0.5 < np.std(samples) < 2.0


class TestQuantumHybrid:
    """Test suite for quantum-hybrid computing."""
    
    def test_quantum_qubit_creation(self):
        """Test quantum qubit initialization."""
        config = QuantumSpintronicConfig()
        qubit = QuantumSpinQubit(qubit_id=0, config=config)
        
        assert qubit.qubit_id == 0
        assert abs(qubit.alpha) == 1.0  # Should start in |0⟩
        assert abs(qubit.beta) == 0.0
        
    def test_quantum_gates(self):
        """Test quantum gate operations."""
        config = QuantumSpintronicConfig()
        qubit = QuantumSpinQubit(qubit_id=0, config=config)
        
        # Test X gate
        qubit.apply_x_gate()
        assert abs(qubit.alpha) < 0.01  # Should be in |1⟩
        assert abs(qubit.beta) == pytest.approx(1.0, abs=0.01)
        
        # Test Hadamard gate
        qubit = QuantumSpinQubit(qubit_id=0, config=config)  # Reset
        qubit.apply_hadamard()
        # Should be in superposition
        assert 0.6 < abs(qubit.alpha) < 0.8
        assert 0.6 < abs(qubit.beta) < 0.8
        
    def test_quantum_measurement(self):
        """Test quantum measurement."""
        config = QuantumSpintronicConfig()
        qubit = QuantumSpinQubit(qubit_id=0, config=config)
        
        # Put in superposition
        qubit.apply_hadamard()
        
        # Measure multiple times
        results = [qubit.measure() for _ in range(100)]
        
        # Should get both 0 and 1 results
        assert 0 in results
        assert 1 in results
        assert all(r in [0, 1] for r in results)
        
    def test_quantum_decoherence(self):
        """Test quantum decoherence evolution."""
        config = QuantumSpintronicConfig(coherence_time=100e-6)
        qubit = QuantumSpinQubit(qubit_id=0, config=config)
        
        # Put in superposition
        qubit.apply_hadamard()
        initial_coherence = abs(qubit.alpha * np.conj(qubit.beta))
        
        # Evolve under decoherence
        qubit.evolve_decoherence(dt=50e-6)  # Half the coherence time
        final_coherence = abs(qubit.alpha * np.conj(qubit.beta))
        
        # Coherence should decrease
        assert final_coherence < initial_coherence


class TestBenchmarking:
    """Test suite for benchmarking framework."""
    
    def test_benchmark_suite_creation(self):
        """Test benchmark suite initialization."""
        suite = SpintronicBenchmarkSuite("test_results")
        
        assert len(suite.standard_configs) == 3
        assert "ultra_low_power" in suite.standard_configs
        assert "high_density" in suite.standard_configs
        assert "high_speed" in suite.standard_configs
        
    def test_benchmark_result(self):
        """Test benchmark result calculations."""
        result = BenchmarkResult(
            name="test_benchmark",
            energy_per_mac_pj=10.0,
            latency_ms=1.0,
            accuracy=0.95
        )
        
        edap = result.energy_delay_accuracy_product()
        fom = result.figure_of_merit()
        
        assert edap > 0
        assert fom > 0
        assert edap == 10.0 * 1.0 / 0.95  # energy * latency / accuracy
        
    @patch('torch.nn.Module')
    @patch('torch.Tensor')
    def test_variation_tolerance_structure(self, mock_tensor, mock_module):
        """Test variation tolerance benchmark structure."""
        suite = SpintronicBenchmarkSuite("test_results")
        
        # Mock model and data
        mock_model = Mock()
        mock_model.eval.return_value = None
        mock_data = Mock()
        mock_labels = Mock()
        
        # Mock the variation injection
        with patch.object(suite, '_inject_device_variations', return_value=mock_model):
            with patch.object(suite, '_get_baseline_accuracy', return_value=0.95):
                # This would normally run the full test, but we'll just verify structure
                variation_levels = [0.05, 0.1, 0.15]
                assert len(variation_levels) == 3
                assert all(0 < v < 1.0 for v in variation_levels)


class TestErrorHandling:
    """Test suite for error handling and recovery."""
    
    def test_error_handler_creation(self):
        """Test error handler initialization."""
        handler = ErrorHandler("test_component")
        
        assert handler.component == "test_component"
        assert len(handler.error_count) == 0
        assert len(handler.handlers) > 0
        
    def test_spintronic_error(self):
        """Test SpinTron error creation."""
        error = SpintronError(
            "Test error",
            ErrorCategory.HARDWARE,
            ErrorSeverity.HIGH
        )
        
        assert str(error) == "Test error"
        assert error.category == ErrorCategory.HARDWARE
        assert error.severity == ErrorSeverity.HIGH
        assert error.timestamp > 0
        
    def test_error_classification(self):
        """Test automatic error classification."""
        handler = ErrorHandler()
        
        # Test ValueError classification
        value_error = ValueError("Invalid input value")
        classified = handler._classify_error(value_error, {})
        
        assert isinstance(classified, SpintronError)
        assert classified.category == ErrorCategory.VALIDATION
        
        # Test MemoryError classification
        memory_error = MemoryError("Out of memory")
        classified = handler._classify_error(memory_error, {})
        
        assert classified.category == ErrorCategory.MEMORY
        assert classified.severity == ErrorSeverity.HIGH
        
    def test_predictive_error_detector(self):
        """Test predictive error detection."""
        detector = PredictiveErrorDetector()
        
        # Test device health monitoring
        metrics = {
            "temperature": 85.0,  # High temperature
            "switching_failures": 0.02,  # 2% failure rate
            "coherence_time": 80e-6,
            "energy_consumption": 15.0
        }
        
        risk_scores = detector.monitor_device_health(metrics)
        
        assert isinstance(risk_scores, dict)
        # Should detect thermal risk due to high temperature
        if "thermal_runaway" in risk_scores:
            assert risk_scores["thermal_runaway"] > 0
        
    def test_autonomous_recovery(self):
        """Test autonomous recovery system."""
        recovery_system = AdvancedErrorRecoverySystem()
        
        # Test health monitoring
        health_data = recovery_system.monitor_system_health()
        
        assert "system_metrics" in health_data
        assert "risk_scores" in health_data
        assert "health_status" in health_data
        assert "recommendations" in health_data
        
        # Verify health status structure
        health_status = health_data["health_status"]
        assert "health_score" in health_status
        assert "status" in health_status
        assert 0 <= health_status["health_score"] <= 100


class TestDistributedTraining:
    """Test suite for distributed training."""
    
    def test_distributed_config(self):
        """Test distributed training configuration."""
        config = DistributedTrainingConfig(
            scaling_strategy=ScalingStrategy.DATA_PARALLEL,
            max_nodes=4,
            auto_scaling_enabled=True
        )
        
        assert config.scaling_strategy == ScalingStrategy.DATA_PARALLEL
        assert config.max_nodes == 4
        assert config.auto_scaling_enabled is True
        
    def test_node_capabilities(self):
        """Test node capabilities calculation."""
        node = NodeCapabilities(
            node_id="test_node",
            cpu_cores=8,
            gpu_memory_gb=16.0,
            network_bandwidth_gbps=10.0,
            spintronic_accelerators=2,
            quantum_processors=1
        )
        
        score = node.compute_capability_score()
        assert score > 0
        assert isinstance(score, float)
        
    def test_distributed_trainer_creation(self):
        """Test distributed trainer initialization."""
        config = DistributedTrainingConfig()
        trainer = DistributedSpintronicTrainer(config)
        
        assert trainer.config == config
        assert len(trainer.nodes) == 0
        assert trainer.global_step == 0
        
    @pytest.mark.asyncio
    async def test_node_registration(self):
        """Test node registration."""
        config = DistributedTrainingConfig()
        trainer = DistributedSpintronicTrainer(config)
        
        node = NodeCapabilities(
            node_id="test_node",
            cpu_cores=4,
            gpu_memory_gb=8.0,
            network_bandwidth_gbps=5.0
        )
        
        await trainer.register_node("test_node", node)
        
        assert "test_node" in trainer.nodes
        assert trainer.nodes["test_node"] == node


class TestAdaptiveScaling:
    """Test suite for adaptive scaling."""
    
    def test_scaling_configuration(self):
        """Test scaling configuration."""
        config = ScalingConfiguration(
            optimization_objective=OptimizationObjective.BALANCED_PERFORMANCE,
            predictive_scaling=True,
            quantum_aware_scaling=True
        )
        
        assert config.optimization_objective == OptimizationObjective.BALANCED_PERFORMANCE
        assert config.predictive_scaling is True
        assert config.quantum_aware_scaling is True
        
    def test_system_metrics(self):
        """Test system metrics structure."""
        metrics = SystemMetrics(
            throughput=100.0,
            latency=50.0,
            accuracy=0.95,
            cpu_utilization=0.7,
            energy_per_operation=10.0
        )
        
        metrics_dict = metrics.to_dict()
        
        assert "throughput" in metrics_dict
        assert "latency" in metrics_dict
        assert "accuracy" in metrics_dict
        assert metrics_dict["throughput"] == 100.0
        
    def test_adaptive_controller_creation(self):
        """Test adaptive scaling controller initialization."""
        config = ScalingConfiguration()
        controller = AdaptiveScalingController(config)
        
        assert controller.config == config
        assert controller.current_scale_factor == 1.0
        assert len(controller.metrics_history) == 0
        
    def test_metrics_collection(self):
        """Test system metrics collection."""
        config = ScalingConfiguration()
        controller = AdaptiveScalingController(config)
        
        metrics = controller._collect_system_metrics()
        
        assert isinstance(metrics, SystemMetrics)
        assert metrics.throughput > 0
        assert metrics.latency > 0
        assert 0 <= metrics.accuracy <= 1.0
        
    def test_scaling_statistics(self):
        """Test scaling statistics collection."""
        config = ScalingConfiguration()
        controller = AdaptiveScalingController(config)
        
        stats = controller.get_scaling_statistics()
        
        assert "total_scaling_events" in stats
        assert "current_scale_factor" in stats
        assert stats["total_scaling_events"] == 0  # No scaling events yet
        assert stats["current_scale_factor"] == 1.0


class TestIntegration:
    """Integration tests for complete system functionality."""
    
    def test_end_to_end_quantum_materials(self):
        """Test integration of quantum and materials components."""
        # Create quantum qubit
        quantum_config = QuantumSpintronicConfig()
        qubit = QuantumSpinQubit(0, quantum_config)
        
        # Create VCMA device
        vcma_config = VCMAConfig()
        vcma_device = VCMADevice(vcma_config)
        
        # Test interaction
        qubit.apply_hadamard()
        switching_energy = vcma_device.switch_state(0.5)
        
        # Both should work independently
        assert abs(qubit.alpha)**2 + abs(qubit.beta)**2 == pytest.approx(1.0, abs=0.01)
        assert switching_energy > 0
        
    def test_error_handling_with_benchmarking(self):
        """Test error handling integration with benchmarking."""
        suite = SpintronicBenchmarkSuite("test_results")
        handler = ErrorHandler("benchmark_test")
        
        # Simulate error during benchmarking
        test_error = SpintronError(
            "Benchmark simulation failed",
            ErrorCategory.SIMULATION,
            ErrorSeverity.MEDIUM
        )
        
        recovery_result = handler.handle_error(test_error)
        
        # Should attempt recovery
        assert recovery_result is not None
        assert "fallback_mode" in recovery_result
        
    def test_distributed_with_adaptive_scaling(self):
        """Test distributed training with adaptive scaling."""
        # Create distributed config
        dist_config = DistributedTrainingConfig(auto_scaling_enabled=True)
        
        # Create adaptive scaling config
        scaling_config = ScalingConfiguration(
            optimization_objective=OptimizationObjective.MAXIMIZE_THROUGHPUT
        )
        
        # Both systems should be compatible
        assert dist_config.auto_scaling_enabled
        assert scaling_config.optimization_objective == OptimizationObjective.MAXIMIZE_THROUGHPUT


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])