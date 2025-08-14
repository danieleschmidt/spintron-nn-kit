"""
Comprehensive Integration Tests for Spintronic Neural Network Framework.

This test suite validates the complete end-to-end functionality of the
spintronic neural network framework across all major components.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
import asyncio
import tempfile
import json
from pathlib import Path
from unittest.mock import Mock, patch
import time

from spintron_nn.core.mtj_models import MTJConfig, MTJDevice
from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig
from spintron_nn.research.algorithms import PhysicsInformedQuantization, StochasticDeviceModeling
from spintron_nn.research.validation import StatisticalValidator, ExperimentConfig
from spintron_nn.research.autonomous_optimization import AutonomousOptimizer, OptimizationConfig
from spintron_nn.reliability.fault_tolerance import FaultTolerantCrossbar, RedundancyType, FaultModel, FaultType
from spintron_nn.security.secure_computing import SecureCrossbar, SecurityConfig, SecurityLevel
from spintron_nn.scaling.quantum_acceleration import HybridQuantumClassicalProcessor, QuantumResource
from spintron_nn.scaling.cloud_orchestration import CloudOrchestrator, CloudProvider, ScalingPolicy


class TestFrameworkIntegration:
    """Test complete framework integration."""
    
    @pytest.fixture
    def sample_model(self):
        """Create sample neural network model."""
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )
        return model
    
    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        x = torch.randn(100, 10)
        y = torch.randn(100, 1)
        return [(x[i:i+10], y[i:i+10]) for i in range(0, 100, 10)]
    
    @pytest.fixture
    def mtj_config(self):
        """Create MTJ configuration."""
        return MTJConfig(
            resistance_high=10e3,
            resistance_low=5e3,
            switching_voltage=0.3,
            cell_area=40e-9
        )
    
    @pytest.fixture
    def crossbar_config(self, mtj_config):
        """Create crossbar configuration."""
        return CrossbarConfig(
            rows=32,
            cols=32,
            mtj_config=mtj_config
        )
    
    def test_end_to_end_training_and_inference(self, sample_model, sample_dataset, crossbar_config):
        """Test complete training and inference pipeline."""
        
        # Create crossbar
        crossbar = MTJCrossbar(crossbar_config)
        
        # Extract weights from model
        weights = sample_model[0].weight.data.numpy()[:32, :32]  # Match crossbar size
        
        # Program crossbar
        conductances = crossbar.set_weights(weights)
        assert conductances.shape == (32, 32)
        assert np.all(conductances > 0)
        
        # Test inference
        input_voltages = np.random.uniform(-0.5, 0.5, 32)
        output_currents = crossbar.compute_vmm(input_voltages)
        
        assert len(output_currents) == 32
        assert np.all(np.isfinite(output_currents))
        
        # Verify output is reasonable
        expected_output = np.dot(weights.T, input_voltages)
        correlation = np.corrcoef(output_currents, expected_output)[0, 1]
        assert correlation > 0.5, f"Low correlation: {correlation}"
    
    def test_physics_informed_quantization_integration(self, mtj_config):
        """Test physics-informed quantization with real device models."""
        
        # Create quantization algorithm
        quantizer = PhysicsInformedQuantization(mtj_config)
        
        # Test quantization
        weights = torch.randn(16, 16)
        result = quantizer.quantize_layer(weights, target_bits=4)
        
        assert result.quantized_weights.shape == weights.shape
        assert result.energy_cost > 0
        assert 0 <= result.accuracy_loss <= 1
        assert len(result.optimization_history) > 0
        
        # Verify quantized weights are within expected range
        assert torch.all(torch.abs(result.quantized_weights) <= 2.0)
    
    def test_stochastic_device_modeling_integration(self, mtj_config):
        """Test stochastic device modeling with crossbar simulation."""
        
        # Create device modeling
        device_model = StochasticDeviceModeling(mtj_config)
        
        # Generate device array
        array_shape = (8, 8)
        device_params = device_model.generate_device_array(array_shape)
        
        assert 'resistance_high' in device_params
        assert 'resistance_low' in device_params
        assert device_params['resistance_high'].shape == array_shape
        
        # Simulate device dynamics
        input_voltages = torch.randn(8)
        simulation_results = device_model.simulate_device_dynamics(
            device_params, input_voltages, time_steps=100
        )
        
        assert 'current_history' in simulation_results
        assert 'energy_history' in simulation_results
        assert simulation_results['total_energy'] > 0
    
    def test_statistical_validation_integration(self):
        """Test statistical validation framework."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = StatisticalValidator(temp_dir)
            
            # Test experiment design
            design = validator.design_experiment(effect_size=0.5, power=0.8)
            assert design['recommended_sample_size'] > 0
            assert design['actual_power'] >= 0.8
            
            # Test validation
            group1 = np.random.normal(0, 1, 50)
            group2 = np.random.normal(0.5, 1, 50)
            
            exp_config = ExperimentConfig(
                experiment_name="test_validation",
                description="Test statistical validation",
                random_seed=42,
                sample_size=50
            )
            
            results = validator.validate_experiment_results(group1, group2, exp_config)
            assert len(results) > 0
            assert any(r.test_name.startswith('Independent t-test') or 
                      r.test_name.startswith('Welch') or 
                      r.test_name.startswith('Mann-Whitney') for r in results)
    
    def test_autonomous_optimization_integration(self, sample_model, sample_dataset):
        """Test autonomous optimization system."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            config = OptimizationConfig(
                max_iterations=10,  # Reduced for testing
                population_size=5
            )
            
            optimizer = AutonomousOptimizer(config, temp_dir)
            
            # Mock the optimization to avoid long runtime
            with patch.object(optimizer, '_evaluate_configuration', return_value=0.5):
                result = optimizer.optimize_system(sample_model, sample_dataset)
            
            assert 'best_parameters' in result.best_parameters
            assert result.total_time > 0
            assert len(result.optimization_history) > 0
    
    def test_fault_tolerance_integration(self, crossbar_config):
        """Test fault tolerance mechanisms."""
        
        # Create fault-tolerant crossbar
        ft_crossbar = FaultTolerantCrossbar(
            crossbar_config,
            RedundancyType.TMR,
            fault_tolerance_level=0.99
        )
        
        # Set weights
        weights = np.random.uniform(-1, 1, (32, 32))
        conductances = ft_crossbar.set_weights(weights)
        assert conductances.shape == (32, 32)
        
        # Test fault-tolerant computation
        input_voltages = np.random.uniform(-0.5, 0.5, 32)
        output = ft_crossbar.compute_vmm_fault_tolerant(input_voltages)
        assert len(output) == 32
        assert np.all(np.isfinite(output))
        
        # Inject fault and test recovery
        fault_model = FaultModel(
            fault_type=FaultType.STUCK_AT_ZERO,
            probability=1.0,
            affected_devices=[(0, 0), (1, 1)]
        )
        
        ft_crossbar.inject_fault(fault_model)
        
        # Should still work with faults
        output_with_fault = ft_crossbar.compute_vmm_fault_tolerant(input_voltages)
        assert len(output_with_fault) == 32
        assert np.all(np.isfinite(output_with_fault))
        
        # Analyze reliability
        reliability_metrics = ft_crossbar.analyze_reliability()
        assert reliability_metrics.mttf_years > 0
        assert 0 <= reliability_metrics.fault_coverage <= 1
    
    def test_security_framework_integration(self, crossbar_config):
        """Test security framework integration."""
        
        security_config = SecurityConfig(
            security_level=SecurityLevel.HIGH,
            enable_differential_privacy=True,
            dp_epsilon=1.0
        )
        
        # Create secure crossbar
        secure_crossbar = SecureCrossbar(crossbar_config, security_config)
        
        # Authenticate user
        session_id = secure_crossbar.authenticate_user("test_user", "secure_password")
        assert session_id is not None
        
        # Test secure computation
        input_voltages = np.random.uniform(-0.5, 0.5, 32)
        output = secure_crossbar.secure_compute_vmm(input_voltages, session_id)
        
        assert len(output) == 32
        assert np.all(np.isfinite(output))
        
        # Check privacy budget
        budget_status = secure_crossbar.get_privacy_budget_status()
        assert budget_status['spent_epsilon'] > 0
        assert budget_status['remaining_epsilon'] >= 0
    
    @pytest.mark.asyncio
    async def test_quantum_acceleration_integration(self):
        """Test quantum acceleration framework."""
        
        # Create quantum resources
        quantum_resources = [
            QuantumResource(
                qubits=20,
                gate_fidelity=0.99,
                coherence_time_ms=100,
                connectivity_graph={i: [j for j in range(20) if j != i] for i in range(20)},
                quantum_volume=64
            )
        ]
        
        # Create hybrid processor
        processor = HybridQuantumClassicalProcessor(quantum_resources, classical_cores=4)
        
        # Test linear system solving
        matrix_A = np.random.random((8, 8)) + np.eye(8) * 10  # Well-conditioned
        vector_b = np.random.random(8)
        
        result = await processor.compute_optimal(
            'linear_solver',
            matrix_A,
            matrix_A=matrix_A,
            vector_b=vector_b
        )
        
        assert len(result) == 8
        assert np.all(np.isfinite(result))
        
        # Verify solution quality
        residual = np.linalg.norm(matrix_A @ result - vector_b)
        assert residual < 1.0, f"Poor solution quality: residual = {residual}"
        
        # Check performance statistics
        stats = processor.get_performance_statistics()
        assert stats['total_executions'] > 0
        
        processor.shutdown()
    
    @pytest.mark.asyncio
    async def test_cloud_orchestration_integration(self):
        """Test cloud orchestration system."""
        
        # Create orchestrator
        orchestrator = CloudOrchestrator(
            supported_providers=[CloudProvider.AWS, CloudProvider.GCP],
            scaling_policy=ScalingPolicy.REACTIVE,
            cost_budget_per_hour=50.0
        )
        
        await orchestrator.initialize()
        
        # Test workload deployment
        workload_config = {
            'name': 'test_workload',
            'type': 'spintronic_inference'
        }
        
        performance_requirements = {
            'min_compute_units': 4.0,
            'max_latency_ms': 100.0,
            'min_availability': 0.99
        }
        
        cost_constraints = {
            'max_cost_per_hour': 20.0,
            'prefer_spot_instances': True
        }
        
        deployment_result = await orchestrator.deploy_workload(
            workload_config,
            performance_requirements,
            cost_constraints
        )
        
        assert deployment_result['status'] == 'success'
        assert deployment_result['allocated_resources'] > 0
        assert deployment_result['estimated_cost_per_hour'] <= 20.0
        
        # Test auto-scaling
        scaling_decision = await orchestrator.auto_scale()
        assert scaling_decision.action in ['scale_up', 'scale_down', 'no_change']
        assert scaling_decision.target_instances >= 0
        
        # Check orchestration status
        status = orchestrator.get_orchestration_status()
        assert status['active_resources'] >= 0
        assert status['cost_budget_utilization'] >= 0
        
        await orchestrator.shutdown()
    
    def test_cross_component_compatibility(self, mtj_config, crossbar_config):
        """Test compatibility between different framework components."""
        
        # Create components
        crossbar = MTJCrossbar(crossbar_config)
        quantizer = PhysicsInformedQuantization(mtj_config)
        device_model = StochasticDeviceModeling(mtj_config)
        
        # Test data flow between components
        weights = torch.randn(16, 16)
        
        # Quantize weights
        quant_result = quantizer.quantize_layer(weights[:16, :16], target_bits=4)
        quantized_weights = quant_result.quantized_weights.numpy()
        
        # Use quantized weights in crossbar
        crossbar_weights = np.pad(quantized_weights, ((0, 16), (0, 16)), mode='constant')
        conductances = crossbar.set_weights(crossbar_weights)
        
        # Generate device parameters
        device_params = device_model.generate_device_array((32, 32))
        
        # Verify compatibility
        assert conductances.shape == device_params['resistance_high'].shape
        assert np.all(conductances > 0)
        assert np.all(device_params['resistance_high'] > 0)
    
    def test_performance_benchmarking(self, crossbar_config):
        """Test performance benchmarking across components."""
        
        crossbar = MTJCrossbar(crossbar_config)
        
        # Benchmark weight programming
        weights = np.random.uniform(-1, 1, (32, 32))
        
        start_time = time.time()
        conductances = crossbar.set_weights(weights)
        weight_programming_time = time.time() - start_time
        
        assert weight_programming_time < 1.0, "Weight programming too slow"
        
        # Benchmark VMM computation
        input_voltages = np.random.uniform(-0.5, 0.5, 32)
        
        start_time = time.time()
        output = crossbar.compute_vmm(input_voltages)
        vmm_time = time.time() - start_time
        
        assert vmm_time < 0.1, "VMM computation too slow"
        
        # Get performance statistics
        stats = crossbar.get_statistics()
        assert stats['read_operations'] > 0
        assert stats['total_energy_j'] >= 0
    
    def test_error_handling_and_recovery(self, crossbar_config):
        """Test error handling and recovery mechanisms."""
        
        crossbar = MTJCrossbar(crossbar_config)
        
        # Test invalid inputs
        with pytest.raises(ValueError):
            crossbar.set_weights(np.array([[1, 2]]))  # Wrong shape
        
        with pytest.raises(ValueError):
            crossbar.compute_vmm(np.array([1, 2]))  # Wrong size
        
        # Test NaN handling
        weights_with_nan = np.full((32, 32), np.nan)
        with pytest.raises(Exception):
            crossbar.set_weights(weights_with_nan)
        
        # Test recovery after error
        valid_weights = np.random.uniform(-1, 1, (32, 32))
        conductances = crossbar.set_weights(valid_weights)
        assert np.all(np.isfinite(conductances))
    
    def test_memory_and_resource_management(self, crossbar_config):
        """Test memory usage and resource management."""
        
        import psutil
        import gc
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss
        
        # Create multiple crossbars
        crossbars = []
        for i in range(10):
            crossbar = MTJCrossbar(crossbar_config)
            weights = np.random.uniform(-1, 1, (32, 32))
            crossbar.set_weights(weights)
            crossbars.append(crossbar)
        
        mid_memory = process.memory_info().rss
        memory_increase = (mid_memory - initial_memory) / 1024 / 1024  # MB
        
        # Clean up
        del crossbars
        gc.collect()
        
        final_memory = process.memory_info().rss
        memory_recovered = (mid_memory - final_memory) / 1024 / 1024  # MB
        
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.1f} MB"
        assert memory_recovered > 0, "Memory not properly released"
    
    def test_configuration_validation(self):
        """Test configuration validation across components."""
        
        # Test invalid MTJ configuration
        with pytest.raises(Exception):
            MTJConfig(
                resistance_high=-1000,  # Invalid negative resistance
                resistance_low=5e3,
                switching_voltage=0.3
            )
        
        # Test invalid crossbar configuration
        with pytest.raises(Exception):
            CrossbarConfig(
                rows=0,  # Invalid zero size
                cols=32,
                mtj_config=MTJConfig()
            )
        
        # Test security configuration validation
        with pytest.raises(Exception):
            SecurityConfig(
                dp_epsilon=-1.0  # Invalid negative epsilon
            )
    
    def test_serialization_and_persistence(self, crossbar_config):
        """Test serialization and state persistence."""
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create and configure crossbar
            crossbar = MTJCrossbar(crossbar_config)
            weights = np.random.uniform(-1, 1, (32, 32))
            conductances = crossbar.set_weights(weights)
            
            # Save state
            state_file = Path(temp_dir) / "crossbar_state.json"
            state = {
                'config': {
                    'rows': crossbar_config.rows,
                    'cols': crossbar_config.cols,
                    'read_voltage': crossbar_config.read_voltage
                },
                'conductances': conductances.tolist(),
                'statistics': crossbar.get_statistics()
            }
            
            with open(state_file, 'w') as f:
                json.dump(state, f)
            
            # Verify file was created and is valid
            assert state_file.exists()
            
            with open(state_file, 'r') as f:
                loaded_state = json.load(f)
            
            assert 'config' in loaded_state
            assert 'conductances' in loaded_state
            assert loaded_state['config']['rows'] == 32
            
            # Verify conductances can be restored
            restored_conductances = np.array(loaded_state['conductances'])
            assert restored_conductances.shape == (32, 32)
            np.testing.assert_array_almost_equal(conductances, restored_conductances)


class TestScalabilityAndPerformance:
    """Test scalability and performance characteristics."""
    
    @pytest.mark.parametrize("size", [16, 32, 64, 128])
    def test_crossbar_scaling(self, size):
        """Test crossbar performance scaling with size."""
        
        config = CrossbarConfig(rows=size, cols=size, mtj_config=MTJConfig())
        crossbar = MTJCrossbar(config)
        
        # Test weight programming time
        weights = np.random.uniform(-1, 1, (size, size))
        
        start_time = time.time()
        conductances = crossbar.set_weights(weights)
        programming_time = time.time() - start_time
        
        # Programming time should be roughly linear with size
        expected_max_time = size * size * 1e-6  # Very optimistic target
        assert programming_time < max(0.1, expected_max_time), f"Programming too slow for size {size}: {programming_time:.3f}s"
        
        # Test VMM computation time
        input_voltages = np.random.uniform(-0.5, 0.5, size)
        
        start_time = time.time()
        output = crossbar.compute_vmm(input_voltages)
        vmm_time = time.time() - start_time
        
        # VMM time should be roughly quadratic with size
        expected_max_vmm_time = size * size * 1e-7
        assert vmm_time < max(0.01, expected_max_vmm_time), f"VMM too slow for size {size}: {vmm_time:.3f}s"
        
        assert len(output) == size
        assert np.all(np.isfinite(output))
    
    def test_concurrent_operations(self, crossbar_config):
        """Test concurrent operations on crossbar."""
        
        import threading
        import time
        
        crossbar = MTJCrossbar(crossbar_config)
        results = []
        errors = []
        
        def compute_vmm_worker(worker_id):
            try:
                input_voltages = np.random.uniform(-0.5, 0.5, 32)
                output = crossbar.compute_vmm(input_voltages)
                results.append((worker_id, output))
            except Exception as e:
                errors.append((worker_id, str(e)))
        
        # Set initial weights
        weights = np.random.uniform(-1, 1, (32, 32))
        crossbar.set_weights(weights)
        
        # Launch concurrent computations
        threads = []
        for i in range(5):
            thread = threading.Thread(target=compute_vmm_worker, args=(i,))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors in concurrent operations: {errors}"
        assert len(results) == 5
        
        # Verify all outputs are valid
        for worker_id, output in results:
            assert len(output) == 32
            assert np.all(np.isfinite(output))
    
    def test_long_running_stability(self, crossbar_config):
        """Test stability under long-running operations."""
        
        crossbar = MTJCrossbar(crossbar_config)
        weights = np.random.uniform(-1, 1, (32, 32))
        crossbar.set_weights(weights)
        
        # Run many operations
        num_operations = 1000
        outputs = []
        
        start_time = time.time()
        
        for i in range(num_operations):
            input_voltages = np.random.uniform(-0.5, 0.5, 32)
            output = crossbar.compute_vmm(input_voltages)
            outputs.append(output)
            
            # Check for degradation
            if i % 100 == 0:
                assert np.all(np.isfinite(output)), f"Invalid output at iteration {i}"
        
        total_time = time.time() - start_time
        avg_time_per_op = total_time / num_operations
        
        # Check performance hasn't degraded
        assert avg_time_per_op < 0.001, f"Performance degraded: {avg_time_per_op:.6f}s per operation"
        
        # Check output consistency
        output_stds = [np.std([outputs[i][j] for i in range(num_operations)]) for j in range(32)]
        max_std = max(output_stds)
        assert max_std < 1.0, f"Output instability detected: max std = {max_std}"
        
        # Check statistics
        stats = crossbar.get_statistics()
        assert stats['read_operations'] >= num_operations


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])
