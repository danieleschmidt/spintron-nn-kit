"""PyTest configuration and shared fixtures."""

import os
import tempfile
from pathlib import Path
from typing import Generator, Dict, Any

import pytest
import torch
import numpy as np


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_model() -> torch.nn.Module:
    """Create a simple PyTorch model for testing."""
    return torch.nn.Sequential(
        torch.nn.Linear(784, 128),
        torch.nn.ReLU(),
        torch.nn.Linear(128, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 10)
    )


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Create sample input tensor."""
    return torch.randn(1, 784)


@pytest.fixture
def mtj_config() -> Dict[str, Any]:
    """Default MTJ configuration for testing."""
    return {
        'resistance_high': 10e3,
        'resistance_low': 5e3,
        'switching_voltage': 0.3,
        'cell_area': 40e-9,
        'retention_time': 10,  # years
        'switching_energy': 1e-15,  # Joules
    }


@pytest.fixture
def crossbar_config() -> Dict[str, Any]:
    """Default crossbar configuration for testing."""
    return {
        'rows': 128,
        'cols': 128,
        'mtj_type': 'perpendicular',
        'reference_layer': 'CoFeB',
        'barrier': 'MgO',
        'free_layer': 'CoFeB',
    }


@pytest.fixture
def verilog_constraints() -> Dict[str, Any]:
    """Default Verilog generation constraints."""
    return {
        'target_frequency': 50e6,
        'max_area': 1.0,
        'io_voltage': 1.8,
        'core_voltage': 0.8,
        'technology': '28nm',
    }


@pytest.fixture
def test_data_dir() -> Path:
    """Path to test data directory."""
    return Path(__file__).parent / 'data'


@pytest.fixture
def mock_spice_simulator():
    """Mock SPICE simulator for testing."""
    class MockSpiceSimulator:
        def __init__(self):
            self.simulation_results = {}
            
        def simulate(self, circuit, inputs, **kwargs):
            # Return mock simulation results
            return {
                'outputs': np.random.randn(len(inputs)),
                'power': np.random.uniform(1e-6, 1e-3),
                'delay': np.random.uniform(1e-9, 1e-6),
            }
            
        def verify_timing(self, results, target_freq):
            return results['delay'] < 1/target_freq
    
    return MockSpiceSimulator()


@pytest.fixture
def hardware_test_env():
    """Setup hardware testing environment."""
    env = {
        'SIMULATION_MODE': 'test',
        'HARDWARE_BACKEND': 'mock',
        'VERIFICATION_LEVEL': 'basic',
    }
    
    # Set environment variables
    original_env = {}
    for key, value in env.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    yield env
    
    # Restore original environment
    for key, value in original_env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


@pytest.fixture(scope="session")
def performance_baseline():
    """Performance baseline for benchmarking tests."""
    return {
        'energy_per_mac_pj': 10.0,
        'inference_latency_ms': 1.0,
        'memory_usage_mb': 100.0,
        'accuracy_threshold': 0.95,
    }


# Pytest markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "e2e: marks tests as end-to-end tests"
    )
    config.addinivalue_line(
        "markers", "hardware: marks tests that require hardware simulation"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )
    config.addinivalue_line(
        "markers", "requires_tools: marks tests that require external tools"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test paths."""
    for item in items:
        # Add integration marker for tests in integration directory
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Add e2e marker for tests in e2e directory
        if "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
        
        # Add hardware marker for hardware-related tests
        if "hardware" in str(item.fspath) or "simulation" in str(item.fspath):
            item.add_marker(pytest.mark.hardware)


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Setup common test environment."""
    # Set deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Disable CUDA for consistent testing
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    yield
    
    # Cleanup after test
    torch.cuda.empty_cache() if torch.cuda.is_available() else None