"""Sample models and test data for testing."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Any


class KeywordSpottingNet(nn.Module):
    """Sample keyword spotting network."""
    
    def __init__(self, num_keywords: int = 10, input_features: int = 40):
        super().__init__()
        self.features = input_features
        self.num_keywords = num_keywords
        
        self.network = nn.Sequential(
            nn.Linear(input_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_keywords)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class TinyConvNet(nn.Module):
    """Sample tiny convolutional network for vision tasks."""
    
    def __init__(self, num_classes: int = 10, input_channels: int = 1):
        super().__init__()
        self.num_classes = num_classes
        
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(32 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class MicroNet(nn.Module):
    """Ultra-small network for testing."""
    
    def __init__(self, input_size: int = 784, hidden_size: int = 32, num_classes: int = 10):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def generate_test_data(model_type: str, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate test data for different model types."""
    
    if model_type == "keyword_spotting":
        # MFCC features: (batch, features)
        inputs = torch.randn(batch_size, 40)
        targets = torch.randint(0, 10, (batch_size,))
        
    elif model_type == "vision":
        # Small images: (batch, channels, height, width)
        inputs = torch.randn(batch_size, 1, 32, 32)
        targets = torch.randint(0, 10, (batch_size,))
        
    elif model_type == "micro":
        # Flattened input: (batch, features)
        inputs = torch.randn(batch_size, 784)
        targets = torch.randint(0, 10, (batch_size,))
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return inputs, targets


def create_mtj_test_parameters() -> Dict[str, Any]:
    """Create test parameters for MTJ devices."""
    return {
        "resistance_high": 10e3,      # 10 kOhm
        "resistance_low": 5e3,        # 5 kOhm
        "switching_voltage": 0.3,     # 300 mV
        "switching_current": 50e-6,   # 50 μA
        "cell_area": 40e-9,          # 40 nm²
        "retention_time": 10,         # years
        "switching_energy": 1e-15,    # 1 fJ
        "write_time": 1e-9,          # 1 ns
        "read_time": 100e-12,        # 100 ps
        "endurance": 1e12,           # 1T cycles
    }


def create_crossbar_test_config() -> Dict[str, Any]:
    """Create test configuration for crossbar arrays."""
    return {
        "rows": 128,
        "cols": 128,
        "mtj_type": "perpendicular",
        "reference_layer": "CoFeB",
        "barrier": "MgO", 
        "free_layer": "CoFeB",
        "thickness_barrier": 1.2e-9,  # 1.2 nm
        "thickness_free": 2.0e-9,     # 2.0 nm
        "thickness_ref": 3.0e-9,      # 3.0 nm
        "anisotropy_field": 1000,     # Oe
        "saturation_magnetization": 1000,  # emu/cm³
    }


def create_simulation_test_data() -> Dict[str, Any]:
    """Create test data for hardware simulation."""
    return {
        "input_voltages": np.random.uniform(0, 1.8, 128),  # 1.8V supply
        "weight_matrix": np.random.uniform(-1, 1, (128, 128)),
        "expected_currents": np.random.uniform(0, 100e-6, 128),  # μA range
        "noise_level": 0.1,  # 10% noise
        "temperature": 25,   # Celsius
        "supply_voltage": 1.8,  # Volts
        "frequency": 50e6,   # 50 MHz
    }


def create_benchmark_test_data() -> Dict[str, Any]:
    """Create test data for benchmarking."""
    return {
        "energy_targets": {
            "mac_energy_pj": 10.0,
            "inference_energy_nj": 100.0,
            "standby_power_uw": 1.0,
        },
        "performance_targets": {
            "inference_latency_ms": 1.0,
            "throughput_inferences_per_sec": 1000,
            "accuracy_threshold": 0.95,
        },
        "area_targets": {
            "total_area_mm2": 1.0,
            "crossbar_density_cells_per_mm2": 1e6,
        },
    }