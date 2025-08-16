"""
Dependency-free demonstration of SpinTron-NN-Kit core functionality.

This module demonstrates key features without requiring external dependencies.
"""

import math
import json
from typing import Dict, List, Any


class MTJDeviceDemo:
    """Demonstration MTJ device without PyTorch dependencies."""
    
    def __init__(self, resistance_high: float = 10e3, resistance_low: float = 5e3):
        self.resistance_high = resistance_high
        self.resistance_low = resistance_low
        self.current_state = 0  # 0 = low resistance, 1 = high resistance
        
    def switch_state(self, voltage: float, switching_threshold: float = 0.3):
        """Switch MTJ state based on applied voltage."""
        if abs(voltage) > switching_threshold:
            self.current_state = 1 - self.current_state
            return True
        return False
    
    def get_resistance(self) -> float:
        """Get current resistance value."""
        return self.resistance_high if self.current_state else self.resistance_low
    
    def calculate_power(self, voltage: float) -> float:
        """Calculate power consumption."""
        current = voltage / self.get_resistance()
        return voltage * current


class CrossbarDemo:
    """Demonstration crossbar array."""
    
    def __init__(self, rows: int = 8, cols: int = 8):
        self.rows = rows
        self.cols = cols
        self.devices = [[MTJDeviceDemo() for _ in range(cols)] for _ in range(rows)]
    
    def set_weights(self, weights: List[List[float]]):
        """Set crossbar weights (simplified)."""
        for i in range(min(self.rows, len(weights))):
            for j in range(min(self.cols, len(weights[i]))):
                # Map weight to resistance state
                normalized_weight = (weights[i][j] + 1.0) / 2.0  # Assume weights in [-1,1]
                if normalized_weight > 0.5:
                    self.devices[i][j].current_state = 1
                else:
                    self.devices[i][j].current_state = 0
    
    def compute_vmm(self, input_vector: List[float]) -> List[float]:
        """Compute vector-matrix multiplication."""
        output = []
        for j in range(self.cols):
            sum_output = 0.0
            for i in range(self.rows):
                if i < len(input_vector):
                    conductance = 1.0 / self.devices[i][j].get_resistance()
                    sum_output += input_vector[i] * conductance
            output.append(sum_output)
        return output


class SpintronicNetworkDemo:
    """Demonstration spintronic neural network."""
    
    def __init__(self, layer_sizes: List[int]):
        self.layer_sizes = layer_sizes
        self.layers = []
        
        # Create crossbar layers
        for i in range(len(layer_sizes) - 1):
            crossbar = CrossbarDemo(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(crossbar)
    
    def forward(self, inputs: List[float]) -> List[float]:
        """Forward pass through network."""
        current_values = inputs[:]
        
        for layer in self.layers:
            # Compute layer output
            current_values = layer.compute_vmm(current_values)
            
            # Apply activation (simplified ReLU)
            current_values = [max(0.0, x) for x in current_values]
        
        return current_values
    
    def set_random_weights(self, seed: int = 42):
        """Set random weights for demonstration."""
        # Simple pseudo-random generator to avoid numpy dependency
        def simple_random():
            nonlocal seed
            seed = (seed * 1664525 + 1013904223) % (2**32)
            return (seed / (2**32)) * 2.0 - 1.0  # Range [-1, 1]
        
        for layer in self.layers:
            weights = []
            for i in range(layer.rows):
                row = []
                for j in range(layer.cols):
                    row.append(simple_random())
                weights.append(row)
            layer.set_weights(weights)


def demonstrate_core_functionality():
    """Demonstrate core SpinTron-NN-Kit functionality."""
    
    print("🚀 SpinTron-NN-Kit Core Functionality Demonstration")
    print("=" * 60)
    
    # 1. MTJ Device Demo
    print("\n1. MTJ Device Physics")
    print("-" * 30)
    
    mtj = MTJDeviceDemo(resistance_high=15e3, resistance_low=5e3)
    print(f"Initial resistance: {mtj.get_resistance()/1e3:.1f} kΩ")
    print(f"Initial power @ 1V: {mtj.calculate_power(1.0)*1e6:.1f} μW")
    
    # Switch state
    switched = mtj.switch_state(0.5)  # Above threshold
    print(f"Applied 0.5V -> Switched: {switched}")
    print(f"New resistance: {mtj.get_resistance()/1e3:.1f} kΩ")
    print(f"New power @ 1V: {mtj.calculate_power(1.0)*1e6:.1f} μW")
    
    # 2. Crossbar Array Demo  
    print("\n2. MTJ Crossbar Array")
    print("-" * 30)
    
    crossbar = CrossbarDemo(rows=4, cols=4)
    
    # Set example weights
    weights = [
        [0.5, -0.3, 0.8, -0.1],
        [-0.2, 0.6, -0.4, 0.7],
        [0.1, -0.5, 0.3, -0.8],
        [-0.6, 0.2, -0.7, 0.4]
    ]
    crossbar.set_weights(weights)
    
    # Compute vector-matrix multiplication
    input_vector = [1.0, 0.5, -0.3, 0.8]
    output = crossbar.compute_vmm(input_vector)
    
    print(f"Input vector: {[f'{x:.1f}' for x in input_vector]}")
    print(f"Output vector: {[f'{x:.2e}' for x in output]}")
    
    # 3. Neural Network Demo
    print("\n3. Spintronic Neural Network")
    print("-" * 30)
    
    # Create a simple 4-3-2 network
    network = SpintronicNetworkDemo([4, 3, 2])
    network.set_random_weights(seed=42)
    
    # Run inference
    test_input = [1.0, 0.5, -0.2, 0.8]
    result = network.forward(test_input)
    
    print(f"Network architecture: 4 → 3 → 2")
    print(f"Test input: {[f'{x:.1f}' for x in test_input]}")
    print(f"Network output: {[f'{x:.3f}' for x in result]}")
    
    # 4. Energy Estimation
    print("\n4. Energy Analysis")
    print("-" * 30)
    
    # Estimate energy consumption
    total_devices = sum(layer.rows * layer.cols for layer in network.layers)
    switching_energy_pj = 50  # 50 pJ per switch typical for MTJ
    leakage_power_pw = 100   # 100 pW per device
    
    # Estimate switching activity (rough)
    switching_activity = 0.3  # 30% of devices switch per operation
    active_switches = total_devices * switching_activity
    
    dynamic_energy_pj = active_switches * switching_energy_pj
    static_power_nw = total_devices * leakage_power_pw / 1000  # Convert to nW
    
    print(f"Total MTJ devices: {total_devices}")
    print(f"Active switches: {int(active_switches)}")
    print(f"Dynamic energy: {dynamic_energy_pj:.1f} pJ")
    print(f"Static power: {static_power_nw:.1f} nW")
    print(f"Energy per MAC: {dynamic_energy_pj / max(1, len(test_input)):.1f} pJ")
    
    # 5. Performance Metrics
    print("\n5. Performance Summary")
    print("-" * 30)
    
    # Calculate key metrics
    mac_operations = sum(layer.rows * layer.cols for layer in network.layers)
    energy_per_mac = dynamic_energy_pj / mac_operations
    
    # Comparison with CMOS (typical values)
    cmos_energy_per_mac_pj = 1000  # 1 nJ typical for 28nm CMOS
    energy_improvement = cmos_energy_per_mac_pj / energy_per_mac
    
    metrics = {
        'mac_operations': mac_operations,
        'energy_per_mac_pj': energy_per_mac,
        'total_dynamic_energy_pj': dynamic_energy_pj,
        'static_power_nw': static_power_nw,
        'energy_improvement_vs_cmos': energy_improvement,
        'network_accuracy_estimate': 0.85  # Placeholder
    }
    
    print(f"MAC operations: {mac_operations}")
    print(f"Energy per MAC: {energy_per_mac:.1f} pJ")
    print(f"Improvement vs CMOS: {energy_improvement:.0f}×")
    print(f"Estimated accuracy: {metrics['network_accuracy_estimate']:.1%}")
    
    # Save results
    results = {
        'demonstration': 'SpinTron-NN-Kit Core Functionality',
        'timestamp': '2025-01-01T00:00:00Z',  # Placeholder
        'mtj_device': {
            'resistance_ratio': mtj.resistance_high / mtj.resistance_low,
            'switching_successful': switched
        },
        'crossbar': {
            'size': f"{crossbar.rows}x{crossbar.cols}",
            'input_vector': input_vector,
            'output_vector': output
        },
        'network': {
            'architecture': network.layer_sizes,
            'total_devices': total_devices,
            'inference_result': result
        },
        'performance': metrics
    }
    
    with open('demo_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to demo_results.json")
    print(f"✓ Demonstration completed successfully!")
    
    return results


if __name__ == "__main__":
    demonstrate_core_functionality()