# SpinTron-NN-Kit Examples

This document provides comprehensive examples for using SpinTron-NN-Kit to develop ultra-low-power spintronic neural networks.

## Quick Start

```python
from spintron_nn import *
import torch
import torch.nn as nn

# 1. Define your PyTorch model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# 2. Convert to spintronic implementation
model = SimpleNet()
converter = PyTorchConverter()
spintronic_model = converter.convert(model)

# 3. Generate hardware
generator = VerilogGenerator()
verilog_files = generator.generate_from_model(spintronic_model)

print(f"Energy per inference: {spintronic_model.estimate_energy():.2e}J")
```

## Core Examples

### 1. MTJ Device Physics

```python
from spintron_nn.core.mtj_models import MTJConfig, MTJDevice

# Configure MTJ parameters for your process
config = MTJConfig(
    resistance_high=10e3,      # 10kΩ high resistance
    resistance_low=5e3,        # 5kΩ low resistance
    switching_voltage=0.3,     # 300mV switching
    thermal_stability=85.0,    # 85°C operating temp
    retention_time=10.0        # 10 year retention
)

# Create device and analyze
device = MTJDevice(config)
print(f"Resistance ratio: {device.resistance_ratio:.1f}")
print(f"Switching energy: {device.switching_energy:.2e}J")
print(f"Read energy: {device.read_energy:.2e}J")
```

### 2. Crossbar Array Operations

```python
from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig
import numpy as np

# Create 128x128 crossbar for neural network layer
config = CrossbarConfig(rows=128, cols=128, mtj_config=mtj_config)
crossbar = MTJCrossbar(config)

# Program weights from trained model
weights = np.random.randn(128, 128) * 0.5
crossbar.set_weights(weights)

# Perform vector-matrix multiplication (neural network forward pass)
input_vector = np.random.randn(128)
output = crossbar.compute_vmm(input_vector, include_nonidealities=True)

print(f"VMM latency: {crossbar.get_vmm_latency():.2f}ns")
print(f"Energy consumption: {crossbar.get_energy_consumption():.2e}J")
```

### 3. Model Conversion Pipeline

```python
from spintron_nn.converter.pytorch_parser import PyTorchConverter
from spintron_nn.converter.optimization import ModelOptimizer

# Load pre-trained PyTorch model
model = torch.load('pretrained_model.pth')

# Convert with optimization
converter = PyTorchConverter()
optimizer = ModelOptimizer()

# Apply spintronic-specific optimizations
optimized_model = optimizer.optimize_for_spintronic(model)
spintronic_model = converter.convert(optimized_model)

# Analyze conversion results
print(f"Original parameters: {sum(p.numel() for p in model.parameters())}")
print(f"Spintronic crossbars: {len(spintronic_model.crossbars)}")
print(f"Energy reduction: {optimizer.get_energy_reduction():.1f}x")
```

## Training Examples

### 4. Quantization-Aware Training

```python
from spintron_nn.training.qat import SpintronicTrainer, QuantizationConfig
from torch.utils.data import DataLoader

# Configure quantization for MTJ constraints
qconfig = QuantizationConfig(
    weight_bits=8,              # 8-bit weights
    activation_bits=8,          # 8-bit activations
    enable_mtj_aware=True,      # MTJ-specific quantization
    symmetric_weights=True,     # Better for crossbars
    signed_activations=False    # Unsigned for power efficiency
)

# Setup trainer with spintronic optimizations
trainer = SpintronicTrainer(model, qconfig)

# Train with hardware constraints
history = trainer.train(
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    learning_rate=0.001,
    energy_constraint=1e-12,    # 1pJ energy budget
    accuracy_threshold=0.95     # Target 95% accuracy
)

print(f"Final accuracy: {history['val_acc'][-1]:.2f}%")
print(f"Energy per inference: {trainer.estimate_energy():.2e}J")
```

### 5. Variation-Aware Training

```python
from spintron_nn.training.variation_aware import VariationAwareTraining, VariationModel

# Model device variations
variation_model = VariationModel(
    resistance_std=0.1,         # 10% resistance variation
    switching_voltage_std=0.05, # 5% switching variation
    temperature_coefficient=-0.001,  # -0.1%/°C
    operating_temperature_range=(-40, 85)  # Industrial range
)

# Setup variation-aware training
va_trainer = VariationAwareTraining(
    model=model,
    variation_model=variation_model,
    monte_carlo_samples=1000
)

# Train with robustness optimization
history = va_trainer.train(
    train_loader, val_loader,
    num_epochs=50,
    robustness_weight=0.2  # 20% robustness loss weight
)

# Evaluate robustness
robustness = va_trainer.evaluate_robustness(test_loader)
print(f"Mean accuracy: {robustness['accuracy']:.2f}%")
print(f"Worst-case accuracy: {robustness['worst_case']:.2f}%")
```

## Hardware Generation

### 6. Verilog Generation

```python
from spintron_nn.hardware.verilog_gen import VerilogGenerator, VerilogConfig

# Configure hardware generation
hw_config = VerilogConfig(
    target_technology="45nm",
    operating_frequency=100e6,    # 100 MHz
    pipeline_stages=3,
    include_testbench=True,
    synthesis_constraints=True
)

# Generate complete hardware
generator = VerilogGenerator(hw_config)
files = generator.generate_from_model(
    spintronic_model,
    output_dir="./hardware_output"
)

# Generated files:
# - crossbar_array.v (MTJ crossbar implementation)
# - control_unit.v (Control logic)
# - adc_interface.v (Analog-to-digital conversion)
# - top_module.v (Complete system)
# - testbench.v (Verification environment)

print(f"Generated {len(files)} Verilog files")
print(f"Estimated area: {generator.estimate_area():.2f} mm²")
print(f"Power consumption: {generator.estimate_power():.2f} mW")
```

### 7. Hardware-Software Co-design

```python
from spintron_nn.hardware.codesign import HardwareSoftwareCodesign

# Co-optimize hardware and model
codesign = HardwareSoftwareCodesign(
    model=model,
    hardware_constraints={
        'area_budget_mm2': 5.0,
        'power_budget_mw': 10.0,
        'frequency_mhz': 100,
        'technology_node': '28nm'
    }
)

# Run co-design optimization
optimized_system = codesign.optimize(
    accuracy_threshold=0.90,
    optimization_iterations=50
)

print(f"Optimized accuracy: {optimized_system.accuracy:.2f}%")
print(f"Hardware area: {optimized_system.area_mm2:.2f} mm²")
print(f"Power consumption: {optimized_system.power_mw:.2f} mW")
```

## Simulation and Analysis

### 8. Behavioral Simulation

```python
from spintron_nn.simulation.behavioral import BehavioralSimulator, SimulationConfig

# Configure realistic simulation
sim_config = SimulationConfig(
    include_device_variations=True,
    include_thermal_effects=True,
    include_quantization_noise=True,
    operating_temperature=25.0,
    device_variation_std=0.05,
    enable_power_tracking=True
)

# Create behavioral simulator
simulator = BehavioralSimulator(spintronic_model, sim_config, device="cuda")

# Run comprehensive test suite
test_results = simulator.run_test_suite(
    test_inputs=test_dataset,
    expected_outputs=expected_outputs,
    tolerance=0.01
)

print(f"Test pass rate: {test_results['accuracy_metrics']['pass_rate']:.2%}")
print(f"Average inference time: {test_results['performance_metrics']['avg_inference_time_s']:.2e}s")
```

### 9. Monte Carlo Analysis

```python
# Monte Carlo analysis for robustness evaluation
mc_results = simulator.monte_carlo_analysis(
    test_input=sample_input,
    num_samples=10000,
    analyze_variations=True
)

# Statistical analysis
print(f"Output mean: {mc_results['global_statistics']['overall_mean']:.3f}")
print(f"Output std: {mc_results['global_statistics']['overall_std']:.3f}")
print(f"SNR: {mc_results['global_statistics']['signal_to_noise_ratio']:.1f} dB")
print(f"Coefficient of variation: {mc_results['global_statistics']['coefficient_of_variation']:.3f}")
```

### 10. Power Analysis

```python
from spintron_nn.simulation.power import PowerAnalyzer, PowerConfig

# Configure power analysis
power_config = PowerConfig(
    supply_voltage=1.0,         # 1V supply
    operating_frequency=100e6,  # 100 MHz
    technology_node="28nm",
    temperature=25.0,
    include_leakage=True,
    include_peripheral_power=True
)

# Analyze power consumption
analyzer = PowerAnalyzer(spintronic_model, power_config)
power_report = analyzer.analyze_inference(test_inputs)

print(f"Total energy per inference: {power_report['total_energy_j']:.2e}J")
print(f"Energy per MAC: {power_report['energy_per_mac_j']:.2e}J")
print(f"Peak power: {power_report['peak_power_w']:.2e}W")
print(f"Energy breakdown:")
for component, energy in power_report['energy_breakdown'].items():
    print(f"  {component}: {energy:.2e}J ({energy/power_report['total_energy_j']*100:.1f}%)")
```

## Pre-optimized Models

### 11. Vision Models

```python
from spintron_nn.models.vision import TinyConvNet, OptimizedMobileNetV2

# Ultra-low-power CIFAR-10 classifier
model = TinyConvNet(
    num_classes=10,
    power_budget_uw=50,  # 50μW power budget
    accuracy_target=0.88
)

# Pre-trained and optimized
model.load_pretrained()
print(f"Model accuracy: {model.test_accuracy:.2f}%")
print(f"Estimated power: {model.estimated_power_uw:.1f}μW")
print(f"Inference latency: {model.inference_latency_ms:.2f}ms")

# Mobile vision with extreme efficiency
mobile_model = OptimizedMobileNetV2(
    num_classes=1000,
    width_multiplier=0.25,    # Quarter-width for efficiency
    spintronic_optimized=True
)
```

### 12. Audio Models

```python
from spintron_nn.models.audio import KeywordSpottingNet, WakeWordDetector

# Always-on keyword spotting
kws_model = KeywordSpottingNet(
    num_keywords=12,
    power_budget_uw=20,      # 20μW always-on
    false_positive_rate=0.01  # 1% false positive rate
)

# Wake word detection for voice assistants
wake_model = WakeWordDetector(
    wake_words=["hey_assistant", "wake_up"],
    detection_threshold=0.95,
    power_budget_uw=10       # 10μW standby power
)

print(f"Wake word sensitivity: {wake_model.sensitivity:.3f}")
print(f"Background rejection: {wake_model.background_rejection:.3f}")
```

## Performance Optimization

### 13. Caching and Parallelization

```python
from spintron_nn.utils.performance import PerformanceOptimizer, PerformanceConfig

# Configure performance optimization
perf_config = PerformanceConfig(
    enable_result_caching=True,
    cache_size_mb=500,
    max_workers=8,
    enable_memory_mapping=True,
    prefetch_inputs=True,
    batch_optimization=True
)

# Create optimizer
optimizer = PerformanceOptimizer(perf_config)

# Optimize inference pipeline
@optimizer.optimize_inference
def run_batch_inference(model, batch_data):
    results = []
    for data in batch_data:
        result = model(data)
        results.append(result)
    return results

# Automatically parallelized and cached
batch_results = run_batch_inference(spintronic_model, test_batch)
print(f"Batch processing speedup: {optimizer.get_speedup():.1f}x")
```

### 14. Auto-scaling

```python
from spintron_nn.utils.performance import AutoScaler

# Setup auto-scaling for varying workloads
scaler = AutoScaler(
    min_workers=1,
    max_workers=16,
    target_latency_ms=10.0,
    scale_up_threshold=0.8,
    scale_down_threshold=0.3
)

# Register model for auto-scaling
scaler.register_model("spintronic_classifier", spintronic_model)

# Process requests with automatic scaling
for request_batch in request_stream:
    results = scaler.process_batch("spintronic_classifier", request_batch)
    print(f"Current workers: {scaler.get_current_workers()}")
    print(f"Average latency: {scaler.get_average_latency():.2f}ms")
```

## Command Line Interface

### 15. CLI Usage Examples

```bash
# Convert PyTorch model to spintronic
spintron-nn convert \
    --model pretrained_model.pth \
    --output spintronic_model.json \
    --optimize-energy \
    --target-accuracy 0.90

# Generate Verilog hardware
spintron-nn generate-verilog \
    --model spintronic_model.json \
    --output ./hardware \
    --technology 28nm \
    --frequency 100MHz \
    --include-testbench

# Run behavioral simulation
spintron-nn simulate \
    --model spintronic_model.json \
    --input test_data.npy \
    --output simulation_results.json \
    --monte-carlo 1000 \
    --include-variations

# Benchmark performance
spintron-nn benchmark \
    --model spintronic_model.json \
    --target-device asic \
    --batch-size 32 \
    --report benchmark_report.html

# Train with quantization awareness
spintron-nn train \
    --model model_definition.py \
    --dataset cifar10 \
    --qat \
    --variation-aware \
    --epochs 100 \
    --output trained_spintronic_model.pth
```

## Advanced Examples

### 16. Custom Training Loop

```python
import torch.optim as optim
from spintron_nn.training.losses import SpintronicLoss

# Custom loss function for spintronic constraints
spintronic_loss = SpintronicLoss(
    energy_weight=0.1,
    variation_weight=0.05,
    sparsity_weight=0.02
)

# Custom training loop with energy optimization
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

for epoch in range(100):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # Forward pass
        output = model(data)
        
        # Compute losses
        ce_loss = F.cross_entropy(output, target)
        energy_loss = spintronic_loss.energy_loss(model)
        variation_loss = spintronic_loss.variation_loss(model)
        
        # Combined loss
        total_loss = ce_loss + energy_loss + variation_loss
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
    
    scheduler.step()
    
    # Validate with device variations
    val_accuracy = validate_with_variations(model, val_loader)
    print(f"Epoch {epoch}: Val Acc {val_accuracy:.2f}%")
```

### 17. Hardware-in-the-Loop Testing

```python
from spintron_nn.hardware.hil import HardwareInTheLoop

# Setup hardware-in-the-loop testing
hil = HardwareInTheLoop(
    fpga_board="xilinx_zynq",
    connection="ethernet",
    bitstream="spintronic_nn.bit"
)

# Deploy model to hardware
hil.deploy_model(spintronic_model)

# Run inference on actual hardware
hw_results = hil.run_inference_batch(test_inputs)
sw_results = simulator.run_inference_batch(test_inputs)

# Compare hardware vs simulation
correlation = hil.compare_results(hw_results, sw_results)
print(f"HW/SW correlation: {correlation:.3f}")
print(f"Hardware accuracy: {hil.measure_accuracy(test_dataset):.2f}%")
print(f"Actual power consumption: {hil.measure_power():.2f}mW")
```

## Integration Examples

### 18. Edge Deployment

```python
from spintron_nn.deployment.edge import EdgeDeployer

# Deploy to edge device
deployer = EdgeDeployer(
    target_device="arm_cortex_m4",
    memory_budget_kb=256,
    power_budget_mw=5.0
)

# Optimize model for edge deployment
edge_model = deployer.optimize_for_edge(spintronic_model)
deployment_package = deployer.create_deployment_package(
    model=edge_model,
    runtime="tflite_micro",
    include_calibration=True
)

print(f"Model size: {deployment_package.model_size_kb}KB")
print(f"RAM usage: {deployment_package.ram_usage_kb}KB")
print(f"Inference time: {deployment_package.inference_time_ms}ms")
```

For more examples, see the `examples/` directory in the repository.