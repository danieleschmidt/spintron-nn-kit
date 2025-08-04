# SpinTron-NN-Kit

Ultra-low-power neural inference framework that bridges PyTorch models to spin-orbit-torque (SOT) hardware through automated Verilog generation. Achieve picojoule-level multiply-accumulate operations using magnetic tunnel junction (MTJ) crossbars for edge AI applications.

## Overview

SpinTron-NN-Kit leverages 2025's breakthroughs in spintronic logic devices to enable extreme energy efficiency for neural network inference. The toolkit provides an end-to-end flow from high-level PyTorch models to synthesizable Verilog targeting SOT-based neuromorphic chips, achieving 1000x better energy efficiency than traditional CMOS for small vision and keyword-spotting models.

## Key Features

- **PyTorch to Spintronics**: Automated conversion of neural networks to spintronic hardware
- **MTJ Crossbar Mapping**: Efficient weight mapping to magnetic tunnel junction arrays
- **Picojoule MACs**: Ultra-low power multiply-accumulate operations (~10 pJ/MAC)
- **Hardware Templates**: Pre-validated Verilog modules for common layer types
- **Quantization-Aware Training**: Spintronic-specific QAT for optimal accuracy
- **Multi-Level Cell Support**: 2-4 bit weight precision using domain wall devices

## Installation

```bash
# Basic installation
pip install spintron-nn-kit

# With hardware simulation support
pip install spintron-nn-kit[simulation]

# Development installation
git clone https://github.com/yourusername/spintron-nn-kit
cd spintron-nn-kit
pip install -e ".[dev]"
```

## Quick Start

### Basic Model Conversion

```python
import torch
from spintron_nn import SpintronConverter, MTJConfig

# Load your PyTorch model
model = torch.load('keyword_spotting_model.pth')

# Configure MTJ parameters
mtj_config = MTJConfig(
    resistance_high=10e3,  # 10 kOhm
    resistance_low=5e3,    # 5 kOhm
    switching_voltage=0.3,  # 300 mV
    cell_area=40e-9,       # 40 nm²
)

# Convert to spintronic implementation
converter = SpintronConverter(mtj_config)
spintronic_model = converter.convert(
    model,
    quantization_bits=2,
    crossbar_size=128
)

# Generate Verilog
verilog_code = spintronic_model.to_verilog(
    module_name="keyword_detector",
    target_frequency=10e6  # 10 MHz
)

# Save design files
spintronic_model.save_design("output/keyword_detector")
```

### Vision Model Example

```python
from spintron_nn import VisionSpintronNet
from spintron_nn.models import MobileNetV2_Spintronic

# Pre-optimized spintronic MobileNetV2
model = MobileNetV2_Spintronic(
    num_classes=10,
    input_size=(32, 32),
    mtj_precision=3  # 3-bit weights
)

# Train with spintronic-aware constraints
from spintron_nn.training import SpintronicTrainer

trainer = SpintronicTrainer(
    model,
    device_variations=0.1,  # 10% MTJ variation
    temperature_range=(0, 85)  # Operating temp in Celsius
)

trainer.train(
    train_loader,
    val_loader,
    epochs=100,
    lr=0.001
)

# Export to hardware
hardware_package = model.export_to_hardware(
    include_testbench=True,
    power_analysis=True
)
```

## Architecture

```
spintron-nn-kit/
├── spintron_nn/
│   ├── core/
│   │   ├── mtj_models.py      # MTJ device physics
│   │   ├── crossbar.py        # Crossbar array modeling
│   │   └── sot_physics.py     # Spin-orbit torque calculations
│   ├── converter/
│   │   ├── pytorch_parser.py  # PyTorch model analysis
│   │   ├── graph_optimizer.py # Computation graph optimization
│   │   └── mapping.py         # Neural to spintronic mapping
│   ├── hardware/
│   │   ├── verilog_gen.py     # Verilog code generation
│   │   ├── templates/         # Hardware module templates
│   │   └── constraints.py     # Timing/area constraints
│   ├── training/
│   │   ├── qat.py            # Quantization-aware training
│   │   ├── variation_aware.py # Device variation modeling
│   │   └── energy_opt.py     # Energy optimization
│   ├── simulation/
│   │   ├── spice_interface.py # SPICE simulation interface
│   │   ├── behavioral.py      # Fast behavioral simulation
│   │   └── power_analysis.py  # Power consumption analysis
│   └── models/              # Pre-optimized models
├── examples/
├── tests/
└── benchmarks/
```

## Spintronic Device Modeling

### MTJ Crossbar Configuration

```python
from spintron_nn.core import MTJCrossbar, DomainWallDevice

# Configure crossbar array
crossbar = MTJCrossbar(
    rows=128,
    cols=128,
    mtj_type="perpendicular",
    reference_layer="CoFeB",
    barrier="MgO",
    free_layer="CoFeB"
)

# Add domain wall devices for multi-level cells
dw_device = DomainWallDevice(
    track_length=200e-9,  # 200 nm
    domain_width=20e-9,   # 20 nm
    levels=4              # 2-bit precision
)

crossbar.set_cell_type(dw_device)

# Simulate array behavior
weights = torch.randn(128, 128)
conductances = crossbar.map_weights(weights)
current_output = crossbar.compute_vmm(input_voltages, conductances)
```

### Energy Analysis

```python
from spintron_nn.simulation import EnergyAnalyzer

analyzer = EnergyAnalyzer(spintronic_model)

# Analyze energy per operation
energy_report = analyzer.analyze(
    test_input,
    include_peripheral=True,
    temperature=25
)

print(f"Energy per MAC: {energy_report.mac_energy_pj:.2f} pJ")
print(f"Static power: {energy_report.static_power_uw:.2f} μW")
print(f"Dynamic energy: {energy_report.dynamic_energy_nj:.2f} nJ")

# Generate energy heatmap
analyzer.plot_energy_distribution("energy_heatmap.png")
```

## Hardware Generation

### Verilog Module Hierarchy

```python
from spintron_nn.hardware import VerilogGenerator, DesignConstraints

# Set design constraints
constraints = DesignConstraints(
    target_frequency=50e6,  # 50 MHz
    max_area=1.0,          # 1 mm²
    io_voltage=1.8,        # 1.8V I/O
    core_voltage=0.8       # 0.8V core
)

# Generate hierarchical Verilog
verilog_gen = VerilogGenerator(constraints)
design_files = verilog_gen.generate(
    spintronic_model,
    hierarchy_style="flat",  # or "hierarchical"
    include_assertions=True
)

# Generate synthesis scripts
verilog_gen.generate_synthesis_scripts(
    tool="synopsys",  # or "cadence", "xilinx"
    technology="28nm"
)
```

### Testbench Generation

```python
from spintron_nn.hardware import TestbenchGenerator

tb_gen = TestbenchGenerator(spintronic_model)

# Generate comprehensive testbench
testbench = tb_gen.generate(
    test_vectors=test_dataset[:100],
    coverage_goals=["statement", "branch", "toggle"],
    timing_checks=True
)

# Generate SystemVerilog assertions
assertions = tb_gen.generate_assertions(
    functional_coverage=True,
    protocol_checks=True
)
```

## Training for Spintronics

### Variation-Aware Training

```python
from spintron_nn.training import VariationAwareTraining

# Model MTJ variations
variation_model = {
    'resistance': {'mean': 1.0, 'std': 0.1},
    'switching_voltage': {'mean': 0.3, 'std': 0.05},
    'retention_time': {'mean': 10, 'std': 2}  # years
}

# Train with variation injection
var_trainer = VariationAwareTraining(
    model,
    variation_model,
    monte_carlo_samples=100
)

var_trainer.train(
    train_loader,
    epochs=50,
    robustness_weight=0.1
)
```

### Energy-Optimized Training

```python
from spintron_nn.training import EnergyOptimizedTraining

# Multi-objective optimization
energy_trainer = EnergyOptimizedTraining(
    model,
    energy_weight=0.3,
    accuracy_weight=0.7
)

# Custom energy loss
def energy_loss(model, input, target):
    output = model(input)
    accuracy_loss = F.cross_entropy(output, target)
    
    # Estimate switching energy
    weight_changes = model.get_weight_updates()
    switching_energy = mtj_config.estimate_switching_energy(weight_changes)
    
    return accuracy_loss + energy_weight * switching_energy

energy_trainer.set_custom_loss(energy_loss)
energy_trainer.train(train_loader, val_loader)
```

## Simulation and Verification

### SPICE Co-simulation

```python
from spintron_nn.simulation import SPICESimulator

# Setup SPICE simulation
spice_sim = SPICESimulator(
    simulator="ngspice",  # or "hspice", "spectre"
    mtj_model="stanford_mtj_model.lib"
)

# Extract critical path
critical_path = spintronic_model.extract_critical_path()

# Simulate with SPICE accuracy
spice_results = spice_sim.simulate(
    critical_path,
    input_patterns=test_patterns,
    temperature_corners=[-40, 25, 125],
    voltage_corners=[0.9, 1.0, 1.1]
)

# Verify timing
timing_met = spice_sim.verify_timing(
    spice_results,
    target_frequency=50e6
)
```

### Behavioral Simulation

```python
from spintron_nn.simulation import BehavioralSimulator

# Fast functional simulation
behav_sim = BehavioralSimulator(spintronic_model)

# Run inference simulation
for input_batch in test_loader:
    spintronic_output = behav_sim.forward(input_batch)
    pytorch_output = original_model(input_batch)
    
    # Compare outputs
    error = torch.abs(spintronic_output - pytorch_output).mean()
    assert error < 0.01, f"Simulation mismatch: {error}"

# Generate waveforms
behav_sim.dump_waveforms("simulation.vcd")
```

## Benchmarking

### Performance Metrics

```python
from spintron_nn.benchmarks import SpintronicBenchmark

benchmark = SpintronicBenchmark()

# Compare against CMOS baseline
results = benchmark.compare_implementations(
    spintronic_model,
    cmos_baseline,
    metrics=['energy', 'latency', 'area', 'accuracy']
)

# Generate report
benchmark.generate_report(
    results,
    output_format='latex',
    include_plots=True
)
```

### Standard Models

The toolkit includes pre-optimized spintronic versions of common models:

```python
from spintron_nn.models import (
    KeywordSpottingNet_Spintronic,
    TinyConvNet_Spintronic,
    MicroNet_Spintronic
)

# 10-keyword spotting model
kws_model = KeywordSpottingNet_Spintronic(
    num_keywords=10,
    mtj_bits=2,
    crossbar_size=64
)

# Tiny vision model
vision_model = TinyConvNet_Spintronic(
    input_shape=(32, 32, 1),
    num_classes=10,
    filters=[16, 32, 64],
    mtj_array_size=128
)
```

## Advanced Features

### Hybrid CMOS-Spintronic Designs

```python
from spintron_nn.hybrid import HybridDesigner

# Partition model between CMOS and spintronic
designer = HybridDesigner()
hybrid_model = designer.partition(
    model,
    spintronic_layers=['conv', 'fc'],
    cmos_layers=['batchnorm', 'activation']
)

# Co-optimize hybrid system
hybrid_model.co_optimize(
    spintronic_constraints=mtj_config,
    cmos_technology='28nm',
    interface_overhead=True
)
```

### Fault Tolerance

```python
from spintron_nn.reliability import FaultTolerantDesign

# Add redundancy for critical applications
ft_design = FaultTolerantDesign(spintronic_model)

ft_design.add_redundancy(
    redundancy_type='TMR',  # Triple Modular Redundancy
    critical_layers=['fc_final'],
    voter_implementation='spintronic'
)

# Analyze reliability
reliability = ft_design.calculate_mttf(
    failure_rate_per_mtj=1e-9,  # FIT
    operating_hours=87600  # 10 years
)
```

## Deployment

### FPGA Prototyping

```python
from spintron_nn.deployment import FPGAPrototype

# Generate FPGA prototype with MTJ emulation
fpga_proto = FPGAPrototype(
    board='zcu104',
    emulation_accuracy='cycle_accurate'
)

fpga_design = fpga_proto.generate(
    spintronic_model,
    include_debug_ports=True,
    mtj_emulation_method='lookup_table'
)

# Generate programming files
fpga_proto.build(output_dir='fpga_build/')
```

### ASIC Tape-out Package

```python
from spintron_nn.deployment import TapeoutPackage

# Prepare for fabrication
tapeout = TapeoutPackage(spintronic_model)

# Generate all required files
tapeout.generate(
    foundry='tsmc',
    process='28nm',
    include_files=[
        'gds',
        'lef',
        'lib',
        'spice_netlist',
        'timing_constraints',
        'power_intent'
    ]
)

# Run final DRC/LVS checks
tapeout.run_physical_verification()
```

## Examples

### Keyword Spotting on MTJ Crossbar

```python
# Complete example: "Hey Assistant" wake word detection
import torchaudio
from spintron_nn.models import WakeWordDetector_Spintronic

# Load pre-trained model
detector = WakeWordDetector_Spintronic.from_pretrained('hey_assistant')

# Process audio
waveform, sample_rate = torchaudio.load('audio_sample.wav')
mfcc = torchaudio.transforms.MFCC(sample_rate)(waveform)

# Run spintronic inference
detection = detector(mfcc)
print(f"Wake word detected: {detection.probability:.2%}")

# Analyze power consumption
power = detector.estimate_power_consumption(
    duty_cycle=0.1,  # 10% active
    voltage=0.8
)
print(f"Average power: {power:.2f} μW")
```

### Always-On Vision Sensor

```python
from spintron_nn.models import AlwaysOnVision_Spintronic

# Ultra-low power person detection
vision_model = AlwaysOnVision_Spintronic(
    resolution=(64, 64),
    detect_classes=['person', 'vehicle'],
    target_power_uw=100  # 100 μW budget
)

# Deploy to edge device
vision_model.optimize_for_deployment(
    batch_size=1,
    latency_constraint_ms=100
)

# Generate complete edge AI system
edge_system = vision_model.generate_system_design(
    include_image_sensor_interface=True,
    include_spi_output=True
)
```

## Citation

If you use SpinTron-NN-Kit in your research, please cite:

```bibtex
@software{spintron_nn_kit,
  title={SpinTron-NN-Kit: Neural Inference on Spin-Orbit-Torque Hardware},
  author={Daniel Schmidt},
  year={2025},
  url={https://github.com/danieleschmidt/spintron-nn-kit}
}
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- NIST Spintronics Group for MTJ device models
- PyTorch team for the excellent deep learning framework
- The neuromorphic computing community

## Contact

- GitHub Issues: [https://github.com/yourusername/spintron-nn-kit/issues](https://github.com/yourusername/spintron-nn-kit/issues)
- Email: spintron-nn@example.com
- Documentation: [https://spintron-nn-kit.readthedocs.io](https://spintron-nn-kit.readthedocs.io)
