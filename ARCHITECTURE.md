# SpinTron-NN-Kit Architecture

## System Overview

SpinTron-NN-Kit provides an end-to-end flow from PyTorch neural networks to spintronic hardware implementations using magnetic tunnel junction (MTJ) crossbars. The system achieves picojoule-level energy efficiency for edge AI applications.

## High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   PyTorch       │───▶│  SpinTron-NN     │───▶│   Hardware      │
│   Model         │    │  Converter       │    │   Generation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  Quantization &  │
                       │  Optimization    │
                       └──────────────────┘
                                │
                                ▼
                       ┌──────────────────┐
                       │  MTJ Crossbar    │
                       │  Mapping         │
                       └──────────────────┘
```

## Core Components

### 1. Model Converter (`spintron_nn/converter/`)

**Purpose**: Transforms PyTorch models into spintronic-compatible representations.

**Key Modules**:
- `pytorch_parser.py`: Analyzes PyTorch model structure and extracts computation graph
- `graph_optimizer.py`: Optimizes computation graphs for spintronic execution
- `mapping.py`: Maps neural operations to MTJ crossbar operations

**Data Flow**:
```
PyTorch Model → Graph Analysis → Optimization → Spintronic Mapping
```

### 2. MTJ Device Physics (`spintron_nn/core/`)

**Purpose**: Models the physical behavior of magnetic tunnel junctions and crossbar arrays.

**Key Modules**:
- `mtj_models.py`: MTJ resistance, switching, and retention modeling
- `crossbar.py`: Crossbar array simulation with realistic device variations
- `sot_physics.py`: Spin-orbit torque calculations for switching operations

**Physical Models**:
- TMR (Tunnel Magnetoresistance) characteristics
- Thermal stability and retention
- Process variations and aging effects

### 3. Hardware Generation (`spintron_nn/hardware/`)

**Purpose**: Generates synthesizable Verilog for ASIC/FPGA implementation.

**Key Modules**:
- `verilog_gen.py`: Hierarchical Verilog code generation
- `templates/`: Parameterizable hardware module templates
- `constraints.py`: Timing, area, and power constraints

**Generation Flow**:
```
Spintronic Model → Module Selection → Parameter Configuration → Verilog Output
```

### 4. Training Framework (`spintron_nn/training/`)

**Purpose**: Specialized training techniques for spintronic constraints.

**Key Modules**:
- `qat.py`: Quantization-aware training with spintronic constraints
- `variation_aware.py`: Training robust to device variations
- `energy_opt.py`: Energy-aware training optimization

### 5. Simulation Engine (`spintron_nn/simulation/`)

**Purpose**: Multi-level simulation from behavioral to circuit-level accuracy.

**Simulation Hierarchy**:
1. **Behavioral**: Fast functional simulation
2. **Device-Aware**: Includes MTJ variations and non-idealities
3. **Circuit-Level**: SPICE co-simulation for critical paths

## Data Flow Architecture

### Forward Inference Path

```
Input Data
    │
    ▼
┌─────────────────┐
│  Input Buffer   │
│  & ADC          │
└─────────────────┘
    │
    ▼
┌─────────────────┐    ┌─────────────────┐
│  Weight Memory  │───▶│  MTJ Crossbar   │
│  (MTJ Array)    │    │  Computation    │
└─────────────────┘    └─────────────────┘
    │                           │
    ▼                           ▼
┌─────────────────┐    ┌─────────────────┐
│  Activation     │    │  Current        │
│  Functions      │    │  Summation      │
└─────────────────┘    └─────────────────┘
    │                           │
    └───────────┬───────────────┘
                ▼
    ┌─────────────────┐
    │  Output Buffer  │
    │  & Processing   │
    └─────────────────┘
```

### Training/Programming Path

```
Weight Updates
    │
    ▼
┌─────────────────┐
│  Quantization   │
│  & Mapping      │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  Write Pulse    │
│  Generation     │
└─────────────────┘
    │
    ▼
┌─────────────────┐
│  MTJ State      │
│  Programming    │
└─────────────────┘
```

## MTJ Crossbar Architecture

### Physical Organization

```
     Word Lines (Rows)
         │ │ │ │
    ─────┼─┼─┼─┼──── Bit Line 0
         │ │ │ │
    ─────┼─┼─┼─┼──── Bit Line 1
         │ │ │ │
    ─────┼─┼─┼─┼──── Bit Line 2
         │ │ │ │
         MTJ MTJ MTJ
```

### Electrical Model

Each MTJ junction acts as a voltage-controlled resistor:
- **High Resistance State (HRS)**: Anti-parallel magnetic alignment
- **Low Resistance State (LRS)**: Parallel magnetic alignment
- **Switching**: Controlled by spin-orbit torque (SOT)

### Multi-Level Cells

Domain wall devices enable multi-bit storage per cell:
```
┌─────────────────────────────┐
│ DW1  │ DW2  │ DW3  │ DW4  │  ← 4 domain walls = 2-bit precision
└─────────────────────────────┘
  00     01     10     11       ← Resistance levels
```

## System Integration

### CMOS Interface

```
┌─────────────────┐    ┌─────────────────┐
│  CMOS Control   │───▶│  Spintronic     │
│  Logic          │    │  Compute Array  │
└─────────────────┘    └─────────────────┘
         │                       │
         ▼                       ▼
┌─────────────────┐    ┌─────────────────┐
│  ADC/DAC        │    │  Sense          │
│  Interfaces     │    │  Amplifiers     │
└─────────────────┘    └─────────────────┘
```

### Power Management

- **Always-On Mode**: Ultra-low static power (~nW)
- **Active Inference**: Optimized dynamic power
- **Programming Mode**: Higher power for weight updates

### Clock Domains

1. **High-Speed Clock**: CMOS control logic (10-100 MHz)
2. **Medium-Speed Clock**: Data movement and ADC (1-10 MHz)
3. **Low-Speed Clock**: Weight programming (kHz range)

## Scalability Architecture

### Hierarchical Design

```
System Level
    │
    ├── Tile Level (Multiple Cores)
    │       │
    │       ├── Core Level (MTJ Arrays + CMOS)
    │       │       │
    │       │       ├── Array Level (128x128 MTJ)
    │       │       │       │
    │       │       │       └── Cell Level (Single MTJ)
    │       │       │
    │       │       └── Peripheral Circuits
    │       │
    │       └── Inter-Core Communication
    │
    └── System-Level Interconnect
```

### Memory Hierarchy

1. **L0**: MTJ crossbar arrays (weights)
2. **L1**: Local SRAM buffers (activations)
3. **L2**: Shared memory (intermediate results)
4. **L3**: External memory interface

## Design Constraints

### Physical Constraints

- **MTJ Size**: 40-100 nm diameter
- **Array Density**: Up to 128x128 per tile
- **Write Current**: 50-200 μA per MTJ
- **Retention**: >10 years at operating temperature

### Performance Constraints

- **Frequency**: 1-50 MHz operation
- **Latency**: <1ms for typical inference
- **Throughput**: 1-100 TOPS/W
- **Accuracy**: Within 1% of floating-point baseline

### Environmental Constraints

- **Temperature**: -40°C to +85°C operation
- **Voltage**: 0.6V to 1.2V supply
- **Process**: 28nm and below CMOS compatibility

## Verification Strategy

### Multi-Level Verification

1. **Unit Level**: Individual module verification
2. **Integration Level**: Interface compatibility
3. **System Level**: End-to-end functionality
4. **Physical Level**: Post-layout simulation

### Coverage Metrics

- **Functional Coverage**: All model types and layer configurations
- **Code Coverage**: >95% line and branch coverage
- **Corner Coverage**: PVT variations and device mismatch
- **Fault Coverage**: Stuck-at and transition faults

## Future Enhancements

### Planned Improvements

1. **3D Integration**: Vertical MTJ stacking
2. **In-Memory Computing**: Enhanced compute-in-memory operations  
3. **Neuromorphic Features**: Spike-based processing
4. **Auto-Optimization**: Automated design space exploration

### Research Directions

- Advanced MTJ materials (e.g., voltage-controlled magnetic anisotropy)
- Probabilistic computing with stochastic MTJs
- Federated learning with privacy-preserving hardware
- Quantum-classical hybrid architectures