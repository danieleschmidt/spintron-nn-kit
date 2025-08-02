# ADR-0001: MTJ Crossbar Architecture for Neural Network Weights

## Status

Accepted

## Context

SpinTron-NN-Kit requires an efficient hardware substrate for storing and computing with neural network weights. Traditional SRAM-based weight storage consumes significant static power and area. Emerging spintronic devices, particularly Magnetic Tunnel Junctions (MTJs), offer non-volatile weight storage with ultra-low static power consumption.

Key requirements:
- Non-volatile weight storage to eliminate static power
- High density for large neural networks
- Compute-in-memory capability for energy efficiency
- Multi-bit precision support (2-4 bits per weight)
- Compatibility with standard CMOS processes

## Decision

We will use MTJ crossbar arrays as the primary weight storage and computation substrate with the following architecture:

1. **MTJ Device Type**: Perpendicular Magnetic Anisotropy (PMA) MTJs with spin-orbit torque (SOT) switching
2. **Array Organization**: 128x128 crossbar tiles with hierarchical interconnect
3. **Multi-Level Cells**: Domain wall devices for 2-4 bit weight precision
4. **Write Mechanism**: Current-induced SOT switching with separate write transistors
5. **Read Mechanism**: Voltage-mode sensing with integrated current-to-voltage conversion

## Consequences

### Positive Consequences
- **Ultra-low static power**: ~nW static power vs. mW for SRAM
- **High density**: 4-10x higher density than SRAM at equivalent technology node
- **Non-volatile**: Instant-on capability and power-off data retention
- **Compute-in-memory**: Vector-matrix multiplication in crossbar arrays
- **Scalability**: Proven scaling to advanced technology nodes

### Negative Consequences
- **Write energy**: Higher energy required for weight updates vs. SRAM
- **Write latency**: Slower weight programming (μs vs. ns)
- **Variability**: Device-to-device variations affect accuracy
- **Limited endurance**: Finite number of write cycles (~10^12)
- **Temperature sensitivity**: Retention time decreases with temperature

### Neutral Consequences
- Requires co-design with CMOS peripheral circuits
- New design methodology needed for MTJ-based systems
- Additional simulation models required for verification

## Alternatives Considered

### SRAM-based Weight Storage
- **Pros**: Fast access, mature technology, high endurance
- **Cons**: High static power, large area, volatile storage
- **Rejected**: Does not meet ultra-low power requirements

### Flash-based Weight Storage
- **Pros**: Non-volatile, high density, mature technology
- **Cons**: High write voltage, slow programming, limited endurance
- **Rejected**: Write characteristics incompatible with training workflows

### ReRAM Crossbars
- **Pros**: High density, fast switching, compute-in-memory
- **Cons**: High variability, retention issues, immature technology
- **Rejected**: Insufficient reliability for neural network applications

### Phase Change Memory (PCM)
- **Pros**: Multi-level storage, fast switching, non-volatile
- **Cons**: High write current, thermal cross-talk, drift
- **Rejected**: Thermal issues incompatible with dense arrays

## Implementation Notes

### Phase 1: Behavioral Modeling
- Develop MTJ device models with variations
- Implement crossbar array simulation framework
- Validate against published experimental data

### Phase 2: Circuit Design
- Design sense amplifiers and write drivers
- Implement peripheral control circuits
- Optimize for power and performance

### Phase 3: System Integration
- Integrate with CMOS logic and memory hierarchy
- Implement system-level power management
- Develop programming and calibration protocols

### Dependencies
- Access to MTJ device models and parameters
- SPICE models for circuit simulation
- Fabrication technology with MTJ integration

## References

- [1] Kent, A.D. & Worledge, D.C. "A new spin on magnetic memories." Nature Nanotechnology (2015)
- [2] Gokmen, T. & Vlasov, Y. "Acceleration of Deep Neural Network Training with Resistive Cross-Point Devices." Frontiers in Neuroscience (2016)
- [3] Dieny, B. et al. "Opportunities and challenges for spintronics in the microelectronics industry." Nature Electronics (2020)

---

**Date**: 2025-08-02  
**Author**: SpinTron-NN-Kit Team  
**Reviewers**: Architecture Review Board