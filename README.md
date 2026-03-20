# spintron-nn-kit 🧲⚡

**Ultra-low-power neural inference via Spin-Orbit Torque (SOT) hardware + Verilog generation.**

Maps neural network weight matrices onto physical MTJ crossbar arrays, estimates picojoule-level inference energy, and generates synthesizable Verilog RTL for FPGA/ASIC integration.

---

## Physics Background

Spin-Orbit Torque (SOT) devices use a heavy-metal/ferromagnet stack to switch a Magnetic Tunnel Junction (MTJ) via the spin Hall effect. This eliminates current flow through the tunnel barrier during write, giving:

- **~10× lower write endurance degradation** vs STT-MRAM
- **~10 pJ per switching event** at 1 ns pulse widths
- **~0.1 pJ per read** (non-destructive)

The MTJ has two resistance states:
| State | Resistance | Conductance | Meaning |
|---|---|---|---|
| Parallel (P) | 1 kΩ | 1.0 mS | Low-R / weight = 1 |
| Anti-Parallel (AP) | 2 kΩ | 0.5 mS | High-R / weight = 0 |

TMR ≈ 100%, thermal stability Δ = Eb/kT ≈ 40, critical current Ic0 ≈ 100 µA.

---

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.9 and NumPy.

---

## Quick Start

```python
import numpy as np
from spintron import CrossbarArray, EnergyEstimator, VerilogGenerator

# 1. Create a 4×4 crossbar and program weights
cb = CrossbarArray(rows=4, cols=4)
weights = np.array([[ 1, -1,  1, -1],
                    [-1,  1, -1,  1],
                    [ 1,  1, -1, -1],
                    [-1, -1,  1,  1]], dtype=float)
cb.program_weights(weights)

# 2. Analog vector-matrix multiply
output = cb.vmm(np.array([1.0, 0.5, -0.5, 1.0]))

# 3. Estimate energy
est = EnergyEstimator()
print(f"Inference: {est.layer_energy_pj(weights):.2f} pJ")
print(f"Programming: {est.programming_energy_pj(weights):.2f} pJ")

# 4. Generate Verilog RTL
vgen = VerilogGenerator(cb, data_width=8)
vgen.save("sot_crossbar.v")
```

Run the full demo:
```bash
python demo.py
```

---

## API Reference

### `SOTDevice`

Single MTJ device with Neel-Brown stochastic switching model.

```python
dev = SOTDevice(state='parallel')   # or 'antiparallel'
dev.conductance()                    # → float (S)
dev.switching_probability(current_ua=80.0, pulse_ns=1.0)  # → float [0,1]
dev.switch(current_ua=120.0)         # → bool (did it switch?)
```

### `CrossbarArray`

NxM array of SOTDevices for analog VMM.

```python
cb = CrossbarArray(rows=N, cols=M)
cb.program_weights(weight_matrix)   # maps ±weights → P/AP states
cb.vmm(input_vector)                # → output currents (np.ndarray)
cb.energy_pj(input_vector)          # → pJ for this VMM pass
cb.conductance_matrix()             # → np.ndarray (N×M)
```

### `VerilogGenerator`

Synthesizable Verilog RTL emitter.

```python
vgen = VerilogGenerator(crossbar, data_width=8)
verilog_str = vgen.generate_module("my_crossbar")
vgen.save("my_crossbar.v")
```

### `EnergyEstimator`

Picojoule-level energy breakdown.

```python
est = EnergyEstimator()
est.mac_energy_pj(n_macs=64)             # → 6.4 pJ
est.layer_energy_pj(weight_matrix)       # → pJ for one forward pass
est.programming_energy_pj(weight_matrix) # → pJ for one-time write
est.total_energy_pj(weight_matrix, n_inferences=1000)
```

---

## Verilog Output

The generated module has:
- `clk` / `rst_n` — synchronous design with active-low reset
- `in_vec[0:ROWS-1]` — `DATA_W`-bit input activations
- `out_vec[0:COLS-1]` — `ACC_W`-bit accumulated outputs
- Hardcoded weight registers initialised from the programmed crossbar state

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Energy Comparison

| Technology | Energy/MAC |
|---|---|
| CMOS GPU (A100) | ~0.5 nJ |
| CMOS ASIC (TPU) | ~10–50 pJ |
| **SOT Crossbar (this kit)** | **~0.1 pJ** |
| Ideal limit | ~0.01 pJ |

SOT crossbars achieve near-ideal efficiency by replacing digital multipliers with analog conductance modulation.

---

## License

MIT
