#!/usr/bin/env python3
"""
spintron-nn-kit demo: Map a 4x4 FC layer to an SOT crossbar.

Demonstrates:
1. Programming a CrossbarArray with a weight matrix
2. Running vector-matrix multiplication via conductance modulation
3. Estimating energy consumption
4. Generating synthesizable Verilog RTL
"""

import numpy as np
from spintron import CrossbarArray, EnergyEstimator, VerilogGenerator

# -----------------------------------------------------------------------
# 1. Define a small fully-connected layer (4 inputs → 4 outputs)
# -----------------------------------------------------------------------
print("=" * 60)
print("spintron-nn-kit Demo: 4x4 SOT Crossbar")
print("=" * 60)

weight_matrix = np.array([
    [ 1.0, -1.0,  0.5, -0.5],
    [-0.3,  0.8,  1.0,  0.2],
    [ 0.7, -0.9, -0.4,  1.0],
    [ 0.1,  0.6, -0.7,  0.9],
], dtype=float)

print("\n[1] Weight matrix (4x4):")
print(weight_matrix)

# -----------------------------------------------------------------------
# 2. Create and program the crossbar
# -----------------------------------------------------------------------
crossbar = CrossbarArray(rows=4, cols=4)
crossbar.program_weights(weight_matrix)

print(f"\n[2] Crossbar: {crossbar}")
print("    Conductance matrix (mS):")
G = crossbar.conductance_matrix() * 1e3
print(np.round(G, 3))

# -----------------------------------------------------------------------
# 3. Run a VMM inference pass
# -----------------------------------------------------------------------
input_vec = np.array([1.0, 0.5, -0.5, 1.0])
print(f"\n[3] Input vector: {input_vec}")
output = crossbar.vmm(input_vec)
print(f"    VMM output (column currents): {np.round(output, 6)}")

# Compare with software VMM
sw_output = input_vec @ weight_matrix
print(f"    Software reference:           {np.round(sw_output * 1e-3, 6)}")
print("    (Crossbar output is analog current; SW output is weight-scale)")

# -----------------------------------------------------------------------
# 4. Energy estimation
# -----------------------------------------------------------------------
estimator = EnergyEstimator()
read_energy = estimator.layer_energy_pj(weight_matrix)
prog_energy = estimator.programming_energy_pj(weight_matrix)
total_1k    = estimator.total_energy_pj(weight_matrix, n_inferences=1000)

print(f"\n[4] Energy estimates:")
print(f"    Single inference (read):   {read_energy:.2f} pJ")
print(f"    One-time programming:      {prog_energy:.2f} pJ")
print(f"    Total (1000 inferences):   {total_1k:.2f} pJ  ({total_1k/1000:.4f} pJ/inference avg)")

# Per-crossbar read energy via crossbar.energy_pj
xbar_energy = crossbar.energy_pj(np.abs(input_vec))
print(f"    Crossbar analog read energy (this input): {xbar_energy:.4f} pJ")

# -----------------------------------------------------------------------
# 5. Generate Verilog
# -----------------------------------------------------------------------
vgen = VerilogGenerator(crossbar, data_width=8)
verilog = vgen.generate_module("sot_crossbar_4x4")

output_path = "/tmp/spintron-nn-kit/sot_crossbar_4x4.v"
vgen.save(output_path)

print(f"\n[5] Verilog RTL generated → {output_path}")
print("    Module preview (first 15 lines):")
for line in verilog.splitlines()[:15]:
    print(f"    {line}")

# -----------------------------------------------------------------------
# 6. Summary
# -----------------------------------------------------------------------
print("\n" + "=" * 60)
print("Summary:")
print(f"  Crossbar size:        {crossbar.rows} × {crossbar.cols} MTJ devices")
print(f"  Total devices:        {crossbar.rows * crossbar.cols}")
print(f"  Parallel (weight≥0):  {sum(1 for r in range(4) for c in range(4) if weight_matrix[r,c] >= 0)}")
print(f"  AntiParallel (w<0):   {sum(1 for r in range(4) for c in range(4) if weight_matrix[r,c] < 0)}")
print(f"  Read energy/MAC:      {EnergyEstimator.READ_ENERGY_PJ} pJ")
print(f"  Switch energy/MTJ:    {EnergyEstimator.MTJ_SWITCH_ENERGY_PJ} pJ")
print(f"  Verilog module:       sot_crossbar_4x4")
print("=" * 60)
