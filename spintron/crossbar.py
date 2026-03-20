"""
SOT Crossbar Array for analog vector-matrix multiplication (VMM).

Architecture:
- N rows × M columns of SOT MTJ devices
- Input voltages applied row-wise
- Column currents summed (Kirchhoff's current law) → output
- Weights encoded as conductance values (P=high G, AP=low G)

Weight encoding:
- Positive weight → parallel state  (high conductance)
- Negative weight → antiparallel state (low conductance)
  (differential pair encoding can be added for signed weights)
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

import numpy as np

from .sot_device import SOTDevice


class CrossbarArray:
    """
    NxM crossbar array of SOT MTJ devices.

    Args:
        rows (int): Number of input rows.
        cols (int): Number of output columns.
    """

    # Input voltage applied during VMM read (V)
    V_READ = 0.1

    def __init__(self, rows: int, cols: int):
        if rows <= 0 or cols <= 0:
            raise ValueError("rows and cols must be positive integers")
        self.rows = rows
        self.cols = cols
        # 2D list of SOTDevice objects, shape [rows][cols]
        self._devices: List[List[SOTDevice]] = [
            [SOTDevice(state="parallel") for _ in range(cols)]
            for _ in range(rows)
        ]

    # ------------------------------------------------------------------
    # Programming
    # ------------------------------------------------------------------

    def program_weights(self, weight_matrix: np.ndarray) -> None:
        """
        Program the crossbar to represent a weight matrix.

        Weights are binarised: positive (≥ 0) → parallel (high G),
        negative (< 0) → antiparallel (low G).

        Args:
            weight_matrix: NumPy array of shape (rows, cols).
        """
        wm = np.asarray(weight_matrix, dtype=float)
        if wm.shape != (self.rows, self.cols):
            raise ValueError(
                f"weight_matrix shape {wm.shape} does not match "
                f"crossbar ({self.rows}, {self.cols})"
            )
        for r in range(self.rows):
            for c in range(self.cols):
                state = "parallel" if wm[r, c] >= 0 else "antiparallel"
                self._devices[r][c].set_state(state)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def vmm(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Perform analog vector-matrix multiplication.

        The input vector is applied as voltages on the rows.
        Column currents are summed: I_j = Σ_i (V_i * G_ij).

        Args:
            input_vector: 1-D array of length `rows`.

        Returns:
            1-D NumPy array of length `cols` (column currents in µA).
        """
        v = np.asarray(input_vector, dtype=float)
        if v.shape != (self.rows,):
            raise ValueError(
                f"input_vector length {len(v)} does not match rows {self.rows}"
            )

        # Build conductance matrix [rows x cols]
        G = self._conductance_matrix()

        # Analog VMM: I = V @ G  (shape: cols)
        currents = v @ G  # µA (when V is scaled to read voltage)
        return currents

    def energy_pj(self, input_vector: np.ndarray) -> float:
        """
        Estimate energy (pJ) for one VMM read operation.

        Energy = Σ_{i,j} V_read^2 / R_ij * t_read
        where t_read is assumed 1 ns per cell access.

        Args:
            input_vector: 1-D array of length `rows` (values 0 or 1).

        Returns:
            Energy in picojoules.
        """
        v = np.asarray(input_vector, dtype=float)
        T_READ_NS = 1.0  # ns
        T_READ_S = T_READ_NS * 1e-9

        total_energy_j = 0.0
        for r in range(self.rows):
            if abs(v[r]) < 1e-12:
                continue
            v_applied = v[r] * self.V_READ
            for c in range(self.cols):
                g = self._devices[r][c].conductance()
                # P = V^2 * G, E = P * t
                total_energy_j += v_applied ** 2 * g * T_READ_S

        return total_energy_j * 1e12  # convert to pJ

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _conductance_matrix(self) -> np.ndarray:
        """Return the conductance matrix G of shape (rows, cols)."""
        G = np.zeros((self.rows, self.cols), dtype=float)
        for r in range(self.rows):
            for c in range(self.cols):
                G[r, c] = self._devices[r][c].conductance()
        return G

    def device(self, row: int, col: int) -> SOTDevice:
        """Access individual device at (row, col)."""
        return self._devices[row][col]

    def conductance_matrix(self) -> np.ndarray:
        """Public accessor for the conductance matrix."""
        return self._conductance_matrix()

    def __repr__(self) -> str:
        return f"CrossbarArray(rows={self.rows}, cols={self.cols})"
