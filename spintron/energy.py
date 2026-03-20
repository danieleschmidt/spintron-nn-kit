"""
Energy estimation for SOT crossbar neural inference.

Reference values:
- MTJ switching energy: ~10 pJ per event (write)
- MTJ read energy:      ~0.1 pJ per access (non-destructive)

For a fully-connected layer mapped to an SOT crossbar:
- Programming (write) energy dominates at inference-time amortised cost
- Each MAC corresponds to one read operation + conductance accumulation
"""

from __future__ import annotations

import numpy as np


class EnergyEstimator:
    """
    Estimate inference energy for SOT-crossbar-mapped neural layers.

    Class Attributes:
        MTJ_SWITCH_ENERGY_PJ: Energy per MTJ state switch (pJ).
        READ_ENERGY_PJ:        Energy per single MTJ read (pJ).
    """

    MTJ_SWITCH_ENERGY_PJ: float = 10.0   # pJ per switching event
    READ_ENERGY_PJ: float = 0.1           # pJ per read

    # ------------------------------------------------------------------
    # Core estimators
    # ------------------------------------------------------------------

    def mac_energy_pj(self, n_macs: int) -> float:
        """
        Estimate energy for n multiply-accumulate operations.

        Each MAC = one MTJ read.

        Args:
            n_macs: Number of MAC operations.

        Returns:
            Energy in picojoules.
        """
        if n_macs < 0:
            raise ValueError("n_macs must be non-negative")
        return n_macs * self.READ_ENERGY_PJ

    def layer_energy_pj(self, weight_matrix: np.ndarray) -> float:
        """
        Estimate total inference energy for a weight matrix layer.

        Accounts for:
        - Read energy for every non-zero weight during a single forward pass
        - No switching energy (weights are fixed after programming)

        Args:
            weight_matrix: 2D NumPy array of shape (in_features, out_features).

        Returns:
            Energy in picojoules for one forward pass.
        """
        wm = np.asarray(weight_matrix, dtype=float)
        if wm.ndim != 2:
            raise ValueError("weight_matrix must be 2-dimensional")
        n_macs = wm.shape[0] * wm.shape[1]
        return self.mac_energy_pj(n_macs)

    def programming_energy_pj(self, weight_matrix: np.ndarray) -> float:
        """
        Estimate one-time programming energy for a weight matrix.

        Each device that needs to be switched (set to AP state) costs
        MTJ_SWITCH_ENERGY_PJ. Devices left in P state cost nothing.

        Args:
            weight_matrix: 2D array — negative values map to AP (switched).

        Returns:
            Total programming energy in pJ.
        """
        wm = np.asarray(weight_matrix, dtype=float)
        if wm.ndim != 2:
            raise ValueError("weight_matrix must be 2-dimensional")
        n_switches = int(np.sum(wm < 0))
        return n_switches * self.MTJ_SWITCH_ENERGY_PJ

    def total_energy_pj(
        self,
        weight_matrix: np.ndarray,
        n_inferences: int = 1,
    ) -> float:
        """
        Total amortised energy = programming + n_inferences × read energy.

        Args:
            weight_matrix:  2D weight array.
            n_inferences:   Number of inference passes to amortise over.

        Returns:
            Total energy in pJ.
        """
        prog = self.programming_energy_pj(weight_matrix)
        read = self.layer_energy_pj(weight_matrix) * n_inferences
        return prog + read

    def __repr__(self) -> str:
        return (
            f"EnergyEstimator("
            f"MTJ_SWITCH_ENERGY_PJ={self.MTJ_SWITCH_ENERGY_PJ}, "
            f"READ_ENERGY_PJ={self.READ_ENERGY_PJ})"
        )
