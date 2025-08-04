"""
MTJ Crossbar Array Simulation.

This module implements comprehensive crossbar array modeling including:
- Vector-matrix multiplication using MTJ conductances
- Device variations and non-idealities
- Peripheral circuit modeling
- Power and timing analysis
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import warnings

from .mtj_models import MTJDevice, MTJConfig, DomainWallDevice


@dataclass
class CrossbarConfig:
    """Configuration for MTJ crossbar arrays."""
    
    rows: int = 128
    cols: int = 128
    mtj_config: MTJConfig = None
    
    # Peripheral circuit parameters
    read_voltage: float = 0.1      # Read voltage (V)
    write_voltage: float = 0.5     # Write voltage (V)
    wire_resistance: float = 10.0  # Wire resistance per cell (Ohm)
    
    # Sense amplifier parameters
    sense_amplifier_gain: float = 1000.0
    sense_amplifier_offset: float = 1e-6  # Offset current (A)
    
    # Timing parameters
    read_time: float = 10e-9       # Read access time (s)
    write_time: float = 100e-9     # Write access time (s)
    
    # Non-idealities
    enable_variations: bool = True
    enable_wire_resistance: bool = True
    enable_sneak_paths: bool = True
    
    def __post_init__(self):
        if self.mtj_config is None:
            self.mtj_config = MTJConfig()


class MTJCrossbar:
    """
    MTJ crossbar array for in-memory computing.
    
    Implements physics-based simulation of MTJ crossbar arrays including:
    - Accurate device modeling with variations
    - Wire resistance and parasitic effects
    - Peripheral circuit simulation
    - Power and energy analysis
    """
    
    def __init__(self, config: CrossbarConfig):
        self.config = config
        self.rows = config.rows
        self.cols = config.cols
        
        # Initialize MTJ devices
        self.devices = []
        for i in range(self.rows):
            row = []
            for j in range(self.cols):
                device = MTJDevice(config.mtj_config)
                row.append(device)
            self.devices.append(row)
        
        # Wire resistance matrices
        if config.enable_wire_resistance:
            self._init_wire_resistance()
        
        # Performance counters
        self.read_count = 0
        self.write_count = 0
        self.total_energy = 0.0
    
    def _init_wire_resistance(self):
        """Initialize wire resistance matrices."""
        # Row wire resistances (word lines)
        self.row_resistances = np.random.normal(
            self.config.wire_resistance,
            self.config.wire_resistance * 0.1,
            (self.rows, self.cols)
        )
        
        # Column wire resistances (bit lines)
        self.col_resistances = np.random.normal(
            self.config.wire_resistance,
            self.config.wire_resistance * 0.1,
            (self.rows, self.cols)
        )
    
    def set_weights(self, weights: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """
        Program crossbar with weight matrix.
        
        Args:
            weights: Weight matrix to program (rows x cols)
            
        Returns:
            Actual programmed conductances
        """
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        
        if weights.shape != (self.rows, self.cols):
            raise ValueError(f"Weight shape {weights.shape} doesn't match crossbar size ({self.rows}, {self.cols})")
        
        # Map weights to resistance values
        conductances = np.zeros((self.rows, self.cols))
        
        # Find weight range for mapping
        w_min, w_max = weights.min(), weights.max()
        if w_max == w_min:
            w_min, w_max = -1.0, 1.0  # Default range
        
        for i in range(self.rows):
            for j in range(self.cols):
                weight = weights[i, j]
                device = self.devices[i][j]
                
                # Map weight to resistance
                if hasattr(device, 'map_weight'):
                    # Use domain wall device if available
                    resistance = device.map_weight(weight, (w_min, w_max))
                else:
                    # Simple binary mapping for regular MTJ
                    if weight > (w_min + w_max) / 2:
                        device._state = 0  # Low resistance
                    else:
                        device._state = 1  # High resistance
                    resistance = device.resistance
                
                conductances[i, j] = 1.0 / resistance
                self.write_count += 1
        
        return conductances
    
    def get_conductances(self) -> np.ndarray:
        """Get current conductance matrix."""
        conductances = np.zeros((self.rows, self.cols))
        
        for i in range(self.rows):
            for j in range(self.cols):
                conductances[i, j] = self.devices[i][j].conductance
        
        return conductances
    
    def compute_vmm(
        self, 
        input_voltages: Union[np.ndarray, torch.Tensor],
        include_nonidealities: bool = True
    ) -> np.ndarray:
        """
        Compute vector-matrix multiplication using crossbar.
        
        Args:
            input_voltages: Input voltage vector (length = rows)
            include_nonidealities: Include wire resistance and variations
            
        Returns:
            Output currents (length = cols)
        """
        if isinstance(input_voltages, torch.Tensor):
            input_voltages = input_voltages.detach().cpu().numpy()
        
        if len(input_voltages) != self.rows:
            raise ValueError(f"Input length {len(input_voltages)} doesn't match rows {self.rows}")
        
        # Get conductance matrix
        conductances = self.get_conductances()
        
        if include_nonidealities and self.config.enable_wire_resistance:
            # Include wire resistance effects
            output_currents = self._compute_vmm_with_wire_resistance(
                input_voltages, conductances
            )
        else:
            # Ideal computation
            output_currents = np.dot(conductances.T, input_voltages)
        
        # Add sense amplifier effects
        if include_nonidealities:
            output_currents = self._apply_sense_amplifier(output_currents)
        
        self.read_count += 1
        return output_currents
    
    def _compute_vmm_with_wire_resistance(
        self, 
        input_voltages: np.ndarray,
        conductances: np.ndarray
    ) -> np.ndarray:
        """Compute VMM including wire resistance effects."""
        output_currents = np.zeros(self.cols)
        
        for j in range(self.cols):  # For each output column
            column_current = 0.0
            
            for i in range(self.rows):  # For each input row
                if abs(input_voltages[i]) < 1e-12:
                    continue
                
                # Effective resistance includes wire resistance
                mtj_resistance = 1.0 / conductances[i, j]
                wire_r = self.row_resistances[i, j] + self.col_resistances[i, j]
                total_resistance = mtj_resistance + wire_r
                
                # Current through this path
                current = input_voltages[i] / total_resistance
                column_current += current
            
            output_currents[j] = column_current
        
        return output_currents
    
    def _apply_sense_amplifier(self, currents: np.ndarray) -> np.ndarray:
        """Apply sense amplifier characteristics."""
        # Add offset current
        currents_with_offset = currents + self.config.sense_amplifier_offset
        
        # Apply gain
        amplified_currents = currents_with_offset * self.config.sense_amplifier_gain
        
        # Add noise (simplified model)
        noise_std = np.abs(amplified_currents) * 0.01  # 1% noise
        noise = np.random.normal(0, noise_std)
        
        return amplified_currents + noise
    
    def analog_read(
        self, 
        row_indices: Union[int, List[int], np.ndarray],
        col_indices: Union[int, List[int], np.ndarray]
    ) -> Union[float, np.ndarray]:
        """
        Read specific cells in analog mode.
        
        Args:
            row_indices: Row index/indices to read
            col_indices: Column index/indices to read
            
        Returns:
            Current values for specified cells
        """
        if isinstance(row_indices, int):
            row_indices = [row_indices]
        if isinstance(col_indices, int):
            col_indices = [col_indices]
        
        currents = []
        read_voltage = self.config.read_voltage
        
        for i in row_indices:
            for j in col_indices:
                if i >= self.rows or j >= self.cols:
                    raise IndexError(f"Index ({i}, {j}) out of bounds for ({self.rows}, {self.cols})")
                
                device = self.devices[i][j]
                current = device.read_current(read_voltage)
                
                # Add wire resistance effect
                if self.config.enable_wire_resistance:
                    wire_r = self.row_resistances[i, j] + self.col_resistances[i, j]
                    effective_voltage = read_voltage * (device.resistance / (device.resistance + wire_r))
                    current = effective_voltage / device.resistance
                
                currents.append(current)
        
        self.read_count += len(row_indices) * len(col_indices)
        
        if len(currents) == 1:
            return currents[0]
        return np.array(currents)
    
    def write_cell(
        self, 
        row: int, 
        col: int, 
        target_state: int,
        write_voltage: Optional[float] = None
    ) -> bool:
        """
        Write to specific cell.
        
        Args:
            row: Row index
            col: Column index
            target_state: Target resistance state (0=low, 1=high)
            write_voltage: Write voltage (uses config default if None)
            
        Returns:
            True if write successful
        """
        if row >= self.rows or col >= self.cols:
            raise IndexError(f"Index ({row}, {col}) out of bounds")
        
        if write_voltage is None:
            write_voltage = self.config.write_voltage
        
        device = self.devices[row][col]
        current_state = device._state
        
        if current_state == target_state:
            return True  # Already in correct state
        
        # Apply write voltage with correct polarity
        voltage = write_voltage if target_state == 1 else -write_voltage
        success = device.switch(voltage, self.config.write_time)
        
        self.write_count += 1
        
        # Calculate energy consumption
        write_current = voltage / device.resistance
        energy = abs(voltage * write_current * self.config.write_time)
        self.total_energy += energy
        
        return success
    
    def measure_sneak_current(self, target_row: int, target_col: int) -> float:
        """
        Measure sneak path current for given target cell.
        
        Args:
            target_row: Row of target cell
            target_col: Column of target cell
            
        Returns:
            Sneak current magnitude
        """
        if not self.config.enable_sneak_paths:
            return 0.0
        
        read_voltage = self.config.read_voltage
        sneak_current = 0.0
        
        # Sneak paths through other cells in same row/column
        for i in range(self.rows):
            if i != target_row:
                # Sneak through (i, target_col) -> (target_row, target_col)
                device1 = self.devices[i][target_col]
                device2 = self.devices[target_row][i]  # Approximate path
                
                # Simplified sneak current calculation
                total_r = device1.resistance + device2.resistance
                sneak_current += read_voltage / total_r * 0.1  # Reduced by path impedance
        
        return sneak_current
    
    def power_analysis(self, workload: Dict) -> Dict:
        """
        Analyze power consumption for given workload.
        
        Args:
            workload: Dictionary with operation counts and patterns
            
        Returns:
            Power analysis results
        """
        results = {
            'static_power': 0.0,
            'dynamic_power': 0.0,
            'energy_per_read': 0.0,
            'energy_per_write': 0.0,
            'total_energy': self.total_energy
        }
        
        # Static power (leakage)
        leakage_per_cell = 1e-12  # 1 pW per cell
        results['static_power'] = self.rows * self.cols * leakage_per_cell
        
        # Dynamic power
        if 'reads_per_second' in workload:
            read_energy = self.config.read_voltage**2 / (10e3) * self.config.read_time  # Simplified
            results['energy_per_read'] = read_energy
            results['dynamic_power'] += workload['reads_per_second'] * read_energy
        
        if 'writes_per_second' in workload:
            write_energy = self.config.write_voltage**2 / (5e3) * self.config.write_time
            results['energy_per_write'] = write_energy
            results['dynamic_power'] += workload['writes_per_second'] * write_energy
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get crossbar usage and performance statistics."""
        conductances = self.get_conductances()
        
        return {
            'dimensions': (self.rows, self.cols),
            'total_cells': self.rows * self.cols,
            'read_operations': self.read_count,
            'write_operations': self.write_count,
            'total_energy_j': self.total_energy,
            'average_conductance': float(np.mean(conductances)),
            'conductance_std': float(np.std(conductances)),
            'min_conductance': float(np.min(conductances)),
            'max_conductance': float(np.max(conductances))
        }
    
    def reset_statistics(self):
        """Reset performance counters."""
        self.read_count = 0
        self.write_count = 0
        self.total_energy = 0.0