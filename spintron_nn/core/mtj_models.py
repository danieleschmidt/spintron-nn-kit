"""
Magnetic Tunnel Junction (MTJ) device physics models.

This module implements comprehensive MTJ device models including:
- TMR characteristics and resistance states
- Thermal stability and retention
- Process variations and aging effects
- Multi-level cell domain wall devices
"""

import math
import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union
from enum import Enum


class MTJType(Enum):
    """Types of MTJ devices."""
    PERPENDICULAR = "perpendicular"
    IN_PLANE = "in_plane"
    VOLTAGE_CONTROLLED = "voltage_controlled"


@dataclass
class MTJConfig:
    """Configuration parameters for MTJ devices."""
    
    # Physical parameters
    resistance_high: float = 10e3  # High resistance state (Ohm)
    resistance_low: float = 5e3    # Low resistance state (Ohm)
    switching_voltage: float = 0.3  # Switching voltage (V)
    cell_area: float = 40e-9       # Cell area (m²)
    
    # Material properties
    mtj_type: MTJType = MTJType.PERPENDICULAR
    reference_layer: str = "CoFeB"
    barrier: str = "MgO"
    free_layer: str = "CoFeB"
    
    # Thermal properties
    thermal_stability: float = 60.0  # kT
    retention_time: float = 10.0     # years
    operating_temp: float = 25.0     # Celsius
    
    # Variations
    resistance_variation: float = 0.1   # 10% std deviation
    switching_variation: float = 0.05   # 5% std deviation
    
    @property
    def tmr_ratio(self) -> float:
        """Tunnel Magnetoresistance ratio."""
        return (self.resistance_high - self.resistance_low) / self.resistance_low
    
    @property
    def switching_current(self) -> float:
        """Critical switching current (A)."""
        return self.switching_voltage / self.resistance_low
    
    @property
    def switching_energy(self) -> float:
        """Energy per switching event (J)."""
        return 0.5 * self.switching_current * self.switching_voltage * 1e-9  # ~pJ
    

class MTJDevice:
    """
    Comprehensive MTJ device model with physics-based behavior.
    
    Models:
    - Resistance states and switching
    - Thermal stability and retention  
    - Process variations
    - Temperature dependence
    """
    
    def __init__(self, config: MTJConfig):
        self.config = config
        self._state = 0  # 0 = low resistance, 1 = high resistance
        self._last_switch_time = 0.0
        self._write_count = 0
        
        # Generate device-specific variations
        self._resistance_variation = np.random.normal(1.0, config.resistance_variation)
        self._switching_variation = np.random.normal(1.0, config.switching_variation)
    
    @property
    def resistance(self) -> float:
        """Current resistance including variations and temperature effects."""
        base_resistance = (
            self.config.resistance_high if self._state == 1 
            else self.config.resistance_low
        )
        
        # Apply device variation
        resistance = base_resistance * self._resistance_variation
        
        # Temperature coefficient (typical -0.1%/°C for MTJ)
        temp_coeff = -0.001 * (self.config.operating_temp - 25.0)
        resistance *= (1.0 + temp_coeff)
        
        return resistance
    
    @property 
    def conductance(self) -> float:
        """Current conductance (1/resistance)."""
        return 1.0 / self.resistance
    
    def switch(self, voltage: float, pulse_width: float = 1e-9) -> bool:
        """
        Attempt to switch MTJ state with given voltage and pulse width.
        
        Args:
            voltage: Applied voltage (V)
            pulse_width: Pulse duration (s)
            
        Returns:
            True if switching occurred
        """
        # Voltage-dependent switching probability
        switching_voltage = self.config.switching_voltage * self._switching_variation
        
        if abs(voltage) < switching_voltage:
            return False
        
        # Switching probability with voltage and pulse width dependence
        voltage_factor = abs(voltage) / switching_voltage
        time_factor = pulse_width / 1e-9  # Normalize to 1ns
        
        switch_probability = 1.0 - math.exp(-voltage_factor * time_factor * 0.1)
        
        if np.random.random() < switch_probability:
            # Switch state based on voltage polarity
            self._state = 1 if voltage > 0 else 0
            self._write_count += 1
            return True
            
        return False
    
    def read_current(self, read_voltage: float = 0.1) -> float:
        """
        Calculate read current for given read voltage.
        
        Args:
            read_voltage: Read voltage (V)
            
        Returns:
            Read current (A)
        """
        return read_voltage / self.resistance
    
    def retention_time(self, temperature: float = None) -> float:
        """
        Calculate data retention time at given temperature.
        
        Args:
            temperature: Temperature in Celsius (uses config default if None)
            
        Returns:
            Retention time in seconds
        """
        if temperature is None:
            temperature = self.config.operating_temp
            
        # Arrhenius equation for thermal activation
        kb = 1.380649e-23  # Boltzmann constant
        temp_kelvin = temperature + 273.15
        
        # Thermal stability factor (Delta = KV/kBT)
        delta = self.config.thermal_stability * (25 + 273.15) / temp_kelvin
        
        # Retention time (seconds)
        tau0 = 1e-9  # Attempt frequency ~1 GHz
        return tau0 * math.exp(delta)
    
    def endurance_degradation(self) -> float:
        """
        Calculate endurance-based resistance degradation.
        
        Returns:
            Degradation factor (1.0 = no degradation)
        """
        # Typical MTJ endurance ~10^12 cycles
        max_cycles = 1e12
        degradation_rate = 0.1  # 10% degradation at end of life
        
        cycle_fraction = min(self._write_count / max_cycles, 1.0)
        return 1.0 - degradation_rate * cycle_fraction
    
    def power_consumption(self, voltage: float, duration: float) -> float:
        """
        Calculate power consumption for operation.
        
        Args:
            voltage: Applied voltage (V)
            duration: Operation duration (s)
            
        Returns:
            Energy consumption (J)
        """
        current = voltage / self.resistance
        power = voltage * current
        return power * duration


class DomainWallDevice:
    """
    Domain wall device for multi-level cell storage.
    
    Enables 2-4 bit precision per cell using magnetic domain walls.
    """
    
    def __init__(
        self, 
        track_length: float = 200e-9,
        domain_width: float = 20e-9, 
        levels: int = 4
    ):
        self.track_length = track_length
        self.domain_width = domain_width
        self.levels = levels
        self.max_domains = int(track_length / domain_width)
        
        # Current domain configuration
        self._domain_positions = []
        self._current_level = 0
        
    @property
    def resistance_levels(self) -> np.ndarray:
        """Available resistance levels."""
        r_min = 5e3   # Minimum resistance
        r_max = 20e3  # Maximum resistance
        return np.linspace(r_min, r_max, self.levels)
    
    @property
    def current_resistance(self) -> float:
        """Current resistance based on domain configuration."""
        return self.resistance_levels[self._current_level]
    
    def set_level(self, level: int) -> bool:
        """
        Set device to specific resistance level.
        
        Args:
            level: Target resistance level (0 to levels-1)
            
        Returns:
            True if successful
        """
        if 0 <= level < self.levels:
            self._current_level = level
            # Update domain positions based on level
            domains_needed = int(level * self.max_domains / self.levels)
            self._domain_positions = list(range(domains_needed))
            return True
        return False
    
    def program_analog(self, target_resistance: float) -> float:
        """
        Program device to target resistance value.
        
        Args:
            target_resistance: Desired resistance (Ohm)
            
        Returns:
            Actual achieved resistance
        """
        # Find closest level
        levels = self.resistance_levels
        differences = np.abs(levels - target_resistance)
        closest_level = np.argmin(differences)
        
        self.set_level(closest_level)
        return levels[closest_level]
    
    def map_weight(self, weight: float, weight_range: Tuple[float, float]) -> float:
        """
        Map neural network weight to resistance value.
        
        Args:
            weight: Neural network weight
            weight_range: (min_weight, max_weight) tuple
            
        Returns:
            Mapped resistance value
        """
        min_weight, max_weight = weight_range
        
        # Normalize weight to [0, 1]
        normalized = (weight - min_weight) / (max_weight - min_weight)
        normalized = np.clip(normalized, 0.0, 1.0)
        
        # Map to resistance range
        min_r, max_r = self.resistance_levels[0], self.resistance_levels[-1]
        target_resistance = min_r + normalized * (max_r - min_r)
        
        return self.program_analog(target_resistance)
    
    def switching_energy(self, from_level: int, to_level: int) -> float:
        """
        Calculate energy required to switch between levels.
        
        Args:
            from_level: Starting level
            to_level: Target level
            
        Returns:
            Switching energy (J)
        """
        level_change = abs(to_level - from_level)
        # Energy proportional to number of domains to move
        energy_per_domain = 1e-15  # ~fJ per domain wall movement
        return level_change * energy_per_domain * self.max_domains / self.levels


def estimate_switching_energy(
    weight_changes: torch.Tensor, 
    mtj_config: MTJConfig
) -> torch.Tensor:
    """
    Estimate switching energy for weight updates.
    
    Args:
        weight_changes: Tensor of weight changes
        mtj_config: MTJ configuration
        
    Returns:
        Estimated switching energy per weight
    """
    # Only weights that change significantly require switching
    switching_threshold = 0.1  # 10% change threshold
    switching_mask = torch.abs(weight_changes) > switching_threshold
    
    base_energy = mtj_config.switching_energy
    energy = torch.where(
        switching_mask,
        torch.tensor(base_energy),
        torch.tensor(0.0)
    )
    
    return energy