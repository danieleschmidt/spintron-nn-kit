"""
Spin-Orbit Torque (SOT) physics calculations.

This module implements the physical models for spin-orbit torque effects
used in spintronic devices for efficient switching.
"""

import math
import numpy as np
import torch
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass 
class SOTParameters:
    """Physical parameters for SOT calculations."""
    
    # Material parameters
    spin_hall_angle: float = 0.3        # Spin Hall angle
    gilbert_damping: float = 0.02       # Gilbert damping parameter
    gyromagnetic_ratio: float = 2.21e5  # Gyromagnetic ratio (m/A/s)
    
    # Geometric parameters
    free_layer_thickness: float = 1.5e-9  # Free layer thickness (m)
    heavy_metal_thickness: float = 5e-9   # Heavy metal layer thickness (m)
    
    # Magnetic parameters
    saturation_magnetization: float = 1.2e6  # Saturation magnetization (A/m)
    magnetic_anisotropy: float = 8e5         # Effective anisotropy field (A/m)
    
    # Interface parameters
    interface_transparency: float = 0.8   # Spin transparency at interface
    
    
class SOTCalculator:
    """
    Calculator for spin-orbit torque effects in magnetic tunnel junctions.
    
    Implements physics-based models for:
    - Critical switching current
    - Switching probability and dynamics
    - Energy efficiency optimization
    """
    
    def __init__(self, parameters: SOTParameters):
        self.params = parameters
        
    def critical_current(
        self, 
        thermal_stability: float = 60.0,
        pulse_width: float = 1e-9
    ) -> float:
        """
        Calculate critical switching current for SOT switching.
        
        Args:
            thermal_stability: Thermal stability factor (kT units)
            pulse_width: Current pulse width (s)
            
        Returns:
            Critical current (A)
        """
        # Physical constants
        kb = 1.380649e-23  # Boltzmann constant
        mu0 = 4 * math.pi * 1e-7  # Permeability of free space
        hbar = 1.054571817e-34  # Reduced Planck constant
        e = 1.602176634e-19  # Elementary charge
        
        # Calculate critical current density
        # Based on macrospin model for SOT switching
        
        # Volume of free layer
        area = math.pi * (20e-9)**2  # Assume circular MTJ with 40nm diameter
        volume = area * self.params.free_layer_thickness
        
        # Thermal energy
        kbt = kb * 300  # Room temperature
        
        # Critical current density (A/m²)
        j_critical = (
            2 * e * self.params.gilbert_damping * 
            self.params.saturation_magnetization * 
            self.params.free_layer_thickness * 
            thermal_stability * kbt
        ) / (
            hbar * self.params.spin_hall_angle * 
            self.params.interface_transparency
        )
        
        # Account for pulse width dependence
        if pulse_width < 1e-9:
            # For short pulses, higher current needed
            pulse_factor = (1e-9 / pulse_width) ** 0.5
            j_critical *= pulse_factor
        
        # Total critical current
        i_critical = j_critical * area
        
        return i_critical
    
    def switching_probability(
        self,
        applied_current: float,
        pulse_width: float = 1e-9,
        temperature: float = 300.0,
        thermal_stability: float = 60.0
    ) -> float:
        """
        Calculate switching probability for given current and conditions.
        
        Args:
            applied_current: Applied switching current (A)
            pulse_width: Current pulse width (s)
            temperature: Operating temperature (K)
            thermal_stability: Thermal stability factor
            
        Returns:
            Switching probability (0-1)
        """
        i_critical = self.critical_current(thermal_stability, pulse_width)
        
        # Current ratio
        current_ratio = applied_current / i_critical
        
        if current_ratio < 0.5:
            return 0.0  # Below threshold
        
        # Switching probability model
        # Based on thermally activated switching with SOT assistance
        kb = 1.380649e-23
        
        # Effective energy barrier reduction due to SOT
        barrier_reduction = min(current_ratio - 0.5, thermal_stability * 0.8)
        effective_barrier = thermal_stability - barrier_reduction
        
        # Switching rate (1/s)
        attempt_frequency = 1e9  # ~GHz
        switching_rate = attempt_frequency * math.exp(-effective_barrier)
        
        # Switching probability for given pulse width
        probability = 1.0 - math.exp(-switching_rate * pulse_width)
        
        return min(probability, 1.0)
    
    def switching_delay(
        self,
        applied_current: float, 
        thermal_stability: float = 60.0
    ) -> float:
        """
        Calculate average switching delay.
        
        Args:
            applied_current: Applied current (A)
            thermal_stability: Thermal stability factor
            
        Returns:
            Average switching delay (s)
        """
        i_critical = self.critical_current(thermal_stability)
        current_ratio = applied_current / i_critical
        
        if current_ratio < 1.0:
            return float('inf')  # No switching
        
        # Delay inversely proportional to overdrive
        base_delay = 1e-9  # 1 ns base delay
        delay = base_delay / (current_ratio - 1.0 + 0.1)
        
        return delay
    
    def switching_energy(
        self,
        switching_current: float,
        switching_voltage: float,
        pulse_width: float = 1e-9
    ) -> float:
        """
        Calculate energy consumption for switching operation.
        
        Args:
            switching_current: Current during switching (A)
            switching_voltage: Voltage during switching (V) 
            pulse_width: Switching pulse width (s)
            
        Returns:
            Switching energy (J)
        """
        # Power during switching
        power = switching_current * switching_voltage
        
        # Total energy
        energy = power * pulse_width
        
        return energy
    
    def optimize_switching_conditions(
        self,
        target_probability: float = 0.99,
        max_energy: float = 1e-15,  # 1 fJ
        thermal_stability: float = 60.0
    ) -> Tuple[float, float, float]:
        """
        Optimize switching current, voltage, and pulse width for target conditions.
        
        Args:
            target_probability: Target switching probability
            max_energy: Maximum allowed energy (J)
            thermal_stability: Thermal stability factor
            
        Returns:
            Tuple of (current, voltage, pulse_width)
        """
        # Start with minimum conditions
        pulse_widths = np.logspace(-12, -9, 100)  # 1 ps to 1 ns
        
        best_conditions = None
        min_energy = float('inf')
        
        for pulse_width in pulse_widths:
            i_critical = self.critical_current(thermal_stability, pulse_width)
            
            # Binary search for required current
            i_min, i_max = i_critical, i_critical * 5
            
            for _ in range(20):  # Binary search iterations
                i_test = (i_min + i_max) / 2
                prob = self.switching_probability(
                    i_test, pulse_width, 300.0, thermal_stability
                )
                
                if prob < target_probability:
                    i_min = i_test
                else:
                    i_max = i_test
            
            # Use found current
            switching_current = i_max
            
            # Estimate switching voltage (simplified model)
            switching_voltage = switching_current * 100  # ~100 Ohm effective resistance
            
            # Calculate energy
            energy = self.switching_energy(switching_current, switching_voltage, pulse_width)
            
            if energy <= max_energy and energy < min_energy:
                min_energy = energy
                best_conditions = (switching_current, switching_voltage, pulse_width)
        
        if best_conditions is None:
            # Fallback to minimum energy solution
            pulse_width = 1e-9
            switching_current = self.critical_current(thermal_stability, pulse_width) * 2
            switching_voltage = switching_current * 100
            best_conditions = (switching_current, switching_voltage, pulse_width)
        
        return best_conditions
    
    def temperature_dependence(
        self,
        temperature_range: Tuple[float, float] = (223, 398)  # -50°C to 125°C in K
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate temperature dependence of switching parameters.
        
        Args:
            temperature_range: (min_temp, max_temp) in Kelvin
            
        Returns:
            Tuple of (temperatures, critical_currents)
        """
        temperatures = np.linspace(temperature_range[0], temperature_range[1], 50)
        critical_currents = []
        
        for temp in temperatures:
            # Temperature affects thermal stability
            thermal_stability = 60.0 * (300.0 / temp)  # Scale with temperature
            i_crit = self.critical_current(thermal_stability)
            critical_currents.append(i_crit)
        
        return temperatures, np.array(critical_currents)
    
    def process_variation_analysis(
        self,
        variation_params: dict,
        num_samples: int = 1000
    ) -> dict:
        """
        Analyze impact of process variations on SOT switching.
        
        Args:
            variation_params: Dictionary of parameter variations (std dev)
            num_samples: Number of Monte Carlo samples
            
        Returns:
            Dictionary with variation statistics
        """
        results = {
            'critical_currents': [],
            'switching_probabilities': [],
            'switching_energies': []
        }
        
        for _ in range(num_samples):
            # Generate varied parameters
            varied_params = SOTParameters()
            
            if 'spin_hall_angle' in variation_params:
                varied_params.spin_hall_angle = np.random.normal(
                    self.params.spin_hall_angle,
                    variation_params['spin_hall_angle']
                )
            
            if 'gilbert_damping' in variation_params:
                varied_params.gilbert_damping = np.random.normal(
                    self.params.gilbert_damping,
                    variation_params['gilbert_damping']
                )
            
            # Create varied calculator
            varied_calc = SOTCalculator(varied_params)
            
            # Calculate metrics
            i_crit = varied_calc.critical_current()
            prob = varied_calc.switching_probability(i_crit * 1.5)
            energy = varied_calc.switching_energy(i_crit * 1.5, i_crit * 150, 1e-9)
            
            results['critical_currents'].append(i_crit)
            results['switching_probabilities'].append(prob)
            results['switching_energies'].append(energy)
        
        # Calculate statistics
        for key in results:
            values = np.array(results[key])
            results[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        return results