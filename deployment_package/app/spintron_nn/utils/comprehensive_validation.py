"""
Comprehensive validation system for SpinTron-NN-Kit operations.
"""

import math
import time
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum

from .robust_error_handling import SpintronError, ErrorSeverity


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metrics: Dict[str, float]
    validation_time: float


class ComprehensiveValidator:
    """Comprehensive validation for all SpinTron operations."""
    
    def __init__(self, validation_level: ValidationLevel = ValidationLevel.STANDARD):
        self.validation_level = validation_level
        self.validation_history: List[ValidationResult] = []
        
    def validate_mtj_parameters(self, 
                               resistance_high: float,
                               resistance_low: float,
                               switching_voltage: float,
                               cell_area: float) -> ValidationResult:
        """Validate MTJ device parameters."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}
        
        # Basic physics constraints
        if resistance_high <= resistance_low:
            errors.append("High resistance must be greater than low resistance")
            
        resistance_ratio = resistance_high / resistance_low if resistance_low > 0 else float('inf')
        if resistance_ratio < 1.5:
            warnings.append(f"Low resistance ratio ({resistance_ratio:.2f}), may affect readout margin")
        elif resistance_ratio > 10:
            warnings.append(f"Very high resistance ratio ({resistance_ratio:.2f}), may affect switching")
            
        metrics['resistance_ratio'] = resistance_ratio
        
        # Switching voltage validation
        if switching_voltage <= 0:
            errors.append("Switching voltage must be positive")
        elif switching_voltage > 1.0:
            warnings.append(f"High switching voltage ({switching_voltage}V), may increase power consumption")
            
        # Cell area validation
        if cell_area <= 0:
            errors.append("Cell area must be positive")
        elif cell_area < 10e-18:  # 10 nm²
            warnings.append("Very small cell area, may have reliability issues")
        elif cell_area > 1e-12:   # 1 μm²
            warnings.append("Large cell area, may limit density")
            
        metrics['cell_area_nm2'] = cell_area * 1e18
        
        # Advanced validations for higher levels
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            # Thermal stability validation
            thermal_barrier = resistance_ratio * 0.026  # kT at room temperature
            if thermal_barrier < 40:
                warnings.append(f"Low thermal stability factor ({thermal_barrier:.1f})")
            metrics['thermal_stability'] = thermal_barrier
            
        # Paranoid level checks
        if self.validation_level == ValidationLevel.PARANOID:
            # Manufacturing tolerances
            if resistance_ratio > 3 and switching_voltage < 0.2:
                warnings.append("Parameter combination may be challenging to manufacture")
                
        validation_time = time.time() - start_time
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_time=validation_time
        )
        
        self.validation_history.append(result)
        return result
        
    def validate_crossbar_configuration(self,
                                      rows: int,
                                      cols: int,
                                      mtj_parameters: Dict[str, float]) -> ValidationResult:
        """Validate crossbar array configuration."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}
        
        # Size constraints
        if rows <= 0 or cols <= 0:
            errors.append("Crossbar dimensions must be positive")
            
        total_devices = rows * cols
        metrics['total_devices'] = total_devices
        
        if total_devices > 1000000:  # 1M devices
            warnings.append(f"Very large crossbar ({total_devices} devices), may have yield issues")
        elif total_devices < 64:
            warnings.append("Small crossbar may not be efficient")
            
        # Power analysis
        if mtj_parameters:
            switching_power = (mtj_parameters.get('switching_voltage', 0.3) ** 2 / 
                             mtj_parameters.get('resistance_low', 5000))
            total_switching_power = switching_power * total_devices
            metrics['peak_switching_power_w'] = total_switching_power
            
            if total_switching_power > 1.0:  # 1W
                warnings.append(f"High peak switching power ({total_switching_power:.2f}W)")
                
        # Parasitic effects validation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            wire_resistance = 0.1 * max(rows, cols)  # Simplified model
            if wire_resistance > 100:  # 100Ω
                warnings.append(f"High wire resistance ({wire_resistance:.1f}Ω) may affect performance")
            metrics['estimated_wire_resistance_ohm'] = wire_resistance
            
        validation_time = time.time() - start_time
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_time=validation_time
        )
        
        self.validation_history.append(result)
        return result
        
    def validate_neural_network_mapping(self,
                                      layer_sizes: List[int],
                                      crossbar_size: Tuple[int, int],
                                      quantization_bits: int) -> ValidationResult:
        """Validate neural network to crossbar mapping."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}
        
        # Basic constraints
        if not layer_sizes:
            errors.append("Layer sizes cannot be empty")
            return ValidationResult(False, errors, warnings, metrics, time.time() - start_time)
            
        if any(size <= 0 for size in layer_sizes):
            errors.append("All layer sizes must be positive")
            
        if quantization_bits < 1 or quantization_bits > 8:
            errors.append("Quantization bits must be between 1 and 8")
            
        # Mapping feasibility
        max_layer_size = max(layer_sizes)
        crossbar_capacity = min(crossbar_size)
        
        if max_layer_size > crossbar_capacity:
            errors.append(f"Layer size {max_layer_size} exceeds crossbar capacity {crossbar_capacity}")
            
        # Utilization analysis
        total_weights = sum(layer_sizes[i] * layer_sizes[i+1] for i in range(len(layer_sizes)-1))
        crossbar_devices = crossbar_size[0] * crossbar_size[1]
        utilization = total_weights / crossbar_devices if crossbar_devices > 0 else 0
        
        metrics['weight_utilization'] = utilization
        metrics['total_weights'] = total_weights
        
        if utilization < 0.1:
            warnings.append(f"Low crossbar utilization ({utilization:.1%})")
        elif utilization > 0.9:
            warnings.append(f"High crossbar utilization ({utilization:.1%}), may need partitioning")
            
        # Quantization analysis
        weight_levels = 2 ** quantization_bits
        metrics['weight_levels'] = weight_levels
        
        if quantization_bits < 4:
            warnings.append(f"Low quantization ({quantization_bits} bits) may affect accuracy")
            
        # Performance estimation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            estimated_latency_us = len(layer_sizes) * 10  # 10 μs per layer (simplified)
            estimated_energy_nj = total_weights * 10e-3   # 10 pJ per MAC
            
            metrics['estimated_latency_us'] = estimated_latency_us
            metrics['estimated_energy_nj'] = estimated_energy_nj
            
            if estimated_latency_us > 1000:  # 1 ms
                warnings.append(f"High estimated latency ({estimated_latency_us} μs)")
                
        validation_time = time.time() - start_time
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_time=validation_time
        )
        
        self.validation_history.append(result)
        return result
        
    def validate_verilog_generation(self,
                                  module_name: str,
                                  target_frequency: float,
                                  design_constraints: Dict[str, Any]) -> ValidationResult:
        """Validate Verilog generation parameters."""
        start_time = time.time()
        errors = []
        warnings = []
        metrics = {}
        
        # Module name validation
        if not module_name or not module_name.replace('_', '').isalnum():
            errors.append("Invalid module name, must be alphanumeric with underscores")
            
        # Frequency validation
        if target_frequency <= 0:
            errors.append("Target frequency must be positive")
        elif target_frequency > 1e9:  # 1 GHz
            warnings.append(f"Very high frequency ({target_frequency/1e6:.1f} MHz), may be challenging")
        elif target_frequency < 1e3:  # 1 kHz
            warnings.append(f"Very low frequency ({target_frequency:.1f} Hz)")
            
        metrics['target_frequency_mhz'] = target_frequency / 1e6
        
        # Design constraints validation
        required_constraints = ['max_area', 'io_voltage', 'core_voltage']
        for constraint in required_constraints:
            if constraint not in design_constraints:
                warnings.append(f"Missing design constraint: {constraint}")
                
        # Voltage validation
        if 'io_voltage' in design_constraints:
            io_voltage = design_constraints['io_voltage']
            if io_voltage < 0.8 or io_voltage > 3.3:
                warnings.append(f"Unusual I/O voltage ({io_voltage}V)")
                
        if 'core_voltage' in design_constraints:
            core_voltage = design_constraints['core_voltage']
            if core_voltage < 0.5 or core_voltage > 1.8:
                warnings.append(f"Unusual core voltage ({core_voltage}V)")
                
        # Power estimation
        if self.validation_level in [ValidationLevel.STRICT, ValidationLevel.PARANOID]:
            estimated_power_mw = target_frequency / 1e6 * 10  # Simplified estimation
            metrics['estimated_power_mw'] = estimated_power_mw
            
            if estimated_power_mw > 100:  # 100 mW
                warnings.append(f"High estimated power ({estimated_power_mw:.1f} mW)")
                
        validation_time = time.time() - start_time
        result = ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metrics=metrics,
            validation_time=validation_time
        )
        
        self.validation_history.append(result)
        return result
        
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation operations."""
        if not self.validation_history:
            return {'total_validations': 0}
            
        total_validations = len(self.validation_history)
        passed_validations = sum(1 for result in self.validation_history if result.is_valid)
        total_errors = sum(len(result.errors) for result in self.validation_history)
        total_warnings = sum(len(result.warnings) for result in self.validation_history)
        avg_validation_time = sum(result.validation_time for result in self.validation_history) / total_validations
        
        return {
            'total_validations': total_validations,
            'passed_validations': passed_validations,
            'success_rate': passed_validations / total_validations,
            'total_errors': total_errors,
            'total_warnings': total_warnings,
            'average_validation_time_ms': avg_validation_time * 1000,
            'validation_level': self.validation_level.value
        }


def validate_input_data(data: Any, expected_type: type, name: str = "input") -> None:
    """Validate input data type and basic constraints."""
    if not isinstance(data, expected_type):
        raise SpintronError(
            f"{name} must be of type {expected_type.__name__}, got {type(data).__name__}",
            severity=ErrorSeverity.HIGH
        )
        
    if expected_type in [int, float] and data <= 0:
        raise SpintronError(
            f"{name} must be positive, got {data}",
            severity=ErrorSeverity.MEDIUM
        )


def validate_range(value: Union[int, float], min_val: float, max_val: float, name: str) -> None:
    """Validate that value is within specified range."""
    if not (min_val <= value <= max_val):
        raise SpintronError(
            f"{name} must be between {min_val} and {max_val}, got {value}",
            severity=ErrorSeverity.MEDIUM
        )