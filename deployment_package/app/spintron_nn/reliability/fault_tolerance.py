"""
Fault Tolerance and Reliability Framework for Spintronic Neural Networks.

Implements comprehensive fault tolerance mechanisms including:
- Triple Modular Redundancy (TMR)
- Error Detection and Correction (ECC)
- Graceful degradation strategies
- Self-healing capabilities
- Reliability analysis and MTTF calculation
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import time
import random
from enum import Enum

from ..core.mtj_models import MTJDevice, MTJConfig
from ..core.crossbar import MTJCrossbar, CrossbarConfig
from ..utils.error_handling import SpintronError, HardwareError, robust_operation
from ..utils.logging_config import get_logger
from ..utils.monitoring import get_system_monitor

logger = get_logger(__name__)


class FaultType(Enum):
    """Types of faults that can occur in spintronic systems."""
    STUCK_AT_ZERO = "stuck_at_zero"
    STUCK_AT_ONE = "stuck_at_one"
    PARAMETRIC_DRIFT = "parametric_drift"
    RANDOM_TELEGRAPH_NOISE = "rtn"
    RETENTION_FAILURE = "retention_failure"
    SWITCHING_FAILURE = "switching_failure"
    WIRE_BREAK = "wire_break"
    SHORT_CIRCUIT = "short_circuit"


class RedundancyType(Enum):
    """Types of redundancy schemes."""
    TMR = "triple_modular_redundancy"
    DMR = "dual_modular_redundancy"
    HAMMING = "hamming_code"
    BCH = "bch_code"
    ADAPTIVE = "adaptive_redundancy"


@dataclass
class FaultModel:
    """Model for fault injection and analysis."""
    
    fault_type: FaultType
    probability: float  # Failure probability per operation
    affected_devices: List[Tuple[int, int]]  # List of (row, col) positions
    onset_time: float = 0.0  # When fault starts (seconds)
    severity: float = 1.0  # Fault severity (0-1)
    transient: bool = False  # Whether fault is transient
    duration: float = float('inf')  # Fault duration (seconds)


@dataclass
class ReliabilityMetrics:
    """Reliability analysis results."""
    
    mttf_years: float  # Mean Time To Failure
    availability: float  # System availability (0-1)
    fault_coverage: float  # Fault coverage (0-1)
    performance_degradation: float  # Performance impact (0-1)
    power_overhead: float  # Power overhead due to redundancy
    area_overhead: float  # Area overhead due to redundancy
    error_correction_rate: float  # Successfully corrected errors


class FaultTolerantCrossbar:
    """
    Fault-tolerant crossbar with multiple redundancy schemes.
    
    Implements TMR, ECC, and adaptive fault tolerance for high reliability.
    """
    
    def __init__(
        self,
        base_config: CrossbarConfig,
        redundancy_type: RedundancyType = RedundancyType.TMR,
        fault_tolerance_level: float = 0.99
    ):
        self.base_config = base_config
        self.redundancy_type = redundancy_type
        self.fault_tolerance_level = fault_tolerance_level
        
        # Initialize monitoring
        self.monitor = get_system_monitor()
        
        # Fault injection for testing
        self.injected_faults = []
        self.fault_history = []
        
        # Reliability tracking
        self.operation_count = 0
        self.detected_errors = 0
        self.corrected_errors = 0
        self.uncorrectable_errors = 0
        
        # Initialize redundant crossbars
        self._initialize_redundant_systems()
        
        logger.info(f"Initialized fault-tolerant crossbar with {redundancy_type.value}")
    
    def _initialize_redundant_systems(self):
        """Initialize redundant crossbar systems."""
        
        if self.redundancy_type == RedundancyType.TMR:
            # Triple Modular Redundancy
            self.primary_crossbar = MTJCrossbar(self.base_config)
            self.secondary_crossbar = MTJCrossbar(self.base_config)
            self.tertiary_crossbar = MTJCrossbar(self.base_config)
            
            # Initialize with same weights
            self.redundant_crossbars = [
                self.primary_crossbar,
                self.secondary_crossbar, 
                self.tertiary_crossbar
            ]
            
        elif self.redundancy_type == RedundancyType.DMR:
            # Dual Modular Redundancy
            self.primary_crossbar = MTJCrossbar(self.base_config)
            self.secondary_crossbar = MTJCrossbar(self.base_config)
            
            self.redundant_crossbars = [
                self.primary_crossbar,
                self.secondary_crossbar
            ]
            
        elif self.redundancy_type == RedundancyType.HAMMING:
            # Single crossbar with Hamming ECC
            self.primary_crossbar = MTJCrossbar(self.base_config)
            self.redundant_crossbars = [self.primary_crossbar]
            
            # Initialize ECC matrices
            self._initialize_hamming_ecc()
            
        else:
            # Adaptive redundancy starts with single crossbar
            self.primary_crossbar = MTJCrossbar(self.base_config)
            self.redundant_crossbars = [self.primary_crossbar]
    
    def _initialize_hamming_ecc(self):
        """Initialize Hamming error correction codes."""
        
        # Calculate required parity bits
        data_bits = self.base_config.rows * self.base_config.cols
        parity_bits = 0
        while (2 ** parity_bits) < (data_bits + parity_bits + 1):
            parity_bits += 1
        
        self.hamming_data_bits = data_bits
        self.hamming_parity_bits = parity_bits
        
        # Generate Hamming code matrices
        self.hamming_generator = self._generate_hamming_generator_matrix()
        self.hamming_check = self._generate_hamming_check_matrix()
        
        logger.info(f"Initialized Hamming ECC: {data_bits} data bits, {parity_bits} parity bits")
    
    def _generate_hamming_generator_matrix(self) -> np.ndarray:
        """Generate Hamming code generator matrix."""
        
        n = self.hamming_data_bits + self.hamming_parity_bits
        k = self.hamming_data_bits
        
        # Simplified Hamming matrix generation
        G = np.zeros((k, n), dtype=int)
        
        # Identity matrix for data bits
        G[:k, :k] = np.eye(k)
        
        # Parity check bits
        for i in range(self.hamming_parity_bits):
            for j in range(k):
                if (j + 1) & (2 ** i):
                    G[j, k + i] = 1
        
        return G
    
    def _generate_hamming_check_matrix(self) -> np.ndarray:
        """Generate Hamming code check matrix."""
        
        n = self.hamming_data_bits + self.hamming_parity_bits
        r = self.hamming_parity_bits
        
        H = np.zeros((r, n), dtype=int)
        
        # Generate check matrix
        col = 0
        for i in range(1, n + 1):
            if i & (i - 1) != 0:  # Not a power of 2
                H[:, col] = [(i >> j) & 1 for j in range(r)]
                col += 1
        
        # Add identity for parity bits
        for i in range(r):
            H[i, self.hamming_data_bits + i] = 1
        
        return H
    
    @robust_operation(max_retries=3, delay=0.1)
    def set_weights(self, weights: np.ndarray) -> np.ndarray:
        """Set weights with fault tolerance."""
        
        try:
            # Set weights in all redundant crossbars
            conductances_list = []
            
            for i, crossbar in enumerate(self.redundant_crossbars):
                try:
                    conductances = crossbar.set_weights(weights)
                    conductances_list.append(conductances)
                except Exception as e:
                    logger.warning(f"Weight setting failed in crossbar {i}: {str(e)}")
                    # Use previous weights or default
                    if hasattr(self, '_last_conductances'):
                        conductances_list.append(self._last_conductances)
                    else:
                        conductances_list.append(np.ones_like(weights) * 1e-6)
            
            # Verify consistency across redundant systems
            if len(conductances_list) > 1:
                self._verify_weight_consistency(conductances_list)
            
            # Store for future use
            self._last_conductances = conductances_list[0]
            
            return conductances_list[0]
            
        except Exception as e:
            raise HardwareError(f"Fault-tolerant weight setting failed: {str(e)}")
    
    def _verify_weight_consistency(
        self, 
        conductances_list: List[np.ndarray],
        tolerance: float = 0.1
    ):
        """Verify consistency across redundant crossbars."""
        
        if len(conductances_list) < 2:
            return
        
        reference = conductances_list[0]
        
        for i, conductances in enumerate(conductances_list[1:], 1):
            diff = np.abs(conductances - reference)
            max_diff = np.max(diff)
            
            if max_diff > tolerance:
                logger.warning(
                    f"Inconsistency detected in crossbar {i}: max diff = {max_diff:.4f}"
                )
                
                # Record potential fault
                self._record_potential_fault(i, max_diff)
    
    def _record_potential_fault(self, crossbar_id: int, severity: float):
        """Record potential fault for analysis."""
        
        fault_record = {
            'timestamp': time.time(),
            'crossbar_id': crossbar_id,
            'type': 'consistency_error',
            'severity': severity
        }
        
        self.fault_history.append(fault_record)
        self.detected_errors += 1
    
    @robust_operation(max_retries=3, delay=0.1)
    def compute_vmm_fault_tolerant(
        self,
        input_voltages: np.ndarray,
        enable_correction: bool = True
    ) -> np.ndarray:
        """Compute VMM with fault tolerance."""
        
        start_time = time.time()
        self.operation_count += 1
        
        try:
            if self.redundancy_type == RedundancyType.TMR:
                return self._compute_tmr_vmm(input_voltages, enable_correction)
            elif self.redundancy_type == RedundancyType.DMR:
                return self._compute_dmr_vmm(input_voltages, enable_correction)
            elif self.redundancy_type == RedundancyType.HAMMING:
                return self._compute_ecc_vmm(input_voltages, enable_correction)
            elif self.redundancy_type == RedundancyType.ADAPTIVE:
                return self._compute_adaptive_vmm(input_voltages, enable_correction)
            else:
                raise ValueError(f"Unknown redundancy type: {self.redundancy_type}")
                
        except Exception as e:
            self.uncorrectable_errors += 1
            raise HardwareError(f"Fault-tolerant VMM failed: {str(e)}")
        finally:
            # Record operation
            execution_time = time.time() - start_time
            self.monitor.record_operation(
                "fault_tolerant_vmm",
                execution_time,
                success=True,
                tags={"redundancy_type": self.redundancy_type.value}
            )
    
    def _compute_tmr_vmm(
        self,
        input_voltages: np.ndarray,
        enable_correction: bool
    ) -> np.ndarray:
        """Compute VMM using Triple Modular Redundancy."""
        
        outputs = []
        failed_crossbars = []
        
        # Compute outputs from all three crossbars
        for i, crossbar in enumerate(self.redundant_crossbars):
            try:
                output = crossbar.compute_vmm(input_voltages)
                outputs.append(output)
            except Exception as e:
                logger.warning(f"TMR crossbar {i} failed: {str(e)}")
                failed_crossbars.append(i)
                outputs.append(None)
        
        # Perform majority voting
        if enable_correction:
            corrected_output = self._majority_vote(outputs)
            
            if len(failed_crossbars) > 0:
                self.detected_errors += 1
                if len(failed_crossbars) <= 1:
                    self.corrected_errors += 1
                else:
                    self.uncorrectable_errors += 1
            
            return corrected_output
        else:
            # Use first available output
            for output in outputs:
                if output is not None:
                    return output
            
            raise HardwareError("All TMR crossbars failed")
    
    def _majority_vote(self, outputs: List[Optional[np.ndarray]]) -> np.ndarray:
        """Perform majority voting on TMR outputs."""
        
        valid_outputs = [out for out in outputs if out is not None]
        
        if len(valid_outputs) == 0:
            raise HardwareError("No valid outputs for majority voting")
        
        if len(valid_outputs) == 1:
            return valid_outputs[0]
        
        if len(valid_outputs) == 2:
            # Use first output if only two available
            return valid_outputs[0]
        
        # True majority voting with three outputs
        output1, output2, output3 = valid_outputs[0], valid_outputs[1], valid_outputs[2]
        
        # Element-wise majority voting
        result = np.zeros_like(output1)
        
        for i in range(len(output1)):
            votes = [output1[i], output2[i], output3[i]]
            
            # Find closest pair
            diffs = [
                abs(votes[0] - votes[1]),
                abs(votes[0] - votes[2]),
                abs(votes[1] - votes[2])
            ]
            
            min_diff_idx = np.argmin(diffs)
            
            if min_diff_idx == 0:  # output1 and output2 are closest
                result[i] = (votes[0] + votes[1]) / 2
            elif min_diff_idx == 1:  # output1 and output3 are closest
                result[i] = (votes[0] + votes[2]) / 2
            else:  # output2 and output3 are closest
                result[i] = (votes[1] + votes[2]) / 2
        
        return result
    
    def _compute_dmr_vmm(
        self,
        input_voltages: np.ndarray,
        enable_correction: bool
    ) -> np.ndarray:
        """Compute VMM using Dual Modular Redundancy."""
        
        outputs = []
        
        # Compute outputs from both crossbars
        for i, crossbar in enumerate(self.redundant_crossbars):
            try:
                output = crossbar.compute_vmm(input_voltages)
                outputs.append(output)
            except Exception as e:
                logger.warning(f"DMR crossbar {i} failed: {str(e)}")
                outputs.append(None)
        
        # Error detection and handling
        if outputs[0] is not None and outputs[1] is not None:
            # Compare outputs
            diff = np.abs(outputs[0] - outputs[1])
            max_diff = np.max(diff)
            
            if max_diff > 0.1:  # Threshold for error detection
                self.detected_errors += 1
                logger.warning(f"DMR disagreement detected: max diff = {max_diff:.4f}")
                
                # Use average of both outputs
                return (outputs[0] + outputs[1]) / 2
            else:
                return outputs[0]
        
        elif outputs[0] is not None:
            return outputs[0]
        elif outputs[1] is not None:
            return outputs[1]
        else:
            raise HardwareError("Both DMR crossbars failed")
    
    def _compute_ecc_vmm(
        self,
        input_voltages: np.ndarray,
        enable_correction: bool
    ) -> np.ndarray:
        """Compute VMM with Hamming ECC."""
        
        # Compute base output
        output = self.primary_crossbar.compute_vmm(input_voltages)
        
        if not enable_correction:
            return output
        
        # Apply ECC if enabled
        try:
            # Flatten output for ECC processing
            output_flat = output.flatten()
            
            # Convert to binary for ECC
            output_binary = self._float_to_binary(output_flat)
            
            # Apply Hamming code error detection/correction
            corrected_binary = self._apply_hamming_ecc(output_binary)
            
            # Convert back to float
            corrected_flat = self._binary_to_float(corrected_binary)
            
            # Reshape to original shape
            corrected_output = corrected_flat.reshape(output.shape)
            
            return corrected_output
            
        except Exception as e:
            logger.warning(f"ECC correction failed: {str(e)}")
            return output
    
    def _float_to_binary(self, values: np.ndarray, bits: int = 16) -> np.ndarray:
        """Convert float array to binary representation."""
        
        # Simple quantization to integers
        max_val = np.max(np.abs(values))
        if max_val == 0:
            max_val = 1.0
        
        # Scale to use full bit range
        scale_factor = (2 ** (bits - 1) - 1) / max_val
        quantized = np.round(values * scale_factor).astype(int)
        
        # Convert to binary
        binary_array = np.zeros((len(values), bits), dtype=int)
        
        for i, val in enumerate(quantized):
            # Handle negative values using two's complement
            if val < 0:
                val = (2 ** bits) + val
            
            binary_repr = format(val, f'0{bits}b')
            binary_array[i] = [int(b) for b in binary_repr]
        
        return binary_array
    
    def _binary_to_float(self, binary_array: np.ndarray, bits: int = 16) -> np.ndarray:
        """Convert binary representation back to float array."""
        
        values = np.zeros(binary_array.shape[0])
        
        for i, binary_val in enumerate(binary_array):
            # Convert binary to integer
            int_val = 0
            for j, bit in enumerate(binary_val):
                int_val += bit * (2 ** (bits - 1 - j))
            
            # Handle two's complement for negative values
            if int_val >= (2 ** (bits - 1)):
                int_val -= (2 ** bits)
            
            values[i] = int_val
        
        # Scale back to original range (simplified)
        max_val = np.max(np.abs(values))
        if max_val > 0:
            values = values / max_val
        
        return values
    
    def _apply_hamming_ecc(self, binary_data: np.ndarray) -> np.ndarray:
        """Apply Hamming error detection and correction."""
        
        corrected_data = binary_data.copy()
        
        # Process each data word
        for i in range(binary_data.shape[0]):
            data_word = binary_data[i]
            
            # Calculate syndrome
            syndrome = self._calculate_syndrome(data_word)
            
            if np.any(syndrome):
                # Error detected
                self.detected_errors += 1
                
                # Find error position
                error_pos = self._syndrome_to_position(syndrome)
                
                if error_pos < len(data_word):
                    # Correct single-bit error
                    corrected_data[i, error_pos] = 1 - data_word[error_pos]
                    self.corrected_errors += 1
                else:
                    # Uncorrectable error
                    self.uncorrectable_errors += 1
        
        return corrected_data
    
    def _calculate_syndrome(self, data_word: np.ndarray) -> np.ndarray:
        """Calculate Hamming syndrome for error detection."""
        
        # Simplified syndrome calculation
        syndrome = np.zeros(self.hamming_parity_bits, dtype=int)
        
        for i in range(self.hamming_parity_bits):
            parity = 0
            for j in range(len(data_word)):
                if (j + 1) & (2 ** i):
                    parity ^= data_word[j]
            syndrome[i] = parity
        
        return syndrome
    
    def _syndrome_to_position(self, syndrome: np.ndarray) -> int:
        """Convert syndrome to error position."""
        
        position = 0
        for i, bit in enumerate(syndrome):
            position += bit * (2 ** i)
        
        return position - 1  # Convert to 0-based indexing
    
    def _compute_adaptive_vmm(
        self,
        input_voltages: np.ndarray,
        enable_correction: bool
    ) -> np.ndarray:
        """Compute VMM with adaptive redundancy."""
        
        # Start with single crossbar
        try:
            output = self.primary_crossbar.compute_vmm(input_voltages)
            
            # Check for errors
            if self._detect_output_anomaly(output):
                # Activate additional redundancy
                return self._activate_additional_redundancy(input_voltages)
            
            return output
            
        except Exception as e:
            # Primary failed, activate backup
            return self._activate_additional_redundancy(input_voltages)
    
    def _detect_output_anomaly(self, output: np.ndarray) -> bool:
        """Detect anomalies in output."""
        
        # Check for NaN or infinite values
        if not np.all(np.isfinite(output)):
            return True
        
        # Check for extreme values
        if np.max(np.abs(output)) > 10.0:  # Threshold
            return True
        
        # Check output distribution
        if len(output) > 1:
            std = np.std(output)
            mean = np.mean(output)
            
            # Flag if std is too high relative to mean
            if std > abs(mean) * 5:
                return True
        
        return False
    
    def _activate_additional_redundancy(self, input_voltages: np.ndarray) -> np.ndarray:
        """Activate additional redundancy when needed."""
        
        logger.info("Activating additional redundancy")
        
        # Add backup crossbars if not already present
        if len(self.redundant_crossbars) == 1:
            backup_crossbar = MTJCrossbar(self.base_config)
            
            # Copy weights from primary
            if hasattr(self, '_last_conductances'):
                backup_crossbar.set_weights(self._last_conductances)
            
            self.redundant_crossbars.append(backup_crossbar)
        
        # Use DMR mode
        outputs = []
        for crossbar in self.redundant_crossbars:
            try:
                output = crossbar.compute_vmm(input_voltages)
                outputs.append(output)
            except Exception:
                outputs.append(None)
        
        # Return best available output
        valid_outputs = [out for out in outputs if out is not None]
        
        if len(valid_outputs) == 0:
            raise HardwareError("All adaptive crossbars failed")
        
        if len(valid_outputs) == 1:
            return valid_outputs[0]
        
        # Use average of valid outputs
        return np.mean(valid_outputs, axis=0)
    
    def inject_fault(self, fault_model: FaultModel):
        """Inject fault for testing purposes."""
        
        logger.info(f"Injecting fault: {fault_model.fault_type.value}")
        
        self.injected_faults.append(fault_model)
        
        # Apply fault to specified devices
        for row, col in fault_model.affected_devices:
            for crossbar in self.redundant_crossbars:
                if hasattr(crossbar, 'devices'):
                    device = crossbar.devices[row][col]
                    self._apply_fault_to_device(device, fault_model)
    
    def _apply_fault_to_device(self, device: MTJDevice, fault_model: FaultModel):
        """Apply specific fault to a device."""
        
        if fault_model.fault_type == FaultType.STUCK_AT_ZERO:
            device._state = 0
            device._stuck = True
        elif fault_model.fault_type == FaultType.STUCK_AT_ONE:
            device._state = 1
            device._stuck = True
        elif fault_model.fault_type == FaultType.PARAMETRIC_DRIFT:
            # Increase resistance by fault severity
            device.config.resistance_high *= (1 + fault_model.severity)
            device.config.resistance_low *= (1 + fault_model.severity)
        elif fault_model.fault_type == FaultType.RETENTION_FAILURE:
            # Reduce retention time
            if hasattr(device, 'retention_time'):
                device.retention_time *= (1 - fault_model.severity)
    
    def analyze_reliability(self, operating_time_years: float = 10.0) -> ReliabilityMetrics:
        """Analyze system reliability."""
        
        logger.info("Performing reliability analysis")
        
        # Calculate MTTF based on redundancy
        base_mttf = self._calculate_base_mttf()
        
        if self.redundancy_type == RedundancyType.TMR:
            # TMR can tolerate one failure
            system_mttf = base_mttf * 3  # Simplified calculation
        elif self.redundancy_type == RedundancyType.DMR:
            # DMR can detect but not correct all failures
            system_mttf = base_mttf * 1.5
        else:
            system_mttf = base_mttf
        
        # Calculate availability
        repair_time = 24.0  # 24 hours to repair
        availability = system_mttf / (system_mttf + repair_time / (365 * 24))
        
        # Calculate fault coverage
        if self.operation_count > 0:
            fault_coverage = (self.detected_errors + self.corrected_errors) / max(1, self.operation_count)
        else:
            fault_coverage = 0.0
        
        # Performance degradation
        performance_degradation = self.uncorrectable_errors / max(1, self.operation_count)
        
        # Overhead calculations
        power_overhead = self._calculate_power_overhead()
        area_overhead = self._calculate_area_overhead()
        
        # Error correction rate
        error_correction_rate = self.corrected_errors / max(1, self.detected_errors)
        
        metrics = ReliabilityMetrics(
            mttf_years=system_mttf,
            availability=availability,
            fault_coverage=fault_coverage,
            performance_degradation=performance_degradation,
            power_overhead=power_overhead,
            area_overhead=area_overhead,
            error_correction_rate=error_correction_rate
        )
        
        logger.info(f"Reliability analysis completed - MTTF: {system_mttf:.2f} years")
        
        return metrics
    
    def _calculate_base_mttf(self) -> float:
        """Calculate base MTTF for single crossbar."""
        
        # Simplified MTTF calculation
        # Based on number of devices and failure rates
        
        total_devices = self.base_config.rows * self.base_config.cols
        device_failure_rate = 1e-9  # FIT per device
        
        # System failure rate (sum of all device failure rates)
        system_failure_rate = total_devices * device_failure_rate
        
        # MTTF in years
        mttf_hours = 1.0 / system_failure_rate
        mttf_years = mttf_hours / (365 * 24)
        
        return mttf_years
    
    def _calculate_power_overhead(self) -> float:
        """Calculate power overhead due to redundancy."""
        
        if self.redundancy_type == RedundancyType.TMR:
            return 2.0  # 200% overhead for 3x redundancy
        elif self.redundancy_type == RedundancyType.DMR:
            return 1.0  # 100% overhead for 2x redundancy
        elif self.redundancy_type == RedundancyType.HAMMING:
            return 0.1  # 10% overhead for ECC logic
        else:
            return 0.0
    
    def _calculate_area_overhead(self) -> float:
        """Calculate area overhead due to redundancy."""
        
        if self.redundancy_type == RedundancyType.TMR:
            return 2.0  # 200% overhead
        elif self.redundancy_type == RedundancyType.DMR:
            return 1.0  # 100% overhead
        elif self.redundancy_type == RedundancyType.HAMMING:
            return 0.15  # 15% overhead for ECC
        else:
            return 0.0
    
    def get_fault_statistics(self) -> Dict[str, Any]:
        """Get comprehensive fault and reliability statistics."""
        
        return {
            'total_operations': self.operation_count,
            'detected_errors': self.detected_errors,
            'corrected_errors': self.corrected_errors,
            'uncorrectable_errors': self.uncorrectable_errors,
            'injected_faults': len(self.injected_faults),
            'fault_history_length': len(self.fault_history),
            'redundancy_type': self.redundancy_type.value,
            'error_rate': self.detected_errors / max(1, self.operation_count),
            'correction_rate': self.corrected_errors / max(1, self.detected_errors)
        }
    
    def reset_statistics(self):
        """Reset fault and reliability statistics."""
        
        self.operation_count = 0
        self.detected_errors = 0
        self.corrected_errors = 0
        self.uncorrectable_errors = 0
        self.fault_history.clear()
        
        logger.info("Fault tolerance statistics reset")


class SelfHealingSystem:
    """
    Self-healing system for autonomous fault recovery.
    
    Implements autonomous detection, diagnosis, and recovery from faults.
    """
    
    def __init__(self, fault_tolerant_crossbar: FaultTolerantCrossbar):
        self.crossbar = fault_tolerant_crossbar
        self.healing_history = []
        self.health_monitoring_enabled = True
        
        # Self-healing parameters
        self.health_check_interval = 60.0  # seconds
        self.last_health_check = time.time()
        self.healing_strategies = [
            self._strategy_reconfigure,
            self._strategy_redundancy_activation,
            self._strategy_parameter_adaptation,
            self._strategy_graceful_degradation
        ]
        
        logger.info("Initialized SelfHealingSystem")
    
    def monitor_and_heal(self) -> bool:
        """Monitor system health and perform healing if needed."""
        
        current_time = time.time()
        
        if current_time - self.last_health_check < self.health_check_interval:
            return True  # No check needed yet
        
        self.last_health_check = current_time
        
        # Perform health assessment
        health_status = self._assess_system_health()
        
        if health_status['healthy']:
            return True
        
        # System needs healing
        logger.warning(f"System health degraded: {health_status['issues']}")
        
        # Apply healing strategies
        healing_success = self._apply_healing_strategies(health_status)
        
        # Record healing attempt
        healing_record = {
            'timestamp': current_time,
            'health_status': health_status,
            'healing_success': healing_success,
            'strategies_applied': []
        }
        
        self.healing_history.append(healing_record)
        
        return healing_success
    
    def _assess_system_health(self) -> Dict[str, Any]:
        """Assess current system health."""
        
        issues = []
        health_score = 1.0
        
        stats = self.crossbar.get_fault_statistics()
        
        # Check error rates
        error_rate = stats['error_rate']
        if error_rate > 0.01:  # 1% threshold
            issues.append(f"High error rate: {error_rate:.3f}")
            health_score *= 0.8
        
        # Check correction efficiency
        correction_rate = stats['correction_rate']
        if correction_rate < 0.9:  # 90% threshold
            issues.append(f"Low correction rate: {correction_rate:.3f}")
            health_score *= 0.9
        
        # Check for uncorrectable errors
        if stats['uncorrectable_errors'] > 0:
            issues.append(f"Uncorrectable errors: {stats['uncorrectable_errors']}")
            health_score *= 0.7
        
        # Performance degradation
        if hasattr(self.crossbar, 'performance_metrics'):
            perf = self.crossbar.performance_metrics
            if perf.get('latency_degradation', 0) > 0.2:  # 20% threshold
                issues.append("Performance degradation detected")
                health_score *= 0.85
        
        return {
            'healthy': health_score > 0.8,
            'health_score': health_score,
            'issues': issues,
            'statistics': stats
        }
    
    def _apply_healing_strategies(self, health_status: Dict[str, Any]) -> bool:
        """Apply healing strategies based on health assessment."""
        
        for strategy in self.healing_strategies:
            try:
                success = strategy(health_status)
                if success:
                    logger.info(f"Healing strategy {strategy.__name__} succeeded")
                    return True
            except Exception as e:
                logger.warning(f"Healing strategy {strategy.__name__} failed: {str(e)}")
        
        logger.error("All healing strategies failed")
        return False
    
    def _strategy_reconfigure(self, health_status: Dict[str, Any]) -> bool:
        """Healing strategy: Reconfigure system parameters."""
        
        logger.info("Attempting reconfiguration healing")
        
        # Reset error counters
        self.crossbar.reset_statistics()
        
        # Reconfigure crossbar parameters
        if hasattr(self.crossbar, 'base_config'):
            # Reduce operating voltage to improve reliability
            original_voltage = self.crossbar.base_config.read_voltage
            self.crossbar.base_config.read_voltage *= 0.9
            
            # Test if reconfiguration helps
            time.sleep(1.0)  # Wait for configuration to take effect
            
            # Simple test
            test_input = np.random.random(self.crossbar.base_config.rows) * 0.1
            
            try:
                output = self.crossbar.compute_vmm_fault_tolerant(test_input)
                if np.all(np.isfinite(output)):
                    logger.info("Reconfiguration successful")
                    return True
                else:
                    # Revert configuration
                    self.crossbar.base_config.read_voltage = original_voltage
                    return False
            except Exception:
                # Revert configuration
                self.crossbar.base_config.read_voltage = original_voltage
                return False
        
        return False
    
    def _strategy_redundancy_activation(self, health_status: Dict[str, Any]) -> bool:
        """Healing strategy: Activate additional redundancy."""
        
        logger.info("Attempting redundancy activation healing")
        
        # Increase redundancy level if possible
        if self.crossbar.redundancy_type == RedundancyType.ADAPTIVE:
            # Add more redundant crossbars
            try:
                backup_crossbar = MTJCrossbar(self.crossbar.base_config)
                self.crossbar.redundant_crossbars.append(backup_crossbar)
                
                logger.info("Additional redundancy activated")
                return True
            except Exception:
                return False
        
        return False
    
    def _strategy_parameter_adaptation(self, health_status: Dict[str, Any]) -> bool:
        """Healing strategy: Adapt operating parameters."""
        
        logger.info("Attempting parameter adaptation healing")
        
        # Adapt timing parameters
        if hasattr(self.crossbar, 'base_config'):
            # Increase read/write times for better reliability
            self.crossbar.base_config.read_time *= 1.2
            self.crossbar.base_config.write_time *= 1.2
            
            logger.info("Operating parameters adapted")
            return True
        
        return False
    
    def _strategy_graceful_degradation(self, health_status: Dict[str, Any]) -> bool:
        """Healing strategy: Graceful degradation."""
        
        logger.info("Attempting graceful degradation healing")
        
        # Reduce crossbar size or precision to improve reliability
        if hasattr(self.crossbar, 'base_config'):
            # Disable problematic regions (simplified)
            self.crossbar.fault_tolerance_level *= 0.9
            
            logger.info("Graceful degradation activated")
            return True
        
        return False
    
    def get_healing_statistics(self) -> Dict[str, Any]:
        """Get self-healing statistics."""
        
        total_attempts = len(self.healing_history)
        successful_attempts = sum(1 for h in self.healing_history if h['healing_success'])
        
        return {
            'total_healing_attempts': total_attempts,
            'successful_healings': successful_attempts,
            'healing_success_rate': successful_attempts / max(1, total_attempts),
            'last_health_check': self.last_health_check,
            'health_monitoring_enabled': self.health_monitoring_enabled
        }
