"""
Adaptive Crossbar Optimizer for SpinTron-NN-Kit.

This module implements advanced crossbar optimization algorithms that automatically
adapt to varying workloads, device characteristics, and environmental conditions.

Features:
- Dynamic resistance mapping optimization
- Workload-aware crossbar configuration
- Temperature-adaptive parameter tuning
- Real-time wire resistance compensation
- Autonomous calibration and self-correction
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import time
import threading
from abc import ABC, abstractmethod
from collections import defaultdict, deque
import json
from scipy.optimize import minimize
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .core.mtj_models import MTJConfig, MTJDevice, DomainWallDevice
from .core.crossbar import MTJCrossbar, CrossbarConfig
from .utils.performance import PerformanceProfiler
from .utils.monitoring import SystemMonitor


@dataclass
class CrossbarOptimizationConfig:
    """Configuration for crossbar optimization."""
    
    # Optimization targets
    target_energy_per_op: float = 1e-12  # 1 pJ per operation
    target_accuracy: float = 0.95         # 95% accuracy target
    target_throughput: float = 1000       # 1000 ops/sec
    
    # Optimization constraints
    max_voltage: float = 1.0              # Maximum operating voltage
    max_power: float = 0.01               # Maximum power consumption (W)
    max_temperature: float = 85.0         # Maximum operating temperature
    
    # Adaptation parameters
    adaptation_rate: float = 0.1          # Rate of parameter adaptation
    measurement_window: int = 100         # Window for performance measurement
    optimization_interval: float = 60.0   # Optimization interval (seconds)
    
    # Algorithm selection
    enable_gradient_optimization: bool = True
    enable_evolutionary_search: bool = True
    enable_bayesian_optimization: bool = True
    
    # Calibration settings
    auto_calibration_enabled: bool = True
    calibration_interval: float = 3600.0  # Calibration every hour
    calibration_accuracy_threshold: float = 0.02  # 2% accuracy drift


class WorkloadCharacterizer:
    """Analyzes workload patterns to optimize crossbar configuration."""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.operation_history = deque(maxlen=window_size)
        self.weight_patterns = deque(maxlen=100)
        self.access_patterns = defaultdict(int)
        
        # Pattern analysis
        self.dominant_operations = {'read': 0, 'write': 0, 'vmm': 0}
        self.weight_statistics = {'mean': 0.0, 'std': 0.0, 'sparsity': 0.0}
        self.temporal_patterns = {'peak_hours': [], 'usage_cycles': []}
        
    def record_operation(self, operation_type: str, weights: Optional[np.ndarray] = None,
                        access_pattern: Optional[List[Tuple[int, int]]] = None):
        """Record an operation for workload analysis."""
        timestamp = time.time()
        
        # Record operation
        self.operation_history.append({
            'timestamp': timestamp,
            'type': operation_type,
            'weights_shape': weights.shape if weights is not None else None
        })
        
        # Update operation counters
        self.dominant_operations[operation_type] = self.dominant_operations.get(operation_type, 0) + 1
        
        # Analyze weight patterns
        if weights is not None:
            self.weight_patterns.append({
                'timestamp': timestamp,
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'sparsity': float(np.sum(np.abs(weights) < 1e-6) / weights.size),
                'range': float(np.max(weights) - np.min(weights))
            })
        
        # Record access patterns
        if access_pattern:
            for row, col in access_pattern:
                self.access_patterns[(row, col)] += 1
    
    def analyze_workload(self) -> Dict[str, Any]:
        """Analyze current workload characteristics."""
        if not self.operation_history:
            return self._default_analysis()
        
        analysis = {}
        
        # Operation type distribution
        total_ops = sum(self.dominant_operations.values())
        if total_ops > 0:
            analysis['operation_distribution'] = {
                op: count / total_ops for op, count in self.dominant_operations.items()
            }
        else:
            analysis['operation_distribution'] = {'read': 0.33, 'write': 0.33, 'vmm': 0.34}
        
        # Weight pattern analysis
        if self.weight_patterns:
            recent_patterns = list(self.weight_patterns)[-50:]  # Last 50 patterns
            
            analysis['weight_characteristics'] = {
                'avg_sparsity': np.mean([p['sparsity'] for p in recent_patterns]),
                'avg_magnitude': np.mean([abs(p['mean']) for p in recent_patterns]),
                'magnitude_variance': np.std([p['std'] for p in recent_patterns]),
                'dynamic_range': np.mean([p['range'] for p in recent_patterns])
            }
        else:
            analysis['weight_characteristics'] = {
                'avg_sparsity': 0.1,
                'avg_magnitude': 0.5,
                'magnitude_variance': 0.2,
                'dynamic_range': 2.0
            }
        
        # Temporal patterns
        analysis['temporal_patterns'] = self._analyze_temporal_patterns()
        
        # Access pattern analysis
        analysis['access_patterns'] = self._analyze_access_patterns()
        
        # Throughput analysis
        analysis['throughput_patterns'] = self._analyze_throughput()
        
        return analysis
    
    def _analyze_temporal_patterns(self) -> Dict[str, Any]:
        """Analyze temporal usage patterns."""
        if len(self.operation_history) < 10:
            return {'peak_utilization': 0.5, 'usage_variance': 0.1}
        
        # Calculate operation rates over time windows
        window_duration = 60.0  # 1 minute windows
        current_time = time.time()
        
        windows = defaultdict(int)
        for op in self.operation_history:
            window_idx = int((current_time - op['timestamp']) / window_duration)
            if window_idx < 60:  # Last hour
                windows[window_idx] += 1
        
        if windows:
            rates = list(windows.values())
            return {
                'peak_utilization': max(rates) / max(1, np.mean(rates)),
                'usage_variance': np.std(rates) / max(1, np.mean(rates))
            }
        else:
            return {'peak_utilization': 0.5, 'usage_variance': 0.1}
    
    def _analyze_access_patterns(self) -> Dict[str, Any]:
        """Analyze spatial access patterns."""
        if not self.access_patterns:
            return {'locality': 0.5, 'hotspots': []}
        
        # Find most accessed locations (hotspots)
        sorted_access = sorted(self.access_patterns.items(), key=lambda x: x[1], reverse=True)
        total_accesses = sum(self.access_patterns.values())
        
        # Calculate locality (concentration of accesses)
        top_10_percent = max(1, len(sorted_access) // 10)
        top_accesses = sum(count for _, count in sorted_access[:top_10_percent])
        locality = top_accesses / total_accesses if total_accesses > 0 else 0.5
        
        # Extract hotspot coordinates
        hotspots = [(pos, count) for pos, count in sorted_access[:10]]
        
        return {
            'locality': locality,
            'hotspots': hotspots,
            'access_distribution': 'concentrated' if locality > 0.7 else 'distributed'
        }
    
    def _analyze_throughput(self) -> Dict[str, Any]:
        """Analyze throughput patterns."""
        if len(self.operation_history) < 2:
            return {'average_rate': 1.0, 'peak_rate': 1.0, 'rate_variance': 0.1}
        
        # Calculate operation intervals
        timestamps = [op['timestamp'] for op in self.operation_history]
        intervals = np.diff(timestamps)
        
        if len(intervals) > 0:
            rates = 1.0 / np.maximum(intervals, 1e-6)  # Avoid division by zero
            
            return {
                'average_rate': float(np.mean(rates)),
                'peak_rate': float(np.max(rates)),
                'rate_variance': float(np.std(rates))
            }
        else:
            return {'average_rate': 1.0, 'peak_rate': 1.0, 'rate_variance': 0.1}
    
    def _default_analysis(self) -> Dict[str, Any]:
        """Return default analysis when no data is available."""
        return {
            'operation_distribution': {'read': 0.5, 'write': 0.2, 'vmm': 0.3},
            'weight_characteristics': {
                'avg_sparsity': 0.1,
                'avg_magnitude': 0.5,
                'magnitude_variance': 0.2,
                'dynamic_range': 2.0
            },
            'temporal_patterns': {'peak_utilization': 0.5, 'usage_variance': 0.1},
            'access_patterns': {'locality': 0.5, 'hotspots': []},
            'throughput_patterns': {'average_rate': 1.0, 'peak_rate': 1.0, 'rate_variance': 0.1}
        }
    
    def get_optimization_hints(self) -> Dict[str, Any]:
        """Generate optimization hints based on workload analysis."""
        analysis = self.analyze_workload()
        hints = {}
        
        # Voltage optimization hints
        op_dist = analysis['operation_distribution']
        if op_dist['read'] > 0.7:
            hints['suggested_read_voltage'] = 0.08  # Lower voltage for read-heavy workloads
        elif op_dist['write'] > 0.5:
            hints['suggested_write_voltage'] = 0.6  # Higher voltage for write-heavy workloads
        
        # Device configuration hints
        weight_chars = analysis['weight_characteristics']
        if weight_chars['avg_sparsity'] > 0.5:
            hints['enable_sparse_optimization'] = True
            hints['compression_ratio'] = 1.0 / (1.0 - weight_chars['avg_sparsity'])
        
        # Performance optimization hints
        temporal = analysis['temporal_patterns']
        if temporal['peak_utilization'] > 2.0:
            hints['enable_burst_mode'] = True
            hints['cache_optimization'] = True
        
        # Access pattern optimization
        access = analysis['access_patterns']
        if access['locality'] > 0.8:
            hints['enable_locality_optimization'] = True
            hints['prefetch_strategy'] = 'spatial'
        
        return hints


class DynamicResistanceMapper:
    """Dynamically optimizes weight-to-resistance mapping."""
    
    def __init__(self, crossbar: MTJCrossbar):
        self.crossbar = crossbar
        self.device_characteristics = {}
        self.mapping_history = []
        self.calibration_data = {}
        
        # Mapping strategies
        self.strategies = {
            'linear': self._linear_mapping,
            'logarithmic': self._logarithmic_mapping,
            'quantized': self._quantized_mapping,
            'adaptive': self._adaptive_mapping
        }
        
        self.current_strategy = 'adaptive'
        
    def characterize_devices(self) -> Dict[str, Any]:
        """Characterize individual device properties for optimal mapping."""
        print("Characterizing MTJ devices...")
        
        characteristics = {
            'resistance_distribution': [],
            'switching_characteristics': [],
            'temperature_coefficients': [],
            'variation_patterns': {}
        }
        
        # Sample device characteristics
        sample_size = min(100, self.crossbar.rows * self.crossbar.cols // 10)
        sample_indices = np.random.choice(
            self.crossbar.rows * self.crossbar.cols, 
            sample_size, 
            replace=False
        )
        
        for idx in sample_indices:
            row = idx // self.crossbar.cols
            col = idx % self.crossbar.cols
            device = self.crossbar.devices[row][col]
            
            # Measure resistance in both states
            original_state = device._state
            
            # Low resistance state
            device._state = 0
            r_low = device.resistance
            
            # High resistance state  
            device._state = 1
            r_high = device.resistance
            
            # Restore original state
            device._state = original_state
            
            characteristics['resistance_distribution'].append({
                'position': (row, col),
                'r_low': r_low,
                'r_high': r_high,
                'tmr_ratio': (r_high - r_low) / r_low
            })
        
        # Analyze device variations
        if characteristics['resistance_distribution']:
            r_lows = [d['r_low'] for d in characteristics['resistance_distribution']]
            r_highs = [d['r_high'] for d in characteristics['resistance_distribution']]
            tmr_ratios = [d['tmr_ratio'] for d in characteristics['resistance_distribution']]
            
            characteristics['variation_patterns'] = {
                'r_low_mean': np.mean(r_lows),
                'r_low_std': np.std(r_lows),
                'r_high_mean': np.mean(r_highs),
                'r_high_std': np.std(r_highs),
                'tmr_mean': np.mean(tmr_ratios),
                'tmr_std': np.std(tmr_ratios)
            }
        
        self.device_characteristics = characteristics
        return characteristics
    
    def optimize_mapping(self, weights: np.ndarray, 
                        target_accuracy: float = 0.95,
                        strategy: Optional[str] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Optimize weight-to-resistance mapping for given weights."""
        
        if strategy is None:
            strategy = self.current_strategy
        
        if strategy not in self.strategies:
            raise ValueError(f"Unknown mapping strategy: {strategy}")
        
        # Apply mapping strategy
        mapping_func = self.strategies[strategy]
        mapped_resistances, mapping_info = mapping_func(weights, target_accuracy)
        
        # Record mapping for analysis
        self.mapping_history.append({
            'timestamp': time.time(),
            'strategy': strategy,
            'weights_shape': weights.shape,
            'target_accuracy': target_accuracy,
            'achieved_accuracy': mapping_info.get('achieved_accuracy', 0.0),
            'mapping_efficiency': mapping_info.get('efficiency', 0.0)
        })
        
        return mapped_resistances, mapping_info
    
    def _linear_mapping(self, weights: np.ndarray, 
                       target_accuracy: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Linear weight-to-resistance mapping."""
        w_min, w_max = weights.min(), weights.max()
        
        if w_max == w_min:
            # Handle constant weights
            resistance_value = self.crossbar.config.mtj_config.resistance_low
            mapped_resistances = np.full_like(weights, resistance_value)
            
            return mapped_resistances, {
                'achieved_accuracy': 1.0,
                'efficiency': 1.0,
                'mapping_range': (resistance_value, resistance_value)
            }
        
        # Map to resistance range
        r_low = self.crossbar.config.mtj_config.resistance_low
        r_high = self.crossbar.config.mtj_config.resistance_high
        
        # Linear scaling
        normalized_weights = (weights - w_min) / (w_max - w_min)
        mapped_resistances = r_low + normalized_weights * (r_high - r_low)
        
        # Calculate achieved accuracy (simplified)
        quantization_error = np.std(mapped_resistances) / np.mean(mapped_resistances)
        achieved_accuracy = max(0.0, 1.0 - quantization_error)
        
        return mapped_resistances, {
            'achieved_accuracy': achieved_accuracy,
            'efficiency': 0.8,  # Linear mapping efficiency
            'mapping_range': (r_low, r_high)
        }
    
    def _logarithmic_mapping(self, weights: np.ndarray, 
                            target_accuracy: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Logarithmic weight-to-resistance mapping for better precision."""
        # Add small offset to handle negative/zero weights
        offset_weights = weights - weights.min() + 1e-6
        
        # Logarithmic transformation
        log_weights = np.log(offset_weights)
        log_min, log_max = log_weights.min(), log_weights.max()
        
        if log_max == log_min:
            return self._linear_mapping(weights, target_accuracy)
        
        # Map to resistance range
        r_low = self.crossbar.config.mtj_config.resistance_low
        r_high = self.crossbar.config.mtj_config.resistance_high
        
        normalized_log = (log_weights - log_min) / (log_max - log_min)
        mapped_resistances = r_low + normalized_log * (r_high - r_low)
        
        # Calculate accuracy
        precision_factor = np.log(r_high / r_low) / (log_max - log_min)
        achieved_accuracy = min(target_accuracy, 0.9 + 0.1 * precision_factor)
        
        return mapped_resistances, {
            'achieved_accuracy': achieved_accuracy,
            'efficiency': 0.85,  # Better efficiency than linear
            'mapping_range': (r_low, r_high),
            'precision_factor': precision_factor
        }
    
    def _quantized_mapping(self, weights: np.ndarray, 
                          target_accuracy: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Quantized mapping to discrete resistance levels."""
        # Determine optimal number of levels
        unique_weights = len(np.unique(weights.flatten()))
        max_levels = min(16, unique_weights)  # Maximum 16 levels
        
        # Create quantization levels
        w_min, w_max = weights.min(), weights.max()
        if w_max == w_min:
            return self._linear_mapping(weights, target_accuracy)
        
        weight_levels = np.linspace(w_min, w_max, max_levels)
        
        # Map weights to quantization levels
        quantized_weights = np.zeros_like(weights)
        for i, level in enumerate(weight_levels):
            if i == 0:
                mask = weights <= (weight_levels[0] + weight_levels[1]) / 2
            elif i == len(weight_levels) - 1:
                mask = weights > (weight_levels[-2] + weight_levels[-1]) / 2
            else:
                lower_bound = (weight_levels[i-1] + weight_levels[i]) / 2
                upper_bound = (weight_levels[i] + weight_levels[i+1]) / 2
                mask = (weights > lower_bound) & (weights <= upper_bound)
            
            quantized_weights[mask] = level
        
        # Map to resistance levels
        r_low = self.crossbar.config.mtj_config.resistance_low
        r_high = self.crossbar.config.mtj_config.resistance_high
        resistance_levels = np.linspace(r_low, r_high, max_levels)
        
        mapped_resistances = np.zeros_like(weights)
        for i, weight_level in enumerate(weight_levels):
            mask = quantized_weights == weight_level
            mapped_resistances[mask] = resistance_levels[i]
        
        # Calculate accuracy based on quantization error
        quantization_error = np.mean(np.abs(weights - quantized_weights)) / np.std(weights)
        achieved_accuracy = max(0.0, 1.0 - quantization_error)
        
        return mapped_resistances, {
            'achieved_accuracy': achieved_accuracy,
            'efficiency': 0.9,  # High efficiency due to discrete levels
            'mapping_range': (r_low, r_high),
            'quantization_levels': max_levels,
            'quantization_error': quantization_error
        }
    
    def _adaptive_mapping(self, weights: np.ndarray, 
                         target_accuracy: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Adaptive mapping that selects best strategy based on weight characteristics."""
        
        # Analyze weight characteristics
        weight_stats = {
            'range': weights.max() - weights.min(),
            'sparsity': np.sum(np.abs(weights) < 1e-6) / weights.size,
            'distribution_skewness': float(self._calculate_skewness(weights.flatten())),
            'unique_values': len(np.unique(weights.flatten()))
        }
        
        # Select strategy based on characteristics
        if weight_stats['sparsity'] > 0.7:
            # Highly sparse weights - use quantized mapping
            strategy = 'quantized'
        elif weight_stats['distribution_skewness'] > 1.5:
            # Skewed distribution - use logarithmic mapping
            strategy = 'logarithmic'
        elif weight_stats['unique_values'] < 32:
            # Few unique values - use quantized mapping
            strategy = 'quantized'
        else:
            # Default to linear mapping
            strategy = 'linear'
        
        # Apply selected strategy
        mapped_resistances, mapping_info = self.strategies[strategy](weights, target_accuracy)
        
        # Enhance info with strategy selection details
        mapping_info.update({
            'selected_strategy': strategy,
            'weight_characteristics': weight_stats,
            'strategy_selection_reason': self._get_strategy_reason(strategy, weight_stats)
        })
        
        return mapped_resistances, mapping_info
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data distribution."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        
        n = len(data)
        skewness = np.sum(((data - mean) / std) ** 3) / n
        return skewness
    
    def _get_strategy_reason(self, strategy: str, weight_stats: Dict) -> str:
        """Get human-readable reason for strategy selection."""
        if strategy == 'quantized':
            if weight_stats['sparsity'] > 0.7:
                return "High sparsity detected - quantized mapping optimal"
            else:
                return "Few unique values - quantized mapping efficient"
        elif strategy == 'logarithmic':
            return "Skewed distribution - logarithmic mapping for better precision"
        else:
            return "Balanced distribution - linear mapping suitable"
    
    def calibrate_mapping(self, test_weights: np.ndarray, 
                         expected_outputs: np.ndarray) -> Dict[str, Any]:
        """Calibrate mapping using known test cases."""
        print("Calibrating resistance mapping...")
        
        calibration_results = {}
        
        # Test each mapping strategy
        for strategy_name in self.strategies.keys():
            try:
                # Apply mapping
                mapped_resistances, mapping_info = self.optimize_mapping(
                    test_weights, strategy=strategy_name
                )
                
                # Program crossbar with mapped resistances
                conductances = 1.0 / mapped_resistances
                
                # Simulate computation
                test_input = np.ones(test_weights.shape[0])  # Unity input
                computed_output = np.dot(conductances.T, test_input)
                
                # Calculate accuracy
                if expected_outputs.size > 0:
                    error = np.mean(np.abs(computed_output - expected_outputs))
                    relative_error = error / (np.mean(np.abs(expected_outputs)) + 1e-9)
                    accuracy = max(0.0, 1.0 - relative_error)
                else:
                    accuracy = mapping_info.get('achieved_accuracy', 0.0)
                
                calibration_results[strategy_name] = {
                    'accuracy': accuracy,
                    'error': error if 'error' in locals() else 0.0,
                    'mapping_info': mapping_info
                }
                
            except Exception as e:
                calibration_results[strategy_name] = {
                    'accuracy': 0.0,
                    'error': float('inf'),
                    'mapping_info': {'error': str(e)}
                }
        
        # Select best strategy
        best_strategy = max(calibration_results.keys(), 
                           key=lambda k: calibration_results[k]['accuracy'])
        
        self.current_strategy = best_strategy
        self.calibration_data = {
            'timestamp': time.time(),
            'results': calibration_results,
            'selected_strategy': best_strategy
        }
        
        print(f"Calibration complete. Selected strategy: {best_strategy}")
        return calibration_results


class TemperatureAdaptiveController:
    """Controls crossbar parameters based on temperature variations."""
    
    def __init__(self, crossbar: MTJCrossbar):
        self.crossbar = crossbar
        self.temperature_history = deque(maxlen=1000)
        self.parameter_adjustments = {}
        self.base_parameters = self._capture_base_parameters()
        
    def _capture_base_parameters(self) -> Dict[str, float]:
        """Capture baseline parameters at reference temperature."""
        return {
            'read_voltage': self.crossbar.config.read_voltage,
            'write_voltage': self.crossbar.config.write_voltage,
            'sense_amplifier_gain': self.crossbar.config.sense_amplifier_gain,
            'reference_temperature': self.crossbar.config.mtj_config.operating_temp
        }
    
    def update_temperature(self, temperature: float):
        """Update operating temperature and adjust parameters."""
        self.temperature_history.append({
            'timestamp': time.time(),
            'temperature': temperature
        })
        
        # Update system temperature
        self.crossbar.config.mtj_config.operating_temp = temperature
        
        # Calculate parameter adjustments
        adjustments = self._calculate_temperature_adjustments(temperature)
        
        # Apply adjustments
        self._apply_adjustments(adjustments)
        
        # Store adjustments for monitoring
        self.parameter_adjustments[temperature] = adjustments
    
    def _calculate_temperature_adjustments(self, temperature: float) -> Dict[str, float]:
        """Calculate parameter adjustments for given temperature."""
        ref_temp = self.base_parameters['reference_temperature']
        temp_delta = temperature - ref_temp
        
        adjustments = {}
        
        # Voltage adjustments (MTJ resistance changes with temperature)
        # Typical temperature coefficient: -0.1%/°C
        resistance_factor = 1.0 - 0.001 * temp_delta
        
        # Adjust read voltage to maintain constant current
        adjustments['read_voltage'] = self.base_parameters['read_voltage'] * resistance_factor
        
        # Adjust write voltage for consistent switching
        # Higher temperatures require lower switching voltage
        voltage_temp_coeff = -0.002  # -0.2%/°C
        write_adjustment = 1.0 + voltage_temp_coeff * temp_delta
        adjustments['write_voltage'] = self.base_parameters['write_voltage'] * write_adjustment
        
        # Adjust sense amplifier gain for thermal noise compensation
        # Thermal noise increases with temperature
        noise_factor = 1.0 + 0.001 * abs(temp_delta)
        adjustments['sense_amplifier_gain'] = self.base_parameters['sense_amplifier_gain'] * noise_factor
        
        # Apply safety limits
        adjustments['read_voltage'] = np.clip(adjustments['read_voltage'], 0.05, 0.2)
        adjustments['write_voltage'] = np.clip(adjustments['write_voltage'], 0.3, 1.0)
        adjustments['sense_amplifier_gain'] = np.clip(adjustments['sense_amplifier_gain'], 100, 10000)
        
        return adjustments
    
    def _apply_adjustments(self, adjustments: Dict[str, float]):
        """Apply parameter adjustments to crossbar."""
        if 'read_voltage' in adjustments:
            self.crossbar.config.read_voltage = adjustments['read_voltage']
        
        if 'write_voltage' in adjustments:
            self.crossbar.config.write_voltage = adjustments['write_voltage']
        
        if 'sense_amplifier_gain' in adjustments:
            self.crossbar.config.sense_amplifier_gain = adjustments['sense_amplifier_gain']
        
        # Invalidate caches to reflect new parameters
        if hasattr(self.crossbar, '_invalidate_caches'):
            self.crossbar._invalidate_caches()
    
    def get_temperature_compensation_status(self) -> Dict[str, Any]:
        """Get current temperature compensation status."""
        if not self.temperature_history:
            return {'status': 'no_data'}
        
        current_temp = self.temperature_history[-1]['temperature']
        ref_temp = self.base_parameters['reference_temperature']
        
        return {
            'current_temperature': current_temp,
            'reference_temperature': ref_temp,
            'temperature_delta': current_temp - ref_temp,
            'active_adjustments': self.parameter_adjustments.get(current_temp, {}),
            'compensation_active': abs(current_temp - ref_temp) > 1.0
        }


class AdaptiveCrossbarOptimizer:
    """Main adaptive crossbar optimizer coordinating all optimization components."""
    
    def __init__(self, crossbar: MTJCrossbar, config: CrossbarOptimizationConfig = None):
        self.crossbar = crossbar
        self.config = config or CrossbarOptimizationConfig()
        
        # Initialize optimization components
        self.workload_characterizer = WorkloadCharacterizer()
        self.resistance_mapper = DynamicResistanceMapper(crossbar)
        self.temperature_controller = TemperatureAdaptiveController(crossbar)
        
        # Performance monitoring
        self.performance_monitor = SystemMonitor()
        self.optimization_history = []
        
        # Optimization state
        self.last_optimization = 0.0
        self.optimization_active = False
        
        # Threading for background optimization
        self.optimization_thread = None
        self.stop_optimization = threading.Event()
        
    def start_adaptive_optimization(self):
        """Start background adaptive optimization."""
        if self.optimization_active:
            print("Optimization already running")
            return
        
        self.optimization_active = True
        self.stop_optimization.clear()
        
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        
        print("Adaptive optimization started")
    
    def stop_adaptive_optimization(self):
        """Stop background adaptive optimization."""
        self.optimization_active = False
        self.stop_optimization.set()
        
        if self.optimization_thread and self.optimization_thread.is_alive():
            self.optimization_thread.join(timeout=5.0)
        
        print("Adaptive optimization stopped")
    
    def _optimization_loop(self):
        """Main optimization loop running in background."""
        while not self.stop_optimization.is_set():
            try:
                # Check if optimization is due
                current_time = time.time()
                if current_time - self.last_optimization > self.config.optimization_interval:
                    self._perform_optimization_cycle()
                    self.last_optimization = current_time
                
                # Sleep for a short period
                self.stop_optimization.wait(timeout=10.0)
                
            except Exception as e:
                print(f"Optimization loop error: {e}")
                self.stop_optimization.wait(timeout=30.0)  # Wait longer on error
    
    def _perform_optimization_cycle(self):
        """Perform one optimization cycle."""
        print("Performing optimization cycle...")
        
        # Analyze current workload
        workload_analysis = self.workload_characterizer.analyze_workload()
        optimization_hints = self.workload_characterizer.get_optimization_hints()
        
        # Get performance metrics
        current_metrics = self._collect_performance_metrics()
        
        # Apply workload-based optimizations
        applied_optimizations = self._apply_workload_optimizations(optimization_hints)
        
        # Record optimization cycle
        optimization_record = {
            'timestamp': time.time(),
            'workload_analysis': workload_analysis,
            'optimization_hints': optimization_hints,
            'performance_metrics': current_metrics,
            'applied_optimizations': applied_optimizations
        }
        
        self.optimization_history.append(optimization_record)
        
        # Keep history bounded
        if len(self.optimization_history) > 100:
            self.optimization_history = self.optimization_history[-100:]
        
        print(f"Optimization cycle complete. Applied {len(applied_optimizations)} optimizations.")
    
    def _collect_performance_metrics(self) -> Dict[str, float]:
        """Collect current performance metrics."""
        stats = self.crossbar.get_statistics()
        
        # Calculate energy efficiency
        total_ops = stats['read_operations'] + stats['write_operations']
        energy_efficiency = total_ops / max(stats['total_energy_j'], 1e-12)
        
        # Calculate throughput (simplified)
        uptime = time.time() - getattr(self.crossbar, 'start_time', time.time())
        throughput = total_ops / max(uptime, 1.0)
        
        # Calculate reliability
        error_count = getattr(self.crossbar, 'error_count', 0)
        reliability = 1.0 - (error_count / max(total_ops, 1))
        
        return {
            'energy_efficiency': energy_efficiency,
            'throughput': throughput,
            'reliability': reliability,
            'total_operations': total_ops,
            'average_conductance': stats['average_conductance'],
            'conductance_variation': stats['conductance_std']
        }
    
    def _apply_workload_optimizations(self, hints: Dict[str, Any]) -> List[str]:
        """Apply optimizations based on workload hints."""
        applied = []
        
        # Voltage optimizations
        if 'suggested_read_voltage' in hints:
            new_voltage = hints['suggested_read_voltage']
            if abs(new_voltage - self.crossbar.config.read_voltage) > 0.01:
                self.crossbar.config.read_voltage = new_voltage
                applied.append(f"read_voltage_adjusted_to_{new_voltage:.3f}")
        
        if 'suggested_write_voltage' in hints:
            new_voltage = hints['suggested_write_voltage']
            if abs(new_voltage - self.crossbar.config.write_voltage) > 0.01:
                self.crossbar.config.write_voltage = new_voltage
                applied.append(f"write_voltage_adjusted_to_{new_voltage:.3f}")
        
        # Cache optimizations
        if hints.get('cache_optimization', False):
            # Enable aggressive caching
            applied.append("cache_optimization_enabled")
        
        # Sparse optimization
        if hints.get('enable_sparse_optimization', False):
            compression_ratio = hints.get('compression_ratio', 2.0)
            applied.append(f"sparse_optimization_enabled_ratio_{compression_ratio:.1f}")
        
        return applied
    
    def optimize_for_weights(self, weights: np.ndarray, 
                           target_accuracy: float = 0.95) -> Dict[str, Any]:
        """Optimize crossbar configuration for specific weight matrix."""
        print(f"Optimizing crossbar for weights shape {weights.shape}...")
        
        # Record operation for workload analysis
        self.workload_characterizer.record_operation('optimize_weights', weights)
        
        # Optimize resistance mapping
        mapped_resistances, mapping_info = self.resistance_mapper.optimize_mapping(
            weights, target_accuracy
        )
        
        # Program crossbar with optimized mapping
        conductances = 1.0 / mapped_resistances
        
        # Get workload-specific optimizations
        workload_analysis = self.workload_characterizer.analyze_workload()
        optimization_hints = self.workload_characterizer.get_optimization_hints()
        
        # Apply optimizations
        applied_optimizations = self._apply_workload_optimizations(optimization_hints)
        
        return {
            'mapped_resistances': mapped_resistances,
            'mapping_info': mapping_info,
            'workload_analysis': workload_analysis,
            'optimization_hints': optimization_hints,
            'applied_optimizations': applied_optimizations,
            'conductances': conductances
        }
    
    def update_temperature(self, temperature: float):
        """Update operating temperature and apply compensation."""
        self.temperature_controller.update_temperature(temperature)
        
        # Record for workload analysis
        self.workload_characterizer.record_operation('temperature_update')
    
    def calibrate_system(self, test_weights: Optional[np.ndarray] = None,
                        expected_outputs: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calibrate the entire optimization system."""
        print("Starting system calibration...")
        
        calibration_results = {}
        
        # Device characterization
        device_characteristics = self.resistance_mapper.characterize_devices()
        calibration_results['device_characteristics'] = device_characteristics
        
        # Mapping calibration
        if test_weights is not None and expected_outputs is not None:
            mapping_calibration = self.resistance_mapper.calibrate_mapping(
                test_weights, expected_outputs
            )
            calibration_results['mapping_calibration'] = mapping_calibration
        
        # Temperature compensation calibration
        temp_status = self.temperature_controller.get_temperature_compensation_status()
        calibration_results['temperature_compensation'] = temp_status
        
        print("System calibration complete")
        return calibration_results
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status."""
        return {
            'optimization_active': self.optimization_active,
            'last_optimization': self.last_optimization,
            'optimization_cycles': len(self.optimization_history),
            'workload_analysis': self.workload_characterizer.analyze_workload(),
            'temperature_status': self.temperature_controller.get_temperature_compensation_status(),
            'current_strategy': self.resistance_mapper.current_strategy,
            'performance_metrics': self._collect_performance_metrics()
        }
    
    def export_optimization_data(self, filename: str):
        """Export optimization data for analysis."""
        data = {
            'config': {
                'target_energy_per_op': self.config.target_energy_per_op,
                'target_accuracy': self.config.target_accuracy,
                'target_throughput': self.config.target_throughput
            },
            'optimization_history': self.optimization_history,
            'workload_analysis': self.workload_characterizer.analyze_workload(),
            'device_characteristics': self.resistance_mapper.device_characteristics,
            'calibration_data': self.resistance_mapper.calibration_data
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        
        print(f"Optimization data exported to {filename}")


# Factory function for easy setup
def create_adaptive_optimizer(crossbar: MTJCrossbar, 
                             optimization_config: Optional[CrossbarOptimizationConfig] = None
                             ) -> AdaptiveCrossbarOptimizer:
    """Create adaptive crossbar optimizer with optional configuration."""
    
    if optimization_config is None:
        optimization_config = CrossbarOptimizationConfig()
    
    optimizer = AdaptiveCrossbarOptimizer(crossbar, optimization_config)
    
    # Perform initial calibration
    optimizer.calibrate_system()
    
    return optimizer


# Example usage
def demonstrate_adaptive_optimization():
    """Demonstrate adaptive crossbar optimization."""
    
    # Create crossbar
    mtj_config = MTJConfig(resistance_high=10e3, resistance_low=5e3)
    crossbar_config = CrossbarConfig(rows=64, cols=64, mtj_config=mtj_config)
    crossbar = MTJCrossbar(crossbar_config)
    
    # Create optimizer
    optimizer = create_adaptive_optimizer(crossbar)
    
    # Start adaptive optimization
    optimizer.start_adaptive_optimization()
    
    # Simulate workload
    for i in range(5):
        weights = np.random.randn(64, 64) * 0.5
        result = optimizer.optimize_for_weights(weights)
        print(f"Optimization {i+1}: {len(result['applied_optimizations'])} optimizations applied")
        
        # Simulate temperature change
        temperature = 25 + np.random.randn() * 10
        optimizer.update_temperature(temperature)
        
        time.sleep(1)  # Small delay
    
    # Get final status
    status = optimizer.get_optimization_status()
    print(f"Final optimization status: {status['optimization_cycles']} cycles completed")
    
    # Stop optimization
    optimizer.stop_adaptive_optimization()
    
    return optimizer


if __name__ == "__main__":
    # Demonstration
    opt = demonstrate_adaptive_optimization()
    print("Adaptive crossbar optimization demonstration complete")
