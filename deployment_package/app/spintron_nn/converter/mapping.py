"""
Neural to Spintronic Mapping.

This module provides specialized mapping algorithms for neural network
operations to spintronic hardware implementations.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum

from ..core.mtj_models import MTJConfig, DomainWallDevice
from ..core.crossbar import MTJCrossbar, CrossbarConfig


class MappingStrategy(Enum):
    """Strategies for mapping neural operations to spintronic hardware."""
    DIRECT = "direct"                    # Direct weight mapping
    DIFFERENTIAL = "differential"        # Differential pairs for signed weights
    TEMPORAL = "temporal"               # Time-based encoding
    AMPLITUDE = "amplitude"             # Amplitude modulation
    HYBRID = "hybrid"                   # Combination of strategies


@dataclass
class MappingConfig:
    """Configuration for neural-to-spintronic mapping."""
    
    strategy: MappingStrategy = MappingStrategy.DIRECT
    
    # Weight mapping parameters
    weight_precision: int = 8
    use_differential_pairs: bool = False
    enable_weight_sharing: bool = True
    
    # Activation mapping
    activation_precision: int = 8
    activation_encoding: str = "voltage"  # "voltage", "current", "time"
    
    # Hardware constraints
    max_crossbar_size: int = 128
    min_crossbar_utilization: float = 0.5
    
    # Optimization targets
    optimize_for: str = "energy"  # "energy", "area", "speed", "accuracy"


class NeuralMapping:
    """Maps neural network operations to spintronic hardware."""
    
    def __init__(self, config: MappingConfig, mtj_config: MTJConfig):
        self.config = config
        self.mtj_config = mtj_config
        
    def map_weights_to_conductances(
        self,
        weights: np.ndarray,
        target_precision: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Map neural network weights to MTJ conductance values.
        
        Args:
            weights: Weight matrix to map
            target_precision: Target precision in bits
            
        Returns:
            Tuple of (conductances, mapping_info)
        """
        if target_precision is None:
            target_precision = self.config.weight_precision
        
        if self.config.strategy == MappingStrategy.DIRECT:
            return self._direct_mapping(weights, target_precision)
        elif self.config.strategy == MappingStrategy.DIFFERENTIAL:
            return self._differential_mapping(weights, target_precision)
        elif self.config.strategy == MappingStrategy.HYBRID:
            return self._hybrid_mapping(weights, target_precision)
        else:
            raise ValueError(f"Unsupported mapping strategy: {self.config.strategy}")
    
    def _direct_mapping(
        self, 
        weights: np.ndarray, 
        precision: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Direct mapping of weights to conductance values."""
        # Find weight range
        w_min, w_max = weights.min(), weights.max()
        
        # Map to conductance range
        g_min = 1.0 / self.mtj_config.resistance_high
        g_max = 1.0 / self.mtj_config.resistance_low
        
        # Handle negative weights by shifting to positive range
        if w_min < 0:
            # Shift weights to [0, w_max - w_min]
            shifted_weights = weights - w_min
            shifted_max = w_max - w_min
            
            # Map to conductance range
            normalized = shifted_weights / shifted_max
            conductances = g_min + normalized * (g_max - g_min)
            
            mapping_info = {
                'strategy': 'direct_shifted',
                'weight_range': (w_min, w_max),
                'conductance_range': (g_min, g_max),
                'offset': w_min,
                'scale': shifted_max
            }
        else:
            # Direct positive mapping
            normalized = weights / w_max
            conductances = g_min + normalized * (g_max - g_min)
            
            mapping_info = {
                'strategy': 'direct_positive',
                'weight_range': (0, w_max),
                'conductance_range': (g_min, g_max),
                'offset': 0,
                'scale': w_max
            }
        
        # Quantize to target precision
        if precision < 32:
            n_levels = 2 ** precision
            conductance_step = (g_max - g_min) / (n_levels - 1)
            conductances = g_min + np.round((conductances - g_min) / conductance_step) * conductance_step
        
        return conductances, mapping_info
    
    def _differential_mapping(
        self, 
        weights: np.ndarray, 
        precision: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Differential pair mapping for signed weights."""
        # Split into positive and negative components
        positive_weights = np.maximum(weights, 0)
        negative_weights = np.maximum(-weights, 0)
        
        # Map both to conductances
        g_min = 1.0 / self.mtj_config.resistance_high
        g_max = 1.0 / self.mtj_config.resistance_low
        
        # Normalize to full range
        w_max = np.maximum(positive_weights.max(), negative_weights.max())
        
        if w_max > 0:
            pos_conductances = g_min + (positive_weights / w_max) * (g_max - g_min)
            neg_conductances = g_min + (negative_weights / w_max) * (g_max - g_min)
        else:
            pos_conductances = np.full_like(weights, g_min)
            neg_conductances = np.full_like(weights, g_min)
        
        # Combine into differential pairs
        # Shape: (rows, cols, 2) where [:,:,0] is positive, [:,:,1] is negative
        differential_conductances = np.stack([pos_conductances, neg_conductances], axis=-1)
        
        # Quantize
        if precision < 32:
            n_levels = 2 ** precision
            conductance_step = (g_max - g_min) / (n_levels - 1)
            differential_conductances = g_min + np.round(
                (differential_conductances - g_min) / conductance_step
            ) * conductance_step
        
        mapping_info = {
            'strategy': 'differential',
            'weight_range': (weights.min(), weights.max()),
            'conductance_range': (g_min, g_max),
            'requires_differential_pairs': True,
            'scale': w_max
        }
        
        return differential_conductances, mapping_info
    
    def _hybrid_mapping(
        self, 
        weights: np.ndarray, 
        precision: int
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Hybrid mapping combining multiple strategies."""
        # Analyze weight distribution
        weight_stats = {
            'mean': weights.mean(),
            'std': weights.std(),
            'min': weights.min(),
            'max': weights.max(),
            'negative_ratio': (weights < 0).mean()
        }
        
        # Choose strategy based on weight characteristics
        if weight_stats['negative_ratio'] > 0.3:
            # High proportion of negative weights - use differential
            return self._differential_mapping(weights, precision)
        elif weight_stats['min'] >= 0:
            # All positive weights - use direct
            return self._direct_mapping(weights, precision)
        else:
            # Mixed but mostly positive - use shifted direct
            return self._direct_mapping(weights, precision)
    
    def map_activations(
        self,
        activations: np.ndarray,
        encoding: str = "voltage"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Map neural network activations to hardware signals.
        
        Args:
            activations: Activation values to map
            encoding: Encoding scheme ("voltage", "current", "time")
            
        Returns:
            Tuple of (encoded_signals, encoding_info)
        """
        if encoding == "voltage":
            return self._voltage_encoding(activations)
        elif encoding == "current":
            return self._current_encoding(activations)
        elif encoding == "time":
            return self._temporal_encoding(activations)
        else:
            raise ValueError(f"Unsupported activation encoding: {encoding}")
    
    def _voltage_encoding(
        self, 
        activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode activations as voltage levels."""
        # Map to voltage range (0 to read voltage)
        v_max = self.mtj_config.switching_voltage * 0.3  # Safe read voltage
        
        # Normalize activations to [0, 1]
        a_min, a_max = activations.min(), activations.max()
        if a_max > a_min:
            normalized = (activations - a_min) / (a_max - a_min)
        else:
            normalized = np.zeros_like(activations)
        
        voltages = normalized * v_max
        
        # Quantize based on activation precision
        if self.config.activation_precision < 32:
            n_levels = 2 ** self.config.activation_precision
            voltage_step = v_max / (n_levels - 1)
            voltages = np.round(voltages / voltage_step) * voltage_step
        
        encoding_info = {
            'encoding': 'voltage',
            'voltage_range': (0, v_max),
            'activation_range': (a_min, a_max),
            'precision_bits': self.config.activation_precision
        }
        
        return voltages, encoding_info
    
    def _current_encoding(
        self, 
        activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode activations as current levels."""
        # Map to current range
        i_max = self.mtj_config.switching_current * 0.1  # Safe read current
        
        a_min, a_max = activations.min(), activations.max()
        if a_max > a_min:
            normalized = (activations - a_min) / (a_max - a_min)
        else:
            normalized = np.zeros_like(activations)
        
        currents = normalized * i_max
        
        encoding_info = {
            'encoding': 'current',
            'current_range': (0, i_max),
            'activation_range': (a_min, a_max)
        }
        
        return currents, encoding_info
    
    def _temporal_encoding(
        self, 
        activations: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Encode activations as pulse widths."""
        # Map to time range (pulse width modulation)
        t_min = 1e-9   # 1 ns minimum
        t_max = 100e-9 # 100 ns maximum
        
        a_min, a_max = activations.min(), activations.max()
        if a_max > a_min:
            normalized = (activations - a_min) / (a_max - a_min)
        else:
            normalized = np.zeros_like(activations)
        
        pulse_widths = t_min + normalized * (t_max - t_min)
        
        encoding_info = {
            'encoding': 'temporal',
            'time_range': (t_min, t_max),
            'activation_range': (a_min, a_max)
        }
        
        return pulse_widths, encoding_info
    
    def optimize_crossbar_allocation(
        self,
        weight_matrices: List[np.ndarray],
        target_utilization: float = 0.8
    ) -> List[Dict[str, Any]]:
        """
        Optimize allocation of weight matrices to crossbar arrays.
        
        Args:
            weight_matrices: List of weight matrices to allocate
            target_utilization: Target crossbar utilization
            
        Returns:
            List of allocation plans
        """
        allocations = []
        max_size = self.config.max_crossbar_size
        
        for i, weights in enumerate(weight_matrices):
            rows, cols = weights.shape
            
            if rows <= max_size and cols <= max_size:
                # Single crossbar allocation
                utilization = (rows * cols) / (max_size * max_size)
                
                allocation = {
                    'layer_index': i,
                    'crossbars': [{
                        'size': (rows, cols),
                        'utilization': utilization,
                        'row_range': (0, rows),
                        'col_range': (0, cols)
                    }],
                    'total_crossbars': 1
                }
            else:
                # Multi-crossbar allocation
                crossbars = []
                
                for row_start in range(0, rows, max_size):
                    row_end = min(row_start + max_size, rows)
                    
                    for col_start in range(0, cols, max_size):
                        col_end = min(col_start + max_size, cols)
                        
                        cb_rows = row_end - row_start
                        cb_cols = col_end - col_start
                        utilization = (cb_rows * cb_cols) / (max_size * max_size)
                        
                        crossbars.append({
                            'size': (cb_rows, cb_cols),
                            'utilization': utilization,
                            'row_range': (row_start, row_end),
                            'col_range': (col_start, col_end)
                        })
                
                allocation = {
                    'layer_index': i,
                    'crossbars': crossbars,
                    'total_crossbars': len(crossbars)
                }
            
            allocations.append(allocation)
        
        return allocations
    
    def estimate_mapping_overhead(
        self,
        weights: np.ndarray,
        mapping_info: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate overhead costs of the mapping strategy.
        
        Args:
            weights: Original weight matrix
            mapping_info: Mapping configuration
            
        Returns:
            Dictionary of overhead metrics
        """
        strategy = mapping_info['strategy']
        
        overhead = {
            'area_overhead': 1.0,      # Relative to direct mapping
            'power_overhead': 1.0,     # Relative to direct mapping
            'latency_overhead': 1.0,   # Relative to direct mapping
            'accuracy_loss': 0.0       # Quantization error
        }
        
        if 'differential' in strategy:
            # Differential pairs double the hardware
            overhead['area_overhead'] = 2.0
            overhead['power_overhead'] = 2.0
            
        # Quantization accuracy loss
        if 'conductance_range' in mapping_info:
            g_min, g_max = mapping_info['conductance_range']
            n_levels = 2 ** self.config.weight_precision
            quantization_step = (g_max - g_min) / (n_levels - 1)
            
            # Estimate quantization noise
            overhead['accuracy_loss'] = quantization_step / (g_max - g_min) * 100  # Percentage
        
        return overhead
    
    def validate_mapping(
        self,
        original_weights: np.ndarray,
        conductances: np.ndarray,
        mapping_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Validate the quality of weight-to-conductance mapping.
        
        Args:
            original_weights: Original weight values
            conductances: Mapped conductance values
            mapping_info: Mapping configuration
            
        Returns:
            Validation metrics
        """
        # Reconstruct weights from conductances
        if mapping_info['strategy'] == 'differential':
            # Handle differential pairs
            pos_conductances = conductances[:, :, 0]
            neg_conductances = conductances[:, :, 1]
            
            # Convert back to weights
            g_min = mapping_info['conductance_range'][0]
            g_max = mapping_info['conductance_range'][1]
            scale = mapping_info['scale']
            
            pos_weights = (pos_conductances - g_min) / (g_max - g_min) * scale
            neg_weights = (neg_conductances - g_min) / (g_max - g_min) * scale
            
            reconstructed_weights = pos_weights - neg_weights
        else:
            # Direct mapping reconstruction
            g_min, g_max = mapping_info['conductance_range']
            offset = mapping_info.get('offset', 0)
            scale = mapping_info['scale']
            
            normalized = (conductances - g_min) / (g_max - g_min)
            reconstructed_weights = normalized * scale + offset
        
        # Calculate metrics
        mse = np.mean((original_weights - reconstructed_weights) ** 2)
        mae = np.mean(np.abs(original_weights - reconstructed_weights))
        max_error = np.max(np.abs(original_weights - reconstructed_weights))
        
        # Correlation
        correlation = np.corrcoef(
            original_weights.flatten(),
            reconstructed_weights.flatten()
        )[0, 1]
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'max_error': float(max_error),
            'correlation': float(correlation),
            'snr_db': float(10 * np.log10(np.var(original_weights) / mse)) if mse > 0 else float('inf')
        }