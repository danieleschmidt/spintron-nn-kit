"""
PyTorch Model Parser and Converter.

This module provides functionality to parse PyTorch models and convert them
to spintronic hardware-compatible representations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import numpy as np
from collections import OrderedDict

from ..core.mtj_models import MTJConfig
from ..core.crossbar import MTJCrossbar, CrossbarConfig


@dataclass
class ConversionConfig:
    """Configuration for PyTorch to spintronic conversion."""
    
    quantization_bits: int = 8
    crossbar_size: int = 128
    mtj_config: MTJConfig = None
    
    # Mapping options
    map_activations: bool = True
    map_batch_norm: bool = False  # Usually keep in CMOS
    map_pooling: bool = False     # Usually keep in CMOS
    
    # Optimization options
    enable_layer_fusion: bool = True
    enable_weight_sharing: bool = True
    
    def __post_init__(self):
        if self.mtj_config is None:
            self.mtj_config = MTJConfig()


class SpintronicLayer:
    """Represents a neural network layer mapped to spintronic hardware."""
    
    def __init__(
        self,
        name: str,
        layer_type: str,
        input_shape: Tuple[int, ...],
        output_shape: Tuple[int, ...],
        weights: Optional[np.ndarray] = None,
        bias: Optional[np.ndarray] = None
    ):
        self.name = name
        self.layer_type = layer_type
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.weights = weights
        self.bias = bias
        
        # Hardware mapping
        self.crossbars: List[MTJCrossbar] = []
        self.hardware_config: Optional[CrossbarConfig] = None
        
    def map_to_crossbars(self, config: ConversionConfig) -> List[MTJCrossbar]:
        """Map layer weights to MTJ crossbar arrays."""
        if self.weights is None:
            return []
        
        crossbars = []
        
        if self.layer_type == 'Linear':
            # Single crossbar for fully connected layer
            rows, cols = self.weights.shape
            
            # Split into multiple crossbars if too large
            if rows > config.crossbar_size or cols > config.crossbar_size:
                crossbars.extend(self._split_linear_layer(config))
            else:
                crossbar_config = CrossbarConfig(
                    rows=rows, 
                    cols=cols,
                    mtj_config=config.mtj_config
                )
                crossbar = MTJCrossbar(crossbar_config)
                crossbar.set_weights(self.weights)
                crossbars.append(crossbar)
                
        elif self.layer_type == 'Conv2d':
            # Multiple crossbars for convolutional layer
            crossbars.extend(self._map_conv_layer(config))
        
        self.crossbars = crossbars
        return crossbars
    
    def _split_linear_layer(self, config: ConversionConfig) -> List[MTJCrossbar]:
        """Split large linear layer across multiple crossbars."""
        crossbars = []
        rows, cols = self.weights.shape
        max_size = config.crossbar_size
        
        # Split by rows first, then by columns
        for row_start in range(0, rows, max_size):
            row_end = min(row_start + max_size, rows)
            
            for col_start in range(0, cols, max_size):
                col_end = min(col_start + max_size, cols)
                
                # Extract weight sub-matrix
                weight_slice = self.weights[row_start:row_end, col_start:col_end]
                
                # Create crossbar for this slice
                crossbar_config = CrossbarConfig(
                    rows=weight_slice.shape[0],
                    cols=weight_slice.shape[1],
                    mtj_config=config.mtj_config
                )
                crossbar = MTJCrossbar(crossbar_config)
                crossbar.set_weights(weight_slice)
                
                # Store mapping information
                crossbar.row_slice = (row_start, row_end)
                crossbar.col_slice = (col_start, col_end)
                
                crossbars.append(crossbar)
        
        return crossbars
    
    def _map_conv_layer(self, config: ConversionConfig) -> List[MTJCrossbar]:
        """Map convolutional layer to crossbar arrays."""
        crossbars = []
        
        # Convert conv weights to matrix form (im2col approach)
        if len(self.weights.shape) == 4:  # [out_channels, in_channels, kernel_h, kernel_w]
            out_ch, in_ch, kh, kw = self.weights.shape
            
            # Reshape to 2D matrix
            weight_matrix = self.weights.reshape(out_ch, in_ch * kh * kw)
            
            # Map as linear layer
            rows, cols = weight_matrix.shape
            
            if rows > config.crossbar_size or cols > config.crossbar_size:
                # Split large conv layer
                for row_start in range(0, rows, config.crossbar_size):
                    row_end = min(row_start + config.crossbar_size, rows)
                    
                    for col_start in range(0, cols, config.crossbar_size):
                        col_end = min(col_start + config.crossbar_size, cols)
                        
                        weight_slice = weight_matrix[row_start:row_end, col_start:col_end]
                        
                        crossbar_config = CrossbarConfig(
                            rows=weight_slice.shape[0],
                            cols=weight_slice.shape[1],
                            mtj_config=config.mtj_config
                        )
                        crossbar = MTJCrossbar(crossbar_config)
                        crossbar.set_weights(weight_slice)
                        crossbars.append(crossbar)
            else:
                crossbar_config = CrossbarConfig(
                    rows=rows,
                    cols=cols, 
                    mtj_config=config.mtj_config
                )
                crossbar = MTJCrossbar(crossbar_config)
                crossbar.set_weights(weight_matrix)
                crossbars.append(crossbar)
        
        return crossbars


class SpintronicModel:
    """Spintronic neural network model."""
    
    def __init__(self, name: str, config: ConversionConfig):
        self.name = name
        self.config = config
        self.layers: List[SpintronicLayer] = []
        self.total_crossbars = 0
        self.total_mtj_cells = 0
        
    def add_layer(self, layer: SpintronicLayer):
        """Add layer to spintronic model."""
        self.layers.append(layer)
        
        # Map to crossbars
        crossbars = layer.map_to_crossbars(self.config)
        self.total_crossbars += len(crossbars)
        
        for crossbar in crossbars:
            self.total_mtj_cells += crossbar.rows * crossbar.cols
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through spintronic model."""
        current_input = x
        
        for layer in self.layers:
            if layer.layer_type == 'Linear' and layer.crossbars:
                current_input = self._linear_forward(current_input, layer)
            elif layer.layer_type == 'Conv2d' and layer.crossbars:
                current_input = self._conv_forward(current_input, layer)
            elif layer.layer_type == 'ReLU':
                current_input = torch.relu(current_input)
            elif layer.layer_type == 'Flatten':
                current_input = current_input.flatten(1)
        
        return current_input
    
    def _linear_forward(self, x: torch.Tensor, layer: SpintronicLayer) -> torch.Tensor:
        """Forward pass through linear layer using crossbars."""
        if len(layer.crossbars) == 1:
            # Single crossbar case
            crossbar = layer.crossbars[0]
            
            # Convert to input voltages (simplified)
            input_voltages = x.detach().cpu().numpy().flatten()
            
            # Compute using crossbar
            output_currents = crossbar.compute_vmm(input_voltages)
            
            # Convert back to tensor
            output = torch.tensor(output_currents, dtype=x.dtype, device=x.device)
            
        else:
            # Multiple crossbars - need to combine outputs
            outputs = []
            
            for crossbar in layer.crossbars:
                # Get corresponding input slice
                row_slice = getattr(crossbar, 'row_slice', (0, x.shape[-1]))
                col_slice = getattr(crossbar, 'col_slice', (0, layer.output_shape[-1]))
                
                input_slice = x[..., row_slice[0]:row_slice[1]]
                input_voltages = input_slice.detach().cpu().numpy().flatten()
                
                output_currents = crossbar.compute_vmm(input_voltages)
                outputs.append(torch.tensor(output_currents, dtype=x.dtype, device=x.device))
            
            # Combine outputs
            output = torch.cat(outputs, dim=-1)
        
        # Add bias if present
        if layer.bias is not None:
            bias_tensor = torch.tensor(layer.bias, dtype=x.dtype, device=x.device)
            output = output + bias_tensor
        
        return output
    
    def _conv_forward(self, x: torch.Tensor, layer: SpintronicLayer) -> torch.Tensor:
        """Forward pass through convolutional layer using crossbars."""
        # This is a simplified implementation
        # In practice, would need im2col transformation
        
        batch_size = x.shape[0]
        output_shape = (batch_size,) + layer.output_shape[1:]
        output = torch.zeros(output_shape, dtype=x.dtype, device=x.device)
        
        # Simplified conv using first crossbar
        if layer.crossbars:
            crossbar = layer.crossbars[0]
            
            # Flatten spatial dimensions for crossbar computation
            x_flat = x.flatten(2)  # [batch, channels, spatial]
            
            for b in range(batch_size):
                input_voltages = x_flat[b].detach().cpu().numpy().flatten()
                
                # Pad or truncate to match crossbar input size
                if len(input_voltages) > crossbar.rows:
                    input_voltages = input_voltages[:crossbar.rows]
                elif len(input_voltages) < crossbar.rows:
                    padded = np.zeros(crossbar.rows)
                    padded[:len(input_voltages)] = input_voltages
                    input_voltages = padded
                
                output_currents = crossbar.compute_vmm(input_voltages)
                
                # Reshape output (simplified)
                output_channels = min(len(output_currents), output.shape[1])
                output[b, :output_channels] = torch.tensor(
                    output_currents[:output_channels].reshape(-1, 1, 1),
                    dtype=x.dtype, device=x.device
                )
        
        return output
    
    def estimate_power(self, workload: Dict) -> Dict:
        """Estimate power consumption for given workload."""
        total_static = 0.0
        total_dynamic = 0.0
        
        for layer in self.layers:
            for crossbar in layer.crossbars:
                power_analysis = crossbar.power_analysis(workload)
                total_static += power_analysis['static_power']
                total_dynamic += power_analysis['dynamic_power']
        
        return {
            'static_power_w': total_static,
            'dynamic_power_w': total_dynamic,
            'total_power_w': total_static + total_dynamic,
            'layers': len(self.layers),
            'crossbars': self.total_crossbars,
            'mtj_cells': self.total_mtj_cells
        }
    
    def get_model_summary(self) -> Dict:
        """Get comprehensive model summary."""
        return {
            'name': self.name,
            'layers': len(self.layers),
            'total_crossbars': self.total_crossbars,
            'total_mtj_cells': self.total_mtj_cells,
            'quantization_bits': self.config.quantization_bits,
            'crossbar_size': self.config.crossbar_size,
            'layer_details': [
                {
                    'name': layer.name,
                    'type': layer.layer_type,
                    'input_shape': layer.input_shape,
                    'output_shape': layer.output_shape,
                    'crossbars': len(layer.crossbars),
                    'mtj_cells': sum(cb.rows * cb.cols for cb in layer.crossbars)
                }
                for layer in self.layers
            ]
        }


class SpintronConverter:
    """Main converter class for PyTorch to spintronic conversion."""
    
    def __init__(self, mtj_config: MTJConfig):
        self.mtj_config = mtj_config
        
    def convert(
        self,
        pytorch_model: nn.Module,
        quantization_bits: int = 8,
        crossbar_size: int = 128,
        model_name: Optional[str] = None
    ) -> SpintronicModel:
        """
        Convert PyTorch model to spintronic implementation.
        
        Args:
            pytorch_model: PyTorch model to convert
            quantization_bits: Quantization precision
            crossbar_size: Maximum crossbar array size
            model_name: Name for converted model
            
        Returns:
            SpintronicModel instance
        """
        if model_name is None:
            model_name = pytorch_model.__class__.__name__ + "_Spintronic"
        
        config = ConversionConfig(
            quantization_bits=quantization_bits,
            crossbar_size=crossbar_size,
            mtj_config=self.mtj_config
        )
        
        spintronic_model = SpintronicModel(model_name, config)
        
        # Analyze model structure
        dummy_input = self._get_dummy_input(pytorch_model)
        layer_info = self._extract_layer_info(pytorch_model, dummy_input)
        
        # Convert each layer
        for layer_name, info in layer_info.items():
            layer = self._convert_layer(layer_name, info, config)
            if layer is not None:
                spintronic_model.add_layer(layer)
        
        return spintronic_model
    
    def _get_dummy_input(self, model: nn.Module) -> torch.Tensor:
        """Generate dummy input for model analysis."""
        # Try to infer input shape from first layer
        first_layer = next(model.children())
        
        if isinstance(first_layer, nn.Conv2d):
            # Convolutional model
            in_channels = first_layer.in_channels
            return torch.randn(1, in_channels, 32, 32)
        elif isinstance(first_layer, nn.Linear):
            # Fully connected model  
            in_features = first_layer.in_features
            return torch.randn(1, in_features)
        else:
            # Default assumption
            return torch.randn(1, 3, 224, 224)
    
    def _extract_layer_info(
        self, 
        model: nn.Module, 
        dummy_input: torch.Tensor
    ) -> Dict[str, Dict[str, Any]]:
        """Extract layer information by forward pass analysis."""
        layer_info = {}
        
        def hook_fn(module, input, output):
            layer_name = f"{module.__class__.__name__}_{id(module)}"
            layer_info[layer_name] = {
                'module': module,
                'input_shape': input[0].shape if input else None,
                'output_shape': output.shape if hasattr(output, 'shape') else None,
                'layer_type': module.__class__.__name__
            }
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d, nn.ReLU, nn.Flatten, nn.AdaptiveAvgPool2d)):
                hook = module.register_forward_hook(hook_fn)
                hooks.append(hook)
        
        # Forward pass to collect information
        with torch.no_grad():
            _ = model(dummy_input)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        return layer_info
    
    def _convert_layer(
        self,
        layer_name: str,
        layer_info: Dict[str, Any],
        config: ConversionConfig
    ) -> Optional[SpintronicLayer]:
        """Convert individual layer to spintronic representation."""
        module = layer_info['module']
        layer_type = layer_info['layer_type']
        input_shape = layer_info['input_shape']
        output_shape = layer_info['output_shape']
        
        if isinstance(module, nn.Linear):
            weights = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
            
            return SpintronicLayer(
                name=layer_name,
                layer_type='Linear',
                input_shape=input_shape,
                output_shape=output_shape,
                weights=weights,
                bias=bias
            )
        
        elif isinstance(module, nn.Conv2d):
            weights = module.weight.detach().cpu().numpy()
            bias = module.bias.detach().cpu().numpy() if module.bias is not None else None
            
            return SpintronicLayer(
                name=layer_name,
                layer_type='Conv2d',
                input_shape=input_shape,
                output_shape=output_shape,
                weights=weights,
                bias=bias
            )
        
        elif isinstance(module, (nn.ReLU, nn.Flatten)):
            # These don't have weights but are part of computation
            return SpintronicLayer(
                name=layer_name,
                layer_type=layer_type,
                input_shape=input_shape,
                output_shape=output_shape
            )
        
        # Skip other layer types (BatchNorm, Pooling, etc. - keep in CMOS)
        return None