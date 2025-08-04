"""
Pre-optimized Vision Models for Spintronic Hardware.

This module provides vision models specifically optimized for 
spintronic neural network implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from ..core.mtj_models import MTJConfig
from ..converter.pytorch_parser import SpintronConverter
from ..training.qat import QuantizationConfig, SpintronicTrainer


logger = logging.getLogger(__name__)


@dataclass
class VisionModelConfig:
    """Configuration for spintronic vision models."""
    
    # Model architecture
    input_size: Tuple[int, int] = (32, 32)
    num_classes: int = 10
    channels: int = 1
    
    # Spintronic optimization
    mtj_precision: int = 3  # 3-bit weights using domain walls
    crossbar_size: int = 128
    enable_weight_sharing: bool = True
    
    # Energy optimization
    target_power_uw: float = 100.0  # 100 μW power budget
    optimize_for_energy: bool = True
    
    # Performance targets
    target_latency_ms: float = 10.0
    target_accuracy: float = 90.0  # 90% minimum accuracy


class SpintronicVisionBase(nn.Module):
    """Base class for spintronic vision models."""
    
    def __init__(self, config: VisionModelConfig):
        super().__init__()
        self.config = config
        self.mtj_config = MTJConfig()
        
        # Track model statistics
        self.model_stats = {
            'total_parameters': 0,
            'spintronic_layers': 0,
            'estimated_power_uw': 0.0,
            'estimated_latency_ms': 0.0
        }
    
    def _make_spintronic_conv(
        self, 
        in_channels: int, 
        out_channels: int, 
        kernel_size: int,
        stride: int = 1,
        padding: int = 0
    ) -> nn.Module:
        """Create spintronic-optimized convolutional layer."""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        
        # Initialize weights for spintronic efficiency
        with torch.no_grad():
            # Use smaller weight magnitudes for better MTJ mapping
            nn.init.uniform_(conv.weight, -0.5, 0.5)
            
            # Encourage sparse weights for energy efficiency
            mask = torch.rand_like(conv.weight) < 0.7  # 70% sparsity
            conv.weight.data *= mask.float()
        
        return conv
    
    def _make_spintronic_linear(
        self, 
        in_features: int, 
        out_features: int
    ) -> nn.Module:
        """Create spintronic-optimized linear layer."""
        linear = nn.Linear(in_features, out_features, bias=False)
        
        # Initialize for spintronic constraints
        with torch.no_grad():
            # Smaller weights for better quantization
            nn.init.uniform_(linear.weight, -0.3, 0.3)
            
            # Apply structured sparsity
            block_size = 8
            for i in range(0, linear.weight.size(0), block_size):
                for j in range(0, linear.weight.size(1), block_size):
                    block = linear.weight[i:i+block_size, j:j+block_size]
                    if torch.rand(1) < 0.3:  # 30% of blocks are zero
                        block.zero_()
        
        return linear
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        
        summary = {
            'model_type': self.__class__.__name__,
            'input_size': self.config.input_size,
            'num_classes': self.config.num_classes,
            'total_parameters': total_params,
            'mtj_precision': self.config.mtj_precision,
            'crossbar_size': self.config.crossbar_size,
            'estimated_power_uw': self._estimate_power(),
            'estimated_latency_ms': self._estimate_latency(),
            'target_power_uw': self.config.target_power_uw,
            'target_accuracy': self.config.target_accuracy
        }
        
        return summary
    
    def _estimate_power(self) -> float:
        """Estimate power consumption."""
        total_ops = 0
        
        # Count operations (simplified)
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                kernel_ops = np.prod(module.weight.shape)
                # Assume 32x32 feature maps on average
                feature_map_ops = kernel_ops * 32 * 32
                total_ops += feature_map_ops
            elif isinstance(module, nn.Linear):
                total_ops += module.weight.numel()
        
        # Power estimation (10 fJ/MAC for spintronic)
        energy_per_op = 10e-15  # 10 fJ
        frequency = 10e6  # 10 MHz typical
        power_watts = total_ops * energy_per_op * frequency
        
        return power_watts * 1e6  # Convert to μW
    
    def _estimate_latency(self) -> float:
        """Estimate inference latency."""
        # Simplified latency model
        total_layers = len([m for m in self.modules() if isinstance(m, (nn.Conv2d, nn.Linear))])
        latency_per_layer = 0.5  # 0.5 ms per layer typical
        
        return total_layers * latency_per_layer
    
    def optimize_for_hardware(self) -> 'SpintronicVisionBase':
        """Apply hardware-specific optimizations."""
        logger.info("Applying spintronic hardware optimizations...")
        
        # Quantization-aware optimization
        qconfig = QuantizationConfig(
            weight_bits=self.config.mtj_precision,
            enable_mtj_constraints=True,
            mtj_levels=2**self.config.mtj_precision
        )
        
        # This would typically involve QAT training
        # For now, we apply post-training optimizations
        self._apply_weight_clustering()
        self._apply_pruning()
        
        logger.info("Hardware optimizations applied")
        return self
    
    def _apply_weight_clustering(self):
        """Apply weight clustering for MTJ efficiency."""
        from sklearn.cluster import KMeans
        
        for name, module in self.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weight_data = module.weight.data
                original_shape = weight_data.shape
                
                # Flatten weights
                flat_weights = weight_data.flatten().cpu().numpy()
                
                # Cluster to MTJ levels
                n_clusters = 2**self.config.mtj_precision
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                
                # Fit and predict
                clustered = kmeans.fit_predict(flat_weights.reshape(-1, 1))
                
                # Replace weights with cluster centers
                new_weights = kmeans.cluster_centers_[clustered].flatten()
                
                # Update module weights
                module.weight.data = torch.tensor(
                    new_weights.reshape(original_shape),
                    dtype=weight_data.dtype,
                    device=weight_data.device
                )
    
    def _apply_pruning(self):
        """Apply structured pruning for energy efficiency."""
        sparsity_target = 0.5  # 50% sparsity
        
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weight = module.weight.data
                
                # Magnitude-based pruning
                weight_magnitude = torch.abs(weight)
                threshold = torch.quantile(weight_magnitude, sparsity_target)
                
                # Create mask
                mask = weight_magnitude > threshold
                
                # Apply pruning
                module.weight.data *= mask.float()


class TinyConvNet_Spintronic(SpintronicVisionBase):
    """Tiny convolutional network optimized for spintronic hardware."""
    
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 32, 32),
        num_classes: int = 10,
        filters: List[int] = [16, 32, 64],
        mtj_array_size: int = 128
    ):
        config = VisionModelConfig(
            input_size=input_shape[1:],
            num_classes=num_classes,
            channels=input_shape[0],
            crossbar_size=mtj_array_size
        )
        super().__init__(config)
        
        # Convolutional layers
        self.conv1 = self._make_spintronic_conv(input_shape[0], filters[0], 3, padding=1)
        self.conv2 = self._make_spintronic_conv(filters[0], filters[1], 3, padding=1)
        self.conv3 = self._make_spintronic_conv(filters[1], filters[2], 3, padding=1)
        
        # Activation and pooling
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Classifier
        self.flatten = nn.Flatten()
        self.classifier = self._make_spintronic_linear(filters[2] * 4 * 4, num_classes)
        
        # Apply initial optimizations
        self.optimize_for_hardware()
        
        logger.info(f"Created TinyConvNet with {sum(p.numel() for p in self.parameters())} parameters")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool(x)
        
        # Third conv block
        x = self.conv3(x)
        x = self.relu(x)
        x = self.adaptive_pool(x)
        
        # Classifier
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x


class MobileNetV2_Spintronic(SpintronicVisionBase):
    """Spintronic-optimized MobileNetV2 for edge inference."""
    
    def __init__(
        self,
        num_classes: int = 1000,
        input_size: Tuple[int, int] = (224, 224),
        width_mult: float = 0.25,  # Reduced for spintronic efficiency
        mtj_precision: int = 3
    ):
        config = VisionModelConfig(
            input_size=input_size,
            num_classes=num_classes,
            mtj_precision=mtj_precision
        )
        super().__init__(config)
        
        self.width_mult = width_mult
        
        # First layer
        input_channel = int(32 * width_mult)
        self.conv1 = self._make_spintronic_conv(3, input_channel, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(input_channel)
        self.relu = nn.ReLU6(inplace=True)
        
        # Inverted residual blocks (simplified)
        self.blocks = nn.ModuleList()
        
        # Block configurations: [expansion, output_channels, num_blocks, stride]
        block_configs = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 2, 1],  # Reduced from original
        ]
        
        in_channels = input_channel
        for expansion, out_channels, num_blocks, stride in block_configs:
            out_channels = int(out_channels * width_mult)
            
            for i in range(num_blocks):
                block_stride = stride if i == 0 else 1
                block = self._make_inverted_residual_block(
                    in_channels, out_channels, expansion, block_stride
                )
                self.blocks.append(block)
                in_channels = out_channels
        
        # Final layers
        final_channels = int(512 * width_mult)  # Reduced from 1280
        self.conv_final = self._make_spintronic_conv(in_channels, final_channels, 1)
        self.bn_final = nn.BatchNorm2d(final_channels)
        
        # Classifier
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = self._make_spintronic_linear(final_channels, num_classes)
        self.dropout = nn.Dropout(0.2)
        
        # Apply optimizations
        self.optimize_for_hardware()
        
        logger.info(f"Created MobileNetV2_Spintronic with width_mult={width_mult}")
    
    def _make_inverted_residual_block(
        self,
        in_channels: int,
        out_channels: int,
        expansion: int,
        stride: int
    ) -> nn.Module:
        """Create inverted residual block optimized for spintronic hardware."""
        
        class InvertedResidualBlock(nn.Module):
            def __init__(self, in_ch, out_ch, exp, stride):
                super().__init__()
                hidden_dim = in_ch * exp
                self.use_residual = stride == 1 and in_ch == out_ch
                
                layers = []
                
                # Expansion
                if exp != 1:
                    layers.extend([
                        nn.Conv2d(in_ch, hidden_dim, 1, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.ReLU6(inplace=True)
                    ])
                
                # Depthwise + Pointwise
                layers.extend([
                    nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, 
                             groups=hidden_dim, bias=False),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(hidden_dim, out_ch, 1, bias=False),
                    nn.BatchNorm2d(out_ch)
                ])
                
                self.conv = nn.Sequential(*layers)
            
            def forward(self, x):
                if self.use_residual:
                    return x + self.conv(x)
                else:
                    return self.conv(x)
        
        return InvertedResidualBlock(in_channels, out_channels, expansion, stride)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Inverted residual blocks
        for block in self.blocks:
            x = block(x)
        
        # Final conv
        x = self.conv_final(x)
        x = self.bn_final(x)
        x = self.relu(x)
        
        # Classifier
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x


class AlwaysOnVision_Spintronic(SpintronicVisionBase):
    """Ultra-low-power always-on vision model for spintronic hardware."""
    
    def __init__(
        self,
        resolution: Tuple[int, int] = (64, 64),
        detect_classes: List[str] = ['person', 'vehicle'],
        target_power_uw: float = 100  # 100 μW budget
    ):
        config = VisionModelConfig(
            input_size=resolution,
            num_classes=len(detect_classes),
            target_power_uw=target_power_uw,
            optimize_for_energy=True
        )
        super().__init__(config)
        
        self.detect_classes = detect_classes
        
        # Ultra-efficient architecture
        # Use very small filters and aggressive quantization
        self.stem = nn.Sequential(
            self._make_spintronic_conv(1, 8, 5, stride=2, padding=2),  # 64->32
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32->16
        )
        
        # Efficient feature extraction
        self.features = nn.Sequential(
            self._make_spintronic_conv(8, 16, 3, padding=1),
            nn.ReLU(),
            self._make_spintronic_conv(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16->8
            
            self._make_spintronic_conv(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        
        # Minimal classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            self._make_spintronic_linear(24 * 4 * 4, 32),
            nn.ReLU(),
            self._make_spintronic_linear(32, len(detect_classes))
        )
        
        # Apply aggressive optimizations for power
        self._apply_aggressive_optimizations()
        
        logger.info(f"Created AlwaysOnVision for {detect_classes} detection")
        logger.info(f"Target power: {target_power_uw} μW")
    
    def _apply_aggressive_optimizations(self):
        """Apply aggressive optimizations for ultra-low power."""
        # Extreme quantization (2-bit weights)
        self.config.mtj_precision = 2
        
        # High sparsity (80%)
        sparsity = 0.8
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weight = module.weight.data
                # Keep only the largest weights
                flat_weight = weight.flatten()
                threshold = torch.quantile(torch.abs(flat_weight), sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
        
        # Weight clustering to 4 levels (2-bit)
        self._apply_weight_clustering()
        
        logger.info("Applied aggressive power optimizations")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.classifier(x)
        return x
    
    def detect(self, image: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
        """Perform detection and return class probabilities."""
        with torch.no_grad():
            logits = self.forward(image)
            probabilities = torch.sigmoid(logits)
            
            results = {}
            for i, class_name in enumerate(self.detect_classes):
                prob = probabilities[0, i].item() if len(probabilities.shape) > 1 else probabilities[i].item()
                results[class_name] = prob
            
            return results
    
    def optimize_for_deployment(
        self,
        batch_size: int = 1,
        latency_constraint_ms: float = 100
    ) -> Dict[str, Any]:
        """Optimize model for deployment with constraints."""
        # Calculate current metrics
        current_power = self._estimate_power()
        current_latency = self._estimate_latency()
        
        optimization_report = {
            'original_power_uw': current_power,
            'original_latency_ms': current_latency,
            'target_power_uw': self.config.target_power_uw,
            'target_latency_ms': latency_constraint_ms,
            'optimizations_applied': []
        }
        
        # Apply optimizations if needed
        if current_power > self.config.target_power_uw:
            # Increase sparsity
            self._increase_sparsity(0.9)
            optimization_report['optimizations_applied'].append('increased_sparsity')
            
            # Reduce precision further if needed
            if current_power > self.config.target_power_uw * 1.2:
                self.config.mtj_precision = 1  # Binary weights
                self._apply_weight_clustering()
                optimization_report['optimizations_applied'].append('binary_weights')
        
        # Final metrics
        optimization_report['final_power_uw'] = self._estimate_power()
        optimization_report['final_latency_ms'] = self._estimate_latency()
        optimization_report['power_savings'] = (
            current_power - optimization_report['final_power_uw']
        ) / current_power * 100
        
        logger.info(f"Deployment optimization completed")
        logger.info(f"Power: {current_power:.1f} -> {optimization_report['final_power_uw']:.1f} μW")
        
        return optimization_report
    
    def _increase_sparsity(self, target_sparsity: float):
        """Increase model sparsity for power reduction."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, 'weight'):
                weight = module.weight.data
                flat_weight = weight.flatten()
                threshold = torch.quantile(torch.abs(flat_weight), target_sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
    
    def generate_system_design(
        self,
        include_image_sensor_interface: bool = True,
        include_spi_output: bool = True
    ) -> Dict[str, Any]:
        """Generate complete edge AI system design."""
        system_design = {
            'compute_core': {
                'model': self.__class__.__name__,
                'parameters': sum(p.numel() for p in self.parameters()),
                'estimated_power_uw': self._estimate_power(),
                'crossbars_needed': self._estimate_crossbars_needed()
            },
            'interfaces': {},
            'power_budget': {
                'compute_core_uw': self._estimate_power(),
                'image_sensor_uw': 50.0 if include_image_sensor_interface else 0.0,
                'interfaces_uw': 10.0,
                'total_system_uw': 0.0
            }
        }
        
        # Add interfaces
        if include_image_sensor_interface:
            system_design['interfaces']['image_sensor'] = {
                'type': 'MIPI CSI-2',
                'resolution': self.config.input_size,
                'frame_rate': 30,
                'power_uw': 50.0
            }
        
        if include_spi_output:
            system_design['interfaces']['output'] = {
                'type': 'SPI',
                'data_rate_mbps': 10,
                'power_uw': 5.0
            }
        
        # Calculate total power
        system_design['power_budget']['total_system_uw'] = sum(
            system_design['power_budget'].values()
        )
        
        logger.info(f"Generated system design with {system_design['power_budget']['total_system_uw']:.1f} μW total power")
        
        return system_design
    
    def _estimate_crossbars_needed(self) -> int:
        """Estimate number of crossbar arrays needed."""
        total_weights = sum(p.numel() for p in self.parameters() if p.dim() >= 2)
        crossbar_capacity = self.config.crossbar_size ** 2
        return int(np.ceil(total_weights / crossbar_capacity))