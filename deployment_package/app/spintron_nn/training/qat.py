"""
Quantization-Aware Training for Spintronic Neural Networks.

This module implements specialized quantization-aware training techniques
optimized for spintronic hardware constraints.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import numpy as np
import logging
from pathlib import Path
import json

from ..core.mtj_models import MTJConfig, estimate_switching_energy
from ..core.crossbar import MTJCrossbar, CrossbarConfig


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for quantization-aware training."""
    
    # Weight quantization
    weight_bits: int = 8
    weight_quantizer: str = "symmetric"  # "symmetric", "asymmetric"
    
    # Activation quantization
    activation_bits: int = 8
    activation_quantizer: str = "relu"  # "relu", "symmetric"
    
    # MTJ-specific parameters
    enable_mtj_constraints: bool = True
    mtj_levels: int = 4  # Number of resistance levels
    
    # Training parameters
    quant_delay: int = 0  # Epochs before enabling quantization
    observer_batch_size: int = 1000  # Batch size for calibration
    
    # Noise modeling
    enable_device_noise: bool = True
    noise_std: float = 0.01  # Standard deviation of device noise


class FakeQuantize(nn.Module):
    """Fake quantization module for QAT."""
    
    def __init__(
        self,
        num_bits: int = 8,
        quant_min: Optional[int] = None,
        quant_max: Optional[int] = None,
        observer: Optional[Callable] = None
    ):
        super().__init__()
        self.num_bits = num_bits
        
        if quant_min is None:
            quant_min = -(2 ** (num_bits - 1))
        if quant_max is None:
            quant_max = 2 ** (num_bits - 1) - 1
            
        self.quant_min = quant_min
        self.quant_max = quant_max
        
        # Learnable scale and zero point
        self.register_buffer('scale', torch.tensor(1.0))
        self.register_buffer('zero_point', torch.tensor(0.0))
        
        # Observer for calibration
        self.observer = observer or self._default_observer
        self.observer_enabled = True
        
    def _default_observer(self, x: torch.Tensor):
        """Default observer for computing scale and zero point."""
        if self.observer_enabled:
            x_min = x.min().item()
            x_max = x.max().item()
            
            # Symmetric quantization
            scale = max(abs(x_min), abs(x_max)) / (2 ** (self.num_bits - 1) - 1)
            self.scale.data = torch.tensor(max(scale, 1e-8))
            self.zero_point.data = torch.tensor(0.0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply fake quantization."""
        # Update observer
        self.observer(x)
        
        # Quantize
        x_quant = torch.round(x / self.scale + self.zero_point)
        x_quant = torch.clamp(x_quant, self.quant_min, self.quant_max)
        
        # Dequantize (fake quantization)
        x_dequant = (x_quant - self.zero_point) * self.scale
        
        return x_dequant
    
    def enable_observer(self):
        """Enable observer for calibration."""
        self.observer_enabled = True
    
    def disable_observer(self):
        """Disable observer after calibration."""
        self.observer_enabled = False


class MTJAwareQuantize(FakeQuantize):
    """MTJ-aware quantization considering device physics."""
    
    def __init__(
        self,
        mtj_config: MTJConfig,
        num_levels: int = 4,
        **kwargs
    ):
        super().__init__(num_bits=int(np.log2(num_levels)), **kwargs)
        self.mtj_config = mtj_config
        self.num_levels = num_levels
        
        # MTJ resistance levels
        r_min = mtj_config.resistance_low
        r_max = mtj_config.resistance_high
        self.resistance_levels = torch.linspace(r_min, r_max, num_levels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply MTJ-aware quantization."""
        # Map weights to conductance levels
        conductance_levels = 1.0 / self.resistance_levels
        
        # Find closest conductance level
        x_expanded = x.unsqueeze(-1)  # [..., 1]
        conductance_expanded = conductance_levels.unsqueeze(0).expand_as(
            x_expanded.expand(*x.shape, len(conductance_levels))
        )
        
        # Find closest level
        distances = torch.abs(x_expanded - conductance_expanded)
        closest_idx = torch.argmin(distances, dim=-1)
        
        # Map to quantized values
        x_quant = conductance_levels[closest_idx]
        
        # Add device noise if enabled
        if self.training and hasattr(self, 'device_noise_enabled'):
            noise = torch.randn_like(x_quant) * self.mtj_config.resistance_variation
            x_quant = x_quant + noise
        
        return x_quant


class SpintronicQATModule(nn.Module):
    """Base module for spintronic QAT."""
    
    def __init__(self, original_module: nn.Module, qconfig: QuantizationConfig):
        super().__init__()
        self.original_module = original_module
        self.qconfig = qconfig
        
        # Add quantizers
        if hasattr(original_module, 'weight'):
            if qconfig.enable_mtj_constraints:
                self.weight_quantizer = MTJAwareQuantize(
                    mtj_config=MTJConfig(),  # Default config
                    num_levels=qconfig.mtj_levels
                )
            else:
                self.weight_quantizer = FakeQuantize(num_bits=qconfig.weight_bits)
        
        self.activation_quantizer = FakeQuantize(num_bits=qconfig.activation_bits)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with quantization."""
        # Quantize input activations
        x_quant = self.activation_quantizer(x)
        
        # Quantize weights if applicable
        if hasattr(self.original_module, 'weight'):
            weight_quant = self.weight_quantizer(self.original_module.weight)
            
            # Replace weight temporarily
            original_weight = self.original_module.weight.data
            self.original_module.weight.data = weight_quant
            
            # Forward pass
            output = self.original_module(x_quant)
            
            # Restore original weight
            self.original_module.weight.data = original_weight
        else:
            output = self.original_module(x_quant)
        
        return output


class SpintronicTrainer:
    """Trainer for spintronic-aware quantization and optimization."""
    
    def __init__(
        self,
        model: nn.Module,
        qconfig: QuantizationConfig,
        mtj_config: Optional[MTJConfig] = None,
        device: str = "cpu"
    ):
        self.model = model
        self.qconfig = qconfig
        self.mtj_config = mtj_config or MTJConfig()
        self.device = device
        
        # Convert model to QAT
        self.qat_model = self._prepare_qat_model()
        
        # Training state
        self.current_epoch = 0
        self.training_history = []
        
        logger.info(f"Initialized SpintronicTrainer with {qconfig.weight_bits}-bit weights")
    
    def _prepare_qat_model(self) -> nn.Module:
        """Convert model to quantization-aware training."""
        qat_model = nn.Module()
        
        # Recursively replace modules with QAT versions
        for name, module in self.model.named_children():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                qat_module = SpintronicQATModule(module, self.qconfig)
                setattr(qat_model, name, qat_module)
            else:
                setattr(qat_model, name, module)
        
        return qat_model.to(self.device)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        learning_rate: float = 0.001,
        optimizer_class: type = optim.Adam,
        loss_fn: Optional[Callable] = None,
        scheduler: Optional[Any] = None
    ) -> Dict[str, List[float]]:
        """
        Train model with spintronic-aware quantization.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            optimizer_class: Optimizer class
            loss_fn: Loss function
            scheduler: Learning rate scheduler
            
        Returns:
            Training history dictionary
        """
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()
        
        optimizer = optimizer_class(self.qat_model.parameters(), lr=learning_rate)
        
        if scheduler is None:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'energy_cost': []
        }
        
        logger.info(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            # Enable quantization after delay
            if epoch >= self.qconfig.quant_delay:
                self._enable_quantization()
            
            # Train epoch
            train_metrics = self._train_epoch(train_loader, optimizer, loss_fn)
            
            # Validate
            val_metrics = self._validate_epoch(val_loader, loss_fn)
            
            # Update scheduler
            scheduler.step()
            
            # Log metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            # Estimate energy cost
            energy_cost = self._estimate_energy_cost()
            history['energy_cost'].append(energy_cost)
            
            # Log progress
            if epoch % 10 == 0 or epoch == num_epochs - 1:
                logger.info(
                    f"Epoch {epoch}: "
                    f"Train Loss={train_metrics['loss']:.4f}, "
                    f"Train Acc={train_metrics['accuracy']:.2f}%, "
                    f"Val Loss={val_metrics['loss']:.4f}, "
                    f"Val Acc={val_metrics['accuracy']:.2f}%, "
                    f"Energy={energy_cost:.2f} pJ"
                )
        
        self.training_history = history
        logger.info("Training completed successfully")
        
        return history
    
    def _train_epoch(
        self, 
        train_loader: DataLoader, 
        optimizer: optim.Optimizer,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.qat_model.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            output = self.qat_model(data)
            
            # Compute loss
            classification_loss = loss_fn(output, target)
            
            # Add energy regularization
            energy_loss = self._compute_energy_loss()
            energy_weight = 0.001  # Small weight for energy term
            
            total_loss_batch = classification_loss + energy_weight * energy_loss
            
            # Backward pass
            total_loss_batch.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.qat_model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Statistics
            total_loss += total_loss_batch.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def _validate_epoch(self, val_loader: DataLoader, loss_fn: Callable) -> Dict[str, float]:
        """Validate for one epoch."""
        self.qat_model.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                output = self.qat_model(data)
                loss = loss_fn(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def _enable_quantization(self):
        """Enable quantization for all QAT modules."""
        for module in self.qat_model.modules():
            if isinstance(module, SpintronicQATModule):
                if hasattr(module, 'weight_quantizer'):
                    module.weight_quantizer.disable_observer()
                module.activation_quantizer.disable_observer()
    
    def _compute_energy_loss(self) -> torch.Tensor:
        """Compute energy regularization loss."""
        total_energy = torch.tensor(0.0, device=self.device)
        
        for module in self.qat_model.modules():
            if isinstance(module, SpintronicQATModule) and hasattr(module, 'weight_quantizer'):
                # Get quantized weights
                weights = module.original_module.weight
                if hasattr(module, '_prev_weights'):
                    # Estimate switching energy based on weight changes
                    weight_changes = weights - module._prev_weights
                    switching_energy = estimate_switching_energy(weight_changes, self.mtj_config)
                    total_energy += switching_energy.sum()
                
                # Store current weights for next iteration
                module._prev_weights = weights.detach().clone()
        
        return total_energy
    
    def _estimate_energy_cost(self) -> float:
        """Estimate energy cost per inference.""" 
        # Simplified energy estimation
        total_ops = 0
        total_params = 0
        
        for module in self.qat_model.modules():
            if isinstance(module.original_module, nn.Linear):
                total_ops += module.original_module.weight.numel()
                total_params += module.original_module.weight.numel()
            elif isinstance(module.original_module, nn.Conv2d):
                # Estimate conv ops (simplified)
                weight = module.original_module.weight
                total_ops += weight.numel() * 32 * 32  # Assume 32x32 feature map
                total_params += weight.numel()
        
        # Energy per operation (pJ/MAC for spintronic)
        energy_per_mac = self.mtj_config.switching_energy * 1e12  # Convert to pJ
        
        return total_ops * energy_per_mac
    
    def calibrate(self, calibration_loader: DataLoader) -> None:
        """Calibrate quantization parameters."""
        logger.info("Calibrating quantization parameters...")
        
        self.qat_model.eval()
        
        # Enable observers
        for module in self.qat_model.modules():
            if isinstance(module, SpintronicQATModule):
                if hasattr(module, 'weight_quantizer'):
                    module.weight_quantizer.enable_observer()
                module.activation_quantizer.enable_observer()
        
        # Run calibration
        with torch.no_grad():
            for batch_idx, (data, _) in enumerate(calibration_loader):
                if batch_idx >= self.qconfig.observer_batch_size // data.size(0):
                    break
                    
                data = data.to(self.device)
                _ = self.qat_model(data)
        
        # Disable observers
        self._enable_quantization()
        
        logger.info("Calibration completed")
    
    def export_quantized_model(self, output_path: str) -> None:
        """Export quantized model for deployment."""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save model state
        torch.save(self.qat_model.state_dict(), output_path / 'quantized_model.pth')
        
        # Save quantization config
        config_dict = {
            'weight_bits': self.qconfig.weight_bits,
            'activation_bits': self.qconfig.activation_bits,
            'mtj_levels': self.qconfig.mtj_levels,
            'enable_mtj_constraints': self.qconfig.enable_mtj_constraints
        }
        
        with open(output_path / 'quantization_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Save training history if available
        if self.training_history:
            with open(output_path / 'training_history.json', 'w') as f:
                json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Quantized model exported to {output_path}")
    
    def get_model_statistics(self) -> Dict[str, Any]:
        """Get statistics about the quantized model."""
        stats = {
            'total_parameters': sum(p.numel() for p in self.qat_model.parameters()),
            'quantized_layers': 0,
            'weight_bits': self.qconfig.weight_bits,
            'activation_bits': self.qconfig.activation_bits,
            'estimated_model_size_mb': 0,
            'compression_ratio': 1.0
        }
        
        # Count quantized layers
        for module in self.qat_model.modules():
            if isinstance(module, SpintronicQATModule):
                stats['quantized_layers'] += 1
        
        # Estimate model size
        original_size = stats['total_parameters'] * 4  # 32-bit floats
        quantized_size = stats['total_parameters'] * (self.qconfig.weight_bits / 8)
        
        stats['estimated_model_size_mb'] = quantized_size / (1024 * 1024)
        stats['compression_ratio'] = original_size / quantized_size
        
        return stats