"""
Variation-Aware Training for Spintronic Neural Networks.

This module implements training techniques that account for device variations
and non-idealities in spintronic hardware.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
import numpy as np
import logging
from pathlib import Path
import json

from ..core.mtj_models import MTJConfig, MTJDevice
from .qat import SpintronicTrainer, QuantizationConfig


logger = logging.getLogger(__name__)


@dataclass
class VariationModel:
    """Model for device variations in spintronic hardware."""
    
    # Resistance variations
    resistance_mean: float = 1.0
    resistance_std: float = 0.1
    
    # Switching variations
    switching_voltage_mean: float = 1.0
    switching_voltage_std: float = 0.05
    
    # Retention variations
    retention_time_mean: float = 10.0  # years
    retention_time_std: float = 2.0
    
    # Temperature effects
    temperature_coefficient: float = -0.001  # per °C
    operating_temperature_range: Tuple[float, float] = (-40.0, 85.0)
    
    # Aging effects
    endurance_cycles: float = 1e12
    aging_rate: float = 0.1  # 10% degradation at end of life
    
    # Process variations
    process_corner: str = "typical"  # "fast", "slow", "typical"
    
    def sample_variations(self, shape: Tuple[int, ...]) -> Dict[str, torch.Tensor]:
        """Sample random variations for given shape."""
        variations = {}
        
        # Resistance variations (log-normal distribution)
        variations['resistance_factor'] = torch.exp(
            torch.normal(0, self.resistance_std, shape)
        )
        
        # Switching voltage variations
        variations['switching_factor'] = torch.normal(
            self.switching_voltage_mean,
            self.switching_voltage_std,
            shape
        )
        
        # Retention variations
        variations['retention_factor'] = torch.normal(
            self.retention_time_mean,
            self.retention_time_std,
            shape
        )
        
        return variations


class VariationAwareLayer(nn.Module):
    """Neural network layer with device variation modeling."""
    
    def __init__(
        self,
        original_layer: nn.Module,
        variation_model: VariationModel,
        enable_variations: bool = True
    ):
        super().__init__()
        self.original_layer = original_layer
        self.variation_model = variation_model
        self.enable_variations = enable_variations
        
        # Sample variations if layer has weights
        if hasattr(original_layer, 'weight') and original_layer.weight is not None:
            weight_shape = original_layer.weight.shape
            self.variations = variation_model.sample_variations(weight_shape)
            
            # Register as buffers so they're moved with the model
            for name, var in self.variations.items():
                self.register_buffer(f'variation_{name}', var)
        else:
            self.variations = {}
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with variation effects."""
        if not self.enable_variations or not self.variations:
            return self.original_layer(x)
        
        # Apply variations to weights
        if hasattr(self.original_layer, 'weight') and self.original_layer.weight is not None:
            # Get variation factors
            resistance_factor = self.variations.get('resistance_factor', 1.0)
            
            # Apply variations (simplified model)
            varied_weight = self.original_layer.weight * resistance_factor
            
            # Temporarily replace weights
            original_weight = self.original_layer.weight.data
            self.original_layer.weight.data = varied_weight
            
            # Forward pass
            output = self.original_layer(x)
            
            # Restore original weights
            self.original_layer.weight.data = original_weight
            
            return output
        else:
            return self.original_layer(x)
    
    def resample_variations(self):
        """Resample device variations."""
        if hasattr(self.original_layer, 'weight') and self.original_layer.weight is not None:
            weight_shape = self.original_layer.weight.shape
            new_variations = self.variation_model.sample_variations(weight_shape)
            
            for name, var in new_variations.items():
                if hasattr(self, f'variation_{name}'):
                    getattr(self, f'variation_{name}').data = var
    
    def set_temperature(self, temperature: float):
        """Update variations based on temperature."""
        if 'resistance_factor' in self.variations:
            temp_effect = 1.0 + self.variation_model.temperature_coefficient * (temperature - 25.0)
            self.variations['resistance_factor'] *= temp_effect


class VariationAwareTraining:
    """Training framework for variation-aware spintronic neural networks."""
    
    def __init__(
        self,
        model: nn.Module,
        variation_model: VariationModel,
        monte_carlo_samples: int = 100,
        qconfig: Optional[QuantizationConfig] = None
    ):
        self.model = model
        self.variation_model = variation_model
        self.monte_carlo_samples = monte_carlo_samples
        self.qconfig = qconfig or QuantizationConfig()
        
        # Convert model to variation-aware
        self.variation_model_obj = self._create_variation_aware_model()
        
        # Base trainer for quantization
        if qconfig:
            self.base_trainer = SpintronicTrainer(
                self.variation_model_obj, 
                qconfig
            )
        
        logger.info(f"Initialized variation-aware training with {monte_carlo_samples} MC samples")
    
    def _create_variation_aware_model(self) -> nn.Module:
        """Convert model to variation-aware version."""
        variation_model = nn.Module()
        
        def convert_layer(layer, name=""):
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                return VariationAwareLayer(layer, self.variation_model)
            elif isinstance(layer, nn.Module) and len(list(layer.children())) > 0:
                # Recursively convert child modules
                new_layer = type(layer)()
                for child_name, child in layer.named_children():
                    setattr(new_layer, child_name, convert_layer(child, child_name))
                return new_layer
            else:
                return layer
        
        # Convert all layers
        for name, layer in self.model.named_children():
            setattr(variation_model, name, convert_layer(layer, name))
        
        return variation_model
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 50,
        robustness_weight: float = 0.1,
        **trainer_kwargs
    ) -> Dict[str, List[float]]:
        """
        Train model with variation awareness.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Number of training epochs
            robustness_weight: Weight for robustness loss term
            **trainer_kwargs: Additional arguments for base trainer
            
        Returns:
            Training history with robustness metrics
        """
        if self.base_trainer:
            # Use quantization-aware training as base
            history = self.base_trainer.train(
                train_loader, val_loader, num_epochs, **trainer_kwargs
            )
        else:
            # Basic training loop
            history = self._basic_training_loop(
                train_loader, val_loader, num_epochs, **trainer_kwargs
            )
        
        # Add robustness training
        history = self._add_robustness_training(
            history, train_loader, val_loader, robustness_weight
        )
        
        return history
    
    def _basic_training_loop(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int,
        learning_rate: float = 0.001,
        **kwargs
    ) -> Dict[str, List[float]]:
        """Basic training loop without quantization."""
        optimizer = torch.optim.Adam(self.variation_model_obj.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Train epoch
            train_metrics = self._train_epoch_with_variations(
                train_loader, optimizer, criterion
            )
            
            # Validate
            val_metrics = self._validate_epoch(val_loader, criterion)
            
            # Record metrics
            history['train_loss'].append(train_metrics['loss'])
            history['train_acc'].append(train_metrics['accuracy'])
            history['val_loss'].append(val_metrics['loss'])
            history['val_acc'].append(val_metrics['accuracy'])
            
            if epoch % 10 == 0:
                logger.info(
                    f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, "
                    f"Val Acc={val_metrics['accuracy']:.2f}%"
                )
        
        return history
    
    def _train_epoch_with_variations(
        self,
        train_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train epoch with device variations."""
        self.variation_model_obj.train()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Sample new variations for each batch
            self._resample_all_variations()
            
            # Forward pass
            output = self.variation_model_obj(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
        
        return {
            'loss': total_loss / len(train_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def _validate_epoch(self, val_loader: DataLoader, criterion: nn.Module) -> Dict[str, float]:
        """Validate with Monte Carlo sampling over variations."""
        self.variation_model_obj.eval()
        
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                # Monte Carlo evaluation
                mc_outputs = []
                
                for _ in range(min(10, self.monte_carlo_samples)):  # Reduced for validation
                    self._resample_all_variations()
                    output = self.variation_model_obj(data)
                    mc_outputs.append(output)
                
                # Average predictions
                avg_output = torch.stack(mc_outputs).mean(dim=0)
                loss = criterion(avg_output, target)
                
                total_loss += loss.item()
                pred = avg_output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        return {
            'loss': total_loss / len(val_loader),
            'accuracy': 100.0 * correct / total
        }
    
    def _add_robustness_training(
        self,
        history: Dict[str, List[float]],
        train_loader: DataLoader,
        val_loader: DataLoader,
        robustness_weight: float
    ) -> Dict[str, List[float]]:
        """Add robustness metrics to training history."""
        # Evaluate robustness
        robustness_metrics = self.evaluate_robustness(val_loader)
        
        # Add to history
        history['robustness_accuracy'] = [robustness_metrics['accuracy']]
        history['robustness_std'] = [robustness_metrics['std']]
        history['worst_case_accuracy'] = [robustness_metrics['worst_case']]
        
        return history
    
    def _resample_all_variations(self):
        """Resample variations for all layers."""
        for module in self.variation_model_obj.modules():
            if isinstance(module, VariationAwareLayer):
                module.resample_variations()
    
    def evaluate_robustness(
        self,
        test_loader: DataLoader,
        temperature_range: Optional[Tuple[float, float]] = None
    ) -> Dict[str, float]:
        """
        Evaluate model robustness to variations.
        
        Args:
            test_loader: Test data loader
            temperature_range: Temperature range for evaluation
            
        Returns:
            Robustness metrics
        """
        self.variation_model_obj.eval()
        
        if temperature_range is None:
            temperature_range = self.variation_model.operating_temperature_range
        
        all_accuracies = []
        all_predictions = []
        
        logger.info(f"Evaluating robustness with {self.monte_carlo_samples} MC samples")
        
        with torch.no_grad():
            for mc_sample in range(self.monte_carlo_samples):
                # Resample variations
                self._resample_all_variations()
                
                # Sample temperature
                temperature = np.random.uniform(*temperature_range)
                self._set_temperature_all_layers(temperature)
                
                # Evaluate on test set
                correct = 0
                total = 0
                sample_predictions = []
                
                for data, target in test_loader:
                    output = self.variation_model_obj(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    sample_predictions.extend(pred.cpu().numpy().flatten())
                
                accuracy = 100.0 * correct / total
                all_accuracies.append(accuracy)
                all_predictions.append(sample_predictions)
        
        # Compute statistics
        accuracies = np.array(all_accuracies)
        
        robustness_metrics = {
            'accuracy': float(np.mean(accuracies)),
            'std': float(np.std(accuracies)),
            'min': float(np.min(accuracies)),
            'max': float(np.max(accuracies)),
            'worst_case': float(np.percentile(accuracies, 5)),  # 5th percentile
            'best_case': float(np.percentile(accuracies, 95)),   # 95th percentile
            'coefficient_of_variation': float(np.std(accuracies) / np.mean(accuracies))
        }
        
        logger.info(f"Robustness evaluation completed:")
        logger.info(f"  Mean accuracy: {robustness_metrics['accuracy']:.2f}%")
        logger.info(f"  Std deviation: {robustness_metrics['std']:.2f}%")
        logger.info(f"  Worst case (5%): {robustness_metrics['worst_case']:.2f}%")
        
        return robustness_metrics
    
    def _set_temperature_all_layers(self, temperature: float):
        """Set temperature for all variation-aware layers."""
        for module in self.variation_model_obj.modules():
            if isinstance(module, VariationAwareLayer):
                module.set_temperature(temperature)
    
    def analyze_sensitivity(
        self,
        test_loader: DataLoader,
        variation_factors: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze sensitivity to different variation factors.
        
        Args:
            test_loader: Test data loader
            variation_factors: Dictionary of variation factors to test
            
        Returns:
            Sensitivity analysis results
        """
        self.variation_model_obj.eval()
        sensitivity_results = {}
        
        logger.info("Running sensitivity analysis...")
        
        with torch.no_grad():
            for factor_name, factor_values in variation_factors.items():
                factor_accuracies = []
                
                for factor_value in factor_values:
                    # Create modified variation model
                    modified_model = VariationModel()
                    
                    # Set the specific factor
                    if factor_name == 'resistance_std':
                        modified_model.resistance_std = factor_value
                    elif factor_name == 'switching_voltage_std':
                        modified_model.switching_voltage_std = factor_value
                    elif factor_name == 'temperature_coefficient':
                        modified_model.temperature_coefficient = factor_value
                    
                    # Update layers with new variation model
                    self._update_variation_model(modified_model)
                    
                    # Evaluate accuracy
                    correct = 0
                    total = 0
                    
                    for data, target in test_loader:
                        output = self.variation_model_obj(data)
                        pred = output.argmax(dim=1, keepdim=True)
                        correct += pred.eq(target.view_as(pred)).sum().item()
                        total += target.size(0)
                    
                    accuracy = 100.0 * correct / total
                    factor_accuracies.append(accuracy)
                
                # Compute sensitivity metrics
                accuracies = np.array(factor_accuracies)
                sensitivity_results[factor_name] = {
                    'values': factor_values,
                    'accuracies': factor_accuracies,
                    'sensitivity': float(np.std(accuracies)),  # Higher std = more sensitive
                    'accuracy_drop': float(max(factor_accuracies) - min(factor_accuracies))
                }
        
        logger.info("Sensitivity analysis completed")
        return sensitivity_results
    
    def _update_variation_model(self, new_variation_model: VariationModel):
        """Update variation model for all layers."""
        for module in self.variation_model_obj.modules():
            if isinstance(module, VariationAwareLayer):
                module.variation_model = new_variation_model
                module.resample_variations()
    
    def export_robust_model(self, output_path: str, include_variations: bool = False):
        """Export variation-aware model."""
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save model
        if include_variations:
            # Save with variation layers
            torch.save(self.variation_model_obj.state_dict(), 
                      output_path / 'robust_model_with_variations.pth')
        else:
            # Extract original model without variation layers
            clean_model = self._extract_clean_model()
            torch.save(clean_model.state_dict(), 
                      output_path / 'robust_model_clean.pth')
        
        # Save variation model configuration
        variation_config = {
            'resistance_std': self.variation_model.resistance_std,
            'switching_voltage_std': self.variation_model.switching_voltage_std,
            'temperature_coefficient': self.variation_model.temperature_coefficient,
            'operating_temperature_range': self.variation_model.operating_temperature_range,
            'monte_carlo_samples': self.monte_carlo_samples
        }
        
        with open(output_path / 'variation_config.json', 'w') as f:
            json.dump(variation_config, f, indent=2)
        
        logger.info(f"Robust model exported to {output_path}")
    
    def _extract_clean_model(self) -> nn.Module:
        """Extract clean model without variation layers."""
        clean_model = nn.Module()
        
        def extract_layer(layer):
            if isinstance(layer, VariationAwareLayer):
                return layer.original_layer
            elif isinstance(layer, nn.Module) and len(list(layer.children())) > 0:
                new_layer = type(layer)()
                for child_name, child in layer.named_children():
                    setattr(new_layer, child_name, extract_layer(child))
                return new_layer
            else:
                return layer
        
        for name, layer in self.variation_model_obj.named_children():
            setattr(clean_model, name, extract_layer(layer))
        
        return clean_model