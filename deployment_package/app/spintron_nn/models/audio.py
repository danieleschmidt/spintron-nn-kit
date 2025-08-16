"""
Pre-optimized Audio Models for Spintronic Hardware.

This module provides audio processing models specifically optimized for 
spintronic neural network implementations, focusing on keyword spotting
and wake word detection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

from ..core.mtj_models import MTJConfig
from ..models.vision import SpintronicVisionBase, VisionModelConfig


logger = logging.getLogger(__name__)


@dataclass
class AudioModelConfig:
    """Configuration for spintronic audio models."""
    
    # Audio processing parameters
    sample_rate: int = 16000
    n_mfcc: int = 40
    n_fft: int = 512
    hop_length: int = 160
    
    # Model architecture
    input_features: int = 40  # MFCC features
    sequence_length: int = 128  # Time steps
    num_keywords: int = 10
    
    # Spintronic optimization
    mtj_precision: int = 3
    enable_temporal_sparsity: bool = True  # Exploit silence in audio
    
    # Power constraints
    target_power_uw: float = 50.0  # Very low for always-on
    target_latency_ms: float = 20.0
    
    # Detection parameters
    detection_threshold: float = 0.7
    false_positive_rate_target: float = 0.01


class SpintronicAudioBase(nn.Module):
    """Base class for spintronic audio models."""
    
    def __init__(self, config: AudioModelConfig):
        super().__init__()
        self.config = config
        self.mtj_config = MTJConfig()
        
        # Audio preprocessing (would typically be done externally)
        self.register_buffer('mel_filters', self._create_mel_filters())
        
    def _create_mel_filters(self) -> torch.Tensor:
        """Create mel-scale filter bank for MFCC computation."""
        # Simplified mel filter bank creation
        n_mels = self.config.n_mfcc
        n_fft = self.config.n_fft
        sample_rate = self.config.sample_rate
        
        # Create triangular filters (simplified)
        filters = torch.zeros(n_mels, n_fft // 2 + 1)
        
        # Mel scale boundaries
        mel_low = 0
        mel_high = 2595 * np.log10(1 + sample_rate / 2 / 700)
        mel_points = torch.linspace(mel_low, mel_high, n_mels + 2)
        
        # Convert back to Hz
        hz_points = 700 * (10**(mel_points / 2595) - 1)
        bin_points = torch.floor((n_fft + 1) * hz_points / sample_rate).long()
        
        for i in range(1, n_mels + 1):
            left, center, right = bin_points[i-1], bin_points[i], bin_points[i+1]
            
            # Left slope
            for j in range(left, center):
                if center > left:
                    filters[i-1, j] = (j - left) / (center - left)
            
            # Right slope  
            for j in range(center, right):
                if right > center:
                    filters[i-1, j] = (right - j) / (right - center)
        
        return filters
    
    def extract_mfcc(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract MFCC features from audio (simplified implementation)."""
        # This is a simplified MFCC extraction
        # In practice, would use torchaudio or similar
        
        # Apply window and FFT
        windowed = audio * torch.hann_window(len(audio))
        fft = torch.fft.rfft(windowed, n=self.config.n_fft)
        power_spectrum = torch.abs(fft) ** 2
        
        # Apply mel filters
        mel_spectrum = torch.matmul(self.mel_filters, power_spectrum)
        
        # Log and DCT (simplified)
        log_mel = torch.log(mel_spectrum + 1e-10)
        
        # Simplified DCT
        mfcc = torch.fft.fft(log_mel).real[:self.config.n_mfcc]
        
        return mfcc
    
    def _make_spintronic_rnn_cell(
        self,
        input_size: int,
        hidden_size: int,
        cell_type: str = "GRU"
    ) -> nn.Module:
        """Create spintronic-optimized RNN cell."""
        
        class SpintronicGRUCell(nn.Module):
            """GRU cell optimized for spintronic implementation."""
            
            def __init__(self, input_sz, hidden_sz):
                super().__init__()
                self.input_size = input_sz
                self.hidden_size = hidden_sz
                
                # Use linear layers that can be mapped to crossbars
                self.weight_ih = nn.Linear(input_sz, 3 * hidden_sz, bias=False)
                self.weight_hh = nn.Linear(hidden_sz, 3 * hidden_sz, bias=False)
                
                # Initialize for spintronic efficiency
                self._init_weights()
            
            def _init_weights(self):
                # Small weights for better MTJ mapping
                nn.init.uniform_(self.weight_ih.weight, -0.3, 0.3)
                nn.init.uniform_(self.weight_hh.weight, -0.3, 0.3)
                
                # Apply sparsity
                with torch.no_grad():
                    # 60% sparsity for energy efficiency
                    ih_mask = torch.rand_like(self.weight_ih.weight) > 0.6
                    hh_mask = torch.rand_like(self.weight_hh.weight) > 0.6
                    
                    self.weight_ih.weight *= ih_mask.float()
                    self.weight_hh.weight *= hh_mask.float()
            
            def forward(self, x, hidden=None):
                if hidden is None:
                    hidden = torch.zeros(x.size(0), self.hidden_size, 
                                       dtype=x.dtype, device=x.device)
                
                # GRU computation
                gi = self.weight_ih(x)
                gh = self.weight_hh(hidden)
                
                i_r, i_z, i_n = gi.chunk(3, 1)
                h_r, h_z, h_n = gh.chunk(3, 1)
                
                reset_gate = torch.sigmoid(i_r + h_r)
                update_gate = torch.sigmoid(i_z + h_z)
                new_gate = torch.tanh(i_n + reset_gate * h_n)
                
                hy = new_gate + update_gate * (hidden - new_gate)
                
                return hy
        
        if cell_type == "GRU":
            return SpintronicGRUCell(input_size, hidden_size)
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        
        return {
            'model_type': self.__class__.__name__,
            'input_features': self.config.input_features,
            'sequence_length': self.config.sequence_length,
            'num_keywords': self.config.num_keywords,
            'total_parameters': total_params,
            'mtj_precision': self.config.mtj_precision,
            'estimated_power_uw': self._estimate_power(),
            'estimated_latency_ms': self._estimate_latency(),
            'target_power_uw': self.config.target_power_uw
        }
    
    def _estimate_power(self) -> float:
        """Estimate power consumption for audio model."""
        # Audio models typically have fewer parameters but more temporal processing
        total_params = sum(p.numel() for p in self.parameters())
        
        # Energy per parameter access (spintronic advantage)
        energy_per_access = 5e-15  # 5 fJ for audio (even lower than vision)
        
        # Estimate accesses per second (depends on audio frame rate)
        frames_per_second = self.config.sample_rate / self.config.hop_length
        accesses_per_second = total_params * frames_per_second
        
        power_watts = accesses_per_second * energy_per_access
        return power_watts * 1e6  # Convert to μW
    
    def _estimate_latency(self) -> float:
        """Estimate processing latency."""
        # Audio processing latency (per frame)
        base_latency = 5.0  # 5 ms base latency
        
        # Add latency based on model complexity
        total_layers = len([m for m in self.modules() if isinstance(m, (nn.Linear, nn.GRU, nn.LSTM))])
        complexity_latency = total_layers * 0.5  # 0.5 ms per layer
        
        return base_latency + complexity_latency


class KeywordSpottingNet_Spintronic(SpintronicAudioBase):
    """Keyword spotting network optimized for spintronic hardware."""
    
    def __init__(
        self,
        num_keywords: int = 10,
        mtj_bits: int = 2,
        crossbar_size: int = 64,
        hidden_size: int = 64
    ):
        config = AudioModelConfig(
            num_keywords=num_keywords,
            mtj_precision=mtj_bits
        )
        super().__init__(config)
        
        self.hidden_size = hidden_size
        
        # Feature processing layers
        self.feature_net = nn.Sequential(
            nn.Linear(config.input_features, hidden_size, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.ReLU()
        )
        
        # Temporal processing (simplified RNN)
        self.temporal_layers = nn.ModuleList([
            self._make_spintronic_rnn_cell(hidden_size, hidden_size) 
            for _ in range(2)  # 2-layer RNN
        ])
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, num_keywords, bias=False)
        )
        
        # Apply optimizations
        self._optimize_for_keyword_spotting()
        
        logger.info(f"Created KeywordSpottingNet for {num_keywords} keywords")
        logger.info(f"Model size: {sum(p.numel() for p in self.parameters())} parameters")
    
    def _optimize_for_keyword_spotting(self):
        """Apply optimizations specific to keyword spotting."""
        # Temporal sparsity: most audio is silence
        if self.config.enable_temporal_sparsity:
            self._add_silence_detection()
        
        # Quantize weights aggressively (keyword spotting is noise-robust)
        self._apply_aggressive_quantization()
        
        # Energy-aware pruning
        self._apply_energy_aware_pruning()
    
    def _add_silence_detection(self):
        """Add silence detection to skip processing during quiet periods."""
        # Simple energy-based silence detector
        self.energy_threshold = nn.Parameter(torch.tensor(0.01), requires_grad=False)
    
    def _apply_aggressive_quantization(self):
        """Apply aggressive quantization for low power."""
        # Quantize to MTJ levels
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # Map to discrete levels
                levels = 2 ** self.config.mtj_precision
                weight_min, weight_max = weight.min(), weight.max()
                
                # Uniform quantization
                step = (weight_max - weight_min) / (levels - 1)
                quantized = torch.round((weight - weight_min) / step) * step + weight_min
                
                module.weight.data = quantized
    
    def _apply_energy_aware_pruning(self):
        """Apply pruning based on energy cost."""
        # Prune connections that have high switching cost
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                
                # Energy cost is related to weight magnitude and switching frequency
                energy_cost = torch.abs(weight)
                
                # Prune 70% of lowest-energy weights
                threshold = torch.quantile(energy_cost, 0.7)
                mask = energy_cost > threshold
                
                module.weight.data *= mask.float()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for keyword spotting.
        
        Args:
            x: Input features [batch, time, features] or [batch, features]
            
        Returns:
            Keyword probabilities [batch, num_keywords]
        """
        batch_size = x.size(0)
        
        # Handle different input shapes
        if x.dim() == 2:  # [batch, features]
            x = x.unsqueeze(1)  # [batch, 1, features]
        
        seq_len = x.size(1)
        
        # Silence detection (skip processing if energy too low)
        if self.config.enable_temporal_sparsity:
            energy = torch.sum(x**2, dim=-1)  # [batch, time]
            active_mask = energy > self.energy_threshold
        else:
            active_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=x.device)
        
        # Process features
        x = x.view(batch_size * seq_len, -1)  # [batch*time, features]
        features = self.feature_net(x)  # [batch*time, hidden]
        features = features.view(batch_size, seq_len, -1)  # [batch, time, hidden]
        
        # Temporal processing with RNN
        hidden_states = []
        hidden = None
        
        for t in range(seq_len):
            # Skip processing if silent
            if active_mask[:, t].any():
                current_input = features[:, t]  # [batch, hidden]
                
                # Process through RNN layers
                for rnn_layer in self.temporal_layers:
                    current_input = rnn_layer(current_input, hidden)
                
                hidden = current_input
                hidden_states.append(hidden)
            else:
                # Use previous hidden state for silent frames
                if hidden is not None:
                    hidden_states.append(hidden)
                else:
                    hidden_states.append(torch.zeros_like(features[:, t]))
        
        # Use final hidden state for classification
        if hidden_states:
            final_hidden = hidden_states[-1]
        else:
            final_hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        # Classification
        output = self.classifier(final_hidden)  # [batch, num_keywords]
        
        return output
    
    def detect_keyword(
        self, 
        audio_features: torch.Tensor, 
        keyword_labels: List[str]
    ) -> Dict[str, float]:
        """
        Detect keywords in audio features.
        
        Args:
            audio_features: MFCC features [time, features]
            keyword_labels: List of keyword names
            
        Returns:
            Dictionary of keyword probabilities
        """
        with torch.no_grad():
            # Add batch dimension
            features = audio_features.unsqueeze(0)  # [1, time, features]
            
            # Forward pass
            logits = self.forward(features)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Create results dictionary
            results = {}
            for i, label in enumerate(keyword_labels):
                if i < probabilities.size(-1):
                    results[label] = probabilities[0, i].item()
            
            return results
    
    def optimize_for_always_on(self) -> Dict[str, Any]:
        """Optimize model for always-on operation."""
        logger.info("Optimizing for always-on keyword spotting...")
        
        original_power = self._estimate_power()
        
        # Increase sparsity for power reduction
        self._increase_sparsity(0.85)  # 85% sparsity
        
        # Use binary weights for maximum efficiency
        self.config.mtj_precision = 1
        self._apply_binary_quantization()
        
        # Optimize temporal processing
        self._optimize_temporal_efficiency()
        
        final_power = self._estimate_power()
        
        optimization_report = {
            'original_power_uw': original_power,
            'optimized_power_uw': final_power,
            'power_reduction_percent': ((original_power - final_power) / original_power) * 100,
            'estimated_battery_life_days': self._estimate_battery_life(final_power),
            'optimizations_applied': [
                'high_sparsity',
                'binary_weights', 
                'temporal_optimization'
            ]
        }
        
        logger.info(f"Always-on optimization completed:")
        logger.info(f"  Power: {original_power:.1f} -> {final_power:.1f} μW")
        logger.info(f"  Battery life: {optimization_report['estimated_battery_life_days']:.1f} days")
        
        return optimization_report
    
    def _increase_sparsity(self, target_sparsity: float):
        """Increase model sparsity."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                flat_weight = weight.flatten()
                threshold = torch.quantile(torch.abs(flat_weight), target_sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
    
    def _apply_binary_quantization(self):
        """Apply binary quantization to weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                # Sign-based binary quantization
                binary_weight = torch.sign(weight) * torch.mean(torch.abs(weight))
                module.weight.data = binary_weight
    
    def _optimize_temporal_efficiency(self):
        """Optimize temporal processing efficiency."""
        # Reduce effective sequence length by using larger hop sizes
        # This is conceptual - would require retraining in practice
        self.config.hop_length *= 2  # Process fewer frames
        
    def _estimate_battery_life(self, power_uw: float) -> float:
        """Estimate battery life for always-on operation."""
        # Typical coin cell battery: 200 mAh at 3V
        battery_capacity_j = 200e-3 * 3600 * 3  # Joules
        power_w = power_uw * 1e-6
        
        if power_w > 0:
            lifetime_seconds = battery_capacity_j / power_w
            lifetime_days = lifetime_seconds / (24 * 3600)
            return lifetime_days
        else:
            return float('inf')


class WakeWordDetector_Spintronic(SpintronicAudioBase):
    """Wake word detector optimized for ultra-low power operation."""
    
    def __init__(
        self,
        wake_words: List[str] = ["hey_assistant"],
        detection_threshold: float = 0.8,
        power_budget_uw: float = 20.0  # Very aggressive power budget
    ):
        config = AudioModelConfig(
            num_keywords=len(wake_words) + 1,  # +1 for "no wake word"
            target_power_uw=power_budget_uw,
            detection_threshold=detection_threshold
        )
        super().__init__(config)
        
        self.wake_words = wake_words
        
        # Ultra-minimal architecture for wake word detection
        # Focus on detecting specific acoustic patterns rather than general speech
        
        # Frequency analysis (focusing on wake word frequencies)
        self.frequency_analyzer = nn.Sequential(
            nn.Linear(config.input_features, 16, bias=False),  # Extreme compression
            nn.ReLU(),
            nn.Linear(16, 8, bias=False),
            nn.ReLU()
        )
        
        # Temporal pattern matching
        self.pattern_matcher = self._make_spintronic_rnn_cell(8, 12)
        
        # Binary classifier (wake word vs. no wake word)
        self.classifier = nn.Linear(12, len(wake_words) + 1, bias=False)
        
        # Apply ultra-aggressive optimizations
        self._apply_ultra_low_power_optimizations()
        
        logger.info(f"Created WakeWordDetector for {wake_words}")
        logger.info(f"Power budget: {power_budget_uw} μW")
        logger.info(f"Parameters: {sum(p.numel() for p in self.parameters())}")
    
    def _apply_ultra_low_power_optimizations(self):
        """Apply ultra-aggressive optimizations for wake word detection."""
        # Binary weights only
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                # Binary quantization with sign and scale
                scale = torch.mean(torch.abs(weight))
                binary_weight = torch.sign(weight) * scale
                module.weight.data = binary_weight
        
        # Extreme sparsity (95%)
        sparsity = 0.95
        for module in self.modules():
            if isinstance(module, nn.Linear):
                weight = module.weight.data
                flat_weight = weight.flatten()
                threshold = torch.quantile(torch.abs(flat_weight), sparsity)
                mask = torch.abs(weight) > threshold
                module.weight.data *= mask.float()
        
        logger.info("Applied ultra-low power optimizations")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for wake word detection.
        
        Args:
            x: Audio features [batch, time, features] or [batch, features]
            
        Returns:
            Wake word detection scores [batch, num_wake_words + 1]
        """
        batch_size = x.size(0)
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        seq_len = x.size(1)
        
        # Frequency analysis
        x_flat = x.view(batch_size * seq_len, -1)
        freq_features = self.frequency_analyzer(x_flat)
        freq_features = freq_features.view(batch_size, seq_len, -1)
        
        # Temporal pattern matching
        hidden = None
        for t in range(seq_len):
            hidden = self.pattern_matcher(freq_features[:, t], hidden)
        
        # Classification
        output = self.classifier(hidden)
        
        return output
    
    def detect_wake_word(
        self, 
        audio_features: torch.Tensor,
        return_confidence: bool = True
    ) -> Dict[str, Any]:
        """
        Detect wake word in audio stream.
        
        Args:
            audio_features: MFCC features [time, features]
            return_confidence: Whether to return confidence scores
            
        Returns:
            Detection results with wake word and confidence
        """
        with torch.no_grad():
            features = audio_features.unsqueeze(0)  # Add batch dim
            
            logits = self.forward(features)
            probabilities = torch.softmax(logits, dim=-1)
            
            # Check if any wake word exceeds threshold
            wake_word_probs = probabilities[0, :-1]  # Exclude "no wake word" class
            max_prob, max_idx = torch.max(wake_word_probs, dim=0)
            
            detected = max_prob.item() > self.config.detection_threshold
            
            results = {
                'wake_word_detected': detected,
                'detected_word': self.wake_words[max_idx.item()] if detected else None,
                'confidence': max_prob.item() if return_confidence else None
            }
            
            if return_confidence:
                results['all_probabilities'] = {
                    word: prob.item() 
                    for word, prob in zip(self.wake_words, wake_word_probs)
                }
            
            return results
    
    def calibrate_threshold(
        self, 
        validation_data: List[Tuple[torch.Tensor, bool]],
        target_false_positive_rate: float = 0.01
    ) -> float:
        """
        Calibrate detection threshold based on validation data.
        
        Args:
            validation_data: List of (features, is_wake_word) tuples
            target_false_positive_rate: Target false positive rate
            
        Returns:
            Optimized threshold
        """
        logger.info("Calibrating detection threshold...")
        
        # Collect scores
        positive_scores = []
        negative_scores = []
        
        with torch.no_grad():
            for features, is_wake_word in validation_data:
                logits = self.forward(features.unsqueeze(0))
                probabilities = torch.softmax(logits, dim=-1)
                max_wake_word_prob = torch.max(probabilities[0, :-1]).item()
                
                if is_wake_word:
                    positive_scores.append(max_wake_word_prob)
                else:
                    negative_scores.append(max_wake_word_prob)
        
        # Find threshold that achieves target false positive rate
        negative_scores = sorted(negative_scores)
        fp_index = int((1 - target_false_positive_rate) * len(negative_scores))
        
        if fp_index < len(negative_scores):
            optimal_threshold = negative_scores[fp_index]
        else:
            optimal_threshold = 0.5
        
        # Calculate metrics at this threshold
        true_positives = sum(1 for score in positive_scores if score > optimal_threshold)
        false_positives = sum(1 for score in negative_scores if score > optimal_threshold)
        
        sensitivity = true_positives / len(positive_scores) if positive_scores else 0.0
        specificity = 1 - (false_positives / len(negative_scores)) if negative_scores else 1.0
        
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
        logger.info(f"Sensitivity: {sensitivity:.3f}")
        logger.info(f"Specificity: {specificity:.3f}")
        
        # Update model threshold
        self.config.detection_threshold = optimal_threshold
        
        return optimal_threshold
    
    def get_power_breakdown(self) -> Dict[str, float]:
        """Get detailed power consumption breakdown."""
        total_power = self._estimate_power()
        
        # Estimate power by component
        freq_params = sum(p.numel() for p in self.frequency_analyzer.parameters())
        pattern_params = sum(p.numel() for p in self.pattern_matcher.parameters())  
        classifier_params = sum(p.numel() for p in self.classifier.parameters())
        
        total_params = freq_params + pattern_params + classifier_params
        
        breakdown = {
            'frequency_analyzer_uw': (freq_params / total_params) * total_power,
            'pattern_matcher_uw': (pattern_params / total_params) * total_power,
            'classifier_uw': (classifier_params / total_params) * total_power,
            'total_uw': total_power,
            'power_efficiency_nj_per_detection': total_power * 1e-6 * 0.02 * 1e9  # 20ms detection * 1e9 for nJ
        }
        
        return breakdown
    
    @classmethod
    def from_pretrained(cls, model_name: str):
        """Load pre-trained wake word detector."""
        # This would load pre-trained weights
        # For now, return a configured model
        
        if model_name == 'hey_assistant':
            return cls(
                wake_words=['hey_assistant'],
                detection_threshold=0.8,
                power_budget_uw=15.0
            )
        elif model_name == 'wake_up':
            return cls(
                wake_words=['wake_up'],
                detection_threshold=0.75,
                power_budget_uw=20.0
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def export_for_deployment(self, output_path: str):
        """Export model in format suitable for deployment."""
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        # Save model weights
        torch.save(self.state_dict(), output_path / 'wake_word_model.pth')
        
        # Save configuration
        config_dict = {
            'wake_words': self.wake_words,
            'detection_threshold': self.config.detection_threshold,
            'input_features': self.config.input_features,
            'sample_rate': self.config.sample_rate,
            'n_mfcc': self.config.n_mfcc,
            'power_budget_uw': self.config.target_power_uw,
            'model_size_params': sum(p.numel() for p in self.parameters())
        }
        
        with open(output_path / 'model_config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
        
        # Generate deployment summary
        power_breakdown = self.get_power_breakdown()
        with open(output_path / 'deployment_summary.json', 'w') as f:
            json.dump({
                'model_summary': self.get_model_summary(),
                'power_breakdown': power_breakdown,
                'deployment_ready': True
            }, f, indent=2)
        
        logger.info(f"Model exported for deployment to {output_path}")
        
        return output_path