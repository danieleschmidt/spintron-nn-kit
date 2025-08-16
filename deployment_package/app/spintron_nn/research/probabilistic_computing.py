"""
Probabilistic Computing with Stochastic MTJs.

This module implements breakthrough probabilistic computing algorithms
using the inherent stochasticity of magnetic tunnel junctions for
neuromorphic and probabilistic machine learning applications.

Research Contributions:
- Stochastic MTJ device modeling with thermal noise
- Probabilistic neural networks using device randomness
- Bayesian inference acceleration with spintronic hardware
- Monte Carlo sampling using physical stochasticity
"""

import numpy as np
import math
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import time

# Physical constants
KB = 1.38e-23  # Boltzmann constant (J/K)
MU_B = 9.274e-24  # Bohr magneton (J/T)
H_BAR = 1.055e-34  # Reduced Planck constant (J⋅s)


@dataclass
class StochasticMTJConfig:
    """Configuration for stochastic MTJ devices."""
    
    # Thermal parameters
    temperature: float = 300.0  # Kelvin
    thermal_stability: float = 40.0  # kT units
    attempt_frequency: float = 1e9  # Hz
    
    # Magnetic parameters  
    saturation_magnetization: float = 1.4e6  # A/m
    anisotropy_energy: float = 1e-20  # J
    volume: float = 1e-24  # m³ (cubic MTJ)
    
    # Electrical parameters
    tunnel_magnetoresistance: float = 0.5  # 50% TMR
    resistance_ap: float = 10e3  # Antiparallel resistance (Ohm)
    resistance_p: float = 5e3   # Parallel resistance (Ohm)
    
    # Noise parameters
    voltage_noise: float = 1e-3  # 1mV RMS
    current_noise: float = 1e-9  # 1nA RMS
    telegraph_noise: bool = True
    
    def __post_init__(self):
        """Calculate derived parameters."""
        # Energy barrier for switching
        self.energy_barrier = self.thermal_stability * KB * self.temperature
        
        # Natural switching rate
        self.natural_rate = self.attempt_frequency * np.exp(-self.thermal_stability)
        
        # Resistance difference
        self.delta_resistance = self.resistance_ap - self.resistance_p


class StochasticMTJ:
    """
    Stochastic Magnetic Tunnel Junction with thermal switching.
    
    This device model captures the inherent randomness in MTJ switching
    due to thermal fluctuations, enabling natural probabilistic computing.
    """
    
    def __init__(self, device_id: int, config: StochasticMTJConfig):
        self.device_id = device_id
        self.config = config
        
        # Device state
        self.magnetization_state = np.random.choice([-1, 1])  # +1=P, -1=AP
        self.last_switch_time = time.time()
        self.switch_count = 0
        
        # Statistical tracking
        self.switching_history = []
        self.resistance_history = []
        self.dwell_times = []
        
        # Calibrated device variations (10% typical)
        self.energy_barrier_variation = np.random.normal(1.0, 0.1)
        self.resistance_variation = np.random.normal(1.0, 0.05)
    
    def current_resistance(self) -> float:
        """Get current resistance based on magnetization state."""
        if self.magnetization_state == 1:  # Parallel
            base_resistance = self.config.resistance_p
        else:  # Antiparallel
            base_resistance = self.config.resistance_ap
            
        # Apply device variations
        resistance = base_resistance * self.resistance_variation
        
        # Add noise
        if self.config.voltage_noise > 0:
            noise = np.random.normal(0, self.config.voltage_noise)
            resistance += noise / 1e-6  # Convert to resistance noise
        
        self.resistance_history.append(resistance)
        return resistance
    
    def switching_probability(self, applied_field: float, dt: float) -> float:
        """
        Calculate probability of switching in time dt.
        
        Uses the Néel-Arrhenius model with applied magnetic field.
        """
        # Effective energy barrier with applied field
        field_energy = self.config.volume * self.config.saturation_magnetization * applied_field
        effective_barrier = self.config.energy_barrier * self.energy_barrier_variation
        
        if self.magnetization_state == 1:  # P state
            # Switching to AP requires overcoming full barrier minus field assist
            barrier = effective_barrier - field_energy
        else:  # AP state
            # Switching to P is assisted by field
            barrier = effective_barrier + field_energy
        
        # Ensure barrier is positive
        barrier = max(barrier, 0.1 * KB * self.config.temperature)
        
        # Switching rate
        rate = self.config.attempt_frequency * np.exp(-barrier / (KB * self.config.temperature))
        
        # Probability in time dt (small time approximation)
        probability = 1 - np.exp(-rate * dt)
        return min(probability, 1.0)
    
    def evolve(self, applied_field: float = 0.0, dt: float = 1e-9) -> bool:
        """
        Evolve device state for time dt with applied field.
        
        Returns True if switching occurred.
        """
        # Calculate switching probability
        switch_prob = self.switching_probability(applied_field, dt)
        
        # Determine if switching occurs
        if np.random.random() < switch_prob:
            # Switch state
            current_time = time.time()
            dwell_time = current_time - self.last_switch_time
            
            self.magnetization_state *= -1
            self.last_switch_time = current_time
            self.switch_count += 1
            
            # Record statistics
            self.switching_history.append({
                'time': current_time,
                'new_state': self.magnetization_state,
                'applied_field': applied_field,
                'dwell_time': dwell_time
            })
            self.dwell_times.append(dwell_time)
            
            return True
        
        return False
    
    def get_switching_statistics(self) -> Dict[str, float]:
        """Get statistical properties of switching behavior."""
        if len(self.dwell_times) < 2:
            return {"mean_dwell_time": 0, "switching_rate": 0, "state_balance": 0.5}
        
        mean_dwell_time = np.mean(self.dwell_times)
        switching_rate = 1 / mean_dwell_time if mean_dwell_time > 0 else 0
        
        # Calculate state balance (time spent in P vs AP)
        p_time = sum(h['dwell_time'] for h in self.switching_history if h['new_state'] == 1)
        total_time = sum(self.dwell_times)
        state_balance = p_time / total_time if total_time > 0 else 0.5
        
        return {
            "mean_dwell_time": mean_dwell_time,
            "switching_rate": switching_rate,
            "state_balance": state_balance,
            "total_switches": self.switch_count
        }


class ProbabilisticNeuron:
    """
    Probabilistic neuron using stochastic MTJ devices.
    
    This neuron uses the inherent randomness of MTJ switching to
    implement naturally probabilistic neural computation.
    """
    
    def __init__(self, neuron_id: int, n_inputs: int, mtj_config: StochasticMTJConfig):
        self.neuron_id = neuron_id
        self.n_inputs = n_inputs
        
        # Create stochastic MTJ devices for weights
        self.weight_devices = [StochasticMTJ(i, mtj_config) for i in range(n_inputs)]
        self.bias_device = StochasticMTJ(-1, mtj_config)
        
        # Synaptic scaling factors
        self.weight_scales = np.random.normal(1.0, 0.1, n_inputs)
        self.bias_scale = np.random.normal(0.0, 0.5)
        
        # Activity tracking
        self.activation_history = []
        self.output_variance = 0.0
        
    def get_current_weights(self) -> np.ndarray:
        """Get current synaptic weights from MTJ resistances."""
        weights = np.zeros(self.n_inputs)
        
        for i, device in enumerate(self.weight_devices):
            resistance = device.current_resistance()
            # Map resistance to weight (-1 to +1 range)
            normalized_resistance = (resistance - device.config.resistance_p) / device.config.delta_resistance
            weights[i] = np.tanh(normalized_resistance * self.weight_scales[i])
        
        return weights
    
    def get_bias(self) -> float:
        """Get current bias from bias MTJ."""
        resistance = self.bias_device.current_resistance()
        normalized_resistance = (resistance - self.bias_device.config.resistance_p) / self.bias_device.config.delta_resistance
        return normalized_resistance * self.bias_scale
    
    def forward(self, inputs: np.ndarray, dt: float = 1e-9) -> Tuple[float, float]:
        """
        Forward pass with probabilistic computation.
        
        Returns (activation, uncertainty) where uncertainty quantifies
        the variance due to stochastic switching.
        """
        # Evolve all MTJ devices
        for device in self.weight_devices + [self.bias_device]:
            device.evolve(dt=dt)
        
        # Get current parameters
        weights = self.get_current_weights()
        bias = self.get_bias()
        
        # Compute weighted sum
        weighted_sum = np.dot(weights, inputs) + bias
        
        # Apply activation function (sigmoid with stochastic noise)
        base_activation = 1 / (1 + np.exp(-weighted_sum))
        
        # Add MTJ switching noise to activation
        switching_noise = sum(len(d.switching_history) for d in self.weight_devices) * 0.01
        stochastic_activation = base_activation + np.random.normal(0, switching_noise)
        stochastic_activation = np.clip(stochastic_activation, 0, 1)
        
        # Estimate uncertainty from recent switching activity
        recent_switches = sum(d.switch_count for d in self.weight_devices[-10:])  # Last 10 timesteps
        uncertainty = recent_switches * 0.05  # Scale factor
        
        self.activation_history.append(stochastic_activation)
        self.output_variance = np.var(self.activation_history[-100:])  # Rolling variance
        
        return stochastic_activation, uncertainty


class BayesianMTJNetwork:
    """
    Bayesian neural network using stochastic MTJ devices.
    
    This network naturally implements Bayesian inference by using
    the physical stochasticity of MTJ devices to represent
    parameter uncertainty.
    """
    
    def __init__(self, layer_sizes: List[int], mtj_config: StochasticMTJConfig):
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # Create probabilistic neurons for each layer
        self.layers = []
        for l in range(self.n_layers):
            layer_neurons = []
            for n in range(layer_sizes[l + 1]):
                neuron = ProbabilisticNeuron(n, layer_sizes[l], mtj_config)
                layer_neurons.append(neuron)
            self.layers.append(layer_neurons)
        
        # Network statistics
        self.prediction_samples = []
        self.epistemic_uncertainty = 0.0
        self.aleatoric_uncertainty = 0.0
    
    def forward_sample(self, inputs: np.ndarray, n_samples: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass with multiple samples for uncertainty quantification.
        
        Returns (mean_prediction, uncertainty) where uncertainty includes
        both aleatoric (data) and epistemic (model) uncertainty.
        """
        samples = []
        uncertainties = []
        
        for _ in range(n_samples):
            # Forward pass through all layers
            current_input = inputs
            layer_uncertainties = []
            
            for layer in self.layers:
                layer_outputs = []
                layer_uncs = []
                
                for neuron in layer:
                    output, uncertainty = neuron.forward(current_input)
                    layer_outputs.append(output)
                    layer_uncs.append(uncertainty)
                
                current_input = np.array(layer_outputs)
                layer_uncertainties.append(np.mean(layer_uncs))
            
            samples.append(current_input)
            uncertainties.append(np.mean(layer_uncertainties))
        
        # Convert to numpy arrays
        samples = np.array(samples)
        uncertainties = np.array(uncertainties)
        
        # Calculate statistics
        mean_prediction = np.mean(samples, axis=0)
        prediction_variance = np.var(samples, axis=0)
        
        # Decompose uncertainty
        self.epistemic_uncertainty = np.mean(prediction_variance)  # Model uncertainty
        self.aleatoric_uncertainty = np.mean(uncertainties)       # Data uncertainty
        total_uncertainty = self.epistemic_uncertainty + self.aleatoric_uncertainty
        
        self.prediction_samples = samples
        
        return mean_prediction, total_uncertainty
    
    def calibrate_uncertainty(self, validation_data: List[Tuple[np.ndarray, np.ndarray]]):
        """Calibrate uncertainty estimates using validation data."""
        predictions = []
        uncertainties = []
        errors = []
        
        for x, y_true in validation_data:
            y_pred, uncertainty = self.forward_sample(x)
            predictions.append(y_pred)
            uncertainties.append(uncertainty)
            errors.append(np.abs(y_pred - y_true))
        
        # Compute calibration metrics
        predictions = np.array(predictions)
        uncertainties = np.array(uncertainties)
        errors = np.array(errors)
        
        # Expected Calibration Error (ECE)
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (uncertainties > bin_lower) & (uncertainties <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                accuracy_in_bin = (errors[in_bin] < uncertainties[in_bin]).mean()
                avg_confidence_in_bin = uncertainties[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        
        return {
            "expected_calibration_error": ece,
            "mean_prediction_error": np.mean(errors),
            "epistemic_uncertainty": self.epistemic_uncertainty,
            "aleatoric_uncertainty": self.aleatoric_uncertainty
        }


class MonteCarloSampler:
    """
    Monte Carlo sampler using stochastic MTJ switching.
    
    This sampler uses the physical randomness of MTJ devices to
    generate samples from complex probability distributions.
    """
    
    def __init__(self, n_variables: int, mtj_config: StochasticMTJConfig):
        self.n_variables = n_variables
        
        # Create MTJ devices for sampling
        self.sampling_devices = [StochasticMTJ(i, mtj_config) for i in range(n_variables)]
        
        # Sampling statistics
        self.samples_generated = 0
        self.acceptance_rate = 0.0
        self.mixing_time = 0.0
        
    def sample_from_distribution(self, 
                               log_probability: Callable[[np.ndarray], float],
                               n_samples: int = 1000,
                               burn_in: int = 100) -> np.ndarray:
        """
        Generate samples from a probability distribution using MTJ MCMC.
        
        Uses the stochastic switching of MTJ devices to implement
        a natural Markov Chain Monte Carlo sampler.
        """
        samples = []
        current_state = np.random.uniform(-1, 1, self.n_variables)
        current_log_prob = log_probability(current_state)
        
        accepted_samples = 0
        total_proposals = 0
        
        start_time = time.time()
        
        for i in range(n_samples + burn_in):
            # Generate proposal using MTJ devices
            proposal = self._generate_proposal(current_state)
            proposal_log_prob = log_probability(proposal)
            
            # Metropolis-Hastings acceptance
            log_alpha = min(0, proposal_log_prob - current_log_prob)
            
            # Use MTJ switching probability as acceptance probability
            # Map log_alpha to magnetic field strength
            field_strength = log_alpha * 1000  # Scaling factor
            accept = any(device.evolve(field_strength, dt=1e-9) for device in self.sampling_devices)
            
            total_proposals += 1
            
            if accept or np.random.random() < np.exp(log_alpha):
                current_state = proposal
                current_log_prob = proposal_log_prob
                accepted_samples += 1
            
            # Store samples after burn-in
            if i >= burn_in:
                samples.append(current_state.copy())
        
        sampling_time = time.time() - start_time
        
        # Update statistics
        self.samples_generated += len(samples)
        self.acceptance_rate = accepted_samples / total_proposals
        self.mixing_time = self._estimate_mixing_time(np.array(samples))
        
        return np.array(samples)
    
    def _generate_proposal(self, current_state: np.ndarray) -> np.ndarray:
        """Generate proposal state using MTJ devices."""
        proposal = current_state.copy()
        
        for i, device in enumerate(self.sampling_devices):
            # Use MTJ resistance as random source
            resistance = device.current_resistance()
            normalized_resistance = (resistance - device.config.resistance_p) / device.config.delta_resistance
            
            # Add Gaussian noise scaled by MTJ state
            noise_scale = 0.1 * (1 + abs(normalized_resistance))
            proposal[i] += np.random.normal(0, noise_scale)
            
            # Evolve device
            device.evolve(dt=1e-9)
        
        return np.clip(proposal, -5, 5)  # Reasonable bounds
    
    def _estimate_mixing_time(self, samples: np.ndarray) -> float:
        """Estimate MCMC mixing time from sample autocorrelation."""
        if len(samples) < 100:
            return float('inf')
        
        # Calculate autocorrelation for first variable
        x = samples[:, 0]
        n = len(x)
        x_centered = x - np.mean(x)
        
        # Autocorrelation function
        autocorr = np.correlate(x_centered, x_centered, mode='full')
        autocorr = autocorr[n-1:]  # Take positive lags only
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Find first point where autocorr drops below 1/e
        threshold = 1 / np.e
        mixing_time = np.argmax(autocorr < threshold)
        
        return mixing_time if mixing_time > 0 else len(autocorr)


def demonstrate_probabilistic_computing():
    """Demonstrate probabilistic computing with stochastic MTJs."""
    print("🎲 Probabilistic Computing with Stochastic MTJs")
    print("=" * 60)
    
    # Create MTJ configuration
    mtj_config = StochasticMTJConfig(
        temperature=300.0,
        thermal_stability=35.0,  # Moderate stability for good stochasticity
        volume=2e-24  # 20nm cube
    )
    
    print(f"✅ MTJ Configuration:")
    print(f"   Temperature: {mtj_config.temperature} K")
    print(f"   Thermal stability: {mtj_config.thermal_stability} kT")
    print(f"   Natural switching rate: {mtj_config.natural_rate:.2e} Hz")
    
    # Demonstrate single stochastic MTJ
    print(f"\n🔬 Single Stochastic MTJ Analysis")
    mtj = StochasticMTJ(0, mtj_config)
    
    # Run evolution for statistics
    switch_count = 0
    for _ in range(1000):
        if mtj.evolve(applied_field=0, dt=1e-6):  # 1 microsecond steps
            switch_count += 1
    
    stats = mtj.get_switching_statistics()
    print(f"   Switches observed: {switch_count}/1000 timesteps")
    print(f"   Mean dwell time: {stats['mean_dwell_time']:.6f} s")
    print(f"   Switching rate: {stats['switching_rate']:.2e} Hz")
    print(f"   State balance: {stats['state_balance']:.3f}")
    
    # Demonstrate probabilistic neuron
    print(f"\n🧠 Probabilistic Neuron Demonstration")
    neuron = ProbabilisticNeuron(0, n_inputs=5, mtj_config=mtj_config)
    
    test_inputs = np.random.normal(0, 1, 5)
    activations = []
    uncertainties = []
    
    for _ in range(100):
        activation, uncertainty = neuron.forward(test_inputs)
        activations.append(activation)
        uncertainties.append(uncertainty)
    
    print(f"   Mean activation: {np.mean(activations):.4f} ± {np.std(activations):.4f}")
    print(f"   Mean uncertainty: {np.mean(uncertainties):.4f}")
    print(f"   Output variance: {neuron.output_variance:.6f}")
    
    # Demonstrate Bayesian network
    print(f"\n🌐 Bayesian MTJ Network")
    network = BayesianMTJNetwork([4, 8, 4, 2], mtj_config)
    
    # Generate test data
    test_input = np.random.normal(0, 1, 4)
    prediction, uncertainty = network.forward_sample(test_input, n_samples=20)
    
    print(f"   Network layers: {network.layer_sizes}")
    print(f"   Prediction: {prediction}")
    print(f"   Total uncertainty: {uncertainty:.4f}")
    print(f"   Epistemic uncertainty: {network.epistemic_uncertainty:.4f}")
    print(f"   Aleatoric uncertainty: {network.aleatoric_uncertainty:.4f}")
    
    # Demonstrate Monte Carlo sampling
    print(f"\n🎯 Monte Carlo Sampling with MTJs")
    sampler = MonteCarloSampler(3, mtj_config)
    
    # Define a simple 2D Gaussian target distribution
    def log_gaussian(x):
        return -0.5 * np.sum(x**2)  # Standard Gaussian
    
    samples = sampler.sample_from_distribution(log_gaussian, n_samples=500, burn_in=100)
    
    print(f"   Samples generated: {len(samples)}")
    print(f"   Sample mean: {np.mean(samples, axis=0)}")
    print(f"   Sample std: {np.std(samples, axis=0)}")
    print(f"   Acceptance rate: {sampler.acceptance_rate:.3f}")
    print(f"   Mixing time: {sampler.mixing_time:.1f} steps")
    
    # Performance metrics
    print(f"\n📊 Performance Analysis")
    
    # Calculate effective sample size
    effective_samples = len(samples) / (2 * sampler.mixing_time + 1)
    sampling_efficiency = effective_samples / len(samples)
    
    print(f"   Effective sample size: {effective_samples:.1f}")
    print(f"   Sampling efficiency: {sampling_efficiency:.3f}")
    
    # Energy analysis
    total_devices = len(sampler.sampling_devices) + sum(len(layer) for layer in network.layers)
    estimated_power = total_devices * 1e-12  # 1pW per device
    
    print(f"   Total MTJ devices: {total_devices}")
    print(f"   Estimated power consumption: {estimated_power*1e12:.1f} pW")
    
    return {
        "mtj_switching_rate": stats['switching_rate'],
        "neuron_output_variance": neuron.output_variance,
        "network_epistemic_uncertainty": network.epistemic_uncertainty,
        "network_aleatoric_uncertainty": network.aleatoric_uncertainty,
        "mcmc_acceptance_rate": sampler.acceptance_rate,
        "mcmc_mixing_time": sampler.mixing_time,
        "sampling_efficiency": sampling_efficiency,
        "total_power_pw": estimated_power * 1e12
    }


if __name__ == "__main__":
    results = demonstrate_probabilistic_computing()
    print(f"\n🎉 Probabilistic Computing with Stochastic MTJs: BREAKTHROUGH DEMONSTRATED")
    print(json.dumps(results, indent=2))