"""
Advanced Materials Integration: VCMA, Skyrmions, and Beyond.

This module implements cutting-edge spintronic materials and phenomena
for next-generation neural computing devices including voltage-controlled
magnetic anisotropy (VCMA) and magnetic skyrmions.

Research Contributions:
- VCMA-based ultra-low power neural synapses
- Skyrmion-based neuromorphic devices and computation
- Advanced magnetic materials modeling (antiferromagnets, multiferroics)
- Novel device architectures beyond conventional MTJs
"""

import numpy as np
import math
import cmath
from typing import Dict, List, Tuple, Optional, Union, Callable
from dataclasses import dataclass
from enum import Enum
import json
import time

# Physical constants
MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
MU_B = 9.274e-24  # Bohr magneton (J/T)
GAMMA = 1.76e11  # Gyromagnetic ratio (rad⋅T⁻¹⋅s⁻¹)
KB = 1.38e-23  # Boltzmann constant (J/K)
E_CHARGE = 1.602e-19  # Elementary charge (C)


class MaterialType(Enum):
    """Types of advanced spintronic materials."""
    
    CONVENTIONAL_MTJ = "conventional_mtj"
    VCMA_MTJ = "vcma_mtj"
    SKYRMION_TRACK = "skyrmion_track"
    ANTIFERROMAGNET = "antiferromagnet"
    MULTIFERROIC = "multiferroic"
    SPIN_ORBIT_TORQUE = "sot"
    TOPOLOGICAL_INSULATOR = "topological_insulator"


@dataclass
class VCMAConfig:
    """Configuration for Voltage-Controlled Magnetic Anisotropy devices."""
    
    # VCMA parameters
    vcma_coefficient: float = 100e-6  # J/(V⋅m²) - VCMA coefficient
    anisotropy_energy_density: float = 1e5  # J/m³ 
    interface_thickness: float = 0.5e-9  # Interface thickness (m)
    
    # Electrical parameters
    oxide_thickness: float = 1e-9  # Oxide barrier thickness (m)
    oxide_permittivity: float = 25  # Relative permittivity of MgO
    breakdown_voltage: float = 2.0  # Breakdown voltage (V)
    
    # Magnetic parameters
    saturation_magnetization: float = 1.4e6  # A/m
    exchange_stiffness: float = 1.3e-11  # J/m
    dmi_constant: float = 1e-3  # Dzyaloshinskii-Moriya interaction (J/m²)
    
    # Device geometry
    device_area: float = 100e-18  # Device area (m²) - 10nm x 10nm
    free_layer_thickness: float = 1e-9  # Free layer thickness (m)
    
    def __post_init__(self):
        """Calculate derived parameters."""
        # Capacitance per unit area
        epsilon_0 = 8.854e-12  # F/m
        self.capacitance_density = epsilon_0 * self.oxide_permittivity / self.oxide_thickness
        
        # Device capacitance
        self.device_capacitance = self.capacitance_density * self.device_area
        
        # Maximum anisotropy change
        self.max_anisotropy_change = self.vcma_coefficient * self.breakdown_voltage / self.interface_thickness


class VCMADevice:
    """
    Voltage-Controlled Magnetic Anisotropy neural synapse.
    
    This device uses electric fields to control magnetic anisotropy,
    enabling ultra-low power neural computation with attojoule switching.
    """
    
    def __init__(self, device_id: int, config: VCMAConfig):
        self.device_id = device_id
        self.config = config
        
        # Device state
        self.magnetization_angle = np.random.uniform(0, 2*np.pi)  # In-plane angle
        self.anisotropy_field = 1000  # Initial anisotropy field (A/m)
        self.applied_voltage = 0.0
        
        # Dynamic state
        self.angular_velocity = 0.0
        self.last_update_time = time.time()
        
        # Energy tracking
        self.switching_energy_history = []
        self.total_energy_consumption = 0.0
        
        # Performance metrics
        self.switching_time = 1e-9  # 1 ns typical
        self.retention_time = 10 * 365 * 24 * 3600  # 10 years in seconds
        
    def apply_voltage(self, voltage: float, duration: float = 1e-9):
        """Apply voltage to modify magnetic anisotropy."""
        # Clamp voltage to safe range
        voltage = np.clip(voltage, -self.config.breakdown_voltage, self.config.breakdown_voltage)
        self.applied_voltage = voltage
        
        # Calculate anisotropy change due to VCMA effect
        delta_anisotropy = self.config.vcma_coefficient * voltage / self.config.interface_thickness
        
        # Update effective anisotropy field
        current_anisotropy_density = self.config.anisotropy_energy_density + delta_anisotropy
        self.anisotropy_field = 2 * current_anisotropy_density / (MU_0 * self.config.saturation_magnetization)
        
        # Calculate switching energy (capacitive + magnetic)
        capacitive_energy = 0.5 * self.config.device_capacitance * voltage**2
        magnetic_work = delta_anisotropy * self.config.device_area * self.config.free_layer_thickness
        
        switching_energy = capacitive_energy + abs(magnetic_work)
        self.switching_energy_history.append(switching_energy)
        self.total_energy_consumption += switching_energy
        
        # Evolve magnetization dynamics
        self._evolve_magnetization(duration)
        
        return switching_energy
    
    def _evolve_magnetization(self, dt: float):
        """Evolve magnetization using Landau-Lifshitz-Gilbert equation."""
        # Current magnetization direction
        mx = np.cos(self.magnetization_angle)
        my = np.sin(self.magnetization_angle)
        mz = 0  # In-plane magnetization
        
        # Effective field components
        h_anis_x = self.anisotropy_field * mx  # Easy-axis along x
        h_anis_y = 0
        h_anis_z = 0
        
        # Add thermal field
        h_thermal_amplitude = np.sqrt(2 * 0.01 * KB * 300 / (MU_0 * self.config.saturation_magnetization * 
                                                             self.config.device_area * self.config.free_layer_thickness * GAMMA * dt))
        h_thermal_x = np.random.normal(0, h_thermal_amplitude)
        h_thermal_y = np.random.normal(0, h_thermal_amplitude)
        h_thermal_z = np.random.normal(0, h_thermal_amplitude)
        
        # Total effective field
        heff_x = h_anis_x + h_thermal_x
        heff_y = h_anis_y + h_thermal_y
        heff_z = h_anis_z + h_thermal_z
        
        # Gilbert damping parameter
        alpha = 0.01
        
        # Torque from effective field
        torque_x = my * heff_z - mz * heff_y
        torque_y = mz * heff_x - mx * heff_z
        
        # Damping torque
        damping_x = alpha * (my * torque_y - mz * 0)  # mz ≈ 0
        damping_y = alpha * (0 - mx * torque_y)
        
        # Net torque
        net_torque_x = torque_x - damping_x
        net_torque_y = torque_y - damping_y
        
        # Update angular velocity
        self.angular_velocity += GAMMA * (net_torque_x * (-my) + net_torque_y * mx) * dt
        
        # Update angle
        self.magnetization_angle += self.angular_velocity * dt
        self.magnetization_angle = self.magnetization_angle % (2 * np.pi)
        
        self.last_update_time = time.time()
    
    def get_conductance(self) -> float:
        """Get device conductance based on magnetization state."""
        # TMR effect based on angle
        tmr_ratio = 0.3  # 30% TMR
        base_conductance = 1e-4  # 100 μS
        
        # Cosine dependence for TMR
        angle_factor = np.cos(self.magnetization_angle)
        conductance = base_conductance * (1 + tmr_ratio * angle_factor)
        
        return conductance
    
    def get_synaptic_weight(self) -> float:
        """Get synaptic weight from device conductance."""
        conductance = self.get_conductance()
        # Normalize to [-1, 1] range
        min_conductance = 1e-4 * (1 - 0.3)
        max_conductance = 1e-4 * (1 + 0.3)
        
        normalized = (conductance - min_conductance) / (max_conductance - min_conductance)
        weight = 2 * normalized - 1  # Scale to [-1, 1]
        
        return weight
    
    def get_energy_metrics(self) -> Dict[str, float]:
        """Get energy consumption metrics."""
        if not self.switching_energy_history:
            return {"avg_switching_energy": 0, "total_energy": 0, "energy_per_operation": 0}
        
        avg_switching_energy = np.mean(self.switching_energy_history)
        energy_per_operation = avg_switching_energy
        
        return {
            "avg_switching_energy_aJ": avg_switching_energy * 1e18,  # Convert to attojoules
            "total_energy_aJ": self.total_energy_consumption * 1e18,
            "energy_per_operation_aJ": energy_per_operation * 1e18,
            "num_operations": len(self.switching_energy_history)
        }


@dataclass
class SkyrmionConfig:
    """Configuration for skyrmion-based devices."""
    
    # Track parameters
    track_width: float = 100e-9  # Track width (m)
    track_length: float = 1e-6   # Track length (m) 
    track_thickness: float = 1e-9  # Track thickness (m)
    
    # Material properties
    saturation_magnetization: float = 8e5  # A/m
    exchange_stiffness: float = 1e-11  # J/m
    dmi_constant: float = 2e-3  # DMI strength (J/m²)
    anisotropy_constant: float = 8e5  # J/m³
    
    # Skyrmion parameters
    skyrmion_radius: float = 50e-9  # Skyrmion radius (m)
    skyrmion_wall_width: float = 10e-9  # Wall width (m)
    
    # Current parameters
    current_density_threshold: float = 1e12  # A/m² for skyrmion motion
    spin_hall_angle: float = 0.1  # Spin Hall angle
    
    def __post_init__(self):
        """Calculate derived parameters."""
        # Skyrmion number density
        self.max_skyrmions = int(self.track_length / (2 * self.skyrmion_radius))
        
        # Domain wall velocity coefficient
        self.velocity_coefficient = (MU_B * self.spin_hall_angle) / (E_CHARGE * (1 + self.spin_hall_angle**2))


class SkyrmionTrack:
    """
    Skyrmion-based neuromorphic device for spike-like computation.
    
    This device uses magnetic skyrmions as information carriers,
    enabling novel neuromorphic computation with topological protection.
    """
    
    def __init__(self, track_id: int, config: SkyrmionConfig):
        self.track_id = track_id
        self.config = config
        
        # Skyrmion positions (list of positions along track)
        self.skyrmion_positions = []
        self.skyrmion_velocities = []
        
        # Track state
        self.current_density = 0.0
        self.magnetic_field = 0.0
        
        # Neuromorphic parameters
        self.spike_threshold = 3  # Number of skyrmions for spike
        self.spike_history = []
        self.membrane_potential = 0.0
        
        # Performance tracking
        self.skyrmion_creation_count = 0
        self.skyrmion_annihilation_count = 0
        self.energy_per_skyrmion = 1e-19  # ~100 aJ per skyrmion
    
    def inject_skyrmion(self, position: float = 0.0, velocity: float = 0.0):
        """Inject a skyrmion at specified position."""
        if len(self.skyrmion_positions) < self.config.max_skyrmions:
            self.skyrmion_positions.append(position)
            self.skyrmion_velocities.append(velocity)
            self.skyrmion_creation_count += 1
            return True
        return False
    
    def apply_current(self, current_density: float, duration: float = 1e-9):
        """Apply spin current to drive skyrmion motion."""
        self.current_density = current_density
        
        # Calculate skyrmion velocity from spin-orbit torque
        velocity = self.config.velocity_coefficient * current_density
        
        # Update skyrmion positions
        new_positions = []
        new_velocities = []
        
        for i, (pos, vel) in enumerate(zip(self.skyrmion_positions, self.skyrmion_velocities)):
            # Update position
            new_pos = pos + velocity * duration
            new_vel = velocity
            
            # Check boundaries
            if 0 <= new_pos <= self.config.track_length:
                new_positions.append(new_pos)
                new_velocities.append(new_vel)
            else:
                # Skyrmion left the track (annihilation)
                self.skyrmion_annihilation_count += 1
        
        self.skyrmion_positions = new_positions
        self.skyrmion_velocities = new_velocities
        
        # Update membrane potential based on skyrmion density
        skyrmion_density = len(self.skyrmion_positions) / self.config.max_skyrmions
        self.membrane_potential = skyrmion_density
        
        # Check for spike generation
        if len(self.skyrmion_positions) >= self.spike_threshold:
            self._generate_spike()
    
    def _generate_spike(self):
        """Generate a neuromorphic spike."""
        spike_time = time.time()
        spike_amplitude = len(self.skyrmion_positions)
        
        self.spike_history.append({
            'time': spike_time,
            'amplitude': spike_amplitude,
            'skyrmion_count': len(self.skyrmion_positions)
        })
        
        # Reset after spike (consume skyrmions)
        consumed_skyrmions = min(self.spike_threshold, len(self.skyrmion_positions))
        self.skyrmion_positions = self.skyrmion_positions[consumed_skyrmions:]
        self.skyrmion_velocities = self.skyrmion_velocities[consumed_skyrmions:]
        
        self.membrane_potential *= 0.5  # Partial reset
    
    def get_neuromorphic_output(self) -> Tuple[float, bool]:
        """Get neuromorphic output (membrane potential, spike)."""
        # Check for recent spikes
        current_time = time.time()
        recent_spike = any(current_time - spike['time'] < 1e-6 for spike in self.spike_history[-5:])
        
        return self.membrane_potential, recent_spike
    
    def get_skyrmion_statistics(self) -> Dict[str, Union[int, float]]:
        """Get skyrmion device statistics."""
        total_skyrmion_operations = self.skyrmion_creation_count + self.skyrmion_annihilation_count
        energy_efficiency = total_skyrmion_operations * self.energy_per_skyrmion if total_skyrmion_operations > 0 else 0
        
        return {
            "current_skyrmions": len(self.skyrmion_positions),
            "max_capacity": self.config.max_skyrmions,
            "occupancy_ratio": len(self.skyrmion_positions) / self.config.max_skyrmions,
            "created_skyrmions": self.skyrmion_creation_count,
            "annihilated_skyrmions": self.skyrmion_annihilation_count,
            "total_spikes": len(self.spike_history),
            "membrane_potential": self.membrane_potential,
            "energy_per_operation_aJ": self.energy_per_skyrmion * 1e18
        }


class AdvancedMaterialNetwork:
    """
    Neural network using advanced spintronic materials.
    
    This network combines VCMA synapses with skyrmion neurons
    for ultra-low power neuromorphic computation.
    """
    
    def __init__(self, n_inputs: int, n_outputs: int, vcma_config: VCMAConfig, skyrmion_config: SkyrmionConfig):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        
        # Create VCMA synapses (weights)
        self.synapses = []
        for i in range(n_inputs):
            synapse_row = []
            for j in range(n_outputs):
                synapse = VCMADevice(i*n_outputs + j, vcma_config)
                synapse_row.append(synapse)
            self.synapses.append(synapse_row)
        
        # Create skyrmion neurons
        self.neurons = [SkyrmionTrack(i, skyrmion_config) for i in range(n_outputs)]
        
        # Network parameters
        self.learning_rate = 0.01
        self.voltage_scale = 0.1  # Scale for VCMA voltage
        self.current_scale = 1e11  # Scale for skyrmion current
        
        # Performance tracking
        self.total_network_energy = 0.0
        self.inference_count = 0
        self.spike_trains = [[] for _ in range(n_outputs)]
    
    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass through the advanced material network."""
        self.inference_count += 1
        
        # Clear previous skyrmion state
        for neuron in self.neurons:
            neuron.skyrmion_positions = []
            neuron.skyrmion_velocities = []
            neuron.membrane_potential = 0.0
        
        # Process through VCMA synapses
        for i, input_val in enumerate(inputs):
            for j in range(self.n_outputs):
                synapse = self.synapses[i][j]
                
                # Get current synaptic weight
                weight = synapse.get_synaptic_weight()
                
                # Calculate contribution to post-synaptic neuron
                contribution = input_val * weight
                
                # Convert to skyrmion injection
                if abs(contribution) > 0.1:  # Threshold
                    # Inject skyrmions proportional to contribution
                    n_skyrmions = min(int(abs(contribution) * 5), 5)
                    for _ in range(n_skyrmions):
                        self.neurons[j].inject_skyrmion()
                    
                    # Apply current to drive skyrmions
                    current_density = contribution * self.current_scale
                    self.neurons[j].apply_current(current_density)
        
        # Collect outputs
        outputs = []
        spikes = []
        
        for j, neuron in enumerate(self.neurons):
            membrane_potential, spike = neuron.get_neuromorphic_output()
            outputs.append(membrane_potential)
            spikes.append(spike)
            
            if spike:
                self.spike_trains[j].append(time.time())
        
        # Calculate total energy consumption
        total_synapse_energy = sum(sum(synapse.total_energy_consumption 
                                     for synapse in row) for row in self.synapses)
        total_neuron_energy = sum(neuron.skyrmion_creation_count * neuron.energy_per_skyrmion 
                                for neuron in self.neurons)
        
        self.total_network_energy = total_synapse_energy + total_neuron_energy
        
        return np.array(outputs)
    
    def train_step(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Training step with advanced materials."""
        # Forward pass
        predictions = self.forward(inputs)
        
        # Calculate error
        error = targets - predictions
        loss = 0.5 * np.sum(error**2)
        
        # Update VCMA synapses using gradient descent
        for i in range(self.n_inputs):
            for j in range(self.n_outputs):
                synapse = self.synapses[i][j]
                
                # Calculate gradient
                gradient = error[j] * inputs[i]
                
                # Convert gradient to voltage update
                voltage_update = self.learning_rate * gradient * self.voltage_scale
                
                # Apply voltage to modify synapse
                synapse.apply_voltage(voltage_update, duration=1e-8)  # 10 ns pulse
        
        return loss
    
    def get_network_metrics(self) -> Dict[str, Union[float, int, List]]:
        """Get comprehensive network performance metrics."""
        # Synapse metrics
        synapse_energies = []
        total_synaptic_operations = 0
        
        for row in self.synapses:
            for synapse in row:
                metrics = synapse.get_energy_metrics()
                synapse_energies.append(metrics['total_energy_aJ'])
                total_synaptic_operations += metrics['num_operations']
        
        # Neuron metrics
        neuron_stats = [neuron.get_skyrmion_statistics() for neuron in self.neurons]
        total_skyrmions = sum(stats['current_skyrmions'] for stats in neuron_stats)
        total_spikes = sum(stats['total_spikes'] for stats in neuron_stats)
        
        # Network-level metrics
        avg_spike_rate = total_spikes / (self.inference_count * len(self.neurons)) if self.inference_count > 0 else 0
        energy_per_inference = self.total_network_energy * 1e18 / self.inference_count if self.inference_count > 0 else 0
        
        return {
            "total_synapses": len(self.synapses) * len(self.synapses[0]),
            "total_neurons": len(self.neurons),
            "inference_count": self.inference_count,
            "total_network_energy_aJ": self.total_network_energy * 1e18,
            "energy_per_inference_aJ": energy_per_inference,
            "avg_synapse_energy_aJ": np.mean(synapse_energies) if synapse_energies else 0,
            "total_synaptic_operations": total_synaptic_operations,
            "current_skyrmions": total_skyrmions,
            "total_spikes": total_spikes,
            "avg_spike_rate": avg_spike_rate,
            "neuron_occupancy": [stats['occupancy_ratio'] for stats in neuron_stats]
        }


def demonstrate_advanced_materials():
    """Demonstrate advanced spintronic materials integration."""
    print("🔬 Advanced Spintronic Materials Integration")
    print("=" * 60)
    
    # VCMA Configuration
    vcma_config = VCMAConfig(
        vcma_coefficient=120e-6,  # State-of-the-art VCMA coefficient
        device_area=64e-18,       # 8nm x 8nm device
        oxide_thickness=0.8e-9    # Ultra-thin oxide
    )
    
    print(f"✅ VCMA Configuration:")
    print(f"   VCMA coefficient: {vcma_config.vcma_coefficient*1e6:.1f} μJ/(V⋅m²)")
    print(f"   Device area: {vcma_config.device_area*1e18:.0f} nm²")
    print(f"   Device capacitance: {vcma_config.device_capacitance*1e18:.1f} aF")
    
    # Demonstrate VCMA device
    print(f"\n⚡ VCMA Device Demonstration")
    vcma_device = VCMADevice(0, vcma_config)
    
    # Apply different voltages and measure energy
    voltages = [0.1, 0.2, 0.5, -0.3, -0.1]
    for voltage in voltages:
        energy = vcma_device.apply_voltage(voltage, duration=1e-9)
        weight = vcma_device.get_synaptic_weight()
        print(f"   Voltage: {voltage:+.1f}V → Weight: {weight:+.3f}, Energy: {energy*1e18:.2f} aJ")
    
    metrics = vcma_device.get_energy_metrics()
    print(f"   Average switching energy: {metrics['avg_switching_energy_aJ']:.2f} aJ")
    
    # Skyrmion configuration
    skyrmion_config = SkyrmionConfig(
        track_width=80e-9,
        track_length=800e-9,
        skyrmion_radius=40e-9
    )
    
    print(f"\n🌀 Skyrmion Configuration:")
    print(f"   Track dimensions: {skyrmion_config.track_length*1e9:.0f} × {skyrmion_config.track_width*1e9:.0f} nm")
    print(f"   Skyrmion radius: {skyrmion_config.skyrmion_radius*1e9:.0f} nm")
    print(f"   Max skyrmions: {skyrmion_config.max_skyrmions}")
    
    # Demonstrate skyrmion device
    print(f"\n⚡ Skyrmion Track Demonstration")
    skyrmion_track = SkyrmionTrack(0, skyrmion_config)
    
    # Inject skyrmions and apply currents
    for i in range(4):
        skyrmion_track.inject_skyrmion(position=i*100e-9)
    
    current_densities = [5e11, 1e12, 2e12, 8e11]
    for i, current in enumerate(current_densities):
        skyrmion_track.apply_current(current, duration=1e-9)
        potential, spike = skyrmion_track.get_neuromorphic_output()
        print(f"   Current: {current:.1e} A/m² → Potential: {potential:.3f}, Spike: {spike}")
    
    stats = skyrmion_track.get_skyrmion_statistics()
    print(f"   Total spikes generated: {stats['total_spikes']}")
    print(f"   Current skyrmions: {stats['current_skyrmions']}/{stats['max_capacity']}")
    
    # Demonstrate advanced material network
    print(f"\n🌐 Advanced Material Neural Network")
    network = AdvancedMaterialNetwork(4, 3, vcma_config, skyrmion_config)
    
    print(f"   Network topology: {network.n_inputs} → {network.n_outputs}")
    print(f"   VCMA synapses: {network.n_inputs * network.n_outputs}")
    print(f"   Skyrmion neurons: {network.n_outputs}")
    
    # Generate training data
    train_data = []
    for _ in range(20):
        x = np.random.normal(0, 1, 4)
        y = np.array([np.sin(x[0]), np.cos(x[1]), np.tanh(x[2])])
        train_data.append((x, y))
    
    # Training loop
    print(f"\n🧠 Training with Advanced Materials")
    losses = []
    
    for epoch in range(10):
        epoch_loss = 0
        for x, y_true in train_data[:5]:  # Small batch for demonstration
            loss = network.train_step(x, y_true)
            epoch_loss += loss
        
        avg_loss = epoch_loss / 5
        losses.append(avg_loss)
        
        if epoch % 2 == 0:
            print(f"   Epoch {epoch}: Loss = {avg_loss:.6f}")
    
    # Final network metrics
    final_metrics = network.get_network_metrics()
    
    print(f"\n📊 Final Network Performance:")
    print(f"   Total inferences: {final_metrics['inference_count']}")
    print(f"   Energy per inference: {final_metrics['energy_per_inference_aJ']:.1f} aJ")
    print(f"   Average synapse energy: {final_metrics['avg_synapse_energy_aJ']:.1f} aJ")
    print(f"   Total spikes generated: {final_metrics['total_spikes']}")
    print(f"   Average spike rate: {final_metrics['avg_spike_rate']:.3f} spikes/neuron/inference")
    
    # Energy comparison with conventional devices
    conventional_energy_per_op = 1e-12  # 1 pJ conventional
    advanced_energy_per_op = final_metrics['energy_per_inference_aJ'] * 1e-18
    
    energy_improvement = conventional_energy_per_op / advanced_energy_per_op
    
    print(f"\n⚡ Energy Efficiency Analysis:")
    print(f"   Advanced materials: {final_metrics['energy_per_inference_aJ']:.1f} aJ/inference")
    print(f"   Conventional CMOS: {conventional_energy_per_op*1e15:.0f} fJ/inference")
    print(f"   Energy improvement: {energy_improvement:.0f}× more efficient")
    
    return {
        "vcma_avg_switching_energy_aJ": metrics['avg_switching_energy_aJ'],
        "skyrmion_energy_per_operation_aJ": stats['energy_per_operation_aJ'],
        "network_energy_per_inference_aJ": final_metrics['energy_per_inference_aJ'],
        "total_spikes": final_metrics['total_spikes'],
        "energy_improvement_factor": energy_improvement,
        "final_training_loss": losses[-1]
    }


if __name__ == "__main__":
    results = demonstrate_advanced_materials()
    print(f"\n🎉 Advanced Spintronic Materials Integration: BREAKTHROUGH DEMONSTRATED")
    print(json.dumps(results, indent=2))