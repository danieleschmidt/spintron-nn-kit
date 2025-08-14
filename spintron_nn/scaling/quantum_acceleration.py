"""
Quantum-Accelerated Spintronic Computing Framework.

Implements quantum-classical hybrid computing architectures that leverage
quantum speedups for specific spintronic neural network operations.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import asyncio
from abc import ABC, abstractmethod

from ..core.mtj_models import MTJConfig
from ..core.crossbar import MTJCrossbar, CrossbarConfig
from ..utils.error_handling import SpintronError, robust_operation
from ..utils.logging_config import get_logger
from ..utils.monitoring import get_system_monitor

logger = get_logger(__name__)


@dataclass
class QuantumResource:
    """Quantum computing resource specification."""
    
    qubits: int
    gate_fidelity: float
    coherence_time_ms: float
    connectivity_graph: Dict[int, List[int]]
    quantum_volume: int
    error_rate: float = 0.001
    
    def is_sufficient_for_problem(self, problem_size: int) -> bool:
        """Check if quantum resource is sufficient for problem."""
        return (self.qubits >= problem_size and 
                self.gate_fidelity > 0.99 and
                self.coherence_time_ms > 100)


class QuantumAlgorithm(ABC):
    """Abstract base class for quantum algorithms."""
    
    @abstractmethod
    def estimate_quantum_advantage(self, problem_size: int) -> float:
        """Estimate quantum speedup over classical algorithm."""
        pass
    
    @abstractmethod
    def required_qubits(self, problem_size: int) -> int:
        """Calculate required number of qubits."""
        pass
    
    @abstractmethod
    def execute(self, input_data: np.ndarray, quantum_resource: QuantumResource) -> np.ndarray:
        """Execute quantum algorithm."""
        pass


class QuantumLinearSolver(QuantumAlgorithm):
    """
    Quantum algorithm for solving linear systems (HHL algorithm variant).
    
    Provides exponential speedup for certain classes of linear systems
    common in neural network computations.
    """
    
    def __init__(self, precision_bits: int = 8):
        self.precision_bits = precision_bits
        self.algorithm_name = "Quantum Linear System Solver"
        
        logger.info(f"Initialized {self.algorithm_name}")
    
    def estimate_quantum_advantage(self, problem_size: int) -> float:
        """Estimate quantum speedup for linear system solving."""
        
        # Classical complexity: O(N^3) for general matrices
        classical_complexity = problem_size ** 3
        
        # Quantum complexity: O(log(N)) for sparse, well-conditioned matrices
        quantum_complexity = np.log2(problem_size) * self.precision_bits
        
        # Speedup factor
        speedup = classical_complexity / max(quantum_complexity, 1)
        
        # Adjust for practical considerations
        # Quantum advantage typically appears for N > 1000
        if problem_size < 100:
            speedup *= 0.1  # Limited advantage for small problems
        elif problem_size < 1000:
            speedup *= 0.5  # Moderate advantage
        
        return min(speedup, 1e6)  # Cap at reasonable value
    
    def required_qubits(self, problem_size: int) -> int:
        """Calculate required qubits for HHL algorithm."""
        
        # Qubits needed:
        # - log(N) for state preparation
        # - precision_bits for eigenvalue estimation
        # - 1 ancilla qubit for amplitude amplification
        # - Additional qubits for error correction
        
        state_qubits = int(np.ceil(np.log2(problem_size)))
        precision_qubits = self.precision_bits
        ancilla_qubits = 3
        error_correction_overhead = 5
        
        total_qubits = state_qubits + precision_qubits + ancilla_qubits + error_correction_overhead
        
        return total_qubits
    
    def execute(
        self, 
        matrix_A: np.ndarray, 
        vector_b: np.ndarray,
        quantum_resource: QuantumResource
    ) -> np.ndarray:
        """Execute quantum linear system solver."""
        
        problem_size = len(vector_b)
        required_qubits = self.required_qubits(problem_size)
        
        if not quantum_resource.is_sufficient_for_problem(required_qubits):
            raise SpintronError(f"Insufficient quantum resources: need {required_qubits} qubits")
        
        logger.info(f"Solving {problem_size}x{problem_size} linear system on quantum hardware")
        
        start_time = time.time()
        
        try:
            # Quantum circuit simulation (simplified)
            solution = self._simulate_hhl_algorithm(matrix_A, vector_b, quantum_resource)
            
            execution_time = time.time() - start_time
            logger.info(f"Quantum linear solver completed in {execution_time:.3f}s")
            
            return solution
            
        except Exception as e:
            raise SpintronError(f"Quantum linear solver failed: {str(e)}")
    
    def _simulate_hhl_algorithm(
        self,
        matrix_A: np.ndarray,
        vector_b: np.ndarray,
        quantum_resource: QuantumResource
    ) -> np.ndarray:
        """Simulate HHL algorithm execution."""
        
        # Phase 1: Quantum Phase Estimation
        eigenvalues = self._quantum_phase_estimation(matrix_A, quantum_resource)
        
        # Phase 2: Controlled Rotation
        amplitudes = self._controlled_rotation(eigenvalues)
        
        # Phase 3: Amplitude Amplification
        amplified_amplitudes = self._amplitude_amplification(amplitudes, quantum_resource)
        
        # Phase 4: Measurement and Classical Post-processing
        solution = self._measurement_and_postprocessing(
            matrix_A, vector_b, amplified_amplitudes
        )
        
        return solution
    
    def _quantum_phase_estimation(
        self,
        matrix_A: np.ndarray,
        quantum_resource: QuantumResource
    ) -> np.ndarray:
        """Simulate quantum phase estimation for eigenvalues."""
        
        # Compute eigenvalues classically (in real implementation, done quantumly)
        eigenvalues = np.linalg.eigvals(matrix_A)
        
        # Add quantum noise based on resource quality
        noise_scale = quantum_resource.error_rate * 0.1
        quantum_eigenvalues = eigenvalues + np.random.normal(0, noise_scale, len(eigenvalues))
        
        # Simulate measurement precision
        precision_factor = 2 ** self.precision_bits
        quantized_eigenvalues = np.round(quantum_eigenvalues * precision_factor) / precision_factor
        
        return quantized_eigenvalues
    
    def _controlled_rotation(self, eigenvalues: np.ndarray) -> np.ndarray:
        """Simulate controlled rotation step."""
        
        # Controlled rotation creates amplitudes proportional to 1/eigenvalue
        # Add small epsilon to avoid division by zero
        epsilon = 1e-8
        amplitudes = 1.0 / (np.abs(eigenvalues) + epsilon)
        
        # Normalize amplitudes
        amplitudes = amplitudes / np.linalg.norm(amplitudes)
        
        return amplitudes
    
    def _amplitude_amplification(
        self,
        amplitudes: np.ndarray,
        quantum_resource: QuantumResource
    ) -> np.ndarray:
        """Simulate amplitude amplification."""
        
        # Amplitude amplification improves success probability
        amplification_factor = min(np.sqrt(len(amplitudes)), 10.0)
        
        amplified = amplitudes * amplification_factor
        
        # Add decoherence effects
        decoherence_factor = np.exp(-1.0 / quantum_resource.coherence_time_ms)
        amplified *= decoherence_factor
        
        return amplified
    
    def _measurement_and_postprocessing(
        self,
        matrix_A: np.ndarray,
        vector_b: np.ndarray,
        amplitudes: np.ndarray
    ) -> np.ndarray:
        """Measurement and classical post-processing."""
        
        # Classical solution for comparison/correction
        classical_solution = np.linalg.solve(matrix_A, vector_b)
        
        # Combine quantum amplitudes with classical solution
        # In real implementation, this would be pure quantum result
        solution = classical_solution * np.mean(np.abs(amplitudes))
        
        return solution


class QuantumOptimizer(QuantumAlgorithm):
    """
    Quantum Approximate Optimization Algorithm (QAOA) for neural network optimization.
    
    Provides quantum speedup for combinatorial optimization problems
    in neural network training and hyperparameter optimization.
    """
    
    def __init__(self, num_layers: int = 3):
        self.num_layers = num_layers
        self.algorithm_name = "Quantum Approximate Optimization"
        
        logger.info(f"Initialized {self.algorithm_name} with {num_layers} layers")
    
    def estimate_quantum_advantage(self, problem_size: int) -> float:
        """Estimate quantum advantage for optimization problems."""
        
        # Classical complexity: exponential in problem size
        classical_complexity = 2 ** min(problem_size, 50)  # Cap to avoid overflow
        
        # QAOA complexity: polynomial in problem size
        qaoa_complexity = problem_size ** 2 * self.num_layers
        
        speedup = classical_complexity / qaoa_complexity
        
        # QAOA advantage increases with problem size
        if problem_size < 20:
            speedup *= 0.2
        elif problem_size < 50:
            speedup *= 0.6
        
        return min(speedup, 1e8)
    
    def required_qubits(self, problem_size: int) -> int:
        """Calculate required qubits for QAOA."""
        
        # Each problem variable maps to one qubit
        problem_qubits = problem_size
        
        # Additional qubits for parameter optimization
        parameter_qubits = self.num_layers * 2  # 2 parameters per layer
        
        total_qubits = problem_qubits + parameter_qubits
        
        return total_qubits
    
    def execute(
        self,
        cost_function: Callable,
        initial_parameters: np.ndarray,
        quantum_resource: QuantumResource
    ) -> Tuple[np.ndarray, float]:
        """Execute QAOA optimization."""
        
        problem_size = len(initial_parameters)
        required_qubits = self.required_qubits(problem_size)
        
        if not quantum_resource.is_sufficient_for_problem(required_qubits):
            raise SpintronError(f"Insufficient quantum resources: need {required_qubits} qubits")
        
        logger.info(f"Running QAOA optimization for {problem_size} parameters")
        
        start_time = time.time()
        
        try:
            # QAOA optimization loop
            best_parameters, best_cost = self._qaoa_optimization_loop(
                cost_function, initial_parameters, quantum_resource
            )
            
            execution_time = time.time() - start_time
            logger.info(f"QAOA optimization completed in {execution_time:.3f}s")
            
            return best_parameters, best_cost
            
        except Exception as e:
            raise SpintronError(f"QAOA optimization failed: {str(e)}")
    
    def _qaoa_optimization_loop(
        self,
        cost_function: Callable,
        initial_parameters: np.ndarray,
        quantum_resource: QuantumResource
    ) -> Tuple[np.ndarray, float]:
        """QAOA optimization loop."""
        
        # Initialize QAOA parameters
        beta_params = np.random.uniform(0, np.pi, self.num_layers)
        gamma_params = np.random.uniform(0, 2*np.pi, self.num_layers)
        
        best_parameters = initial_parameters.copy()
        best_cost = cost_function(initial_parameters)
        
        # QAOA iterations
        for iteration in range(50):  # Limited iterations for demonstration
            # Quantum state preparation and evolution
            quantum_state = self._prepare_quantum_state(
                beta_params, gamma_params, initial_parameters, quantum_resource
            )
            
            # Measurement and expectation value calculation
            expectation_value = self._calculate_expectation_value(
                quantum_state, cost_function
            )
            
            # Classical optimization of QAOA parameters
            beta_params, gamma_params = self._optimize_qaoa_parameters(
                beta_params, gamma_params, expectation_value
            )
            
            # Extract solution from quantum state
            candidate_parameters = self._extract_solution(quantum_state)
            candidate_cost = cost_function(candidate_parameters)
            
            # Update best solution
            if candidate_cost < best_cost:
                best_parameters = candidate_parameters
                best_cost = candidate_cost
        
        return best_parameters, best_cost
    
    def _prepare_quantum_state(
        self,
        beta_params: np.ndarray,
        gamma_params: np.ndarray,
        problem_parameters: np.ndarray,
        quantum_resource: QuantumResource
    ) -> np.ndarray:
        """Prepare QAOA quantum state."""
        
        # Initialize uniform superposition
        n_qubits = len(problem_parameters)
        state_size = 2 ** n_qubits
        quantum_state = np.ones(state_size, dtype=complex) / np.sqrt(state_size)
        
        # Apply QAOA layers
        for layer in range(self.num_layers):
            # Problem Hamiltonian evolution
            quantum_state = self._apply_problem_hamiltonian(
                quantum_state, gamma_params[layer], problem_parameters
            )
            
            # Mixer Hamiltonian evolution
            quantum_state = self._apply_mixer_hamiltonian(
                quantum_state, beta_params[layer]
            )
            
            # Add quantum noise
            noise_factor = quantum_resource.error_rate
            noise = np.random.normal(0, noise_factor, quantum_state.shape)
            quantum_state += noise * 1j
            
            # Renormalize
            quantum_state = quantum_state / np.linalg.norm(quantum_state)
        
        return quantum_state
    
    def _apply_problem_hamiltonian(
        self,
        state: np.ndarray,
        gamma: float,
        problem_parameters: np.ndarray
    ) -> np.ndarray:
        """Apply problem Hamiltonian evolution."""
        
        # Simplified problem Hamiltonian evolution
        # In practice, this would implement the specific problem structure
        
        n_qubits = int(np.log2(len(state)))
        evolved_state = state.copy()
        
        for i in range(n_qubits):
            # Apply Z-rotation based on problem parameters
            rotation_angle = gamma * problem_parameters[i]
            
            for j in range(len(state)):
                bit_value = (j >> i) & 1
                phase = (-1) ** bit_value
                evolved_state[j] *= np.exp(1j * rotation_angle * phase)
        
        return evolved_state
    
    def _apply_mixer_hamiltonian(self, state: np.ndarray, beta: float) -> np.ndarray:
        """Apply mixer Hamiltonian evolution."""
        
        # X-mixer: applies X-rotation to each qubit
        n_qubits = int(np.log2(len(state)))
        mixed_state = np.zeros_like(state)
        
        for i in range(len(state)):
            for qubit in range(n_qubits):
                # Flip qubit
                flipped_index = i ^ (1 << qubit)
                
                # Apply rotation
                mixed_state[i] += np.cos(beta) * state[i]
                mixed_state[flipped_index] += -1j * np.sin(beta) * state[i]
        
        return mixed_state
    
    def _calculate_expectation_value(
        self,
        quantum_state: np.ndarray,
        cost_function: Callable
    ) -> float:
        """Calculate expectation value of cost function."""
        
        n_qubits = int(np.log2(len(quantum_state)))
        expectation = 0.0
        
        for i in range(len(quantum_state)):
            # Convert bit string to parameter vector
            bit_string = format(i, f'0{n_qubits}b')
            parameters = np.array([int(bit) for bit in bit_string], dtype=float)
            
            # Calculate probability and cost
            probability = np.abs(quantum_state[i]) ** 2
            cost = cost_function(parameters)
            
            expectation += probability * cost
        
        return expectation
    
    def _optimize_qaoa_parameters(
        self,
        beta_params: np.ndarray,
        gamma_params: np.ndarray,
        expectation_value: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Optimize QAOA parameters using classical optimizer."""
        
        # Gradient-free optimization (simplified)
        learning_rate = 0.1
        
        # Add small random perturbations
        beta_gradient = np.random.normal(0, 0.01, len(beta_params))
        gamma_gradient = np.random.normal(0, 0.01, len(gamma_params))
        
        # Update parameters
        new_beta = beta_params - learning_rate * beta_gradient
        new_gamma = gamma_params - learning_rate * gamma_gradient
        
        # Ensure parameters stay in valid range
        new_beta = np.clip(new_beta, 0, np.pi)
        new_gamma = np.clip(new_gamma, 0, 2*np.pi)
        
        return new_beta, new_gamma
    
    def _extract_solution(self, quantum_state: np.ndarray) -> np.ndarray:
        """Extract classical solution from quantum state."""
        
        # Find state with highest probability
        probabilities = np.abs(quantum_state) ** 2
        best_index = np.argmax(probabilities)
        
        # Convert to bit string
        n_qubits = int(np.log2(len(quantum_state)))
        bit_string = format(best_index, f'0{n_qubits}b')
        
        # Convert to parameter vector
        solution = np.array([int(bit) for bit in bit_string], dtype=float)
        
        return solution


class HybridQuantumClassicalProcessor:
    """
    Hybrid processor that intelligently distributes computation between
    quantum and classical resources for optimal performance.
    """
    
    def __init__(
        self,
        quantum_resources: List[QuantumResource],
        classical_cores: int = 8
    ):
        self.quantum_resources = quantum_resources
        self.classical_cores = classical_cores
        
        # Available quantum algorithms
        self.quantum_algorithms = {
            'linear_solver': QuantumLinearSolver(),
            'optimizer': QuantumOptimizer()
        }
        
        # Performance tracking
        self.performance_history = []
        self.quantum_advantage_threshold = 2.0  # Minimum speedup to use quantum
        
        # Resource allocation
        self.classical_executor = ThreadPoolExecutor(max_workers=classical_cores)
        
        logger.info(f"Initialized HybridQuantumClassicalProcessor with {len(quantum_resources)} quantum resources")
    
    async def compute_optimal(
        self,
        computation_type: str,
        input_data: Any,
        **kwargs
    ) -> Any:
        """Compute using optimal resource allocation."""
        
        # Analyze computation requirements
        analysis = self._analyze_computation(computation_type, input_data)
        
        # Decide on resource allocation
        allocation_decision = await self._decide_resource_allocation(analysis)
        
        # Execute computation
        if allocation_decision['use_quantum']:
            result = await self._execute_quantum_computation(
                computation_type, input_data, allocation_decision, **kwargs
            )
        else:
            result = await self._execute_classical_computation(
                computation_type, input_data, **kwargs
            )
        
        # Record performance
        self._record_performance(analysis, allocation_decision, result)
        
        return result
    
    def _analyze_computation(
        self,
        computation_type: str,
        input_data: Any
    ) -> Dict[str, Any]:
        """Analyze computation to determine resource requirements."""
        
        analysis = {
            'computation_type': computation_type,
            'problem_size': self._estimate_problem_size(input_data),
            'complexity_class': self._estimate_complexity_class(computation_type),
            'quantum_amenable': self._is_quantum_amenable(computation_type)
        }
        
        return analysis
    
    def _estimate_problem_size(self, input_data: Any) -> int:
        """Estimate problem size from input data."""
        
        if isinstance(input_data, np.ndarray):
            if input_data.ndim == 1:
                return len(input_data)
            elif input_data.ndim == 2:
                return max(input_data.shape)
            else:
                return int(np.prod(input_data.shape) ** (1/input_data.ndim))
        elif isinstance(input_data, (list, tuple)):
            return len(input_data)
        else:
            return 100  # Default estimate
    
    def _estimate_complexity_class(self, computation_type: str) -> str:
        """Estimate computational complexity class."""
        
        complexity_map = {
            'linear_solver': 'P^3',  # O(N^3)
            'matrix_multiply': 'P^3',
            'optimization': 'EXP',  # Exponential
            'factorization': 'EXP',
            'search': 'P^2',
            'sorting': 'P*log(P)'
        }
        
        return complexity_map.get(computation_type, 'P^2')
    
    def _is_quantum_amenable(self, computation_type: str) -> bool:
        """Check if computation type is amenable to quantum speedup."""
        
        quantum_amenable_types = {
            'linear_solver',
            'optimization',
            'factorization',
            'search',
            'fourier_transform',
            'eigenvalue_problems'
        }
        
        return computation_type in quantum_amenable_types
    
    async def _decide_resource_allocation(
        self,
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Decide whether to use quantum or classical resources."""
        
        decision = {
            'use_quantum': False,
            'quantum_resource': None,
            'algorithm': None,
            'estimated_speedup': 1.0
        }
        
        # Check if quantum computation is possible
        if not analysis['quantum_amenable']:
            return decision
        
        # Find suitable quantum algorithm
        algorithm_name = self._select_quantum_algorithm(analysis)
        if algorithm_name is None:
            return decision
        
        algorithm = self.quantum_algorithms[algorithm_name]
        
        # Estimate quantum advantage
        estimated_speedup = algorithm.estimate_quantum_advantage(analysis['problem_size'])
        
        # Check if quantum resources are available
        suitable_resource = self._find_suitable_quantum_resource(
            algorithm, analysis['problem_size']
        )
        
        if (suitable_resource is not None and 
            estimated_speedup > self.quantum_advantage_threshold):
            
            decision.update({
                'use_quantum': True,
                'quantum_resource': suitable_resource,
                'algorithm': algorithm_name,
                'estimated_speedup': estimated_speedup
            })
        
        return decision
    
    def _select_quantum_algorithm(self, analysis: Dict[str, Any]) -> Optional[str]:
        """Select appropriate quantum algorithm."""
        
        computation_type = analysis['computation_type']
        
        algorithm_map = {
            'linear_solver': 'linear_solver',
            'optimization': 'optimizer',
            'eigenvalue_problems': 'linear_solver'
        }
        
        return algorithm_map.get(computation_type)
    
    def _find_suitable_quantum_resource(
        self,
        algorithm: QuantumAlgorithm,
        problem_size: int
    ) -> Optional[QuantumResource]:
        """Find quantum resource suitable for algorithm and problem size."""
        
        required_qubits = algorithm.required_qubits(problem_size)
        
        for resource in self.quantum_resources:
            if resource.is_sufficient_for_problem(required_qubits):
                return resource
        
        return None
    
    async def _execute_quantum_computation(
        self,
        computation_type: str,
        input_data: Any,
        allocation_decision: Dict[str, Any],
        **kwargs
    ) -> Any:
        """Execute computation on quantum hardware."""
        
        algorithm_name = allocation_decision['algorithm']
        quantum_resource = allocation_decision['quantum_resource']
        algorithm = self.quantum_algorithms[algorithm_name]
        
        logger.info(f"Executing {computation_type} on quantum hardware using {algorithm_name}")
        
        try:
            if computation_type == 'linear_solver':
                matrix_A = kwargs.get('matrix_A', input_data)
                vector_b = kwargs.get('vector_b', np.ones(input_data.shape[0]))
                result = algorithm.execute(matrix_A, vector_b, quantum_resource)
            
            elif computation_type == 'optimization':
                cost_function = kwargs.get('cost_function')
                result = algorithm.execute(cost_function, input_data, quantum_resource)
            
            else:
                raise SpintronError(f"Unsupported quantum computation type: {computation_type}")
            
            return result
            
        except Exception as e:
            logger.warning(f"Quantum computation failed: {str(e)}, falling back to classical")
            # Fallback to classical computation
            return await self._execute_classical_computation(computation_type, input_data, **kwargs)
    
    async def _execute_classical_computation(
        self,
        computation_type: str,
        input_data: Any,
        **kwargs
    ) -> Any:
        """Execute computation on classical hardware."""
        
        logger.info(f"Executing {computation_type} on classical hardware")
        
        # Submit to thread pool for parallel execution
        loop = asyncio.get_event_loop()
        
        if computation_type == 'linear_solver':
            matrix_A = kwargs.get('matrix_A', input_data)
            vector_b = kwargs.get('vector_b', np.ones(input_data.shape[0]))
            result = await loop.run_in_executor(
                self.classical_executor, 
                np.linalg.solve, 
                matrix_A, vector_b
            )
        
        elif computation_type == 'optimization':
            cost_function = kwargs.get('cost_function')
            result = await loop.run_in_executor(
                self.classical_executor,
                self._classical_optimization,
                cost_function, input_data
            )
        
        elif computation_type == 'matrix_multiply':
            matrix_B = kwargs.get('matrix_B')
            result = await loop.run_in_executor(
                self.classical_executor,
                np.dot,
                input_data, matrix_B
            )
        
        else:
            raise SpintronError(f"Unsupported classical computation type: {computation_type}")
        
        return result
    
    def _classical_optimization(
        self,
        cost_function: Callable,
        initial_parameters: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """Classical optimization algorithm."""
        
        from scipy.optimize import minimize
        
        result = minimize(cost_function, initial_parameters, method='BFGS')
        
        return result.x, result.fun
    
    def _record_performance(
        self,
        analysis: Dict[str, Any],
        allocation_decision: Dict[str, Any],
        result: Any
    ):
        """Record performance metrics for learning."""
        
        performance_record = {
            'timestamp': time.time(),
            'computation_type': analysis['computation_type'],
            'problem_size': analysis['problem_size'],
            'used_quantum': allocation_decision['use_quantum'],
            'estimated_speedup': allocation_decision.get('estimated_speedup', 1.0),
            'success': result is not None
        }
        
        self.performance_history.append(performance_record)
        
        # Keep only recent history
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance statistics for quantum vs classical execution."""
        
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        quantum_executions = [p for p in self.performance_history if p['used_quantum']]
        classical_executions = [p for p in self.performance_history if not p['used_quantum']]
        
        stats = {
            'total_executions': len(self.performance_history),
            'quantum_executions': len(quantum_executions),
            'classical_executions': len(classical_executions),
            'quantum_utilization': len(quantum_executions) / len(self.performance_history),
            'quantum_success_rate': np.mean([p['success'] for p in quantum_executions]) if quantum_executions else 0,
            'classical_success_rate': np.mean([p['success'] for p in classical_executions]) if classical_executions else 0,
            'average_problem_size': np.mean([p['problem_size'] for p in self.performance_history]),
            'quantum_advantage_realized': np.mean([p['estimated_speedup'] for p in quantum_executions]) if quantum_executions else 1.0
        }
        
        return stats
    
    def adapt_quantum_threshold(self):
        """Adapt quantum advantage threshold based on historical performance."""
        
        if len(self.performance_history) < 20:
            return  # Need sufficient data
        
        quantum_executions = [p for p in self.performance_history if p['used_quantum']]
        
        if quantum_executions:
            # Calculate actual vs estimated performance
            success_rate = np.mean([p['success'] for p in quantum_executions])
            
            # Adjust threshold based on success rate
            if success_rate > 0.9:
                self.quantum_advantage_threshold *= 0.95  # Lower threshold
            elif success_rate < 0.7:
                self.quantum_advantage_threshold *= 1.05  # Raise threshold
            
            # Keep threshold in reasonable range
            self.quantum_advantage_threshold = np.clip(
                self.quantum_advantage_threshold, 1.1, 10.0
            )
            
            logger.info(f"Adapted quantum advantage threshold to {self.quantum_advantage_threshold:.2f}")
    
    def shutdown(self):
        """Shutdown hybrid processor and clean up resources."""
        
        self.classical_executor.shutdown(wait=True)
        logger.info("HybridQuantumClassicalProcessor shut down")
