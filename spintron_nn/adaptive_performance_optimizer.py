"""
Adaptive Performance Optimizer for SpinTron-NN-Kit.

This module provides intelligent performance optimization with machine learning,
adaptive algorithms, and real-time system tuning for maximum efficiency.
"""

import time
import json
import math
import random
import threading
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import statistics


class OptimizationTarget(Enum):
    """Optimization targets."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ENERGY_EFFICIENCY = "energy_efficiency"
    MEMORY_USAGE = "memory_usage"
    ACCURACY = "accuracy"
    COST_EFFICIENCY = "cost_efficiency"
    BALANCED = "balanced"


class OptimizationTechnique(Enum):
    """Optimization techniques."""
    GENETIC_ALGORITHM = "genetic_algorithm"
    SIMULATED_ANNEALING = "simulated_annealing"
    GRADIENT_DESCENT = "gradient_descent"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    NEURAL_ARCHITECTURE_SEARCH = "neural_architecture_search"


class SystemParameter(Enum):
    """System parameters that can be optimized."""
    BATCH_SIZE = "batch_size"
    LEARNING_RATE = "learning_rate"
    CROSSBAR_SIZE = "crossbar_size"
    QUANTIZATION_BITS = "quantization_bits"
    PARALLELISM_FACTOR = "parallelism_factor"
    CACHE_SIZE = "cache_size"
    VOLTAGE_SCALING = "voltage_scaling"
    FREQUENCY_SCALING = "frequency_scaling"


@dataclass
class ParameterSpace:
    """Parameter space definition."""
    
    parameter: SystemParameter
    min_value: float
    max_value: float
    step_size: float
    current_value: float
    optimal_value: Optional[float] = None
    
    def sample_random(self) -> float:
        """Sample random value from parameter space."""
        return random.uniform(self.min_value, self.max_value)
    
    def clip_value(self, value: float) -> float:
        """Clip value to parameter bounds."""
        return max(self.min_value, min(self.max_value, value))


@dataclass
class PerformanceMetrics:
    """Performance metrics snapshot."""
    
    timestamp: float
    throughput: float
    latency: float
    energy_efficiency: float
    memory_usage: float
    accuracy: float
    cost_efficiency: float
    
    def to_dict(self) -> Dict[str, float]:
        return asdict(self)
    
    def get_metric(self, target: OptimizationTarget) -> float:
        """Get specific metric value."""
        metric_map = {
            OptimizationTarget.THROUGHPUT: self.throughput,
            OptimizationTarget.LATENCY: self.latency,
            OptimizationTarget.ENERGY_EFFICIENCY: self.energy_efficiency,
            OptimizationTarget.MEMORY_USAGE: self.memory_usage,
            OptimizationTarget.ACCURACY: self.accuracy,
            OptimizationTarget.COST_EFFICIENCY: self.cost_efficiency,
            OptimizationTarget.BALANCED: self._calculate_balanced_score()
        }
        return metric_map.get(target, 0.0)
    
    def _calculate_balanced_score(self) -> float:
        """Calculate balanced performance score."""
        # Weighted combination of all metrics
        return (
            0.25 * self.throughput +
            0.15 * (1.0 / max(self.latency, 0.001)) +  # Lower latency is better
            0.25 * self.energy_efficiency +
            0.10 * (1.0 / max(self.memory_usage, 0.001)) +  # Lower memory is better
            0.20 * self.accuracy +
            0.05 * self.cost_efficiency
        )


@dataclass
class OptimizationState:
    """Current optimization state."""
    
    parameters: Dict[SystemParameter, float]
    metrics: PerformanceMetrics
    fitness_score: float
    generation: int
    
    def copy(self) -> 'OptimizationState':
        """Create copy of optimization state."""
        return OptimizationState(
            parameters=self.parameters.copy(),
            metrics=self.metrics,
            fitness_score=self.fitness_score,
            generation=self.generation
        )


class GeneticOptimizer:
    """Genetic algorithm optimizer."""
    
    def __init__(self, population_size: int = 50, mutation_rate: float = 0.1):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = 0.8
        self.elite_size = max(2, population_size // 10)
        
        self.population = []
        self.generation = 0
        self.best_fitness_history = []
    
    def initialize_population(self, parameter_spaces: Dict[SystemParameter, ParameterSpace]) -> List[Dict[SystemParameter, float]]:
        """Initialize random population."""
        
        population = []
        
        for _ in range(self.population_size):
            individual = {}
            for param, space in parameter_spaces.items():
                individual[param] = space.sample_random()
            population.append(individual)
        
        return population
    
    def evolve_generation(self, 
                         population: List[OptimizationState],
                         parameter_spaces: Dict[SystemParameter, ParameterSpace]) -> List[Dict[SystemParameter, float]]:
        """Evolve population for one generation."""
        
        # Sort by fitness (descending)
        population.sort(key=lambda x: x.fitness_score, reverse=True)
        
        # Track best fitness
        self.best_fitness_history.append(population[0].fitness_score)
        
        new_population = []
        
        # Elitism: Keep best individuals
        for i in range(self.elite_size):
            new_population.append(population[i].parameters.copy())
        
        # Generate rest through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1.parameters, parent2.parameters)
            else:
                child1, child2 = parent1.parameters.copy(), parent2.parameters.copy()
            
            # Mutation
            child1 = self._mutate(child1, parameter_spaces)
            child2 = self._mutate(child2, parameter_spaces)
            
            new_population.extend([child1, child2])
        
        # Trim to exact population size
        new_population = new_population[:self.population_size]
        self.generation += 1
        
        return new_population
    
    def _tournament_selection(self, population: List[OptimizationState], tournament_size: int = 3) -> OptimizationState:
        """Tournament selection."""
        tournament = random.sample(population, min(tournament_size, len(population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _crossover(self, parent1: Dict[SystemParameter, float], parent2: Dict[SystemParameter, float]) -> Tuple[Dict[SystemParameter, float], Dict[SystemParameter, float]]:
        """Single-point crossover."""
        
        parameters = list(parent1.keys())
        crossover_point = random.randint(1, len(parameters) - 1)
        
        child1 = {}
        child2 = {}
        
        for i, param in enumerate(parameters):
            if i < crossover_point:
                child1[param] = parent1[param]
                child2[param] = parent2[param]
            else:
                child1[param] = parent2[param]
                child2[param] = parent1[param]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[SystemParameter, float], parameter_spaces: Dict[SystemParameter, ParameterSpace]) -> Dict[SystemParameter, float]:
        """Gaussian mutation."""
        
        mutated = individual.copy()
        
        for param, value in mutated.items():
            if random.random() < self.mutation_rate:
                space = parameter_spaces[param]
                
                # Gaussian mutation with adaptive variance
                variance = (space.max_value - space.min_value) * 0.1
                new_value = value + random.gauss(0, variance)
                
                # Clip to bounds
                mutated[param] = space.clip_value(new_value)
        
        return mutated


class BayesianOptimizer:
    """Bayesian optimization with Gaussian processes."""
    
    def __init__(self):
        self.observations = []
        self.acquisition_function = "expected_improvement"
        self.exploration_weight = 0.1
    
    def suggest_next_point(self, parameter_spaces: Dict[SystemParameter, ParameterSpace]) -> Dict[SystemParameter, float]:
        """Suggest next point to evaluate using acquisition function."""
        
        if len(self.observations) < 3:
            # Random exploration for initial points
            return {param: space.sample_random() for param, space in parameter_spaces.items()}
        
        # Simplified acquisition function (in practice, use GPy or similar)
        best_candidate = None
        best_acquisition_value = float('-inf')
        
        # Sample candidate points
        for _ in range(100):
            candidate = {param: space.sample_random() for param, space in parameter_spaces.items()}
            acquisition_value = self._calculate_acquisition(candidate)
            
            if acquisition_value > best_acquisition_value:
                best_acquisition_value = acquisition_value
                best_candidate = candidate
        
        return best_candidate
    
    def update_observations(self, parameters: Dict[SystemParameter, float], fitness: float):
        """Update observations with new evaluation."""
        self.observations.append((parameters.copy(), fitness))
        
        # Keep only recent observations to limit memory
        if len(self.observations) > 100:
            self.observations = self.observations[-100:]
    
    def _calculate_acquisition(self, candidate: Dict[SystemParameter, float]) -> float:
        """Calculate acquisition function value (simplified)."""
        
        # Calculate similarity to existing observations
        distances = []
        fitness_values = []
        
        for obs_params, obs_fitness in self.observations:
            distance = self._calculate_distance(candidate, obs_params)
            distances.append(distance)
            fitness_values.append(obs_fitness)
        
        if not distances:
            return random.random()
        
        # Predict mean and variance (simplified)
        weights = [1.0 / (d + 1e-6) for d in distances]
        total_weight = sum(weights)
        
        if total_weight > 0:
            predicted_mean = sum(w * f for w, f in zip(weights, fitness_values)) / total_weight
            predicted_variance = self.exploration_weight * min(distances)
        else:
            predicted_mean = 0.5
            predicted_variance = 1.0
        
        # Expected improvement acquisition function
        best_observed = max(fitness_values) if fitness_values else 0.0
        improvement = predicted_mean - best_observed
        
        if predicted_variance > 0:
            z_score = improvement / math.sqrt(predicted_variance)
            # Simplified normal CDF and PDF
            expected_improvement = improvement * 0.5 * (1 + math.erf(z_score / math.sqrt(2))) + \
                                 math.sqrt(predicted_variance) * math.exp(-0.5 * z_score**2) / math.sqrt(2 * math.pi)
        else:
            expected_improvement = max(0, improvement)
        
        return expected_improvement
    
    def _calculate_distance(self, params1: Dict[SystemParameter, float], params2: Dict[SystemParameter, float]) -> float:
        """Calculate Euclidean distance between parameter sets."""
        
        distance = 0.0
        for param in params1:
            if param in params2:
                diff = params1[param] - params2[param]
                distance += diff * diff
        
        return math.sqrt(distance)


class AdaptivePerformanceOptimizer:
    """Main adaptive performance optimization system."""
    
    def __init__(self, optimization_target: OptimizationTarget = OptimizationTarget.BALANCED):
        self.optimization_target = optimization_target
        
        # Parameter spaces
        self.parameter_spaces = self._initialize_parameter_spaces()
        
        # Optimizers
        self.genetic_optimizer = GeneticOptimizer(population_size=30)
        self.bayesian_optimizer = BayesianOptimizer()
        
        # Current state
        self.current_state = None
        self.optimization_history = []
        self.best_state = None
        
        # Configuration
        self.optimization_technique = OptimizationTechnique.GENETIC_ALGORITHM
        self.max_iterations = 100
        self.convergence_threshold = 0.001
        self.evaluation_budget = 500
        
        # Performance monitoring
        self.metrics_history = []
        self.optimization_start_time = None
        
        # Adaptive learning
        self.learning_rate_decay = 0.95
        self.exploration_decay = 0.99
        
    def _initialize_parameter_spaces(self) -> Dict[SystemParameter, ParameterSpace]:
        """Initialize parameter spaces for optimization."""
        
        spaces = {
            SystemParameter.BATCH_SIZE: ParameterSpace(
                parameter=SystemParameter.BATCH_SIZE,
                min_value=1, max_value=128, step_size=1, current_value=32
            ),
            SystemParameter.LEARNING_RATE: ParameterSpace(
                parameter=SystemParameter.LEARNING_RATE,
                min_value=1e-5, max_value=1e-1, step_size=1e-5, current_value=1e-3
            ),
            SystemParameter.CROSSBAR_SIZE: ParameterSpace(
                parameter=SystemParameter.CROSSBAR_SIZE,
                min_value=32, max_value=512, step_size=16, current_value=128
            ),
            SystemParameter.QUANTIZATION_BITS: ParameterSpace(
                parameter=SystemParameter.QUANTIZATION_BITS,
                min_value=1, max_value=8, step_size=1, current_value=4
            ),
            SystemParameter.PARALLELISM_FACTOR: ParameterSpace(
                parameter=SystemParameter.PARALLELISM_FACTOR,
                min_value=1, max_value=32, step_size=1, current_value=8
            ),
            SystemParameter.CACHE_SIZE: ParameterSpace(
                parameter=SystemParameter.CACHE_SIZE,
                min_value=64, max_value=2048, step_size=64, current_value=512
            ),
            SystemParameter.VOLTAGE_SCALING: ParameterSpace(
                parameter=SystemParameter.VOLTAGE_SCALING,
                min_value=0.6, max_value=1.2, step_size=0.05, current_value=1.0
            ),
            SystemParameter.FREQUENCY_SCALING: ParameterSpace(
                parameter=SystemParameter.FREQUENCY_SCALING,
                min_value=0.5, max_value=2.0, step_size=0.1, current_value=1.0
            )
        }
        
        return spaces
    
    def optimize_performance(self, max_time_seconds: float = 300) -> OptimizationState:
        """Run adaptive performance optimization."""
        
        print(f"🎯 Starting adaptive optimization for {self.optimization_target.value}")
        print(f"⚙️  Using {self.optimization_technique.value}")
        
        self.optimization_start_time = time.time()
        
        if self.optimization_technique == OptimizationTechnique.GENETIC_ALGORITHM:
            best_state = self._run_genetic_optimization(max_time_seconds)
        elif self.optimization_technique == OptimizationTechnique.BAYESIAN_OPTIMIZATION:
            best_state = self._run_bayesian_optimization(max_time_seconds)
        else:
            # Fallback to genetic algorithm
            best_state = self._run_genetic_optimization(max_time_seconds)
        
        optimization_time = time.time() - self.optimization_start_time
        
        print(f"✅ Optimization complete in {optimization_time:.2f}s")
        print(f"📈 Best fitness: {best_state.fitness_score:.4f}")
        print(f"🔧 Optimal parameters: {best_state.parameters}")
        
        self.best_state = best_state
        return best_state
    
    def _run_genetic_optimization(self, max_time_seconds: float) -> OptimizationState:
        """Run genetic algorithm optimization."""
        
        # Initialize population
        population_params = self.genetic_optimizer.initialize_population(self.parameter_spaces)
        
        best_state = None
        generation = 0
        
        start_time = time.time()
        
        while time.time() - start_time < max_time_seconds and generation < self.max_iterations:
            # Evaluate population
            population_states = []
            
            for params in population_params:
                metrics = self._evaluate_parameters(params)
                fitness = self._calculate_fitness(metrics)
                
                state = OptimizationState(
                    parameters=params,
                    metrics=metrics,
                    fitness_score=fitness,
                    generation=generation
                )
                
                population_states.append(state)
                self.optimization_history.append(state)
                
                # Track best state
                if best_state is None or fitness > best_state.fitness_score:
                    best_state = state.copy()
            
            # Check convergence
            if self._check_convergence():
                print(f"🎯 Converged at generation {generation}")
                break
            
            # Evolve population
            population_params = self.genetic_optimizer.evolve_generation(
                population_states, self.parameter_spaces
            )
            
            generation += 1
            
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_state.fitness_score:.4f}")
        
        return best_state
    
    def _run_bayesian_optimization(self, max_time_seconds: float) -> OptimizationState:
        """Run Bayesian optimization."""
        
        best_state = None
        iteration = 0
        
        start_time = time.time()
        
        while time.time() - start_time < max_time_seconds and iteration < self.max_iterations:
            # Get next point to evaluate
            params = self.bayesian_optimizer.suggest_next_point(self.parameter_spaces)
            
            # Evaluate parameters
            metrics = self._evaluate_parameters(params)
            fitness = self._calculate_fitness(metrics)
            
            # Update Bayesian optimizer
            self.bayesian_optimizer.update_observations(params, fitness)
            
            # Create state
            state = OptimizationState(
                parameters=params,
                metrics=metrics,
                fitness_score=fitness,
                generation=iteration
            )
            
            self.optimization_history.append(state)
            
            # Track best state
            if best_state is None or fitness > best_state.fitness_score:
                best_state = state.copy()
            
            iteration += 1
            
            if iteration % 20 == 0:
                print(f"Iteration {iteration}: Best fitness = {best_state.fitness_score:.4f}")
        
        return best_state
    
    def _evaluate_parameters(self, parameters: Dict[SystemParameter, float]) -> PerformanceMetrics:
        """Evaluate system performance with given parameters."""
        
        # Simulate system evaluation with given parameters
        
        # Base performance values
        base_throughput = 1000.0
        base_latency = 10.0
        base_energy_efficiency = 0.8
        base_memory_usage = 512.0
        base_accuracy = 0.9
        base_cost_efficiency = 0.7
        
        # Parameter effects (simplified model)
        batch_size = parameters.get(SystemParameter.BATCH_SIZE, 32)
        learning_rate = parameters.get(SystemParameter.LEARNING_RATE, 1e-3)
        crossbar_size = parameters.get(SystemParameter.CROSSBAR_SIZE, 128)
        quantization_bits = parameters.get(SystemParameter.QUANTIZATION_BITS, 4)
        parallelism = parameters.get(SystemParameter.PARALLELISM_FACTOR, 8)
        cache_size = parameters.get(SystemParameter.CACHE_SIZE, 512)
        voltage_scaling = parameters.get(SystemParameter.VOLTAGE_SCALING, 1.0)
        frequency_scaling = parameters.get(SystemParameter.FREQUENCY_SCALING, 1.0)
        
        # Calculate performance effects
        throughput = base_throughput * (batch_size / 32) * (parallelism / 8) * frequency_scaling
        throughput *= (crossbar_size / 128) ** 0.5  # Larger crossbars help throughput
        
        latency = base_latency / (frequency_scaling ** 0.8) * (batch_size / 32) ** 0.3
        latency += max(0, (cache_size - 1024) / 1024) * 2  # Large cache increases latency
        
        energy_efficiency = base_energy_efficiency * (voltage_scaling ** -2) * (frequency_scaling ** -1.5)
        energy_efficiency *= (quantization_bits / 4) ** -0.5  # Lower bits = higher efficiency
        
        memory_usage = base_memory_usage * (batch_size / 32) * (crossbar_size / 128) ** 2
        memory_usage += cache_size
        
        accuracy = base_accuracy * (quantization_bits / 4) ** 0.3
        accuracy *= min(1.0, learning_rate * 1000) ** 0.1  # Optimal learning rate
        accuracy = min(0.99, accuracy + random.gauss(0, 0.01))  # Add noise
        
        cost_efficiency = base_cost_efficiency / (voltage_scaling * frequency_scaling)
        cost_efficiency *= (crossbar_size / 128) ** -0.2  # Larger crossbars cost more
        
        # Add realistic noise and constraints
        throughput = max(100, throughput + random.gauss(0, throughput * 0.05))
        latency = max(1.0, latency + random.gauss(0, latency * 0.1))
        energy_efficiency = max(0.1, min(0.99, energy_efficiency + random.gauss(0, 0.02)))
        memory_usage = max(64, memory_usage + random.gauss(0, memory_usage * 0.03))
        accuracy = max(0.5, min(0.99, accuracy))
        cost_efficiency = max(0.1, min(0.99, cost_efficiency + random.gauss(0, 0.02)))
        
        # Simulate evaluation time
        time.sleep(0.01)  # Realistic evaluation delay
        
        return PerformanceMetrics(
            timestamp=time.time(),
            throughput=throughput,
            latency=latency,
            energy_efficiency=energy_efficiency,
            memory_usage=memory_usage,
            accuracy=accuracy,
            cost_efficiency=cost_efficiency
        )
    
    def _calculate_fitness(self, metrics: PerformanceMetrics) -> float:
        """Calculate fitness score based on optimization target."""
        
        if self.optimization_target == OptimizationTarget.BALANCED:
            return metrics.get_metric(self.optimization_target)
        
        # Single-objective optimization
        primary_score = metrics.get_metric(self.optimization_target)
        
        # Apply constraints (penalties for bad performance in other areas)
        penalties = 0.0
        
        # Accuracy constraint
        if metrics.accuracy < 0.8:
            penalties += (0.8 - metrics.accuracy) * 0.5
        
        # Energy efficiency constraint  
        if metrics.energy_efficiency < 0.6:
            penalties += (0.6 - metrics.energy_efficiency) * 0.3
        
        # Memory usage constraint (penalty for excessive memory)
        if metrics.memory_usage > 1024:
            penalties += (metrics.memory_usage - 1024) / 1024 * 0.2
        
        return max(0.0, primary_score - penalties)
    
    def _check_convergence(self) -> bool:
        """Check if optimization has converged."""
        
        if len(self.optimization_history) < 20:
            return False
        
        # Check fitness improvement over last 20 evaluations
        recent_fitness = [state.fitness_score for state in self.optimization_history[-20:]]
        fitness_improvement = max(recent_fitness) - min(recent_fitness)
        
        return fitness_improvement < self.convergence_threshold
    
    def adaptive_technique_selection(self):
        """Adaptively select optimization technique based on problem characteristics."""
        
        if len(self.optimization_history) < 50:
            # Use genetic algorithm for exploration
            self.optimization_technique = OptimizationTechnique.GENETIC_ALGORITHM
        else:
            # Analyze optimization landscape
            fitness_values = [state.fitness_score for state in self.optimization_history[-50:]]
            fitness_variance = statistics.variance(fitness_values) if len(fitness_values) > 1 else 1.0
            
            if fitness_variance > 0.1:
                # High variance - use exploration-heavy method
                self.optimization_technique = OptimizationTechnique.GENETIC_ALGORITHM
            else:
                # Low variance - use exploitation-heavy method
                self.optimization_technique = OptimizationTechnique.BAYESIAN_OPTIMIZATION
    
    def real_time_optimization(self, duration_seconds: float = 300):
        """Run real-time optimization with continuous adaptation."""
        
        print(f"🔄 Starting real-time optimization for {duration_seconds}s")
        
        start_time = time.time()
        iteration = 0
        
        while time.time() - start_time < duration_seconds:
            # Adaptive technique selection
            if iteration % 20 == 0:
                self.adaptive_technique_selection()
                print(f"🔧 Switched to {self.optimization_technique.value}")
            
            # Run short optimization burst
            if self.optimization_technique == OptimizationTechnique.GENETIC_ALGORITHM:
                self._run_genetic_burst()
            else:
                self._run_bayesian_burst()
            
            # Update parameter spaces based on learning
            self._update_parameter_spaces()
            
            iteration += 1
            
            # Brief pause
            time.sleep(1.0)
        
        print(f"✅ Real-time optimization complete")
        
        if self.best_state:
            print(f"📈 Final best fitness: {self.best_state.fitness_score:.4f}")
    
    def _run_genetic_burst(self):
        """Run short genetic algorithm burst."""
        
        # Small population for quick iterations
        burst_optimizer = GeneticOptimizer(population_size=10)
        population_params = burst_optimizer.initialize_population(self.parameter_spaces)
        
        # Single generation
        population_states = []
        for params in population_params:
            metrics = self._evaluate_parameters(params)
            fitness = self._calculate_fitness(metrics)
            
            state = OptimizationState(
                parameters=params,
                metrics=metrics,
                fitness_score=fitness,
                generation=0
            )
            
            population_states.append(state)
            self.optimization_history.append(state)
            
            if self.best_state is None or fitness > self.best_state.fitness_score:
                self.best_state = state.copy()
    
    def _run_bayesian_burst(self):
        """Run short Bayesian optimization burst."""
        
        # Single evaluation
        params = self.bayesian_optimizer.suggest_next_point(self.parameter_spaces)
        metrics = self._evaluate_parameters(params)
        fitness = self._calculate_fitness(metrics)
        
        self.bayesian_optimizer.update_observations(params, fitness)
        
        state = OptimizationState(
            parameters=params,
            metrics=metrics,
            fitness_score=fitness,
            generation=0
        )
        
        self.optimization_history.append(state)
        
        if self.best_state is None or fitness > self.best_state.fitness_score:
            self.best_state = state.copy()
    
    def _update_parameter_spaces(self):
        """Update parameter spaces based on optimization learning."""
        
        if len(self.optimization_history) < 10:
            return
        
        # Analyze successful parameter ranges
        top_states = sorted(self.optimization_history[-50:], key=lambda x: x.fitness_score, reverse=True)[:10]
        
        for param, space in self.parameter_spaces.items():
            successful_values = [state.parameters[param] for state in top_states if param in state.parameters]
            
            if successful_values:
                # Update search space to focus on successful regions
                mean_value = statistics.mean(successful_values)
                std_value = statistics.stdev(successful_values) if len(successful_values) > 1 else 0.1
                
                # Narrow search space around successful region
                new_min = max(space.min_value, mean_value - 2 * std_value)
                new_max = min(space.max_value, mean_value + 2 * std_value)
                
                # Gradually adapt (don't change too quickly)
                space.min_value = 0.9 * space.min_value + 0.1 * new_min
                space.max_value = 0.9 * space.max_value + 0.1 * new_max
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        
        if not self.optimization_history:
            return {"error": "No optimization history available"}
        
        fitness_values = [state.fitness_score for state in self.optimization_history]
        
        report = {
            "optimization_target": self.optimization_target.value,
            "technique_used": self.optimization_technique.value,
            "total_evaluations": len(self.optimization_history),
            "best_fitness": max(fitness_values),
            "average_fitness": statistics.mean(fitness_values),
            "fitness_improvement": max(fitness_values) - min(fitness_values),
            "convergence_achieved": self._check_convergence(),
            "optimization_time": time.time() - self.optimization_start_time if self.optimization_start_time else 0
        }
        
        if self.best_state:
            report["best_parameters"] = self.best_state.parameters
            report["best_metrics"] = self.best_state.metrics.to_dict()
        
        return report


def main():
    """Demonstrate adaptive performance optimizer."""
    
    # Test different optimization targets
    targets = [OptimizationTarget.THROUGHPUT, OptimizationTarget.ENERGY_EFFICIENCY, OptimizationTarget.BALANCED]
    
    for target in targets:
        print(f"\n{'='*60}")
        print(f"Testing optimization for {target.value}")
        print(f"{'='*60}")
        
        optimizer = AdaptivePerformanceOptimizer(optimization_target=target)
        
        # Run optimization
        best_state = optimizer.optimize_performance(max_time_seconds=30)
        
        # Get report
        report = optimizer.get_optimization_report()
        
        print(f"\n📊 Optimization Report:")
        print(f"Evaluations: {report['total_evaluations']}")
        print(f"Best fitness: {report['best_fitness']:.4f}")
        print(f"Improvement: {report['fitness_improvement']:.4f}")
        print(f"Time: {report['optimization_time']:.2f}s")
    
    return optimizer


if __name__ == "__main__":
    main()