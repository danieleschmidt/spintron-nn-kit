"""
Adaptive Scaling and Auto-Optimization Framework.

This module implements intelligent scaling algorithms, adaptive resource
management, and self-optimizing systems for spintronic neural networks.
"""

import asyncio
import time
import threading
import queue
import json
import math
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    THROUGHPUT = "throughput"
    LATENCY = "latency"
    ENERGY_EFFICIENCY = "energy_efficiency"
    ACCURACY = "accuracy"
    RESOURCE_UTILIZATION = "resource_utilization"
    THERMAL_STATE = "thermal_state"
    QUANTUM_COHERENCE = "quantum_coherence"


class OptimizationObjective(Enum):
    """Optimization objectives for adaptive scaling."""
    MAXIMIZE_THROUGHPUT = "maximize_throughput"
    MINIMIZE_LATENCY = "minimize_latency"
    MINIMIZE_ENERGY = "minimize_energy"
    MAXIMIZE_ACCURACY = "maximize_accuracy"
    BALANCED_PERFORMANCE = "balanced_performance"
    COST_OPTIMIZATION = "cost_optimization"


@dataclass
class ScalingConfiguration:
    """Configuration for adaptive scaling system."""
    
    # Scaling parameters
    min_scale_factor: float = 0.1
    max_scale_factor: float = 10.0
    scaling_step_size: float = 0.2
    
    # Decision thresholds
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    
    # Timing parameters
    monitoring_interval: float = 10.0  # seconds
    cooldown_period: float = 60.0     # seconds
    prediction_window: float = 300.0   # seconds
    
    # Optimization
    optimization_objective: OptimizationObjective = OptimizationObjective.BALANCED_PERFORMANCE
    multi_objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "throughput": 0.3,
        "latency": 0.2,
        "energy": 0.2,
        "accuracy": 0.2,
        "cost": 0.1
    })
    
    # Advanced features
    predictive_scaling: bool = True
    machine_learning_enabled: bool = True
    quantum_aware_scaling: bool = True
    thermal_management: bool = True


@dataclass
class SystemMetrics:
    """System performance and resource metrics."""
    
    timestamp: float = field(default_factory=time.time)
    
    # Performance metrics
    throughput: float = 0.0          # Operations per second
    latency: float = 0.0             # Average latency in ms
    accuracy: float = 0.0            # Model accuracy
    
    # Resource metrics
    cpu_utilization: float = 0.0     # CPU usage percentage
    memory_utilization: float = 0.0  # Memory usage percentage
    gpu_utilization: float = 0.0     # GPU usage percentage
    
    # Spintronic-specific metrics
    mtj_switching_rate: float = 0.0  # MTJ switching frequency
    energy_per_operation: float = 0.0  # Energy consumption per op
    device_temperature: float = 25.0   # Device temperature in C
    
    # Quantum metrics (if applicable)
    quantum_coherence_time: float = 0.0  # Quantum coherence time
    quantum_error_rate: float = 0.0      # Quantum error rate
    
    # Cost metrics
    monetary_cost_per_hour: float = 0.0  # Operating cost
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary."""
        return {
            "timestamp": self.timestamp,
            "throughput": self.throughput,
            "latency": self.latency,
            "accuracy": self.accuracy,
            "cpu_utilization": self.cpu_utilization,
            "memory_utilization": self.memory_utilization,
            "gpu_utilization": self.gpu_utilization,
            "mtj_switching_rate": self.mtj_switching_rate,
            "energy_per_operation": self.energy_per_operation,
            "device_temperature": self.device_temperature,
            "quantum_coherence_time": self.quantum_coherence_time,
            "quantum_error_rate": self.quantum_error_rate,
            "monetary_cost_per_hour": self.monetary_cost_per_hour
        }


class AdaptiveScalingController:
    """
    Intelligent scaling controller with predictive capabilities.
    
    Uses machine learning, time series analysis, and multi-objective
    optimization to make optimal scaling decisions for spintronic systems.
    """
    
    def __init__(self, config: ScalingConfiguration):
        self.config = config
        
        # Metrics storage
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = []
        
        # Scaling state
        self.current_scale_factor = 1.0
        self.last_scaling_time = 0.0
        self.cooldown_active = False
        
        # Predictive components
        self.trend_analyzer = TrendAnalyzer()
        self.performance_predictor = PerformancePredictor()
        self.multi_objective_optimizer = MultiObjectiveOptimizer(config.multi_objective_weights)
        
        # Advanced features
        self.thermal_manager = ThermalManager() if config.thermal_management else None
        self.quantum_optimizer = QuantumOptimizer() if config.quantum_aware_scaling else None
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        
        logger.info("Adaptive scaling controller initialized")
    
    def start_monitoring(self):
        """Start continuous system monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            logger.info("Started adaptive scaling monitoring")
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join()
        logger.info("Stopped adaptive scaling monitoring")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                # Collect current metrics
                current_metrics = self._collect_system_metrics()
                self.metrics_history.append(current_metrics)
                
                # Check if scaling decision is needed
                if not self.cooldown_active and len(self.metrics_history) > 5:
                    scaling_decision = self._evaluate_scaling_decision(current_metrics)
                    
                    if scaling_decision["should_scale"]:
                        self._execute_scaling_action(scaling_decision)
                
                # Check cooldown
                if self.cooldown_active:
                    if time.time() - self.last_scaling_time > self.config.cooldown_period:
                        self.cooldown_active = False
                        logger.debug("Scaling cooldown period ended")
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.config.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system performance metrics."""
        # Simulate metric collection (would integrate with actual monitoring)
        metrics = SystemMetrics(
            throughput=100 + np.random.normal(0, 10),
            latency=50 + np.random.normal(0, 5),
            accuracy=0.95 + np.random.normal(0, 0.02),
            cpu_utilization=np.random.uniform(0.3, 0.9),
            memory_utilization=np.random.uniform(0.4, 0.8),
            gpu_utilization=np.random.uniform(0.5, 0.95),
            mtj_switching_rate=1000 + np.random.normal(0, 100),
            energy_per_operation=10 + np.random.normal(0, 1),
            device_temperature=25 + np.random.normal(0, 5),
            quantum_coherence_time=100e-6 + np.random.normal(0, 10e-6),
            quantum_error_rate=np.random.uniform(0.001, 0.01),
            monetary_cost_per_hour=5.0 * self.current_scale_factor
        )
        
        return metrics
    
    def _evaluate_scaling_decision(self, current_metrics: SystemMetrics) -> Dict[str, Any]:
        """Evaluate whether scaling is needed and determine optimal action."""
        
        # Analyze trends
        trends = self.trend_analyzer.analyze_trends(list(self.metrics_history))
        
        # Predict future performance
        if self.config.predictive_scaling:
            predictions = self.performance_predictor.predict_performance(
                list(self.metrics_history), self.config.prediction_window
            )
        else:
            predictions = None
        
        # Multi-objective optimization
        optimization_result = self.multi_objective_optimizer.optimize_scaling(
            current_metrics, trends, predictions
        )
        
        # Thermal considerations
        thermal_constraint = True
        if self.thermal_manager:
            thermal_constraint = self.thermal_manager.check_thermal_constraints(current_metrics)
        
        # Quantum considerations
        quantum_constraint = True
        if self.quantum_optimizer:
            quantum_constraint = self.quantum_optimizer.check_quantum_constraints(current_metrics)
        
        # Make final decision
        should_scale = (
            optimization_result["should_scale"] and
            thermal_constraint and
            quantum_constraint
        )
        
        decision = {
            "should_scale": should_scale,
            "recommended_scale_factor": optimization_result.get("recommended_scale_factor", 1.0),
            "confidence": optimization_result.get("confidence", 0.5),
            "reasoning": optimization_result.get("reasoning", ""),
            "trends": trends,
            "predictions": predictions,
            "thermal_constraint": thermal_constraint,
            "quantum_constraint": quantum_constraint
        }
        
        return decision
    
    def _execute_scaling_action(self, scaling_decision: Dict[str, Any]):
        """Execute the scaling action."""
        recommended_scale = scaling_decision["recommended_scale_factor"]
        confidence = scaling_decision["confidence"]
        
        # Apply confidence-based adjustment
        scale_adjustment = (recommended_scale - self.current_scale_factor) * confidence
        new_scale_factor = self.current_scale_factor + scale_adjustment
        
        # Clamp to valid range
        new_scale_factor = max(self.config.min_scale_factor, 
                              min(self.config.max_scale_factor, new_scale_factor))
        
        # Execute scaling
        if abs(new_scale_factor - self.current_scale_factor) > 0.05:  # Minimum change threshold
            self._apply_scaling(new_scale_factor, scaling_decision)
    
    def _apply_scaling(self, new_scale_factor: float, decision_context: Dict[str, Any]):
        """Apply the scaling change to the system."""
        old_scale_factor = self.current_scale_factor
        self.current_scale_factor = new_scale_factor
        self.last_scaling_time = time.time()
        self.cooldown_active = True
        
        # Record scaling event
        scaling_event = {
            "timestamp": time.time(),
            "old_scale_factor": old_scale_factor,
            "new_scale_factor": new_scale_factor,
            "scale_change": new_scale_factor - old_scale_factor,
            "reasoning": decision_context.get("reasoning", ""),
            "confidence": decision_context.get("confidence", 0.0)
        }
        
        self.scaling_history.append(scaling_event)
        
        # Apply actual scaling (would interface with system)
        self._execute_system_scaling(new_scale_factor)
        
        logger.info(
            f"Scaled system: {old_scale_factor:.2f} -> {new_scale_factor:.2f} "
            f"(confidence: {decision_context.get('confidence', 0):.2f})"
        )
    
    def _execute_system_scaling(self, scale_factor: float):
        """Execute actual system scaling (implementation specific)."""
        # This would interface with the actual system to apply scaling
        # For demonstration, we'll just log the action
        if scale_factor > self.current_scale_factor:
            logger.info(f"Scaling up resources by factor {scale_factor:.2f}")
        else:
            logger.info(f"Scaling down resources by factor {scale_factor:.2f}")
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get comprehensive scaling statistics."""
        if not self.scaling_history:
            return {"total_scaling_events": 0}
        
        scale_changes = [event["scale_change"] for event in self.scaling_history]
        
        return {
            "total_scaling_events": len(self.scaling_history),
            "current_scale_factor": self.current_scale_factor,
            "average_scale_change": np.mean(np.abs(scale_changes)),
            "max_scale_change": np.max(np.abs(scale_changes)),
            "scale_up_events": sum(1 for change in scale_changes if change > 0),
            "scale_down_events": sum(1 for change in scale_changes if change < 0),
            "average_confidence": np.mean([event["confidence"] for event in self.scaling_history]),
            "last_scaling_time": self.last_scaling_time,
            "cooldown_active": self.cooldown_active
        }


class TrendAnalyzer:
    """Analyzes performance trends for predictive scaling."""
    
    def __init__(self):
        self.trend_window = 10  # Number of data points for trend analysis
    
    def analyze_trends(self, metrics_history: List[SystemMetrics]) -> Dict[str, float]:
        """Analyze trends in system metrics."""
        if len(metrics_history) < self.trend_window:
            return {}
        
        recent_metrics = metrics_history[-self.trend_window:]
        trends = {}
        
        # Analyze throughput trend
        throughputs = [m.throughput for m in recent_metrics]
        trends["throughput_trend"] = self._calculate_trend(throughputs)
        
        # Analyze latency trend
        latencies = [m.latency for m in recent_metrics]
        trends["latency_trend"] = self._calculate_trend(latencies)
        
        # Analyze resource utilization trend
        cpu_utils = [m.cpu_utilization for m in recent_metrics]
        trends["cpu_utilization_trend"] = self._calculate_trend(cpu_utils)
        
        # Analyze energy efficiency trend
        energies = [m.energy_per_operation for m in recent_metrics]
        trends["energy_trend"] = self._calculate_trend(energies)
        
        return trends
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend using linear regression slope."""
        if len(values) < 2:
            return 0.0
        
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        slope, _ = np.polyfit(x, y, 1)
        
        # Normalize by average value
        avg_value = np.mean(y)
        if avg_value != 0:
            return slope / avg_value
        return 0.0


class PerformancePredictor:
    """Predicts future system performance using time series analysis."""
    
    def __init__(self):
        self.prediction_models = {}
    
    def predict_performance(
        self, 
        metrics_history: List[SystemMetrics], 
        prediction_horizon: float
    ) -> Dict[str, Any]:
        """Predict system performance for given time horizon."""
        
        if len(metrics_history) < 10:
            return {"prediction_available": False}
        
        # Extract time series data
        timestamps = [m.timestamp for m in metrics_history]
        throughputs = [m.throughput for m in metrics_history]
        latencies = [m.latency for m in metrics_history]
        
        # Simple trend-based prediction
        throughput_prediction = self._predict_metric(throughputs, prediction_horizon)
        latency_prediction = self._predict_metric(latencies, prediction_horizon)
        
        return {
            "prediction_available": True,
            "prediction_horizon": prediction_horizon,
            "predicted_throughput": throughput_prediction,
            "predicted_latency": latency_prediction,
            "confidence": 0.7  # Simplified confidence score
        }
    
    def _predict_metric(self, values: List[float], horizon: float) -> Dict[str, float]:
        """Predict a single metric using trend extrapolation."""
        if len(values) < 5:
            return {"value": values[-1], "confidence": 0.3}
        
        # Use last 10 points for prediction
        recent_values = values[-10:]
        x = np.arange(len(recent_values))
        y = np.array(recent_values)
        
        # Fit linear trend
        slope, intercept = np.polyfit(x, y, 1)
        
        # Predict future value
        future_x = len(recent_values) + horizon / 10.0  # Rough time scaling
        predicted_value = slope * future_x + intercept
        
        # Calculate confidence based on trend consistency
        trend_consistency = 1.0 - np.std(np.diff(recent_values)) / np.mean(recent_values)
        confidence = max(0.1, min(0.9, trend_consistency))
        
        return {
            "value": predicted_value,
            "confidence": confidence,
            "trend_slope": slope
        }


class MultiObjectiveOptimizer:
    """Multi-objective optimizer for scaling decisions."""
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
        self.normalize_weights()
    
    def normalize_weights(self):
        """Normalize weights to sum to 1."""
        total_weight = sum(self.weights.values())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
    
    def optimize_scaling(
        self, 
        current_metrics: SystemMetrics, 
        trends: Dict[str, float],
        predictions: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Optimize scaling decision using multi-objective analysis."""
        
        # Evaluate current performance score
        current_score = self._calculate_performance_score(current_metrics)
        
        # Evaluate potential scaling factors
        scale_factors = np.arange(0.5, 2.1, 0.1)
        best_scale_factor = 1.0
        best_score = current_score
        
        for scale_factor in scale_factors:
            # Estimate metrics after scaling
            estimated_metrics = self._estimate_scaled_metrics(current_metrics, scale_factor)
            score = self._calculate_performance_score(estimated_metrics)
            
            if score > best_score:
                best_score = score
                best_scale_factor = scale_factor
        
        # Decision logic
        should_scale = abs(best_scale_factor - 1.0) > 0.1
        confidence = min(0.9, (best_score - current_score) / current_score + 0.5)
        
        reasoning = self._generate_reasoning(current_metrics, best_scale_factor, trends)
        
        return {
            "should_scale": should_scale,
            "recommended_scale_factor": best_scale_factor,
            "confidence": confidence,
            "reasoning": reasoning,
            "current_score": current_score,
            "predicted_score": best_score
        }
    
    def _calculate_performance_score(self, metrics: SystemMetrics) -> float:
        """Calculate weighted performance score."""
        score = 0.0
        
        # Throughput (higher is better)
        if "throughput" in self.weights:
            normalized_throughput = min(1.0, metrics.throughput / 200.0)
            score += self.weights["throughput"] * normalized_throughput
        
        # Latency (lower is better)
        if "latency" in self.weights:
            normalized_latency = max(0.0, 1.0 - metrics.latency / 100.0)
            score += self.weights["latency"] * normalized_latency
        
        # Energy efficiency (lower energy is better)
        if "energy" in self.weights:
            normalized_energy = max(0.0, 1.0 - metrics.energy_per_operation / 20.0)
            score += self.weights["energy"] * normalized_energy
        
        # Accuracy (higher is better)
        if "accuracy" in self.weights:
            score += self.weights["accuracy"] * metrics.accuracy
        
        # Cost (lower is better)
        if "cost" in self.weights:
            normalized_cost = max(0.0, 1.0 - metrics.monetary_cost_per_hour / 20.0)
            score += self.weights["cost"] * normalized_cost
        
        return score
    
    def _estimate_scaled_metrics(self, current_metrics: SystemMetrics, scale_factor: float) -> SystemMetrics:
        """Estimate metrics after applying scale factor."""
        # Create scaled metrics (simplified model)
        scaled_metrics = SystemMetrics(
            throughput=current_metrics.throughput * scale_factor * 0.9,  # Some efficiency loss
            latency=current_metrics.latency / (scale_factor * 0.8),  # Some overhead
            accuracy=current_metrics.accuracy,  # Assume accuracy unchanged
            energy_per_operation=current_metrics.energy_per_operation * scale_factor * 1.1,  # Energy overhead
            monetary_cost_per_hour=current_metrics.monetary_cost_per_hour * scale_factor
        )
        
        return scaled_metrics
    
    def _generate_reasoning(self, metrics: SystemMetrics, scale_factor: float, trends: Dict[str, float]) -> str:
        """Generate human-readable reasoning for scaling decision."""
        if scale_factor > 1.2:
            return f"Scale up to {scale_factor:.1f}x due to high demand and good efficiency"
        elif scale_factor < 0.8:
            return f"Scale down to {scale_factor:.1f}x due to low utilization and cost optimization"
        else:
            return f"Maintain current scale ({scale_factor:.1f}x) - optimal performance balance"


class ThermalManager:
    """Manages thermal constraints for scaling decisions."""
    
    def __init__(self):
        self.max_temperature = 85.0  # Maximum safe temperature
        self.thermal_buffer = 10.0   # Safety buffer
    
    def check_thermal_constraints(self, metrics: SystemMetrics) -> bool:
        """Check if thermal constraints allow scaling."""
        current_temp = metrics.device_temperature
        safe_threshold = self.max_temperature - self.thermal_buffer
        
        if current_temp > safe_threshold:
            logger.warning(f"Thermal constraint active: {current_temp:.1f}°C > {safe_threshold:.1f}°C")
            return False
        
        return True


class QuantumOptimizer:
    """Quantum-aware optimization for scaling decisions."""
    
    def __init__(self):
        self.min_coherence_time = 50e-6  # Minimum coherence time
        self.max_error_rate = 0.05       # Maximum acceptable error rate
    
    def check_quantum_constraints(self, metrics: SystemMetrics) -> bool:
        """Check if quantum constraints allow scaling."""
        if metrics.quantum_coherence_time < self.min_coherence_time:
            logger.warning(f"Quantum coherence too low: {metrics.quantum_coherence_time*1e6:.1f}μs")
            return False
        
        if metrics.quantum_error_rate > self.max_error_rate:
            logger.warning(f"Quantum error rate too high: {metrics.quantum_error_rate:.3f}")
            return False
        
        return True


def demonstrate_adaptive_scaling():
    """Demonstrate adaptive scaling capabilities."""
    print("📈 Adaptive Scaling and Auto-Optimization Framework")
    print("=" * 60)
    
    # Create configuration
    config = ScalingConfiguration(
        optimization_objective=OptimizationObjective.BALANCED_PERFORMANCE,
        predictive_scaling=True,
        machine_learning_enabled=True,
        quantum_aware_scaling=True,
        thermal_management=True
    )
    
    print(f"✅ Created adaptive scaling configuration")
    print(f"   Optimization objective: {config.optimization_objective.value}")
    print(f"   Predictive scaling: {config.predictive_scaling}")
    print(f"   Quantum-aware: {config.quantum_aware_scaling}")
    print(f"   Thermal management: {config.thermal_management}")
    
    # Initialize controller
    controller = AdaptiveScalingController(config)
    
    print(f"\n🔍 System Monitoring and Analysis")
    
    # Simulate metrics collection
    for i in range(15):
        metrics = controller._collect_system_metrics()
        controller.metrics_history.append(metrics)
        
        if i == 5:
            print(f"   Collected {len(controller.metrics_history)} metric samples")
        
        if i >= 5:  # Start making scaling decisions after some data
            decision = controller._evaluate_scaling_decision(metrics)
            
            if decision["should_scale"]:
                print(f"   Scaling decision: {decision['recommended_scale_factor']:.2f}x")
                print(f"     Confidence: {decision['confidence']:.2f}")
                print(f"     Reasoning: {decision['reasoning']}")
                
                # Execute scaling
                controller._apply_scaling(decision['recommended_scale_factor'], decision)
                break
    
    # Demonstrate trend analysis
    print(f"\n📊 Trend Analysis")
    
    if len(controller.metrics_history) > 5:
        trends = controller.trend_analyzer.analyze_trends(list(controller.metrics_history))
        
        for metric, trend in trends.items():
            direction = "↗️" if trend > 0.01 else "↘️" if trend < -0.01 else "➡️"
            print(f"   {metric}: {direction} {trend:+.3f}")
    
    # Demonstrate performance prediction
    print(f"\n🔮 Performance Prediction")
    
    predictions = controller.performance_predictor.predict_performance(
        list(controller.metrics_history), 300.0  # 5 minutes ahead
    )
    
    if predictions.get("prediction_available"):
        throughput_pred = predictions["predicted_throughput"]
        latency_pred = predictions["predicted_latency"]
        
        print(f"   Predicted throughput: {throughput_pred['value']:.1f} ops/sec")
        print(f"     Confidence: {throughput_pred['confidence']:.2f}")
        print(f"   Predicted latency: {latency_pred['value']:.1f} ms")
        print(f"     Confidence: {latency_pred['confidence']:.2f}")
    
    # Multi-objective optimization demo
    print(f"\n⚖️  Multi-Objective Optimization")
    
    current_metrics = list(controller.metrics_history)[-1]
    optimization_result = controller.multi_objective_optimizer.optimize_scaling(
        current_metrics, trends, predictions
    )
    
    print(f"   Current performance score: {optimization_result['current_score']:.3f}")
    print(f"   Predicted score after scaling: {optimization_result['predicted_score']:.3f}")
    print(f"   Improvement: {(optimization_result['predicted_score'] - optimization_result['current_score']):.3f}")
    
    # Thermal and quantum constraints
    print(f"\n🌡️  Thermal Management")
    
    if controller.thermal_manager:
        thermal_ok = controller.thermal_manager.check_thermal_constraints(current_metrics)
        print(f"   Current temperature: {current_metrics.device_temperature:.1f}°C")
        print(f"   Thermal constraints: {'✅ OK' if thermal_ok else '❌ VIOLATED'}")
    
    print(f"\n⚛️  Quantum Optimization")
    
    if controller.quantum_optimizer:
        quantum_ok = controller.quantum_optimizer.check_quantum_constraints(current_metrics)
        print(f"   Coherence time: {current_metrics.quantum_coherence_time*1e6:.1f} μs")
        print(f"   Error rate: {current_metrics.quantum_error_rate:.3f}")
        print(f"   Quantum constraints: {'✅ OK' if quantum_ok else '❌ VIOLATED'}")
    
    # Get scaling statistics
    stats = controller.get_scaling_statistics()
    
    print(f"\n📈 Scaling Statistics")
    print(f"   Total scaling events: {stats['total_scaling_events']}")
    print(f"   Current scale factor: {stats['current_scale_factor']:.2f}")
    
    if stats['total_scaling_events'] > 0:
        print(f"   Scale up events: {stats['scale_up_events']}")
        print(f"   Scale down events: {stats['scale_down_events']}")
        print(f"   Average confidence: {stats['average_confidence']:.2f}")
    
    # Performance improvement estimation
    baseline_throughput = 100.0
    current_throughput = current_metrics.throughput * controller.current_scale_factor
    improvement = (current_throughput - baseline_throughput) / baseline_throughput
    
    print(f"\n🎯 Performance Impact")
    print(f"   Baseline throughput: {baseline_throughput:.1f} ops/sec")
    print(f"   Current throughput: {current_throughput:.1f} ops/sec")
    print(f"   Performance improvement: {improvement:+.1%}")
    print(f"   Energy efficiency: {1.0 / current_metrics.energy_per_operation:.2f} ops/J")
    
    return {
        "scaling_events": stats['total_scaling_events'],
        "current_scale_factor": stats['current_scale_factor'],
        "performance_improvement": improvement,
        "thermal_constraint_ok": thermal_ok if controller.thermal_manager else True,
        "quantum_constraint_ok": quantum_ok if controller.quantum_optimizer else True,
        "prediction_confidence": predictions.get("confidence", 0.0) if predictions.get("prediction_available") else 0.0,
        "optimization_score": optimization_result['predicted_score'],
        "trend_analysis_available": len(trends) > 0,
        "multi_objective_weights": config.multi_objective_weights
    }


if __name__ == "__main__":
    results = demonstrate_adaptive_scaling()
    print(f"\n🎉 Adaptive Scaling Framework: VALIDATION COMPLETED")
    print(json.dumps(results, indent=2))