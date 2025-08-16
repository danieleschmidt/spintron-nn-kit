"""
Adaptive performance optimization for spintronic neural networks.
Automatically adjusts parameters based on performance metrics and system load.
"""

import time
import math
import json
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from collections import deque
from enum import Enum


class OptimizationStrategy(Enum):
    """Performance optimization strategies."""
    AGGRESSIVE = "aggressive"
    CONSERVATIVE = "conservative" 
    BALANCED = "balanced"
    ADAPTIVE = "adaptive"


@dataclass
class PerformanceMetrics:
    """System performance metrics."""
    latency_ms: float
    throughput_ops_per_sec: float
    energy_consumption_nj: float
    memory_usage_mb: float
    cache_hit_rate: float
    cpu_utilization: float
    timestamp: float


@dataclass
class OptimizationAction:
    """Action taken by the optimizer."""
    action_type: str
    parameters: Dict[str, Any]
    expected_improvement: float
    timestamp: float
    success: Optional[bool] = None


class AdaptivePerformanceOptimizer:
    """Adaptive performance optimizer with machine learning capabilities."""
    
    def __init__(self, strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE):
        self.strategy = strategy
        self.metrics_history = deque(maxlen=1000)
        self.optimization_history: List[OptimizationAction] = []
        
        # Performance baselines
        self.baseline_metrics: Optional[PerformanceMetrics] = None
        self.target_metrics = {
            'max_latency_ms': 100.0,
            'min_throughput_ops_per_sec': 1000.0,
            'max_energy_per_op_pj': 50.0,
            'max_memory_usage_mb': 100.0
        }
        
        # Optimization parameters
        self.optimization_interval = 5.0  # seconds
        self.last_optimization_time = 0.0
        self.learning_rate = 0.1
        self.convergence_threshold = 0.05
        
        # Performance improvement tracking
        self.improvement_trends = {
            'latency': deque(maxlen=10),
            'throughput': deque(maxlen=10),
            'energy': deque(maxlen=10),
            'memory': deque(maxlen=10)
        }
        
    def collect_metrics(self, 
                       latency_ms: float,
                       throughput_ops_per_sec: float,
                       energy_consumption_nj: float,
                       memory_usage_mb: float,
                       cache_hit_rate: float = 0.0,
                       cpu_utilization: float = 0.0) -> PerformanceMetrics:
        """Collect current performance metrics."""
        metrics = PerformanceMetrics(
            latency_ms=latency_ms,
            throughput_ops_per_sec=throughput_ops_per_sec,
            energy_consumption_nj=energy_consumption_nj,
            memory_usage_mb=memory_usage_mb,
            cache_hit_rate=cache_hit_rate,
            cpu_utilization=cpu_utilization,
            timestamp=time.time()
        )
        
        self.metrics_history.append(metrics)
        
        # Set baseline if not set
        if self.baseline_metrics is None:
            self.baseline_metrics = metrics
            
        return metrics
        
    def should_optimize(self) -> bool:
        """Determine if optimization should be triggered."""
        current_time = time.time()
        
        # Time-based trigger
        if current_time - self.last_optimization_time < self.optimization_interval:
            return False
            
        # Performance degradation trigger
        if len(self.metrics_history) < 2:
            return False
            
        recent_metrics = list(self.metrics_history)[-5:]
        if len(recent_metrics) >= 2:
            latest = recent_metrics[-1]
            
            # Check if any metric exceeds targets
            if (latest.latency_ms > self.target_metrics['max_latency_ms'] or
                latest.throughput_ops_per_sec < self.target_metrics['min_throughput_ops_per_sec'] or
                latest.energy_consumption_nj / latest.throughput_ops_per_sec > self.target_metrics['max_energy_per_op_pj'] or
                latest.memory_usage_mb > self.target_metrics['max_memory_usage_mb']):
                return True
                
        return False
        
    def analyze_performance_trends(self) -> Dict[str, float]:
        """Analyze performance trends over recent metrics."""
        if len(self.metrics_history) < 5:
            return {}
            
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate trends (positive = improving, negative = degrading)
        trends = {}
        
        # Latency trend (decreasing is better)
        latencies = [m.latency_ms for m in recent_metrics]
        if len(latencies) >= 3:
            trend = (latencies[0] - latencies[-1]) / latencies[0]
            trends['latency_improvement'] = trend
            
        # Throughput trend (increasing is better)
        throughputs = [m.throughput_ops_per_sec for m in recent_metrics]
        if len(throughputs) >= 3:
            trend = (throughputs[-1] - throughputs[0]) / throughputs[0]
            trends['throughput_improvement'] = trend
            
        # Energy efficiency trend (decreasing energy per op is better)
        if len(recent_metrics) >= 3:
            energy_per_ops = [m.energy_consumption_nj / m.throughput_ops_per_sec 
                             for m in recent_metrics if m.throughput_ops_per_sec > 0]
            if energy_per_ops:
                trend = (energy_per_ops[0] - energy_per_ops[-1]) / energy_per_ops[0]
                trends['energy_efficiency_improvement'] = trend
                
        # Memory usage trend (stable or decreasing is better)
        memory_usages = [m.memory_usage_mb for m in recent_metrics]
        if len(memory_usages) >= 3:
            trend = (memory_usages[0] - memory_usages[-1]) / memory_usages[0]
            trends['memory_improvement'] = trend
            
        return trends
        
    def select_optimization_actions(self, trends: Dict[str, float]) -> List[OptimizationAction]:
        """Select optimization actions based on performance trends."""
        actions = []
        current_time = time.time()
        
        # Latency optimization
        if trends.get('latency_improvement', 0) < -0.1:  # Latency degrading by >10%
            if self.strategy in [OptimizationStrategy.AGGRESSIVE, OptimizationStrategy.ADAPTIVE]:
                actions.append(OptimizationAction(
                    action_type="reduce_crossbar_size",
                    parameters={"size_reduction_factor": 0.8},
                    expected_improvement=0.2,
                    timestamp=current_time
                ))
                
            actions.append(OptimizationAction(
                action_type="increase_cache_size",
                parameters={"cache_multiplier": 1.5},
                expected_improvement=0.15,
                timestamp=current_time
            ))
            
        # Throughput optimization
        if trends.get('throughput_improvement', 0) < -0.05:  # Throughput degrading by >5%
            actions.append(OptimizationAction(
                action_type="parallel_processing",
                parameters={"parallelism_factor": 2},
                expected_improvement=0.4,
                timestamp=current_time
            ))
            
            if self.strategy == OptimizationStrategy.AGGRESSIVE:
                actions.append(OptimizationAction(
                    action_type="frequency_scaling",
                    parameters={"frequency_multiplier": 1.2},
                    expected_improvement=0.2,
                    timestamp=current_time
                ))
                
        # Energy optimization
        if trends.get('energy_efficiency_improvement', 0) < -0.1:
            actions.append(OptimizationAction(
                action_type="voltage_scaling",
                parameters={"voltage_reduction": 0.05},
                expected_improvement=0.1,
                timestamp=current_time
            ))
            
            actions.append(OptimizationAction(
                action_type="dynamic_power_gating",
                parameters={"idle_threshold_ms": 1.0},
                expected_improvement=0.15,
                timestamp=current_time
            ))
            
        # Memory optimization
        if trends.get('memory_improvement', 0) < -0.1:
            actions.append(OptimizationAction(
                action_type="memory_compression",
                parameters={"compression_ratio": 2.0},
                expected_improvement=0.3,
                timestamp=current_time
            ))
            
            actions.append(OptimizationAction(
                action_type="garbage_collection",
                parameters={"gc_threshold": 0.8},
                expected_improvement=0.2,
                timestamp=current_time
            ))
            
        return actions
        
    def apply_optimization_action(self, action: OptimizationAction) -> bool:
        """Apply optimization action and return success status."""
        try:
            if action.action_type == "reduce_crossbar_size":
                # Simulate crossbar size reduction
                print(f"Reducing crossbar size by factor {action.parameters['size_reduction_factor']}")
                
            elif action.action_type == "increase_cache_size":
                # Simulate cache size increase
                print(f"Increasing cache size by factor {action.parameters['cache_multiplier']}")
                
            elif action.action_type == "parallel_processing":
                # Simulate parallel processing enablement
                print(f"Enabling parallel processing with factor {action.parameters['parallelism_factor']}")
                
            elif action.action_type == "frequency_scaling":
                # Simulate frequency scaling
                print(f"Scaling frequency by factor {action.parameters['frequency_multiplier']}")
                
            elif action.action_type == "voltage_scaling":
                # Simulate voltage scaling
                print(f"Reducing voltage by {action.parameters['voltage_reduction']}V")
                
            elif action.action_type == "dynamic_power_gating":
                # Simulate power gating
                print(f"Enabling power gating with threshold {action.parameters['idle_threshold_ms']}ms")
                
            elif action.action_type == "memory_compression":
                # Simulate memory compression
                print(f"Enabling memory compression with ratio {action.parameters['compression_ratio']}")
                
            elif action.action_type == "garbage_collection":
                # Simulate garbage collection
                print(f"Triggering garbage collection at {action.parameters['gc_threshold']} threshold")
                
            else:
                print(f"Unknown optimization action: {action.action_type}")
                return False
                
            action.success = True
            return True
            
        except Exception as e:
            print(f"Failed to apply optimization action {action.action_type}: {e}")
            action.success = False
            return False
            
    def optimize(self) -> List[OptimizationAction]:
        """Perform optimization based on current performance metrics."""
        if not self.should_optimize():
            return []
            
        # Analyze trends
        trends = self.analyze_performance_trends()
        
        # Select actions
        actions = self.select_optimization_actions(trends)
        
        # Apply actions
        applied_actions = []
        for action in actions:
            if self.apply_optimization_action(action):
                applied_actions.append(action)
                self.optimization_history.append(action)
                
        self.last_optimization_time = time.time()
        
        # Update improvement trends
        self._update_improvement_trends(trends)
        
        return applied_actions
        
    def _update_improvement_trends(self, trends: Dict[str, float]) -> None:
        """Update improvement trend tracking."""
        for metric, trend in trends.items():
            if 'latency' in metric:
                self.improvement_trends['latency'].append(trend)
            elif 'throughput' in metric:
                self.improvement_trends['throughput'].append(trend)
            elif 'energy' in metric:
                self.improvement_trends['energy'].append(trend)
            elif 'memory' in metric:
                self.improvement_trends['memory'].append(trend)
                
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary."""
        if not self.metrics_history:
            return {'no_data': True}
            
        recent_metrics = list(self.metrics_history)[-10:]
        
        # Calculate improvements since baseline
        improvements = {}
        if self.baseline_metrics and recent_metrics:
            latest = recent_metrics[-1]
            improvements = {
                'latency_improvement': (self.baseline_metrics.latency_ms - latest.latency_ms) / self.baseline_metrics.latency_ms,
                'throughput_improvement': (latest.throughput_ops_per_sec - self.baseline_metrics.throughput_ops_per_sec) / self.baseline_metrics.throughput_ops_per_sec,
                'energy_efficiency_improvement': ((self.baseline_metrics.energy_consumption_nj / self.baseline_metrics.throughput_ops_per_sec) - 
                                                 (latest.energy_consumption_nj / latest.throughput_ops_per_sec)) / (self.baseline_metrics.energy_consumption_nj / self.baseline_metrics.throughput_ops_per_sec),
                'memory_improvement': (self.baseline_metrics.memory_usage_mb - latest.memory_usage_mb) / self.baseline_metrics.memory_usage_mb
            }
            
        # Calculate action success rate
        successful_actions = sum(1 for action in self.optimization_history if action.success)
        action_success_rate = successful_actions / len(self.optimization_history) if self.optimization_history else 0
        
        # Calculate average improvements
        avg_improvements = {}
        for metric, trends in self.improvement_trends.items():
            if trends:
                avg_improvements[f'avg_{metric}_trend'] = sum(trends) / len(trends)
                
        return {
            'total_optimizations': len(self.optimization_history),
            'successful_optimizations': successful_actions,
            'optimization_success_rate': action_success_rate,
            'baseline_improvements': improvements,
            'average_improvement_trends': avg_improvements,
            'current_strategy': self.strategy.value,
            'metrics_collected': len(self.metrics_history),
            'last_optimization_time': self.last_optimization_time
        }
        
    def export_optimization_data(self) -> Dict[str, Any]:
        """Export optimization data for analysis."""
        return {
            'strategy': self.strategy.value,
            'target_metrics': self.target_metrics,
            'baseline_metrics': asdict(self.baseline_metrics) if self.baseline_metrics else None,
            'metrics_history': [asdict(m) for m in self.metrics_history],
            'optimization_history': [asdict(a) for a in self.optimization_history],
            'improvement_trends': {k: list(v) for k, v in self.improvement_trends.items()},
            'optimization_summary': self.get_optimization_summary()
        }


class AutoScalingManager:
    """Automatic scaling manager for spintronic neural networks."""
    
    def __init__(self, optimizer: AdaptivePerformanceOptimizer):
        self.optimizer = optimizer
        self.scaling_thresholds = {
            'scale_up_cpu': 0.8,      # CPU utilization
            'scale_down_cpu': 0.3,
            'scale_up_memory': 0.8,   # Memory utilization
            'scale_down_memory': 0.4,
            'scale_up_latency': 80.0, # Latency in ms
            'scale_down_latency': 20.0
        }
        self.current_scale = 1.0
        self.min_scale = 0.5
        self.max_scale = 4.0
        
    def should_scale_up(self, metrics: PerformanceMetrics) -> bool:
        """Determine if system should scale up."""
        return (metrics.cpu_utilization > self.scaling_thresholds['scale_up_cpu'] or
                metrics.memory_usage_mb / 100.0 > self.scaling_thresholds['scale_up_memory'] or
                metrics.latency_ms > self.scaling_thresholds['scale_up_latency'])
                
    def should_scale_down(self, metrics: PerformanceMetrics) -> bool:
        """Determine if system should scale down."""
        return (metrics.cpu_utilization < self.scaling_thresholds['scale_down_cpu'] and
                metrics.memory_usage_mb / 100.0 < self.scaling_thresholds['scale_down_memory'] and
                metrics.latency_ms < self.scaling_thresholds['scale_down_latency'])
                
    def scale_system(self, target_scale: float) -> bool:
        """Scale system to target scale factor."""
        if target_scale < self.min_scale or target_scale > self.max_scale:
            return False
            
        scale_action = OptimizationAction(
            action_type="system_scaling",
            parameters={"scale_factor": target_scale, "previous_scale": self.current_scale},
            expected_improvement=(target_scale - self.current_scale) * 0.5,
            timestamp=time.time()
        )
        
        if self.optimizer.apply_optimization_action(scale_action):
            self.current_scale = target_scale
            print(f"System scaled to {target_scale:.2f}x")
            return True
            
        return False
        
    def auto_scale(self, metrics: PerformanceMetrics) -> Optional[float]:
        """Perform automatic scaling based on metrics."""
        if self.should_scale_up(metrics):
            new_scale = min(self.current_scale * 1.5, self.max_scale)
            if self.scale_system(new_scale):
                return new_scale
                
        elif self.should_scale_down(metrics):
            new_scale = max(self.current_scale * 0.75, self.min_scale)
            if self.scale_system(new_scale):
                return new_scale
                
        return None