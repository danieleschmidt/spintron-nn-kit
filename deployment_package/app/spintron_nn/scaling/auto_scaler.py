"""
Auto-scaling system for SpinTron-NN-Kit inference pipelines.

This module provides intelligent auto-scaling based on:
- Request load patterns
- Resource utilization metrics
- Energy efficiency targets
- Response time requirements
"""

import time
import json
import threading
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import statistics


class ScalingMetric(Enum):
    """Metrics used for scaling decisions."""
    CPU_UTILIZATION = "cpu_utilization"
    MEMORY_UTILIZATION = "memory_utilization"
    QUEUE_LENGTH = "queue_length"
    RESPONSE_TIME = "response_time"
    THROUGHPUT = "throughput"
    ENERGY_EFFICIENCY = "energy_efficiency"
    ERROR_RATE = "error_rate"


class ScalingDirection(Enum):
    """Scaling direction decisions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"


@dataclass
class ScalingConfig:
    """Configuration for auto-scaling behavior."""
    
    # Resource limits
    min_instances: int = 1
    max_instances: int = 10
    target_cpu_utilization: float = 0.7  # 70%
    target_memory_utilization: float = 0.8  # 80%
    
    # Performance targets
    max_response_time_ms: float = 1000.0
    min_throughput_qps: float = 10.0
    target_energy_efficiency: float = 15.0  # pJ/MAC
    
    # Scaling behavior
    scale_up_threshold: float = 0.8  # Scale up at 80% of target
    scale_down_threshold: float = 0.5  # Scale down at 50% of target
    cooldown_period_s: float = 60.0  # Wait 60s between scaling decisions
    
    # Monitoring
    metric_window_size: int = 20  # Number of samples for decisions
    polling_interval_s: float = 5.0  # Monitor every 5 seconds


@dataclass
class InstanceMetrics:
    """Metrics for a single instance."""
    instance_id: str
    cpu_utilization: float
    memory_utilization: float
    active_requests: int
    response_times: List[float]
    energy_consumption: float
    error_count: int
    timestamp: float


@dataclass
class ScalingDecision:
    """Auto-scaling decision record."""
    timestamp: float
    current_instances: int
    target_instances: int
    direction: ScalingDirection
    reason: str
    metrics_snapshot: Dict[str, float]
    confidence_score: float


class ResourceMonitor:
    """Monitor system resources and performance metrics."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.metrics_history = deque(maxlen=config.metric_window_size)
        self.running = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.running:
            metrics = self._collect_system_metrics()
            self.metrics_history.append(metrics)
            time.sleep(self.config.polling_interval_s)
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        # Simulated metrics - in production would use psutil, nvidia-ml-py, etc.
        import random
        
        # Simulate realistic metrics with some variation
        base_cpu = 0.6 + random.gauss(0, 0.1)
        base_memory = 0.5 + random.gauss(0, 0.05)
        
        return {
            "cpu_utilization": max(0.0, min(1.0, base_cpu)),
            "memory_utilization": max(0.0, min(1.0, base_memory)),
            "queue_length": max(0, int(random.gauss(5, 2))),
            "response_time_ms": max(10, random.gauss(200, 50)),
            "throughput_qps": max(1, random.gauss(20, 5)),
            "energy_per_inference_pj": max(5, random.gauss(15, 3)),
            "error_rate": max(0, random.gauss(0.01, 0.005)),
            "timestamp": time.time()
        }
    
    def get_current_metrics(self) -> Optional[Dict[str, float]]:
        """Get current metrics snapshot."""
        return list(self.metrics_history)[-1] if self.metrics_history else None
    
    def get_metrics_statistics(self) -> Dict[str, Dict[str, float]]:
        """Get statistical summary of metrics."""
        if not self.metrics_history:
            return {}
        
        stats = {}
        metrics = list(self.metrics_history)
        
        for key in metrics[0].keys():
            if key != "timestamp":
                values = [m[key] for m in metrics]
                stats[key] = {
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0.0,
                    "min": min(values),
                    "max": max(values),
                    "latest": values[-1]
                }
        
        return stats


class LoadBalancer:
    """Intelligent load balancer with energy optimization."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.instances = {}
        self.request_queue = deque()
        self.routing_strategy = "least_loaded"  # or "round_robin", "energy_optimal"
        
    def register_instance(self, instance_id: str, endpoint: str, capabilities: Dict[str, Any]):
        """Register a new instance."""
        self.instances[instance_id] = {
            "endpoint": endpoint,
            "capabilities": capabilities,
            "active_requests": 0,
            "total_requests": 0,
            "response_times": deque(maxlen=100),
            "energy_consumption": 0.0,
            "status": "healthy",
            "last_heartbeat": time.time()
        }
    
    def unregister_instance(self, instance_id: str):
        """Unregister an instance."""
        self.instances.pop(instance_id, None)
    
    def route_request(self, request: Dict[str, Any]) -> Optional[str]:
        """Route request to optimal instance."""
        if not self.instances:
            return None
        
        healthy_instances = {
            iid: info for iid, info in self.instances.items() 
            if info["status"] == "healthy"
        }
        
        if not healthy_instances:
            return None
        
        if self.routing_strategy == "least_loaded":
            return min(healthy_instances.keys(), 
                      key=lambda x: healthy_instances[x]["active_requests"])
        
        elif self.routing_strategy == "energy_optimal":
            return min(healthy_instances.keys(),
                      key=lambda x: healthy_instances[x]["energy_consumption"])
        
        elif self.routing_strategy == "round_robin":
            # Simple round-robin (stateless implementation)
            return list(healthy_instances.keys())[
                hash(str(time.time())) % len(healthy_instances)
            ]
        
        return list(healthy_instances.keys())[0]
    
    def update_instance_metrics(self, instance_id: str, metrics: InstanceMetrics):
        """Update metrics for an instance."""
        if instance_id in self.instances:
            instance = self.instances[instance_id]
            instance["active_requests"] = metrics.active_requests
            instance["response_times"].extend(metrics.response_times)
            instance["energy_consumption"] = metrics.energy_consumption
            instance["last_heartbeat"] = time.time()
            
            # Health check based on response times and errors
            if metrics.response_times:
                avg_response_time = statistics.mean(metrics.response_times)
                if avg_response_time > self.config.max_response_time_ms * 2:
                    instance["status"] = "unhealthy"
                else:
                    instance["status"] = "healthy"
    
    def get_load_distribution(self) -> Dict[str, Any]:
        """Get current load distribution across instances."""
        total_requests = sum(inst["active_requests"] for inst in self.instances.values())
        
        distribution = {}
        for iid, inst in self.instances.items():
            load_percentage = (inst["active_requests"] / max(1, total_requests)) * 100
            distribution[iid] = {
                "load_percentage": load_percentage,
                "active_requests": inst["active_requests"],
                "status": inst["status"],
                "avg_response_time": statistics.mean(inst["response_times"]) if inst["response_times"] else 0
            }
        
        return distribution


class AutoScaler:
    """Intelligent auto-scaler for SpinTron-NN-Kit inference."""
    
    def __init__(self, config: ScalingConfig, 
                 scale_up_callback: Callable[[int], bool] = None,
                 scale_down_callback: Callable[[int], bool] = None):
        """Initialize auto-scaler.
        
        Args:
            config: Scaling configuration
            scale_up_callback: Function to call when scaling up
            scale_down_callback: Function to call when scaling down
        """
        self.config = config
        self.scale_up_callback = scale_up_callback
        self.scale_down_callback = scale_down_callback
        
        self.monitor = ResourceMonitor(config)
        self.load_balancer = LoadBalancer(config)
        
        self.current_instances = config.min_instances
        self.last_scaling_decision = 0
        self.scaling_history = deque(maxlen=100)
        
        self.running = False
        self.scaling_thread = None
    
    def start(self):
        """Start the auto-scaler."""
        self.running = True
        self.monitor.start_monitoring()
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
    
    def stop(self):
        """Stop the auto-scaler."""
        self.running = False
        self.monitor.stop_monitoring()
        if self.scaling_thread:
            self.scaling_thread.join()
    
    def _scaling_loop(self):
        """Main scaling decision loop."""
        while self.running:
            try:
                decision = self._make_scaling_decision()
                if decision and decision.direction != ScalingDirection.MAINTAIN:
                    self._execute_scaling_decision(decision)
                    self.scaling_history.append(decision)
                    
            except Exception as e:
                print(f"Error in scaling loop: {e}")
            
            time.sleep(self.config.polling_interval_s)
    
    def _make_scaling_decision(self) -> Optional[ScalingDecision]:
        """Make intelligent scaling decision based on metrics."""
        # Check cooldown period
        if time.time() - self.last_scaling_decision < self.config.cooldown_period_s:
            return None
        
        # Get current metrics
        current_metrics = self.monitor.get_current_metrics()
        if not current_metrics:
            return None
        
        metrics_stats = self.monitor.get_metrics_statistics()
        if not metrics_stats:
            return None
        
        # Calculate scaling scores
        scale_up_score = self._calculate_scale_up_score(metrics_stats)
        scale_down_score = self._calculate_scale_down_score(metrics_stats)
        
        # Make decision
        direction = ScalingDirection.MAINTAIN
        target_instances = self.current_instances
        reason = "Metrics within target ranges"
        confidence = 0.5
        
        if scale_up_score > self.config.scale_up_threshold:
            if self.current_instances < self.config.max_instances:
                direction = ScalingDirection.SCALE_UP
                target_instances = min(self.config.max_instances, 
                                     self.current_instances + 1)
                reason = f"Scale up needed (score: {scale_up_score:.2f})"
                confidence = min(1.0, scale_up_score)
        
        elif scale_down_score > self.config.scale_down_threshold:
            if self.current_instances > self.config.min_instances:
                direction = ScalingDirection.SCALE_DOWN
                target_instances = max(self.config.min_instances,
                                     self.current_instances - 1)
                reason = f"Scale down possible (score: {scale_down_score:.2f})"
                confidence = min(1.0, scale_down_score)
        
        return ScalingDecision(
            timestamp=time.time(),
            current_instances=self.current_instances,
            target_instances=target_instances,
            direction=direction,
            reason=reason,
            metrics_snapshot=current_metrics,
            confidence_score=confidence
        )
    
    def _calculate_scale_up_score(self, metrics_stats: Dict[str, Dict[str, float]]) -> float:
        """Calculate score indicating need to scale up."""
        score = 0.0
        factors = []
        
        # CPU utilization factor
        if "cpu_utilization" in metrics_stats:
            cpu_mean = metrics_stats["cpu_utilization"]["mean"]
            if cpu_mean > self.config.target_cpu_utilization:
                cpu_factor = (cpu_mean - self.config.target_cpu_utilization) / (1.0 - self.config.target_cpu_utilization)
                factors.append(("cpu", min(1.0, cpu_factor)))
        
        # Memory utilization factor
        if "memory_utilization" in metrics_stats:
            mem_mean = metrics_stats["memory_utilization"]["mean"]
            if mem_mean > self.config.target_memory_utilization:
                mem_factor = (mem_mean - self.config.target_memory_utilization) / (1.0 - self.config.target_memory_utilization)
                factors.append(("memory", min(1.0, mem_factor)))
        
        # Response time factor
        if "response_time_ms" in metrics_stats:
            rt_mean = metrics_stats["response_time_ms"]["mean"]
            if rt_mean > self.config.max_response_time_ms:
                rt_factor = (rt_mean - self.config.max_response_time_ms) / self.config.max_response_time_ms
                factors.append(("response_time", min(1.0, rt_factor)))
        
        # Queue length factor
        if "queue_length" in metrics_stats:
            queue_mean = metrics_stats["queue_length"]["mean"]
            if queue_mean > 10:  # Threshold for queue buildup
                queue_factor = min(1.0, queue_mean / 20)
                factors.append(("queue", queue_factor))
        
        # Calculate weighted average
        if factors:
            score = sum(factor for _, factor in factors) / len(factors)
        
        return score
    
    def _calculate_scale_down_score(self, metrics_stats: Dict[str, Dict[str, float]]) -> float:
        """Calculate score indicating ability to scale down."""
        score = 0.0
        factors = []
        
        # Low CPU utilization
        if "cpu_utilization" in metrics_stats:
            cpu_mean = metrics_stats["cpu_utilization"]["mean"]
            if cpu_mean < self.config.target_cpu_utilization * 0.5:
                cpu_factor = (self.config.target_cpu_utilization * 0.5 - cpu_mean) / (self.config.target_cpu_utilization * 0.5)
                factors.append(("cpu", min(1.0, cpu_factor)))
        
        # Low memory utilization
        if "memory_utilization" in metrics_stats:
            mem_mean = metrics_stats["memory_utilization"]["mean"]
            if mem_mean < self.config.target_memory_utilization * 0.5:
                mem_factor = (self.config.target_memory_utilization * 0.5 - mem_mean) / (self.config.target_memory_utilization * 0.5)
                factors.append(("memory", min(1.0, mem_factor)))
        
        # Low queue length
        if "queue_length" in metrics_stats:
            queue_mean = metrics_stats["queue_length"]["mean"]
            if queue_mean < 2:  # Very low queue
                factors.append(("queue", 0.8))
        
        # Good response times
        if "response_time_ms" in metrics_stats:
            rt_mean = metrics_stats["response_time_ms"]["mean"]
            if rt_mean < self.config.max_response_time_ms * 0.5:
                rt_factor = (self.config.max_response_time_ms * 0.5 - rt_mean) / (self.config.max_response_time_ms * 0.5)
                factors.append(("response_time", min(1.0, rt_factor)))
        
        # Calculate weighted average, but be conservative
        if factors and len(factors) >= 2:  # Need multiple indicators
            score = sum(factor for _, factor in factors) / len(factors) * 0.8  # Conservative factor
        
        return score
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute scaling decision."""
        self.last_scaling_decision = time.time()
        
        try:
            if decision.direction == ScalingDirection.SCALE_UP:
                if self.scale_up_callback:
                    success = self.scale_up_callback(decision.target_instances)
                    if success:
                        self.current_instances = decision.target_instances
                        print(f"Scaled UP to {decision.target_instances} instances: {decision.reason}")
                else:
                    self.current_instances = decision.target_instances
                    print(f"Mock scale UP to {decision.target_instances}: {decision.reason}")
            
            elif decision.direction == ScalingDirection.SCALE_DOWN:
                if self.scale_down_callback:
                    success = self.scale_down_callback(decision.target_instances)
                    if success:
                        self.current_instances = decision.target_instances
                        print(f"Scaled DOWN to {decision.target_instances} instances: {decision.reason}")
                else:
                    self.current_instances = decision.target_instances
                    print(f"Mock scale DOWN to {decision.target_instances}: {decision.reason}")
                    
        except Exception as e:
            print(f"Failed to execute scaling decision: {e}")
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics and metrics."""
        recent_decisions = list(self.scaling_history)[-10:]  # Last 10 decisions
        
        scale_ups = sum(1 for d in recent_decisions if d.direction == ScalingDirection.SCALE_UP)
        scale_downs = sum(1 for d in recent_decisions if d.direction == ScalingDirection.SCALE_DOWN)
        
        return {
            "current_instances": self.current_instances,
            "target_range": [self.config.min_instances, self.config.max_instances],
            "recent_scaling_actions": {
                "scale_ups": scale_ups,
                "scale_downs": scale_downs,
                "total_decisions": len(recent_decisions)
            },
            "load_distribution": self.load_balancer.get_load_distribution(),
            "last_decision_time": self.last_scaling_decision,
            "time_since_last_scaling": time.time() - self.last_scaling_decision,
            "cooldown_remaining": max(0, self.config.cooldown_period_s - (time.time() - self.last_scaling_decision))
        }
    
    def force_scaling_decision(self, target_instances: int, reason: str = "Manual override") -> bool:
        """Force a scaling decision (for testing/manual intervention).
        
        Args:
            target_instances: Target number of instances
            reason: Reason for scaling
            
        Returns:
            Success status
        """
        if not (self.config.min_instances <= target_instances <= self.config.max_instances):
            return False
        
        if target_instances > self.current_instances:
            direction = ScalingDirection.SCALE_UP
        elif target_instances < self.current_instances:
            direction = ScalingDirection.SCALE_DOWN
        else:
            return True  # Already at target
        
        decision = ScalingDecision(
            timestamp=time.time(),
            current_instances=self.current_instances,
            target_instances=target_instances,
            direction=direction,
            reason=reason,
            metrics_snapshot={},
            confidence_score=1.0
        )
        
        self._execute_scaling_decision(decision)
        self.scaling_history.append(decision)
        return True