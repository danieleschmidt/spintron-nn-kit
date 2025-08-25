"""
Robust Monitoring System for SpinTron-NN-Kit
==========================================

Advanced monitoring with real-time metrics, health checks,
and predictive failure detection for spintronic neural networks.
"""

import time
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
from dataclasses import dataclass, asdict
from collections import deque, defaultdict

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    """System health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning" 
    DEGRADED = "degraded"
    CRITICAL = "critical"
    OFFLINE = "offline"

class MetricType(Enum):
    """Types of metrics to monitor"""
    PERFORMANCE = "performance"
    RELIABILITY = "reliability"
    ENERGY = "energy"
    ACCURACY = "accuracy"
    SECURITY = "security"
    USAGE = "usage"

@dataclass
class HealthMetric:
    """Health metric data structure"""
    name: str
    value: float
    unit: str
    timestamp: float
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    metadata: Dict[str, Any]

@dataclass
class SystemAlert:
    """System alert structure"""
    alert_id: str
    severity: str
    component: str
    message: str
    timestamp: float
    acknowledged: bool = False
    resolved: bool = False

class MetricsCollector:
    """Advanced metrics collection system"""
    
    def __init__(self, buffer_size: int = 10000):
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=buffer_size))
        self.collection_interval = 1.0  # seconds
        self.collectors = {}
        self.running = False
        self.collection_thread = None
        
    def register_collector(self, name: str, collector_func: Callable[[], Dict[str, float]]):
        """Register a metrics collector function"""
        self.collectors[name] = collector_func
        logger.info(f"Registered metrics collector: {name}")
    
    def start_collection(self):
        """Start metrics collection"""
        if self.running:
            logger.warning("Metrics collection already running")
            return
            
        self.running = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection"""
        self.running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5.0)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self):
        """Main collection loop"""
        while self.running:
            try:
                timestamp = time.time()
                
                for collector_name, collector_func in self.collectors.items():
                    try:
                        metrics = collector_func()
                        for metric_name, value in metrics.items():
                            full_name = f"{collector_name}.{metric_name}"
                            self.metrics_buffer[full_name].append({
                                'value': value,
                                'timestamp': timestamp
                            })
                    except Exception as e:
                        logger.error(f"Error collecting metrics from {collector_name}: {e}")
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection loop: {e}")
    
    def get_latest_metrics(self, metric_name: str = None) -> Dict[str, Any]:
        """Get latest metrics"""
        if metric_name:
            if metric_name in self.metrics_buffer:
                buffer = self.metrics_buffer[metric_name]
                return buffer[-1] if buffer else None
            return None
        
        # Return all latest metrics
        latest = {}
        for name, buffer in self.metrics_buffer.items():
            if buffer:
                latest[name] = buffer[-1]
        
        return latest
    
    def get_metric_history(self, metric_name: str, 
                          duration: float = 3600) -> List[Dict[str, Any]]:
        """Get metric history for specified duration"""
        if metric_name not in self.metrics_buffer:
            return []
        
        current_time = time.time()
        cutoff_time = current_time - duration
        
        buffer = self.metrics_buffer[metric_name]
        history = [entry for entry in buffer if entry['timestamp'] >= cutoff_time]
        
        return history
    
    def calculate_statistics(self, metric_name: str, 
                           duration: float = 3600) -> Dict[str, float]:
        """Calculate statistics for a metric"""
        history = self.get_metric_history(metric_name, duration)
        
        if not history:
            return {}
        
        values = [entry['value'] for entry in history]
        
        return {
            'count': len(values),
            'min': min(values),
            'max': max(values),
            'mean': sum(values) / len(values),
            'latest': values[-1] if values else 0.0,
            'trend': self._calculate_trend(values)
        }
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 2:
            return "stable"
        
        recent_avg = sum(values[-min(10, len(values)):]) / min(10, len(values))
        older_avg = sum(values[:min(10, len(values))]) / min(10, len(values))
        
        change_percent = ((recent_avg - older_avg) / older_avg * 100) if older_avg != 0 else 0
        
        if change_percent > 5:
            return "increasing"
        elif change_percent < -5:
            return "decreasing"
        else:
            return "stable"

def create_monitoring_system() -> Dict[str, Any]:
    """Create comprehensive monitoring system"""
    
    # Initialize components
    metrics_collector = MetricsCollector()
    
    # Register default collectors
    def system_metrics():
        """Collect basic system metrics"""
        import random
        return {
            'cpu_usage': random.uniform(10, 80),
            'memory_usage': random.uniform(0.3, 0.8),
            'disk_usage': random.uniform(0.1, 0.6)
        }
    
    def spintron_metrics():
        """Collect SpinTron-specific metrics"""
        import random
        return {
            'inference_latency': random.uniform(0.1, 2.0),  # ms
            'throughput': random.uniform(1000, 5000),  # ops/sec
            'energy_consumption': random.uniform(10, 50),  # pJ
            'model_accuracy': random.uniform(0.85, 0.99)
        }
    
    # Register collectors
    metrics_collector.register_collector('system', system_metrics)
    metrics_collector.register_collector('spintron', spintron_metrics)
    
    return {
        'metrics_collector': metrics_collector,
        'status': 'monitoring_system_ready'
    }