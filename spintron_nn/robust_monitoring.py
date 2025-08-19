"""
Robust Monitoring and Health Management System for SpinTron-NN-Kit.

This module implements comprehensive monitoring, health checking, and fault
recovery mechanisms for spintronic neural network systems.

Features:
- Real-time performance monitoring
- Health status tracking and alerting
- Predictive failure detection
- Automatic fault recovery
- Comprehensive logging and metrics
- Resource utilization monitoring
- Quality of Service (QoS) tracking
"""

import time
import threading
import queue
import json
import os
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict
import statistics
import logging
from pathlib import Path
import pickle
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .core.mtj_models import MTJConfig, MTJDevice
from .core.crossbar import MTJCrossbar
from .utils.error_handling import HardwareError, ValidationError


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class PerformanceMetric:
    """Individual performance metric."""
    
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metric to dictionary."""
        return {
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'timestamp': self.timestamp,
            'tags': self.tags
        }


@dataclass
class HealthAlert:
    """Health alert notification."""
    
    alert_id: str
    timestamp: float
    level: AlertLevel
    source: str
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolution_time: Optional[float] = None
    
    def resolve(self):
        """Mark alert as resolved."""
        self.resolved = True
        self.resolution_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            'alert_id': self.alert_id,
            'timestamp': self.timestamp,
            'level': self.level.value,
            'source': self.source,
            'message': self.message,
            'details': self.details,
            'resolved': self.resolved,
            'resolution_time': self.resolution_time
        }


class MetricsCollector:
    """Collects and aggregates performance metrics."""
    
    def __init__(self, max_metrics: int = 100000):
        self.max_metrics = max_metrics
        self.metrics = deque(maxlen=max_metrics)
        self.metric_aggregates = defaultdict(list)
        self.collection_lock = threading.Lock()
        
        # Metric collection configuration
        self.collection_interval = 1.0  # seconds
        self.collection_active = False
        self.collection_thread = None
        
    def record_metric(self, metric: PerformanceMetric):
        """Record a performance metric."""
        with self.collection_lock:
            self.metrics.append(metric)
            
            # Update aggregates
            self.metric_aggregates[metric.name].append(metric.value)
            
            # Keep aggregates bounded
            if len(self.metric_aggregates[metric.name]) > 10000:
                self.metric_aggregates[metric.name] = self.metric_aggregates[metric.name][-5000:]
    
    def get_metrics(self, metric_name: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[PerformanceMetric]:
        """Get metrics with optional filtering."""
        with self.collection_lock:
            filtered_metrics = list(self.metrics)
        
        # Filter by name
        if metric_name:
            filtered_metrics = [m for m in filtered_metrics if m.name == metric_name]
        
        # Filter by time range
        if start_time is not None:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp >= start_time]
        
        if end_time is not None:
            filtered_metrics = [m for m in filtered_metrics if m.timestamp <= end_time]
        
        return filtered_metrics
    
    def get_metric_statistics(self, metric_name: str, 
                            window_seconds: float = 300) -> Dict[str, float]:
        """Get statistical summary of metric over time window."""
        end_time = time.time()
        start_time = end_time - window_seconds
        
        metrics = self.get_metrics(metric_name, start_time, end_time)
        
        if not metrics:
            return {
                'count': 0,
                'mean': 0.0,
                'std': 0.0,
                'min': 0.0,
                'max': 0.0,
                'latest': 0.0
            }
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'latest': values[-1] if values else 0.0
        }
    
    def start_collection(self, target_system: Any):
        """Start automatic metric collection."""
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(
            target=self._collection_loop,
            args=(target_system,),
            daemon=True
        )
        self.collection_thread.start()
        print("Metrics collection started")
    
    def stop_collection(self):
        """Stop automatic metric collection."""
        self.collection_active = False
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        print("Metrics collection stopped")
    
    def _collection_loop(self, target_system: Any):
        """Main metric collection loop."""
        while self.collection_active:
            try:
                # Collect system metrics
                self._collect_system_metrics(target_system)
                
                # Wait for next collection cycle
                time.sleep(self.collection_interval)
                
            except Exception as e:
                print(f"Metric collection error: {e}")
                time.sleep(self.collection_interval * 2)  # Back off on error
    
    def _collect_system_metrics(self, target_system: Any):
        """Collect metrics from target system."""
        timestamp = time.time()
        
        # System resource metrics
        cpu_percent = psutil.cpu_percent()
        memory_info = psutil.virtual_memory()
        
        self.record_metric(PerformanceMetric(
            name="cpu_utilization",
            value=cpu_percent,
            unit="percent",
            timestamp=timestamp
        ))
        
        self.record_metric(PerformanceMetric(
            name="memory_utilization",
            value=memory_info.percent,
            unit="percent",
            timestamp=timestamp
        ))
        
        # System-specific metrics
        if hasattr(target_system, 'get_statistics'):
            stats = target_system.get_statistics()
            
            # Convert stats to metrics
            for stat_name, stat_value in stats.items():
                if isinstance(stat_value, (int, float)):
                    self.record_metric(PerformanceMetric(
                        name=f"system_{stat_name}",
                        value=float(stat_value),
                        unit="count" if "count" in stat_name.lower() else "value",
                        timestamp=timestamp,
                        tags={'source': 'system_stats'}
                    ))
        
        # Performance metrics
        if hasattr(target_system, 'read_count'):
            self.record_metric(PerformanceMetric(
                name="read_operations_rate",
                value=getattr(target_system, 'read_count', 0),
                unit="ops",
                timestamp=timestamp
            ))
        
        if hasattr(target_system, 'write_count'):
            self.record_metric(PerformanceMetric(
                name="write_operations_rate",
                value=getattr(target_system, 'write_count', 0),
                unit="ops",
                timestamp=timestamp
            ))
        
        if hasattr(target_system, 'total_energy'):
            self.record_metric(PerformanceMetric(
                name="total_energy_consumption",
                value=getattr(target_system, 'total_energy', 0.0),
                unit="joules",
                timestamp=timestamp
            ))


class HealthMonitor:
    """Monitors system health and generates alerts."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics_collector = metrics_collector
        self.health_checks = []
        self.alerts = deque(maxlen=10000)
        self.health_status = HealthStatus.HEALTHY
        self.alert_handlers = []
        
        # Health monitoring configuration
        self.monitoring_interval = 5.0  # seconds
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Thresholds
        self.thresholds = {
            'cpu_utilization': {'warning': 80.0, 'critical': 95.0},
            'memory_utilization': {'warning': 85.0, 'critical': 95.0},
            'error_rate': {'warning': 0.01, 'critical': 0.05},
            'response_time': {'warning': 1.0, 'critical': 5.0}
        }
        
        # Initialize default health checks
        self._initialize_health_checks()
    
    def _initialize_health_checks(self):
        """Initialize default health checks."""
        self.health_checks = [
            self._check_cpu_utilization,
            self._check_memory_utilization,
            self._check_error_rate,
            self._check_response_time,
            self._check_device_health
        ]
    
    def add_alert_handler(self, handler: Callable[[HealthAlert], None]):
        """Add alert handler function."""
        self.alert_handlers.append(handler)
    
    def start_monitoring(self, target_system: Any):
        """Start health monitoring."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(target_system,),
            daemon=True
        )
        self.monitoring_thread.start()
        print("Health monitoring started")
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        print("Health monitoring stopped")
    
    def _monitoring_loop(self, target_system: Any):
        """Main health monitoring loop."""
        while self.monitoring_active:
            try:
                # Run health checks
                for health_check in self.health_checks:
                    alerts = health_check(target_system)
                    for alert in alerts:
                        self._handle_alert(alert)
                
                # Update overall health status
                self._update_health_status()
                
                # Wait for next monitoring cycle
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"Health monitoring error: {e}")
                time.sleep(self.monitoring_interval * 2)
    
    def _check_cpu_utilization(self, target_system: Any) -> List[HealthAlert]:
        """Check CPU utilization levels."""
        alerts = []
        
        stats = self.metrics_collector.get_metric_statistics('cpu_utilization', 60)
        if stats['count'] == 0:
            return alerts
        
        cpu_usage = stats['mean']
        
        if cpu_usage > self.thresholds['cpu_utilization']['critical']:
            alerts.append(HealthAlert(
                alert_id=f"cpu_critical_{int(time.time())}",
                timestamp=time.time(),
                level=AlertLevel.CRITICAL,
                source="health_monitor",
                message=f"Critical CPU utilization: {cpu_usage:.1f}%",
                details={'cpu_usage': cpu_usage, 'threshold': self.thresholds['cpu_utilization']['critical']}
            ))
        elif cpu_usage > self.thresholds['cpu_utilization']['warning']:
            alerts.append(HealthAlert(
                alert_id=f"cpu_warning_{int(time.time())}",
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                source="health_monitor",
                message=f"High CPU utilization: {cpu_usage:.1f}%",
                details={'cpu_usage': cpu_usage, 'threshold': self.thresholds['cpu_utilization']['warning']}
            ))
        
        return alerts
    
    def _check_memory_utilization(self, target_system: Any) -> List[HealthAlert]:
        """Check memory utilization levels."""
        alerts = []
        
        stats = self.metrics_collector.get_metric_statistics('memory_utilization', 60)
        if stats['count'] == 0:
            return alerts
        
        memory_usage = stats['mean']
        
        if memory_usage > self.thresholds['memory_utilization']['critical']:
            alerts.append(HealthAlert(
                alert_id=f"memory_critical_{int(time.time())}",
                timestamp=time.time(),
                level=AlertLevel.CRITICAL,
                source="health_monitor",
                message=f"Critical memory utilization: {memory_usage:.1f}%",
                details={'memory_usage': memory_usage, 'threshold': self.thresholds['memory_utilization']['critical']}
            ))
        elif memory_usage > self.thresholds['memory_utilization']['warning']:
            alerts.append(HealthAlert(
                alert_id=f"memory_warning_{int(time.time())}",
                timestamp=time.time(),
                level=AlertLevel.WARNING,
                source="health_monitor",
                message=f"High memory utilization: {memory_usage:.1f}%",
                details={'memory_usage': memory_usage, 'threshold': self.thresholds['memory_utilization']['warning']}
            ))
        
        return alerts
    
    def _check_error_rate(self, target_system: Any) -> List[HealthAlert]:
        """Check system error rates."""
        alerts = []
        
        # Calculate error rate
        if hasattr(target_system, 'error_count') and hasattr(target_system, 'read_count'):
            total_ops = getattr(target_system, 'read_count', 0) + getattr(target_system, 'write_count', 0)
            error_count = getattr(target_system, 'error_count', 0)
            
            if total_ops > 0:
                error_rate = error_count / total_ops
                
                if error_rate > self.thresholds['error_rate']['critical']:
                    alerts.append(HealthAlert(
                        alert_id=f"error_rate_critical_{int(time.time())}",
                        timestamp=time.time(),
                        level=AlertLevel.CRITICAL,
                        source="health_monitor",
                        message=f"Critical error rate: {error_rate:.3f}",
                        details={'error_rate': error_rate, 'error_count': error_count, 'total_ops': total_ops}
                    ))
                elif error_rate > self.thresholds['error_rate']['warning']:
                    alerts.append(HealthAlert(
                        alert_id=f"error_rate_warning_{int(time.time())}",
                        timestamp=time.time(),
                        level=AlertLevel.WARNING,
                        source="health_monitor",
                        message=f"High error rate: {error_rate:.3f}",
                        details={'error_rate': error_rate, 'error_count': error_count, 'total_ops': total_ops}
                    ))
        
        return alerts
    
    def _check_response_time(self, target_system: Any) -> List[HealthAlert]:
        """Check system response times."""
        alerts = []
        
        # This would typically measure actual response times
        # For now, we'll use a placeholder
        if hasattr(target_system, 'monitor'):
            monitor = target_system.monitor
            if hasattr(monitor, 'get_average_response_time'):
                avg_response_time = monitor.get_average_response_time()
                
                if avg_response_time > self.thresholds['response_time']['critical']:
                    alerts.append(HealthAlert(
                        alert_id=f"response_time_critical_{int(time.time())}",
                        timestamp=time.time(),
                        level=AlertLevel.CRITICAL,
                        source="health_monitor",
                        message=f"Critical response time: {avg_response_time:.3f}s",
                        details={'response_time': avg_response_time}
                    ))
                elif avg_response_time > self.thresholds['response_time']['warning']:
                    alerts.append(HealthAlert(
                        alert_id=f"response_time_warning_{int(time.time())}",
                        timestamp=time.time(),
                        level=AlertLevel.WARNING,
                        source="health_monitor",
                        message=f"High response time: {avg_response_time:.3f}s",
                        details={'response_time': avg_response_time}
                    ))
        
        return alerts
    
    def _check_device_health(self, target_system: Any) -> List[HealthAlert]:
        """Check individual device health."""
        alerts = []
        
        if hasattr(target_system, 'devices'):
            # Sample a subset of devices for health checking
            rows, cols = len(target_system.devices), len(target_system.devices[0])
            sample_size = min(20, rows * cols // 10)
            
            failed_devices = 0
            total_checked = 0
            
            for i in range(0, rows, max(1, rows // 10)):
                for j in range(0, cols, max(1, cols // 10)):
                    if total_checked >= sample_size:
                        break
                    
                    device = target_system.devices[i][j]
                    total_checked += 1
                    
                    # Check device health
                    if not self._is_device_healthy(device):
                        failed_devices += 1
            
            if total_checked > 0:
                failure_rate = failed_devices / total_checked
                
                if failure_rate > 0.1:  # >10% failure rate
                    alerts.append(HealthAlert(
                        alert_id=f"device_failure_critical_{int(time.time())}",
                        timestamp=time.time(),
                        level=AlertLevel.CRITICAL,
                        source="health_monitor",
                        message=f"High device failure rate: {failure_rate:.2%}",
                        details={'failure_rate': failure_rate, 'failed_devices': failed_devices, 'total_checked': total_checked}
                    ))
                elif failure_rate > 0.05:  # >5% failure rate
                    alerts.append(HealthAlert(
                        alert_id=f"device_failure_warning_{int(time.time())}",
                        timestamp=time.time(),
                        level=AlertLevel.WARNING,
                        source="health_monitor",
                        message=f"Elevated device failure rate: {failure_rate:.2%}",
                        details={'failure_rate': failure_rate, 'failed_devices': failed_devices, 'total_checked': total_checked}
                    ))
        
        return alerts
    
    def _is_device_healthy(self, device: MTJDevice) -> bool:
        """Check if a device is healthy."""
        try:
            # Check if device has valid resistance
            resistance = device.resistance
            if resistance <= 0 or not np.isfinite(resistance):
                return False
            
            # Check if resistance is within expected range
            expected_min = device.config.resistance_low * 0.5
            expected_max = device.config.resistance_high * 2.0
            
            if resistance < expected_min or resistance > expected_max:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _handle_alert(self, alert: HealthAlert):
        """Handle a generated alert."""
        # Store alert
        self.alerts.append(alert)
        
        # Call alert handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler error: {e}")
    
    def _update_health_status(self):
        """Update overall health status based on recent alerts."""
        # Get recent alerts (last 5 minutes)
        recent_alerts = [
            alert for alert in self.alerts
            if time.time() - alert.timestamp < 300 and not alert.resolved
        ]
        
        if not recent_alerts:
            self.health_status = HealthStatus.HEALTHY
        elif any(alert.level == AlertLevel.CRITICAL for alert in recent_alerts):
            self.health_status = HealthStatus.CRITICAL
        elif any(alert.level == AlertLevel.ERROR for alert in recent_alerts):
            self.health_status = HealthStatus.DEGRADED
        elif any(alert.level == AlertLevel.WARNING for alert in recent_alerts):
            self.health_status = HealthStatus.WARNING
        else:
            self.health_status = HealthStatus.HEALTHY
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health status summary."""
        recent_alerts = [
            alert for alert in self.alerts
            if time.time() - alert.timestamp < 3600  # Last hour
        ]
        
        alert_counts = {level.value: 0 for level in AlertLevel}
        for alert in recent_alerts:
            alert_counts[alert.level.value] += 1
        
        return {
            'health_status': self.health_status.value,
            'monitoring_active': self.monitoring_active,
            'total_alerts': len(self.alerts),
            'recent_alerts': len(recent_alerts),
            'alert_counts': alert_counts,
            'unresolved_alerts': len([a for a in recent_alerts if not a.resolved])
        }


class FaultRecoveryManager:
    """Manages automatic fault recovery and self-healing."""
    
    def __init__(self, target_system: Any):
        self.target_system = target_system
        self.recovery_strategies = {}
        self.recovery_history = []
        self.recovery_active = True
        
        # Initialize recovery strategies
        self._initialize_recovery_strategies()
    
    def _initialize_recovery_strategies(self):
        """Initialize fault recovery strategies."""
        self.recovery_strategies = {
            'cpu_critical': self._recover_cpu_overload,
            'memory_critical': self._recover_memory_overload,
            'error_rate_critical': self._recover_high_error_rate,
            'device_failure_critical': self._recover_device_failures,
            'response_time_critical': self._recover_slow_response
        }
    
    def attempt_recovery(self, alert: HealthAlert) -> bool:
        """Attempt to recover from the issue described in the alert."""
        if not self.recovery_active:
            return False
        
        # Find appropriate recovery strategy
        strategy_key = f"{alert.source}_{alert.level.value}"
        if alert.alert_id.startswith('cpu'):
            strategy_key = 'cpu_critical'
        elif alert.alert_id.startswith('memory'):
            strategy_key = 'memory_critical'
        elif alert.alert_id.startswith('error_rate'):
            strategy_key = 'error_rate_critical'
        elif alert.alert_id.startswith('device_failure'):
            strategy_key = 'device_failure_critical'
        elif alert.alert_id.startswith('response_time'):
            strategy_key = 'response_time_critical'
        
        if strategy_key not in self.recovery_strategies:
            return False
        
        try:
            # Attempt recovery
            recovery_func = self.recovery_strategies[strategy_key]
            success = recovery_func(alert)
            
            # Record recovery attempt
            recovery_record = {
                'timestamp': time.time(),
                'alert_id': alert.alert_id,
                'strategy': strategy_key,
                'success': success,
                'details': alert.details
            }
            self.recovery_history.append(recovery_record)
            
            # Keep history bounded
            if len(self.recovery_history) > 1000:
                self.recovery_history = self.recovery_history[-500:]
            
            return success
            
        except Exception as e:
            print(f"Recovery attempt failed: {e}")
            return False
    
    def _recover_cpu_overload(self, alert: HealthAlert) -> bool:
        """Recover from CPU overload."""
        print("Attempting CPU overload recovery...")
        
        # Reduce processing load
        if hasattr(self.target_system, 'config'):
            # Reduce read/write frequencies
            if hasattr(self.target_system.config, 'read_time'):
                self.target_system.config.read_time *= 1.2  # Slow down operations
            
            if hasattr(self.target_system.config, 'write_time'):
                self.target_system.config.write_time *= 1.2
        
        # Disable non-essential monitoring
        # (Implementation would depend on specific system)
        
        print("CPU overload recovery applied")
        return True
    
    def _recover_memory_overload(self, alert: HealthAlert) -> bool:
        """Recover from memory overload."""
        print("Attempting memory overload recovery...")
        
        # Clear caches
        if hasattr(self.target_system, '_invalidate_caches'):
            self.target_system._invalidate_caches()
        
        # Reduce cache sizes
        if hasattr(self.target_system, '_conductance_cache'):
            self.target_system._conductance_cache = None
        
        # Force garbage collection
        import gc
        gc.collect()
        
        print("Memory overload recovery applied")
        return True
    
    def _recover_high_error_rate(self, alert: HealthAlert) -> bool:
        """Recover from high error rate."""
        print("Attempting error rate recovery...")
        
        # Reset error counters
        if hasattr(self.target_system, 'error_count'):
            self.target_system.error_count = 0
        
        # Recalibrate system
        if hasattr(self.target_system, '_perform_health_check'):
            try:
                self.target_system._perform_health_check()
            except Exception:
                pass  # Continue even if health check fails
        
        # Reset to safe operational parameters
        if hasattr(self.target_system, 'config'):
            config = self.target_system.config
            
            # Reset voltages to safe values
            if hasattr(config, 'read_voltage'):
                config.read_voltage = 0.1  # Safe read voltage
            
            if hasattr(config, 'write_voltage'):
                config.write_voltage = 0.5  # Safe write voltage
        
        print("Error rate recovery applied")
        return True
    
    def _recover_device_failures(self, alert: HealthAlert) -> bool:
        """Recover from device failures."""
        print("Attempting device failure recovery...")
        
        # Implement device redundancy or error correction
        # This is a simplified recovery strategy
        
        # Re-initialize problematic devices
        if hasattr(self.target_system, 'devices'):
            rows, cols = len(self.target_system.devices), len(self.target_system.devices[0])
            
            # Sample and fix a few devices
            for i in range(0, rows, max(1, rows // 5)):
                for j in range(0, cols, max(1, cols // 5)):
                    device = self.target_system.devices[i][j]
                    
                    # Reset device to default state
                    try:
                        device._state = 0  # Reset to low resistance state
                        device._write_count = 0  # Reset write count
                    except Exception:
                        pass  # Continue with other devices
        
        print("Device failure recovery applied")
        return True
    
    def _recover_slow_response(self, alert: HealthAlert) -> bool:
        """Recover from slow response times."""
        print("Attempting response time recovery...")
        
        # Optimize performance settings
        if hasattr(self.target_system, 'config'):
            config = self.target_system.config
            
            # Increase sense amplifier gain for faster reads
            if hasattr(config, 'sense_amplifier_gain'):
                config.sense_amplifier_gain = min(config.sense_amplifier_gain * 1.2, 10000)
        
        # Clear performance bottlenecks
        if hasattr(self.target_system, '_invalidate_caches'):
            self.target_system._invalidate_caches()
        
        print("Response time recovery applied")
        return True
    
    def get_recovery_statistics(self) -> Dict[str, Any]:
        """Get recovery attempt statistics."""
        if not self.recovery_history:
            return {
                'total_attempts': 0,
                'success_rate': 0.0,
                'recent_attempts': 0,
                'strategy_stats': {}
            }
        
        total_attempts = len(self.recovery_history)
        successful_attempts = sum(1 for r in self.recovery_history if r['success'])
        success_rate = successful_attempts / total_attempts
        
        # Recent attempts (last hour)
        recent_attempts = [
            r for r in self.recovery_history
            if time.time() - r['timestamp'] < 3600
        ]
        
        # Strategy statistics
        strategy_stats = defaultdict(lambda: {'attempts': 0, 'successes': 0})
        for record in self.recovery_history:
            strategy = record['strategy']
            strategy_stats[strategy]['attempts'] += 1
            if record['success']:
                strategy_stats[strategy]['successes'] += 1
        
        # Calculate success rates per strategy
        for strategy, stats in strategy_stats.items():
            if stats['attempts'] > 0:
                stats['success_rate'] = stats['successes'] / stats['attempts']
            else:
                stats['success_rate'] = 0.0
        
        return {
            'total_attempts': total_attempts,
            'success_rate': success_rate,
            'recent_attempts': len(recent_attempts),
            'strategy_stats': dict(strategy_stats)
        }


class RobustMonitoringSystem:
    """Main robust monitoring system coordinating all monitoring components."""
    
    def __init__(self, target_system: Any):
        self.target_system = target_system
        
        # Initialize monitoring components
        self.metrics_collector = MetricsCollector()
        self.health_monitor = HealthMonitor(self.metrics_collector)
        self.fault_recovery = FaultRecoveryManager(target_system)
        
        # System state
        self.monitoring_active = False
        self.auto_recovery_enabled = True
        
        # Setup alert handling
        self.health_monitor.add_alert_handler(self._handle_alert)
        
        # Logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup monitoring system logging."""
        self.logger = logging.getLogger("robust_monitoring")
        self.logger.setLevel(logging.INFO)
        
        # Create console handler
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
    
    def start_monitoring(self):
        """Start comprehensive monitoring."""
        if self.monitoring_active:
            return
        
        print("Starting robust monitoring system...")
        
        # Start components
        self.metrics_collector.start_collection(self.target_system)
        self.health_monitor.start_monitoring(self.target_system)
        
        self.monitoring_active = True
        self.logger.info("Robust monitoring system started")
    
    def stop_monitoring(self):
        """Stop comprehensive monitoring."""
        if not self.monitoring_active:
            return
        
        print("Stopping robust monitoring system...")
        
        # Stop components
        self.metrics_collector.stop_collection()
        self.health_monitor.stop_monitoring()
        
        self.monitoring_active = False
        self.logger.info("Robust monitoring system stopped")
    
    def _handle_alert(self, alert: HealthAlert):
        """Handle alerts from health monitor."""
        self.logger.warning(f"Health alert: {alert.message}")
        
        # Attempt automatic recovery for critical alerts
        if (self.auto_recovery_enabled and 
            alert.level in [AlertLevel.CRITICAL, AlertLevel.ERROR]):
            
            recovery_success = self.fault_recovery.attempt_recovery(alert)
            
            if recovery_success:
                self.logger.info(f"Automatic recovery successful for alert: {alert.alert_id}")
                alert.resolve()
            else:
                self.logger.error(f"Automatic recovery failed for alert: {alert.alert_id}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        health_summary = self.health_monitor.get_health_summary()
        recovery_stats = self.fault_recovery.get_recovery_statistics()
        
        # Get recent metrics summary
        cpu_stats = self.metrics_collector.get_metric_statistics('cpu_utilization')
        memory_stats = self.metrics_collector.get_metric_statistics('memory_utilization')
        
        return {
            'monitoring_active': self.monitoring_active,
            'auto_recovery_enabled': self.auto_recovery_enabled,
            'health_status': health_summary,
            'recovery_statistics': recovery_stats,
            'system_metrics': {
                'cpu_utilization': cpu_stats,
                'memory_utilization': memory_stats
            },
            'timestamp': time.time()
        }
    
    def force_health_check(self) -> Dict[str, Any]:
        """Force immediate health check."""
        self.logger.info("Forcing health check...")
        
        # Run all health checks immediately
        alerts = []
        for health_check in self.health_monitor.health_checks:
            try:
                check_alerts = health_check(self.target_system)
                alerts.extend(check_alerts)
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
        
        # Handle alerts
        for alert in alerts:
            self.health_monitor._handle_alert(alert)
        
        return {
            'alerts_generated': len(alerts),
            'alert_details': [alert.to_dict() for alert in alerts],
            'timestamp': time.time()
        }
    
    def export_monitoring_data(self, filename: str, hours: int = 24):
        """Export monitoring data to file."""
        end_time = time.time()
        start_time = end_time - (hours * 3600)
        
        # Get metrics
        metrics = self.metrics_collector.get_metrics(start_time=start_time, end_time=end_time)
        
        # Get alerts
        alerts = [
            alert for alert in self.health_monitor.alerts
            if start_time <= alert.timestamp <= end_time
        ]
        
        # Get recovery history
        recovery_history = [
            record for record in self.fault_recovery.recovery_history
            if start_time <= record['timestamp'] <= end_time
        ]
        
        export_data = {
            'export_info': {
                'timestamp': time.time(),
                'start_time': start_time,
                'end_time': end_time,
                'hours': hours
            },
            'metrics': [metric.to_dict() for metric in metrics],
            'alerts': [alert.to_dict() for alert in alerts],
            'recovery_history': recovery_history,
            'system_status': self.get_system_status()
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(f"Monitoring data exported to {filename}")
        print(f"Exported {len(metrics)} metrics, {len(alerts)} alerts, {len(recovery_history)} recovery records")
    
    def configure_thresholds(self, threshold_config: Dict[str, Dict[str, float]]):
        """Configure monitoring thresholds."""
        self.health_monitor.thresholds.update(threshold_config)
        self.logger.info("Monitoring thresholds updated")
    
    def enable_auto_recovery(self, enabled: bool = True):
        """Enable or disable automatic recovery."""
        self.auto_recovery_enabled = enabled
        self.fault_recovery.recovery_active = enabled
        self.logger.info(f"Auto recovery {'enabled' if enabled else 'disabled'}")


# Factory function for easy setup
def create_robust_monitoring(target_system: Any) -> RobustMonitoringSystem:
    """Create robust monitoring system for target system."""
    monitoring = RobustMonitoringSystem(target_system)
    
    # Configure default thresholds
    default_thresholds = {
        'cpu_utilization': {'warning': 75.0, 'critical': 90.0},
        'memory_utilization': {'warning': 80.0, 'critical': 95.0},
        'error_rate': {'warning': 0.02, 'critical': 0.1},
        'response_time': {'warning': 2.0, 'critical': 10.0}
    }
    monitoring.configure_thresholds(default_thresholds)
    
    return monitoring


# Example usage
def demonstrate_robust_monitoring():
    """Demonstrate robust monitoring capabilities."""
    from .core.mtj_models import MTJConfig
    from .core.crossbar import CrossbarConfig, MTJCrossbar
    
    # Create target system
    mtj_config = MTJConfig()
    crossbar_config = CrossbarConfig(rows=32, cols=32, mtj_config=mtj_config)
    crossbar = MTJCrossbar(crossbar_config)
    
    # Create monitoring system
    monitoring = create_robust_monitoring(crossbar)
    
    # Start monitoring
    monitoring.start_monitoring()
    
    print("Monitoring system active - simulating operations...")
    
    # Simulate some operations
    for i in range(10):
        # Simulate workload
        weights = np.random.randn(32, 32)
        crossbar.set_weights(weights)
        
        input_vector = np.random.randn(32)
        output = crossbar.compute_vmm(input_vector)
        
        time.sleep(0.5)
    
    # Force health check
    health_check_result = monitoring.force_health_check()
    print(f"Health check: {health_check_result['alerts_generated']} alerts generated")
    
    # Get system status
    status = monitoring.get_system_status()
    print(f"System health: {status['health_status']['health_status']}")
    
    # Export data
    monitoring.export_monitoring_data("monitoring_demo.json", hours=1)
    
    # Stop monitoring
    monitoring.stop_monitoring()
    
    return monitoring


if __name__ == "__main__":
    # Demonstration
    monitoring_demo = demonstrate_robust_monitoring()
    print("Robust monitoring demonstration complete")
