"""
Advanced Monitoring and Health Check System for SpinTron-NN-Kit.

This module provides comprehensive system monitoring including:
- Real-time performance metrics
- Health checks and system status
- Alerting and anomaly detection  
- Resource usage monitoring
- Distributed system telemetry
"""

import time
import threading
import psutil
import json
import os
from typing import Any, Dict, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import queue
from concurrent.futures import ThreadPoolExecutor

from .logging_config import get_logger
from .error_handling import SpintronError, ErrorSeverity


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class MetricType(Enum):
    """Types of metrics being tracked."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class HealthCheck:
    """Individual health check definition."""
    name: str
    check_function: Callable[[], bool]
    description: str
    timeout_seconds: float = 5.0
    critical: bool = False
    interval_seconds: float = 30.0
    last_check_time: float = 0.0
    last_result: Optional[bool] = None
    last_error: Optional[str] = None


@dataclass
class Metric:
    """Metric data point."""
    name: str
    value: Union[int, float]
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str]
    unit: Optional[str] = None


@dataclass
class Alert:
    """System alert definition."""
    id: str
    level: str
    message: str
    component: str
    timestamp: float
    resolved: bool = False
    resolution_time: Optional[float] = None


@dataclass
class SystemStatus:
    """Overall system status."""
    status: HealthStatus
    timestamp: float
    component_statuses: Dict[str, HealthStatus]
    metrics_summary: Dict[str, Any]
    active_alerts: List[Alert]
    uptime_seconds: float


class MetricsCollector:
    """Collects and manages system metrics."""
    
    def __init__(self):
        self.logger = get_logger("monitoring.metrics")
        self.metrics = queue.Queue(maxsize=10000)
        self.metric_history = {}
        self.start_time = time.time()
        
        # Performance counters
        self.counters = {}
        self.gauges = {}
        self.timers = {}
        
    def record_counter(self, name: str, value: int = 1, 
                      tags: Dict[str, str] = None, unit: str = None):
        """Record counter metric."""
        tags = tags or {}
        current_value = self.counters.get(name, 0)
        self.counters[name] = current_value + value
        
        metric = Metric(
            name=name,
            value=current_value + value,
            metric_type=MetricType.COUNTER,
            timestamp=time.time(),
            tags=tags,
            unit=unit
        )
        
        self._store_metric(metric)
    
    def record_gauge(self, name: str, value: Union[int, float],
                    tags: Dict[str, str] = None, unit: str = None):
        """Record gauge metric."""
        tags = tags or {}
        self.gauges[name] = value
        
        metric = Metric(
            name=name,
            value=value,
            metric_type=MetricType.GAUGE,
            timestamp=time.time(),
            tags=tags,
            unit=unit
        )
        
        self._store_metric(metric)
    
    def record_timer(self, name: str, duration: float,
                    tags: Dict[str, str] = None, unit: str = "seconds"):
        """Record timer metric."""
        tags = tags or {}
        
        if name not in self.timers:
            self.timers[name] = []
        
        self.timers[name].append(duration)
        
        metric = Metric(
            name=name,
            value=duration,
            metric_type=MetricType.TIMER,
            timestamp=time.time(),
            tags=tags,
            unit=unit
        )
        
        self._store_metric(metric)
    
    def _store_metric(self, metric: Metric):
        """Store metric in queue and history."""
        try:
            self.metrics.put_nowait(metric)
            
            # Store in history for trend analysis
            if metric.name not in self.metric_history:
                self.metric_history[metric.name] = []
            
            self.metric_history[metric.name].append(metric)
            
            # Keep only last 1000 points per metric
            if len(self.metric_history[metric.name]) > 1000:
                self.metric_history[metric.name] = self.metric_history[metric.name][-1000:]
                
        except queue.Full:
            self.logger.warning("Metrics queue full, dropping metric",
                              component="metrics_collector")
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metric values."""
        system_metrics = self._collect_system_metrics()
        
        return {
            'counters': dict(self.counters),
            'gauges': dict(self.gauges),
            'timer_summaries': self._summarize_timers(),
            'system': system_metrics,
            'uptime_seconds': time.time() - self.start_time
        }
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-level metrics."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            network = psutil.net_io_counters()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_free_gb': disk.free / (1024**3),
                'network_bytes_sent': network.bytes_sent if network else 0,
                'network_bytes_recv': network.bytes_recv if network else 0,
                'load_avg': os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
            }
        except Exception as e:
            self.logger.error(f"Failed to collect system metrics: {e}",
                            component="metrics_collector")
            return {}
    
    def _summarize_timers(self) -> Dict[str, Dict[str, float]]:
        """Summarize timer metrics."""
        summaries = {}
        
        for name, values in self.timers.items():
            if values:
                summaries[name] = {
                    'count': len(values),
                    'mean': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'total': sum(values)
                }
        
        return summaries
    
    def get_metric_history(self, metric_name: str, 
                          limit: int = 100) -> List[Metric]:
        """Get historical values for a metric."""
        history = self.metric_history.get(metric_name, [])
        return history[-limit:]


class HealthChecker:
    """Manages health checks for system components."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.logger = get_logger("monitoring.health_checker")
        self.metrics = metrics_collector
        self.health_checks = {}
        self.component_status = {}
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="health_check")
        
        # Register default health checks
        self._register_default_checks()
    
    def register_health_check(self, health_check: HealthCheck):
        """Register a new health check."""
        self.health_checks[health_check.name] = health_check
        self.logger.info(f"Registered health check: {health_check.name}",
                        component="health_checker")
    
    def _register_default_checks(self):
        """Register default system health checks."""
        
        # CPU usage check
        self.register_health_check(HealthCheck(
            name="cpu_usage",
            check_function=lambda: psutil.cpu_percent(interval=1) < 90.0,
            description="CPU usage below 90%",
            critical=True,
            interval_seconds=10.0
        ))
        
        # Memory usage check
        self.register_health_check(HealthCheck(
            name="memory_usage", 
            check_function=lambda: psutil.virtual_memory().percent < 90.0,
            description="Memory usage below 90%",
            critical=True,
            interval_seconds=10.0
        ))
        
        # Disk space check
        self.register_health_check(HealthCheck(
            name="disk_space",
            check_function=lambda: psutil.disk_usage('/').percent < 95.0,
            description="Disk space usage below 95%",
            critical=True,
            interval_seconds=60.0
        ))
        
        # Load average check
        def check_load_average():
            if hasattr(os, 'getloadavg'):
                load = os.getloadavg()[0]  # 1-minute load average
                cpu_count = psutil.cpu_count()
                return load < cpu_count * 2.0  # Load should be < 2x CPU count
            return True
        
        self.register_health_check(HealthCheck(
            name="load_average",
            check_function=check_load_average,
            description="System load average within acceptable range",
            critical=False,
            interval_seconds=30.0
        ))
    
    def start_monitoring(self):
        """Start continuous health monitoring."""
        self.is_running = True
        self.logger.info("Starting health monitoring", component="health_checker")
        
        # Start monitoring thread
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="health_monitor",
            daemon=True
        )
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring."""
        self.is_running = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=5.0)
        
        self.executor.shutdown(wait=True)
        self.logger.info("Stopped health monitoring", component="health_checker")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self.is_running:
            try:
                current_time = time.time()
                
                # Check which health checks need to run
                checks_to_run = []
                for name, check in self.health_checks.items():
                    if current_time - check.last_check_time >= check.interval_seconds:
                        checks_to_run.append((name, check))
                
                # Run checks in parallel
                if checks_to_run:
                    futures = []
                    for name, check in checks_to_run:
                        future = self.executor.submit(self._run_health_check, name, check)
                        futures.append(future)
                    
                    # Collect results
                    for future in futures:
                        try:
                            future.result(timeout=10.0)  # Overall timeout
                        except Exception as e:
                            self.logger.error(f"Health check execution failed: {e}",
                                            component="health_checker")
                
                # Sleep before next iteration
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}",
                                component="health_checker")
                time.sleep(5.0)  # Back off on errors
    
    def _run_health_check(self, name: str, check: HealthCheck):
        """Run a single health check."""
        try:
            start_time = time.time()
            
            # Run the check with timeout
            result = self._execute_check_with_timeout(check)
            
            # Record execution time
            execution_time = time.time() - start_time
            self.metrics.record_timer(
                f"health_check.{name}.duration",
                execution_time,
                tags={"check": name}
            )
            
            # Update check status
            check.last_check_time = time.time()
            check.last_result = result
            check.last_error = None
            
            # Update component status
            if result:
                self.component_status[name] = HealthStatus.HEALTHY
                self.metrics.record_counter(
                    f"health_check.{name}.success",
                    tags={"check": name}
                )
            else:
                status = HealthStatus.CRITICAL if check.critical else HealthStatus.WARNING
                self.component_status[name] = status
                self.metrics.record_counter(
                    f"health_check.{name}.failure",
                    tags={"check": name}
                )
                
                self.logger.warning(f"Health check failed: {name} - {check.description}",
                                  component="health_checker")
            
        except Exception as e:
            # Check failed with exception
            check.last_check_time = time.time()
            check.last_result = False
            check.last_error = str(e)
            
            status = HealthStatus.CRITICAL if check.critical else HealthStatus.WARNING
            self.component_status[name] = status
            
            self.metrics.record_counter(
                f"health_check.{name}.error",
                tags={"check": name, "error": type(e).__name__}
            )
            
            self.logger.error(f"Health check error: {name} - {str(e)}",
                            component="health_checker")
    
    def _execute_check_with_timeout(self, check: HealthCheck) -> bool:
        """Execute health check with timeout."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Health check timed out after {check.timeout_seconds}s")
        
        # Set up timeout (Unix only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(check.timeout_seconds))
        
        try:
            result = check.check_function()
            return bool(result)
        finally:
            if hasattr(signal, 'SIGALRM'):
                signal.alarm(0)  # Cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status."""
        overall_status = HealthStatus.HEALTHY
        
        # Determine overall status from components
        for status in self.component_status.values():
            if status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                break
            elif status == HealthStatus.WARNING and overall_status == HealthStatus.HEALTHY:
                overall_status = HealthStatus.WARNING
        
        return {
            'overall_status': overall_status.value,
            'component_statuses': {k: v.value for k, v in self.component_status.items()},
            'health_checks': {
                name: {
                    'last_result': check.last_result,
                    'last_check_time': check.last_check_time,
                    'last_error': check.last_error,
                    'description': check.description,
                    'critical': check.critical
                }
                for name, check in self.health_checks.items()
            },
            'timestamp': time.time()
        }


class AlertManager:
    """Manages system alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.logger = get_logger("monitoring.alert_manager")
        self.metrics = metrics_collector
        self.alerts = {}
        self.alert_handlers = []
        self.thresholds = {}
        
        # Configure default thresholds
        self._configure_default_thresholds()
    
    def _configure_default_thresholds(self):
        """Configure default alert thresholds."""
        self.thresholds = {
            'cpu_percent': {'warning': 80.0, 'critical': 90.0},
            'memory_percent': {'warning': 80.0, 'critical': 90.0},
            'disk_percent': {'warning': 85.0, 'critical': 95.0},
            'error_rate': {'warning': 10.0, 'critical': 50.0}  # errors per minute
        }
    
    def register_alert_handler(self, handler: Callable[[Alert], None]):
        """Register alert notification handler."""
        self.alert_handlers.append(handler)
    
    def check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds."""
        system_metrics = metrics.get('system', {})
        
        for metric_name, thresholds in self.thresholds.items():
            if metric_name in system_metrics:
                value = system_metrics[metric_name]
                self._evaluate_threshold(metric_name, value, thresholds)
    
    def _evaluate_threshold(self, metric_name: str, value: float, 
                           thresholds: Dict[str, float]):
        """Evaluate a single metric against thresholds."""
        alert_id = f"threshold_{metric_name}"
        
        # Determine alert level
        alert_level = None
        if value >= thresholds.get('critical', float('inf')):
            alert_level = 'CRITICAL'
        elif value >= thresholds.get('warning', float('inf')):
            alert_level = 'WARNING'
        
        # Handle alert state
        if alert_level:
            if alert_id not in self.alerts or self.alerts[alert_id].resolved:
                # New or resolved alert
                alert = Alert(
                    id=alert_id,
                    level=alert_level,
                    message=f"{metric_name} is {value:.1f} (threshold: {thresholds[alert_level.lower()]:.1f})",
                    component="system",
                    timestamp=time.time()
                )
                
                self.alerts[alert_id] = alert
                self._notify_handlers(alert)
                
                self.logger.warning(f"Alert triggered: {alert.message}",
                                  component="alert_manager", 
                                  alert_id=alert_id,
                                  alert_level=alert_level)
        else:
            # Value is below thresholds - resolve alert if active
            if alert_id in self.alerts and not self.alerts[alert_id].resolved:
                self.alerts[alert_id].resolved = True
                self.alerts[alert_id].resolution_time = time.time()
                
                self.logger.info(f"Alert resolved: {alert_id}",
                               component="alert_manager")
    
    def trigger_alert(self, alert_id: str, level: str, message: str, 
                     component: str):
        """Manually trigger an alert."""
        alert = Alert(
            id=alert_id,
            level=level,
            message=message,
            component=component,
            timestamp=time.time()
        )
        
        self.alerts[alert_id] = alert
        self._notify_handlers(alert)
        
        self.logger.warning(f"Manual alert triggered: {message}",
                          component="alert_manager",
                          alert_id=alert_id,
                          alert_level=level)
    
    def resolve_alert(self, alert_id: str):
        """Manually resolve an alert."""
        if alert_id in self.alerts and not self.alerts[alert_id].resolved:
            self.alerts[alert_id].resolved = True
            self.alerts[alert_id].resolution_time = time.time()
            
            self.logger.info(f"Alert manually resolved: {alert_id}",
                           component="alert_manager")
    
    def _notify_handlers(self, alert: Alert):
        """Notify all registered alert handlers."""
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Alert handler failed: {e}",
                                component="alert_manager")
    
    def get_active_alerts(self) -> List[Alert]:
        """Get list of active (unresolved) alerts."""
        return [alert for alert in self.alerts.values() if not alert.resolved]
    
    def get_alert_summary(self) -> Dict[str, Any]:
        """Get alert summary statistics."""
        active_alerts = self.get_active_alerts()
        
        return {
            'total_alerts': len(self.alerts),
            'active_alerts': len(active_alerts),
            'critical_alerts': len([a for a in active_alerts if a.level == 'CRITICAL']),
            'warning_alerts': len([a for a in active_alerts if a.level == 'WARNING']),
            'recent_alerts': [asdict(a) for a in list(self.alerts.values())[-10:]]
        }


class SystemMonitor:
    """Main monitoring system coordinator."""
    
    def __init__(self, export_path: Optional[str] = None):
        self.logger = get_logger("monitoring.system_monitor")
        self.start_time = time.time()
        
        # Initialize components
        self.metrics_collector = MetricsCollector()
        self.health_checker = HealthChecker(self.metrics_collector)
        self.alert_manager = AlertManager(self.metrics_collector)
        
        # Export configuration
        self.export_path = Path(export_path) if export_path else Path("./monitoring")
        self.export_path.mkdir(exist_ok=True)
        
        # Monitoring state
        self.is_running = False
        self.monitor_thread = None
        
        # Register default alert handler
        self.alert_manager.register_alert_handler(self._log_alert_handler)
        
        self.logger.info("System monitor initialized", component="system_monitor")
    
    def start_monitoring(self):
        """Start all monitoring components."""
        self.is_running = True
        
        # Start health checker
        self.health_checker.start_monitoring()
        
        # Start main monitoring loop
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="system_monitor",
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("System monitoring started", component="system_monitor")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        self.is_running = False
        
        # Stop health checker
        self.health_checker.stop_monitoring()
        
        # Wait for monitor thread
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("System monitoring stopped", component="system_monitor")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        export_counter = 0
        
        while self.is_running:
            try:
                # Collect current metrics
                metrics = self.metrics_collector.get_current_metrics()
                
                # Check alert thresholds
                self.alert_manager.check_thresholds(metrics)
                
                # Export metrics periodically (every 60 seconds)
                export_counter += 1
                if export_counter >= 60:  # 60 iterations * 1 second = 60 seconds
                    self._export_metrics(metrics)
                    export_counter = 0
                
                # Sleep between iterations
                time.sleep(1.0)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}",
                                component="system_monitor")
                time.sleep(5.0)  # Back off on errors
    
    def _export_metrics(self, metrics: Dict[str, Any]):
        """Export metrics to file."""
        try:
            export_file = self.export_path / f"metrics_{int(time.time())}.json"
            
            # Include health status and alerts
            export_data = {
                'timestamp': time.time(),
                'metrics': metrics,
                'health_status': self.health_checker.get_health_status(),
                'alert_summary': self.alert_manager.get_alert_summary()
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            # Keep only last 24 hours of exports (cleanup old files)
            self._cleanup_old_exports()
            
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}",
                            component="system_monitor")
    
    def _cleanup_old_exports(self):
        """Clean up old metric export files."""
        try:
            cutoff_time = time.time() - (24 * 60 * 60)  # 24 hours ago
            
            for file_path in self.export_path.glob("metrics_*.json"):
                try:
                    # Extract timestamp from filename
                    timestamp = int(file_path.stem.split('_')[1])
                    if timestamp < cutoff_time:
                        file_path.unlink()
                except (ValueError, IndexError):
                    # Skip files with unexpected names
                    continue
                    
        except Exception as e:
            self.logger.warning(f"Failed to cleanup old exports: {e}",
                              component="system_monitor")
    
    def _log_alert_handler(self, alert: Alert):
        """Default alert handler that logs to system logger."""
        level_map = {
            'CRITICAL': 'critical',
            'WARNING': 'warning',
            'INFO': 'info'
        }
        
        log_level = level_map.get(alert.level, 'warning')
        getattr(self.logger, log_level)(
            f"ALERT: {alert.message}",
            component="alert_handler",
            alert_id=alert.id,
            alert_level=alert.level,
            alert_component=alert.component
        )
    
    def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status."""
        health_status = self.health_checker.get_health_status()
        metrics = self.metrics_collector.get_current_metrics()
        active_alerts = self.alert_manager.get_active_alerts()
        
        # Determine overall status
        overall_status = HealthStatus(health_status['overall_status'])
        
        # If there are critical alerts, override status
        if any(alert.level == 'CRITICAL' for alert in active_alerts):
            overall_status = HealthStatus.CRITICAL
        
        component_statuses = {
            name: HealthStatus(status) 
            for name, status in health_status['component_statuses'].items()
        }
        
        return SystemStatus(
            status=overall_status,
            timestamp=time.time(),
            component_statuses=component_statuses,
            metrics_summary=metrics,
            active_alerts=active_alerts,
            uptime_seconds=time.time() - self.start_time
        )
    
    def record_operation(self, operation_name: str, duration: float,
                        success: bool = True, tags: Dict[str, str] = None):
        """Record operation metrics."""
        tags = tags or {}
        tags['operation'] = operation_name
        
        # Record timing
        self.metrics_collector.record_timer(
            f"operation.{operation_name}.duration",
            duration,
            tags=tags
        )
        
        # Record success/failure
        if success:
            self.metrics_collector.record_counter(
                f"operation.{operation_name}.success",
                tags=tags
            )
        else:
            self.metrics_collector.record_counter(
                f"operation.{operation_name}.failure", 
                tags=tags
            )
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get comprehensive performance report."""
        system_status = self.get_system_status()
        
        return {
            'system_status': system_status.status.value,
            'uptime_seconds': system_status.uptime_seconds,
            'component_health': {k: v.value for k, v in system_status.component_statuses.items()},
            'metrics_summary': system_status.metrics_summary,
            'active_alerts_count': len(system_status.active_alerts),
            'critical_alerts': [a for a in system_status.active_alerts if a.level == 'CRITICAL'],
            'timestamp': system_status.timestamp
        }


# Global monitoring instance
_global_monitor = None


def get_system_monitor(export_path: Optional[str] = None) -> SystemMonitor:
    """Get global system monitor instance."""
    global _global_monitor
    
    if _global_monitor is None:
        _global_monitor = SystemMonitor(export_path)
    
    return _global_monitor


def start_monitoring(export_path: Optional[str] = None):
    """Start global system monitoring."""
    monitor = get_system_monitor(export_path)
    monitor.start_monitoring()


def stop_monitoring():
    """Stop global system monitoring."""
    global _global_monitor
    
    if _global_monitor:
        _global_monitor.stop_monitoring()


def record_operation(operation_name: str, duration: float, 
                    success: bool = True, tags: Dict[str, str] = None):
    """Record operation metrics globally."""
    monitor = get_system_monitor()
    monitor.record_operation(operation_name, duration, success, tags)


def get_system_status() -> Dict[str, Any]:
    """Get global system status."""
    monitor = get_system_monitor()
    return monitor.get_performance_report()