"""
Comprehensive logging configuration for SpinTron-NN-Kit.

This module provides structured logging with security event tracking,
performance monitoring, and audit trails.
"""

import logging
import logging.handlers
import json
import os
import sys
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum


class LogLevel(Enum):
    """Logging levels with security context."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    SECURITY = "SECURITY"


class EventCategory(Enum):
    """Event categories for structured logging."""
    SYSTEM = "system"
    SECURITY = "security"
    PERFORMANCE = "performance"
    USER_ACTION = "user_action"
    HARDWARE = "hardware"
    SIMULATION = "simulation"
    TRAINING = "training"
    CONVERSION = "conversion"


@dataclass
class LogEvent:
    """Structured log event."""
    timestamp: str
    level: str
    category: str
    component: str
    event_type: str
    message: str
    details: Dict[str, Any]
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    trace_id: Optional[str] = None


class SecurityAuditLogger:
    """Specialized logger for security events and audit trails."""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = log_dir
        self.ensure_log_directory()
        
        # Create security-specific logger
        self.security_logger = logging.getLogger("spintron_security")
        self.security_logger.setLevel(logging.INFO)
        
        # Security log file (separate from main logs)
        security_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "security_audit.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        
        # JSON formatter for structured logging
        security_formatter = JsonFormatter()
        security_handler.setFormatter(security_formatter)
        self.security_logger.addHandler(security_handler)
        
        # Performance logger
        self.perf_logger = logging.getLogger("spintron_performance")
        perf_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "performance.log"),
            maxBytes=50*1024*1024,  # 50MB
            backupCount=5
        )
        perf_handler.setFormatter(security_formatter)
        self.perf_logger.addHandler(perf_handler)
    
    def ensure_log_directory(self):
        """Ensure log directory exists with proper permissions."""
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir, mode=0o750)  # Secure permissions
    
    def log_security_event(self, event_type: str, severity: str, details: Dict[str, Any],
                          component: str = "unknown", user_id: str = None):
        """Log security-related events.
        
        Args:
            event_type: Type of security event
            severity: Severity level
            details: Event details
            component: Component generating the event
            user_id: User identifier (if applicable)
        """
        event = LogEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=severity,
            category=EventCategory.SECURITY.value,
            component=component,
            event_type=event_type,
            message=f"Security event: {event_type}",
            details=details,
            user_id=user_id
        )
        
        self.security_logger.info(json.dumps(asdict(event)))
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str,
                             component: str, details: Dict[str, Any] = None):
        """Log performance metrics.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            unit: Unit of measurement
            component: Component being measured
            details: Additional details
        """
        event = LogEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=LogLevel.INFO.value,
            category=EventCategory.PERFORMANCE.value,
            component=component,
            event_type="performance_metric",
            message=f"{metric_name}: {value} {unit}",
            details={
                "metric_name": metric_name,
                "value": value,
                "unit": unit,
                **(details or {})
            }
        )
        
        self.perf_logger.info(json.dumps(asdict(event)))
    
    def log_user_action(self, action: str, component: str, user_id: str = None,
                       details: Dict[str, Any] = None):
        """Log user actions for audit trail.
        
        Args:
            action: Action performed
            component: Component where action occurred
            user_id: User identifier
            details: Action details
        """
        event = LogEvent(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=LogLevel.INFO.value,
            category=EventCategory.USER_ACTION.value,
            component=component,
            event_type="user_action",
            message=f"User action: {action}",
            details={
                "action": action,
                **(details or {})
            },
            user_id=user_id
        )
        
        self.security_logger.info(json.dumps(asdict(event)))


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON.
        
        Args:
            record: Log record to format
            
        Returns:
            JSON-formatted log entry
        """
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields if present
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                          'filename', 'module', 'lineno', 'funcName', 'created',
                          'msecs', 'relativeCreated', 'thread', 'threadName',
                          'processName', 'process', 'getMessage', 'exc_info',
                          'exc_text', 'stack_info']:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


class SpintronLogger:
    """Main logger for SpinTron-NN-Kit with comprehensive features."""
    
    def __init__(self, name: str = "spintron_nn", log_level: str = "INFO", 
                 log_dir: str = "logs", enable_console: bool = True):
        """Initialize SpinTron logger.
        
        Args:
            name: Logger name
            log_level: Logging level
            log_dir: Log directory path
            enable_console: Enable console output
        """
        self.name = name
        self.log_dir = log_dir
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Create log directory
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, mode=0o750)
        
        # File handler with rotation
        file_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "spintron_nn.log"),
            maxBytes=20*1024*1024,  # 20MB
            backupCount=10,
            encoding='utf-8'
        )
        file_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(file_handler)
        
        # Console handler (if enabled)
        if enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Error file handler (separate file for errors)
        error_handler = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, "errors.log"),
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(error_handler)
        
        # Initialize security audit logger
        self.security_logger = SecurityAuditLogger(log_dir)
        
        # Log startup
        self.info("SpinTron-NN-Kit logger initialized", 
                 component="logging", 
                 details={"log_level": log_level, "log_dir": log_dir})
    
    def debug(self, message: str, component: str = "general", **kwargs):
        """Log debug message."""
        self._log(logging.DEBUG, message, component, **kwargs)
    
    def info(self, message: str, component: str = "general", **kwargs):
        """Log info message."""
        self._log(logging.INFO, message, component, **kwargs)
    
    def warning(self, message: str, component: str = "general", **kwargs):
        """Log warning message."""
        self._log(logging.WARNING, message, component, **kwargs)
    
    def error(self, message: str, component: str = "general", exception: Exception = None, **kwargs):
        """Log error message."""
        if exception:
            kwargs["exception_type"] = type(exception).__name__
            kwargs["exception_message"] = str(exception)
        self._log(logging.ERROR, message, component, **kwargs)
    
    def critical(self, message: str, component: str = "general", **kwargs):
        """Log critical message."""
        self._log(logging.CRITICAL, message, component, **kwargs)
        # Also log as security event for critical issues
        self.security_logger.log_security_event(
            "critical_error", "CRITICAL", 
            {"message": message, "component": component, **kwargs}
        )
    
    def security_event(self, event_type: str, severity: str, details: Dict[str, Any],
                      component: str = "security", user_id: str = None):
        """Log security event."""
        self.security_logger.log_security_event(event_type, severity, details, component, user_id)
    
    def performance_metric(self, metric_name: str, value: float, unit: str,
                         component: str, **kwargs):
        """Log performance metric."""
        self.security_logger.log_performance_metric(metric_name, value, unit, component, kwargs)
    
    def user_action(self, action: str, component: str, user_id: str = None, **kwargs):
        """Log user action."""
        self.security_logger.log_user_action(action, component, user_id, kwargs)
    
    def _log(self, level: int, message: str, component: str, **kwargs):
        """Internal logging method."""
        extra = {"component": component, **kwargs}
        self.logger.log(level, message, extra=extra)
    
    def log_function_call(self, func_name: str, component: str, 
                         args: Dict[str, Any] = None, duration: float = None):
        """Log function call for debugging and performance analysis.
        
        Args:
            func_name: Name of the function
            component: Component name
            args: Function arguments (sanitized)
            duration: Execution duration in seconds
        """
        details = {"function": func_name}
        if args:
            details["args"] = {k: str(v)[:100] for k, v in args.items()}  # Sanitize and limit
        if duration:
            details["duration_ms"] = duration * 1000
            
        self.debug(f"Function call: {func_name}", component, **details)
        
        if duration and duration > 1.0:  # Slow function (>1 second)
            self.performance_metric("function_duration", duration, "seconds", component,
                                   function=func_name)


# Global logger instance
_global_logger: Optional[SpintronLogger] = None


def get_logger(component: str = "general") -> logging.Logger:
    """Get logger instance for a component.
    
    Args:
        component: Component name
        
    Returns:
        Logger instance
    """
    global _global_logger
    if _global_logger is None:
        initialize_logging()
    
    # Return a child logger for the component
    return logging.getLogger(f"spintron_nn.{component}")


def initialize_logging(log_level: str = "INFO", log_dir: str = "logs", 
                      enable_console: bool = True) -> SpintronLogger:
    """Initialize global logging configuration.
    
    Args:
        log_level: Logging level
        log_dir: Log directory
        enable_console: Enable console output
        
    Returns:
        Configured logger instance
    """
    global _global_logger
    _global_logger = SpintronLogger("spintron_nn", log_level, log_dir, enable_console)
    return _global_logger


def log_performance(component: str):
    """Decorator to automatically log function performance.
    
    Args:
        component: Component name
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = datetime.utcnow()
            try:
                result = func(*args, **kwargs)
                duration = (datetime.utcnow() - start_time).total_seconds()
                
                logger = get_logger(component)
                if hasattr(logger, 'performance_metric'):
                    logger.performance_metric(
                        f"{func.__name__}_duration", duration, "seconds", component
                    )
                
                return result
            except Exception as e:
                duration = (datetime.utcnow() - start_time).total_seconds()
                logger = get_logger(component)
                logger.error(f"Function {func.__name__} failed", 
                           component=component, exception=e, duration=duration)
                raise
        return wrapper
    return decorator


# Initialize logging on module import
initialize_logging()