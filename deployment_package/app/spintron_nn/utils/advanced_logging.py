"""
Advanced logging system for SpinTron-NN-Kit with structured logging and performance tracking.
"""

import json
import logging
import time
import traceback
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict
from contextlib import contextmanager
from enum import Enum


class LogLevel(Enum):
    """Enhanced log levels."""
    TRACE = 5
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


@dataclass
class PerformanceMetrics:
    """Performance metrics for logging."""
    execution_time_ms: float
    memory_usage_mb: float
    cpu_utilization: float
    operation_count: int
    
    
@dataclass
class SpintronLogEntry:
    """Structured log entry for SpinTron operations."""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    line_number: int
    context: Dict[str, Any]
    performance_metrics: Optional[PerformanceMetrics] = None
    stack_trace: Optional[str] = None


class SpintronLogger:
    """Advanced logger for SpinTron-NN-Kit operations."""
    
    def __init__(self, name: str = "SpintronNN", log_dir: str = "logs"):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create loggers for different purposes
        self.general_logger = self._create_logger("general")
        self.performance_logger = self._create_logger("performance")
        self.error_logger = self._create_logger("error")
        self.security_logger = self._create_logger("security")
        
        # Performance tracking
        self.operation_stack: List[Dict[str, Any]] = []
        self.performance_history: List[PerformanceMetrics] = []
        
        # Context stack for nested operations
        self.context_stack: List[Dict[str, Any]] = []
        
    def _create_logger(self, logger_type: str) -> logging.Logger:
        """Create specialized logger with appropriate formatting."""
        logger = logging.getLogger(f"{self.name}.{logger_type}")
        logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers
        logger.handlers.clear()
        
        # File handler with JSON formatting
        log_file = self.log_dir / f"{logger_type}.jsonl"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler with human-readable formatting
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Custom formatters
        json_formatter = self._create_json_formatter()
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        file_handler.setFormatter(json_formatter)
        console_handler.setFormatter(console_formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
        
    def _create_json_formatter(self):
        """Create JSON formatter for structured logging."""
        class JsonFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    'timestamp': datetime.fromtimestamp(record.created).isoformat(),
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function': record.funcName,
                    'line_number': record.lineno,
                    'logger_name': record.name
                }
                
                # Add context if available
                if hasattr(record, 'context'):
                    log_entry['context'] = record.context
                    
                # Add performance metrics if available
                if hasattr(record, 'performance_metrics'):
                    log_entry['performance_metrics'] = asdict(record.performance_metrics)
                    
                # Add stack trace for errors
                if record.exc_info:
                    log_entry['stack_trace'] = self.formatException(record.exc_info)
                    
                return json.dumps(log_entry)
                
        return JsonFormatter()
        
    @contextmanager
    def operation_context(self, operation_name: str, **context):
        """Context manager for tracking operation performance."""
        start_time = time.time()
        start_context = {
            'operation': operation_name,
            'start_time': start_time,
            **context
        }
        
        self.operation_stack.append(start_context)
        self.context_stack.append(context)
        
        try:
            self.info(f"Starting operation: {operation_name}", context=context)
            yield
            
        except Exception as e:
            self.error(f"Operation failed: {operation_name}", 
                      context={'error': str(e), **context},
                      exc_info=True)
            raise
            
        finally:
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to ms
            
            # Calculate performance metrics
            metrics = PerformanceMetrics(
                execution_time_ms=execution_time,
                memory_usage_mb=self._get_memory_usage(),
                cpu_utilization=0.0,  # Would need psutil for accurate measurement
                operation_count=1
            )
            
            self.performance_history.append(metrics)
            
            self.info(f"Completed operation: {operation_name}",
                     context={'execution_time_ms': execution_time, **context},
                     performance_metrics=metrics)
            
            self.operation_stack.pop()
            self.context_stack.pop()
            
    def _get_memory_usage(self) -> float:
        """Get current memory usage (simplified)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return 0.0  # Return 0 if psutil not available
            
    def _get_current_context(self) -> Dict[str, Any]:
        """Get current context from stack."""
        if self.context_stack:
            return self.context_stack[-1]
        return {}
        
    def trace(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log trace level message."""
        self._log(LogLevel.TRACE, message, context, **kwargs)
        
    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug level message."""
        self._log(LogLevel.DEBUG, message, context, **kwargs)
        
    def info(self, message: str, context: Optional[Dict[str, Any]] = None, 
             performance_metrics: Optional[PerformanceMetrics] = None, **kwargs):
        """Log info level message."""
        self._log(LogLevel.INFO, message, context, performance_metrics=performance_metrics, **kwargs)
        
    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning level message."""
        self._log(LogLevel.WARNING, message, context, **kwargs)
        
    def error(self, message: str, context: Optional[Dict[str, Any]] = None, 
              exc_info: bool = False, **kwargs):
        """Log error level message."""
        self._log(LogLevel.ERROR, message, context, exc_info=exc_info, **kwargs)
        
    def critical(self, message: str, context: Optional[Dict[str, Any]] = None, 
                 exc_info: bool = False, **kwargs):
        """Log critical level message."""
        self._log(LogLevel.CRITICAL, message, context, exc_info=exc_info, **kwargs)
        
    def _log(self, level: LogLevel, message: str, context: Optional[Dict[str, Any]] = None,
             performance_metrics: Optional[PerformanceMetrics] = None,
             exc_info: bool = False, **kwargs):
        """Internal logging method."""
        # Combine provided context with current context
        full_context = {**self._get_current_context(), **(context or {})}
        
        # Select appropriate logger
        logger = self.general_logger
        if level in [LogLevel.ERROR, LogLevel.CRITICAL]:
            logger = self.error_logger
        elif performance_metrics:
            logger = self.performance_logger
            
        # Create log record with extra attributes
        extra = {
            'context': full_context,
            'performance_metrics': performance_metrics
        }
        
        # Map our log levels to standard logging levels
        logging_level = {
            LogLevel.TRACE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }[level]
        
        logger.log(logging_level, message, extra=extra, exc_info=exc_info)
        
    def log_mtj_operation(self, operation: str, mtj_params: Dict[str, Any], 
                         result: Any = None, error: Optional[Exception] = None):
        """Specialized logging for MTJ operations."""
        context = {
            'operation_type': 'mtj_operation',
            'mtj_parameters': mtj_params,
            'result_type': type(result).__name__ if result else None
        }
        
        if error:
            self.error(f"MTJ operation failed: {operation}", 
                      context={**context, 'error': str(error)},
                      exc_info=True)
        else:
            self.info(f"MTJ operation completed: {operation}", context=context)
            
    def log_crossbar_operation(self, operation: str, crossbar_config: Dict[str, Any],
                              performance_data: Optional[Dict[str, float]] = None):
        """Specialized logging for crossbar operations."""
        context = {
            'operation_type': 'crossbar_operation',
            'crossbar_configuration': crossbar_config
        }
        
        if performance_data:
            context['performance_data'] = performance_data
            
        self.info(f"Crossbar operation: {operation}", context=context)
        
    def log_conversion_step(self, step: str, input_info: Dict[str, Any], 
                           output_info: Dict[str, Any]):
        """Specialized logging for model conversion steps."""
        context = {
            'operation_type': 'conversion_step',
            'input_info': input_info,
            'output_info': output_info
        }
        
        self.info(f"Conversion step: {step}", context=context)
        
    def log_security_event(self, event_type: str, details: Dict[str, Any], 
                          severity: str = "info"):
        """Log security-related events."""
        context = {
            'event_type': 'security_event',
            'security_event_type': event_type,
            'severity': severity,
            'details': details
        }
        
        if severity == "critical":
            self.critical(f"Security event: {event_type}", context=context)
        elif severity == "error":
            self.error(f"Security event: {event_type}", context=context)
        elif severity == "warning":
            self.warning(f"Security event: {event_type}", context=context)
        else:
            self.info(f"Security event: {event_type}", context=context)
            
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from logged metrics."""
        if not self.performance_history:
            return {'no_performance_data': True}
            
        execution_times = [m.execution_time_ms for m in self.performance_history]
        memory_usage = [m.memory_usage_mb for m in self.performance_history]
        
        return {
            'total_operations': len(self.performance_history),
            'avg_execution_time_ms': sum(execution_times) / len(execution_times),
            'max_execution_time_ms': max(execution_times),
            'min_execution_time_ms': min(execution_times),
            'avg_memory_usage_mb': sum(memory_usage) / len(memory_usage) if memory_usage else 0,
            'max_memory_usage_mb': max(memory_usage) if memory_usage else 0
        }
        
    def export_logs(self, format_type: str = "json") -> str:
        """Export logs in specified format."""
        export_file = self.log_dir / f"export_{int(time.time())}.{format_type}"
        
        if format_type == "json":
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'performance_summary': self.get_performance_summary(),
                'log_files': [str(f) for f in self.log_dir.glob("*.jsonl")]
            }
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2)
                
        return str(export_file)


# Global logger instance
spintron_logger = SpintronLogger()


def log_operation(operation_name: str):
    """Decorator for automatic operation logging."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with spintron_logger.operation_context(operation_name, 
                                                  function=func.__name__,
                                                  args_count=len(args),
                                                  kwargs_keys=list(kwargs.keys())):
                return func(*args, **kwargs)
        return wrapper
    return decorator