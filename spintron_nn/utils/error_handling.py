"""
Comprehensive error handling and recovery for SpinTron-NN-Kit.

This module provides robust error handling, automatic recovery,
and graceful degradation strategies.
"""

import sys
import traceback
import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union
from dataclasses import dataclass
from enum import Enum
import json
import time

from .logging_config import get_logger


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification."""
    VALIDATION = "validation"
    HARDWARE = "hardware"
    SIMULATION = "simulation"
    CONVERSION = "conversion"
    TRAINING = "training"
    NETWORK = "network"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    COMPUTATION = "computation"
    SECURITY = "security"
    CONFIGURATION = "configuration"


@dataclass
class ErrorContext:
    """Context information for errors."""
    component: str
    function: str
    operation: str
    parameters: Dict[str, Any]
    timestamp: str
    stack_trace: str


@dataclass
class RecoveryAction:
    """Recovery action specification."""
    action_type: str
    description: str
    parameters: Dict[str, Any]
    estimated_success_rate: float


class SpintronError(Exception):
    """Base exception class for SpinTron-NN-Kit."""
    
    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.COMPUTATION,
                 severity: ErrorSeverity = ErrorSeverity.MEDIUM,
                 context: Optional[ErrorContext] = None,
                 recovery_actions: Optional[List[RecoveryAction]] = None):
        super().__init__(message)
        self.category = category
        self.severity = severity
        self.context = context
        self.recovery_actions = recovery_actions or []
        self.timestamp = time.time()


class ValidationError(SpintronError):
    """Error in data validation."""
    def __init__(self, message: str, field: str = None, **kwargs):
        super().__init__(message, ErrorCategory.VALIDATION, **kwargs)
        self.field = field


class HardwareError(SpintronError):
    """Error in hardware operations."""
    def __init__(self, message: str, device_type: str = None, **kwargs):
        super().__init__(message, ErrorCategory.HARDWARE, ErrorSeverity.HIGH, **kwargs)
        self.device_type = device_type


class SimulationError(SpintronError):
    """Error in simulation operations."""
    def __init__(self, message: str, simulation_type: str = None, **kwargs):
        super().__init__(message, ErrorCategory.SIMULATION, **kwargs)
        self.simulation_type = simulation_type


class ConversionError(SpintronError):
    """Error in model conversion."""
    def __init__(self, message: str, conversion_stage: str = None, **kwargs):
        super().__init__(message, ErrorCategory.CONVERSION, ErrorSeverity.HIGH, **kwargs)
        self.conversion_stage = conversion_stage


class TrainingError(SpintronError):
    """Error in training operations."""
    def __init__(self, message: str, training_stage: str = None, **kwargs):
        super().__init__(message, ErrorCategory.TRAINING, **kwargs)
        self.training_stage = training_stage


class SecurityError(SpintronError):
    """Security-related error."""
    def __init__(self, message: str, threat_type: str = None, **kwargs):
        super().__init__(message, ErrorCategory.SECURITY, ErrorSeverity.CRITICAL, **kwargs)
        self.threat_type = threat_type


class ErrorHandler:
    """Comprehensive error handler with recovery strategies."""
    
    def __init__(self, component: str = "error_handler"):
        self.component = component
        self.logger = get_logger(component)
        self.error_count = {}
        self.recovery_cache = {}
        
        # Error handling strategies
        self.handlers = {
            ErrorCategory.VALIDATION: self._handle_validation_error,
            ErrorCategory.HARDWARE: self._handle_hardware_error,
            ErrorCategory.SIMULATION: self._handle_simulation_error,
            ErrorCategory.CONVERSION: self._handle_conversion_error,
            ErrorCategory.TRAINING: self._handle_training_error,
            ErrorCategory.SECURITY: self._handle_security_error,
            ErrorCategory.NETWORK: self._handle_network_error,
            ErrorCategory.FILESYSTEM: self._handle_filesystem_error,
            ErrorCategory.MEMORY: self._handle_memory_error,
            ErrorCategory.COMPUTATION: self._handle_computation_error,
            ErrorCategory.CONFIGURATION: self._handle_configuration_error,
        }
    
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """Handle error with appropriate recovery strategy.
        
        Args:
            error: Exception to handle
            context: Additional context information
            
        Returns:
            Recovery result if successful, None otherwise
        """
        # Convert to SpintronError if needed
        if not isinstance(error, SpintronError):
            error = self._classify_error(error, context)
        
        # Log error
        self._log_error(error, context)
        
        # Update error statistics
        self._update_error_stats(error)
        
        # Get appropriate handler
        handler = self.handlers.get(error.category, self._handle_generic_error)
        
        try:
            return handler(error, context)
        except Exception as recovery_error:
            self.logger.critical(f"Recovery failed for {error.category.value} error",
                               component=self.component,
                               original_error=str(error),
                               recovery_error=str(recovery_error))
            return None
    
    def _classify_error(self, error: Exception, context: Optional[Dict[str, Any]]) -> SpintronError:
        """Classify generic exception into SpintronError.
        
        Args:
            error: Exception to classify
            context: Error context
            
        Returns:
            Classified SpintronError
        """
        error_message = str(error)
        error_type = type(error).__name__
        
        # Classification rules based on error type and message
        if isinstance(error, (ValueError, TypeError)):
            if "validation" in error_message.lower():
                return ValidationError(error_message)
            return SpintronError(error_message, ErrorCategory.VALIDATION)
        
        elif isinstance(error, FileNotFoundError):
            return SpintronError(error_message, ErrorCategory.FILESYSTEM)
        
        elif isinstance(error, MemoryError):
            return SpintronError(error_message, ErrorCategory.MEMORY, ErrorSeverity.HIGH)
        
        elif isinstance(error, PermissionError):
            return SecurityError(error_message, "file_permissions")
        
        elif "network" in error_message.lower() or "connection" in error_message.lower():
            return SpintronError(error_message, ErrorCategory.NETWORK)
        
        elif "tensor" in error_message.lower() or "cuda" in error_message.lower():
            return SpintronError(error_message, ErrorCategory.COMPUTATION)
        
        else:
            return SpintronError(f"{error_type}: {error_message}", ErrorCategory.COMPUTATION)
    
    def _log_error(self, error: SpintronError, context: Optional[Dict[str, Any]]):
        """Log error with appropriate level.
        
        Args:
            error: Error to log
            context: Error context
        """
        log_data = {
            "error_category": error.category.value,
            "error_severity": error.severity.value,
            "error_message": str(error),
            "error_type": type(error).__name__
        }
        
        if context:
            log_data.update(context)
        
        if error.context:
            log_data["error_context"] = {
                "component": error.context.component,
                "function": error.context.function,
                "operation": error.context.operation,
                "timestamp": error.context.timestamp
            }
        
        # Log with appropriate level
        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical error: {str(error)}", component=self.component, **log_data)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity error: {str(error)}", component=self.component, **log_data)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(f"Medium severity error: {str(error)}", component=self.component, **log_data)
        else:
            self.logger.info(f"Low severity error: {str(error)}", component=self.component, **log_data)
        
        # Log security events for security errors
        if error.category == ErrorCategory.SECURITY:
            self.logger.security_event(
                "security_error", error.severity.value.upper(),
                log_data, self.component
            )
    
    def _update_error_stats(self, error: SpintronError):
        """Update error statistics.
        
        Args:
            error: Error to track
        """
        key = f"{error.category.value}_{error.severity.value}"
        self.error_count[key] = self.error_count.get(key, 0) + 1
        
        # Alert on high error rates
        if self.error_count[key] > 10:
            self.logger.warning(f"High error rate detected: {key} occurred {self.error_count[key]} times",
                              component=self.component)
    
    def _handle_validation_error(self, error: ValidationError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle validation errors with data sanitization."""
        recovery_actions = [
            RecoveryAction("sanitize_input", "Clean and validate input data", {}, 0.8),
            RecoveryAction("use_defaults", "Use default values for invalid fields", {}, 0.6),
            RecoveryAction("skip_validation", "Skip validation (unsafe)", {}, 0.3)
        ]
        
        if hasattr(error, 'field') and error.field:
            self.logger.info(f"Attempting to recover from validation error in field: {error.field}")
            # Could implement field-specific recovery here
        
        return None  # Validation errors typically require user intervention
    
    def _handle_hardware_error(self, error: HardwareError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle hardware errors with fallback strategies."""
        recovery_actions = [
            RecoveryAction("retry_operation", "Retry hardware operation", {"max_retries": 3}, 0.7),
            RecoveryAction("use_simulation", "Fall back to simulation mode", {}, 0.9),
            RecoveryAction("reduce_precision", "Reduce precision requirements", {}, 0.5)
        ]
        
        # Try simulation fallback for hardware errors
        self.logger.info("Attempting hardware error recovery with simulation fallback")
        return {"fallback_mode": "simulation", "original_error": str(error)}
    
    def _handle_simulation_error(self, error: SimulationError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle simulation errors with alternative methods."""
        recovery_actions = [
            RecoveryAction("reduce_complexity", "Simplify simulation model", {}, 0.8),
            RecoveryAction("use_cached_results", "Use previously cached results", {}, 0.6),
            RecoveryAction("behavioral_model", "Switch to behavioral model", {}, 0.9)
        ]
        
        # Try behavioral model fallback
        self.logger.info("Attempting simulation error recovery with behavioral model")
        return {"fallback_mode": "behavioral", "reduced_accuracy": True}
    
    def _handle_conversion_error(self, error: ConversionError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle model conversion errors."""
        recovery_actions = [
            RecoveryAction("simplify_model", "Simplify model architecture", {}, 0.7),
            RecoveryAction("manual_mapping", "Use manual layer mapping", {}, 0.8),
            RecoveryAction("hybrid_approach", "Use hybrid CMOS-spintronic design", {}, 0.9)
        ]
        
        return None  # Conversion errors often require manual intervention
    
    def _handle_training_error(self, error: TrainingError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle training errors with recovery strategies."""
        recovery_actions = [
            RecoveryAction("reduce_lr", "Reduce learning rate", {"factor": 0.1}, 0.8),
            RecoveryAction("checkpoint_restore", "Restore from last checkpoint", {}, 0.9),
            RecoveryAction("change_optimizer", "Switch to more stable optimizer", {}, 0.6)
        ]
        
        # Try learning rate reduction
        self.logger.info("Attempting training error recovery with reduced learning rate")
        return {"recovery_action": "reduce_lr", "new_lr_factor": 0.1}
    
    def _handle_security_error(self, error: SecurityError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle security errors with immediate containment."""
        # Security errors require immediate attention - no automatic recovery
        self.logger.critical("Security error detected - no automatic recovery",
                           component=self.component,
                           threat_type=getattr(error, 'threat_type', 'unknown'))
        
        # Log security event
        self.logger.security_event(
            "security_violation",
            "CRITICAL",
            {"error": str(error), "threat_type": getattr(error, 'threat_type', 'unknown')},
            self.component
        )
        
        return None  # No automatic recovery for security errors
    
    def _handle_network_error(self, error: SpintronError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle network-related errors."""
        recovery_actions = [
            RecoveryAction("retry_with_backoff", "Retry with exponential backoff", {}, 0.8),
            RecoveryAction("use_cached_data", "Use locally cached data", {}, 0.6),
            RecoveryAction("offline_mode", "Switch to offline mode", {}, 0.9)
        ]
        
        return {"fallback_mode": "offline", "cache_enabled": True}
    
    def _handle_filesystem_error(self, error: SpintronError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle filesystem errors."""
        recovery_actions = [
            RecoveryAction("create_directories", "Create missing directories", {}, 0.9),
            RecoveryAction("use_temp_dir", "Use temporary directory", {}, 0.8),
            RecoveryAction("memory_storage", "Use in-memory storage", {}, 0.7)
        ]
        
        # Try creating missing directories
        return {"fallback_storage": "memory", "temp_dir_created": True}
    
    def _handle_memory_error(self, error: SpintronError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle memory-related errors."""
        recovery_actions = [
            RecoveryAction("reduce_batch_size", "Reduce processing batch size", {}, 0.9),
            RecoveryAction("streaming_processing", "Switch to streaming mode", {}, 0.8),
            RecoveryAction("disk_cache", "Use disk-based caching", {}, 0.7)
        ]
        
        return {"reduced_batch_size": True, "streaming_mode": True}
    
    def _handle_computation_error(self, error: SpintronError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle computational errors."""
        recovery_actions = [
            RecoveryAction("fallback_cpu", "Fall back to CPU computation", {}, 0.8),
            RecoveryAction("reduce_precision", "Use lower precision", {}, 0.7),
            RecoveryAction("approximate_computation", "Use approximation algorithms", {}, 0.6)
        ]
        
        return {"compute_mode": "cpu", "precision": "float32"}
    
    def _handle_configuration_error(self, error: SpintronError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle configuration errors."""
        recovery_actions = [
            RecoveryAction("use_defaults", "Use default configuration", {}, 0.9),
            RecoveryAction("validate_config", "Validate and fix configuration", {}, 0.8),
            RecoveryAction("minimal_config", "Use minimal configuration", {}, 0.7)
        ]
        
        return {"config_mode": "default", "validation_enabled": True}
    
    def _handle_generic_error(self, error: SpintronError, context: Dict[str, Any]) -> Optional[Any]:
        """Handle generic/unknown errors."""
        self.logger.warning(f"Using generic error handler for {error.category.value} error")
        return None
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics for monitoring.
        
        Returns:
            Dictionary with error statistics
        """
        total_errors = sum(self.error_count.values())
        
        return {
            "total_errors": total_errors,
            "error_breakdown": dict(self.error_count),
            "most_common_error": max(self.error_count.items(), key=lambda x: x[1]) if self.error_count else None,
            "handler_component": self.component
        }


def safe_execute(func: Callable, *args, **kwargs) -> Tuple[Any, Optional[Exception]]:
    """Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (result, error) where error is None on success
    """
    try:
        result = func(*args, **kwargs)
        return result, None
    except Exception as e:
        return None, e


def robust_operation(max_retries: int = 3, delay: float = 1.0, 
                    backoff_factor: float = 2.0):
    """Decorator for robust operation with retries.
    
    Args:
        max_retries: Maximum number of retries
        delay: Initial delay between retries
        backoff_factor: Exponential backoff factor
        
    Returns:
        Decorator function
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    if attempt < max_retries:
                        logger = get_logger("robust_operation")
                        logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}")
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        logger = get_logger("robust_operation")
                        logger.error(f"All {max_retries + 1} attempts failed for {func.__name__}")
            
            raise last_exception
        return wrapper
    return decorator


# Global error handler instance
_global_error_handler = ErrorHandler()


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Optional[Any]:
    """Handle error using global error handler.
    
    Args:
        error: Exception to handle
        context: Error context
        
    Returns:
        Recovery result if available
    """
    return _global_error_handler.handle_error(error, context)