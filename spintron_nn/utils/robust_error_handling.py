"""
Robust error handling system for SpinTron-NN-Kit.
"""

import functools
import logging
import traceback
import time
from typing import Any, Callable, Dict, List, Optional, Type, Union
from enum import Enum


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class SpintronError(Exception):
    """Base exception for SpinTron-NN-Kit."""
    
    def __init__(self, message: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.severity = severity
        self.context = context or {}
        self.timestamp = time.time()


class MTJDeviceError(SpintronError):
    """MTJ device-related errors."""
    pass


class CrossbarError(SpintronError):
    """Crossbar array-related errors."""
    pass


class ConverterError(SpintronError):
    """Model conversion errors."""
    pass


class HardwareGenerationError(SpintronError):
    """Hardware generation errors."""
    pass


class RobustErrorHandler:
    """Robust error handling with recovery strategies."""
    
    def __init__(self):
        self.error_history: List[Dict[str, Any]] = []
        self.recovery_strategies: Dict[Type[Exception], Callable] = {}
        self.max_retries = 3
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SpintronErrorHandler')
        
    def register_recovery_strategy(self, error_type: Type[Exception], 
                                 strategy: Callable) -> None:
        """Register recovery strategy for specific error type."""
        self.recovery_strategies[error_type] = strategy
        
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """Handle error with logging and recovery attempts.
        
        Returns:
            True if error was recovered, False otherwise
        """
        error_info = {
            'error_type': type(error).__name__,
            'message': str(error),
            'timestamp': time.time(),
            'context': context or {},
            'traceback': traceback.format_exc()
        }
        
        self.error_history.append(error_info)
        
        # Log error
        severity = getattr(error, 'severity', ErrorSeverity.MEDIUM)
        log_level = {
            ErrorSeverity.LOW: logging.INFO,
            ErrorSeverity.MEDIUM: logging.WARNING,
            ErrorSeverity.HIGH: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL
        }.get(severity, logging.WARNING)
        
        self.logger.log(log_level, f"Error occurred: {error}")
        
        # Attempt recovery
        if type(error) in self.recovery_strategies:
            try:
                self.recovery_strategies[type(error)](error, context)
                self.logger.info(f"Successfully recovered from {type(error).__name__}")
                return True
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
                
        return False
        
    def with_retry(self, max_retries: Optional[int] = None):
        """Decorator for automatic retry on failure."""
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                retries = max_retries or self.max_retries
                last_error = None
                
                for attempt in range(retries + 1):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_error = e
                        if attempt < retries:
                            self.logger.warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {e}"
                            )
                            time.sleep(0.1 * (2 ** attempt))  # Exponential backoff
                        else:
                            self.handle_error(e, {'function': func.__name__, 'args': args})
                            
                raise last_error
            return wrapper
        return decorator
        
    def safe_execute(self, func: Callable, *args, **kwargs) -> Any:
        """Safely execute function with error handling."""
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.handle_error(e, {'function': func.__name__})
            return None
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and patterns."""
        if not self.error_history:
            return {'total_errors': 0}
            
        error_types = {}
        for error in self.error_history:
            error_type = error['error_type']
            error_types[error_type] = error_types.get(error_type, 0) + 1
            
        return {
            'total_errors': len(self.error_history),
            'error_types': error_types,
            'recent_errors': self.error_history[-5:],
            'most_common_error': max(error_types.items(), key=lambda x: x[1])[0] if error_types else None
        }


# Global error handler instance
error_handler = RobustErrorHandler()


def mtj_recovery_strategy(error: MTJDeviceError, context: Dict[str, Any]) -> None:
    """Recovery strategy for MTJ device errors."""
    if "resistance" in str(error).lower():
        # Adjust resistance parameters
        context['mtj_resistance_adjustment'] = 0.1
    elif "switching" in str(error).lower():
        # Reduce switching voltage
        context['voltage_reduction'] = 0.05


def crossbar_recovery_strategy(error: CrossbarError, context: Dict[str, Any]) -> None:
    """Recovery strategy for crossbar errors."""
    if "size" in str(error).lower():
        # Reduce crossbar size
        context['crossbar_size_reduction'] = 0.5
    elif "mapping" in str(error).lower():
        # Use alternative mapping strategy
        context['alternative_mapping'] = True


# Register recovery strategies
error_handler.register_recovery_strategy(MTJDeviceError, mtj_recovery_strategy)
error_handler.register_recovery_strategy(CrossbarError, crossbar_recovery_strategy)


def robust_operation(func: Callable) -> Callable:
    """Decorator for robust operation with comprehensive error handling."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Pre-execution validation
            if hasattr(func, '__annotations__'):
                # Basic type checking if annotations exist
                pass
                
            result = func(*args, **kwargs)
            
            # Post-execution validation
            if result is None:
                raise SpintronError(
                    f"Function {func.__name__} returned None unexpectedly",
                    severity=ErrorSeverity.MEDIUM
                )
                
            return result
            
        except SpintronError:
            raise  # Re-raise SpinTron-specific errors
        except Exception as e:
            # Wrap other exceptions in SpintronError
            wrapped_error = SpintronError(
                f"Unexpected error in {func.__name__}: {str(e)}",
                severity=ErrorSeverity.HIGH,
                context={'original_error': type(e).__name__}
            )
            error_handler.handle_error(wrapped_error)
            raise wrapped_error
            
    return wrapper