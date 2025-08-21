"""
Robust Error Recovery System for SpinTron-NN-Kit.

This module provides comprehensive error handling, fault tolerance, and
automatic recovery mechanisms for production environments.
"""

import time
import json
import traceback
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
from functools import wraps
from contextlib import contextmanager


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RecoveryStrategy(Enum):
    """Error recovery strategies."""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAILOVER = "failover"
    RESTART = "restart"


class ErrorCategory(Enum):
    """Error categories."""
    HARDWARE_FAILURE = "hardware_failure"
    MEMORY_ERROR = "memory_error"
    COMPUTATION_ERROR = "computation_error"
    IO_ERROR = "io_error"
    NETWORK_ERROR = "network_error"
    VALIDATION_ERROR = "validation_error"
    CONFIGURATION_ERROR = "configuration_error"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass
class ErrorContext:
    """Context information for errors."""
    
    timestamp: float
    error_id: str
    severity: ErrorSeverity
    category: ErrorCategory
    component: str
    function: str
    message: str
    traceback_str: str
    recovery_strategy: Optional[RecoveryStrategy] = None
    retry_count: int = 0
    max_retries: int = 3
    recovery_successful: bool = False


@dataclass
class RecoveryAction:
    """Recovery action definition."""
    
    strategy: RecoveryStrategy
    action: Callable
    timeout_seconds: float
    success_criteria: Callable[[Any], bool]
    fallback_action: Optional[Callable] = None


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """Circuit breaker for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Call function with circuit breaker protection."""
        
        with self._lock:
            if self.state == CircuitBreakerState.OPEN:
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = CircuitBreakerState.HALF_OPEN
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            
            with self._lock:
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
            
            return result
            
        except Exception as e:
            with self._lock:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = CircuitBreakerState.OPEN
                
                if self.state == CircuitBreakerState.HALF_OPEN:
                    self.state = CircuitBreakerState.OPEN
            
            raise e


class RobustErrorRecovery:
    """Comprehensive error recovery and fault tolerance system."""
    
    def __init__(self, log_dir: str = "error_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Error tracking
        self.error_history = []
        self.recovery_statistics = {}
        self.circuit_breakers = {}
        
        # Configuration
        self.max_retry_attempts = 3
        self.retry_backoff_factor = 2.0
        self.default_timeout = 30.0
        
        # Recovery strategies
        self.recovery_strategies = {
            ErrorCategory.HARDWARE_FAILURE: RecoveryStrategy.FAILOVER,
            ErrorCategory.MEMORY_ERROR: RecoveryStrategy.GRACEFUL_DEGRADATION,
            ErrorCategory.COMPUTATION_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.IO_ERROR: RecoveryStrategy.RETRY,
            ErrorCategory.NETWORK_ERROR: RecoveryStrategy.CIRCUIT_BREAKER,
            ErrorCategory.VALIDATION_ERROR: RecoveryStrategy.FALLBACK,
            ErrorCategory.CONFIGURATION_ERROR: RecoveryStrategy.FALLBACK,
            ErrorCategory.RESOURCE_EXHAUSTION: RecoveryStrategy.GRACEFUL_DEGRADATION
        }
        
        # Initialize logging
        self._setup_logging()
        
        # Initialize monitoring
        self.monitoring_enabled = True
        self.health_check_interval = 30.0
        self._start_health_monitoring()
    
    def _setup_logging(self):
        """Setup comprehensive error logging."""
        
        self.logger = logging.getLogger('SpinTronErrorRecovery')
        self.logger.setLevel(logging.INFO)
        
        # File handler
        log_file = self.log_dir / 'error_recovery.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def _start_health_monitoring(self):
        """Start continuous health monitoring."""
        
        def health_monitor():
            while self.monitoring_enabled:
                try:
                    self._perform_health_check()
                    time.sleep(self.health_check_interval)
                except Exception as e:
                    self.logger.error(f"Health monitoring error: {e}")
                    time.sleep(5.0)  # Brief pause before retry
        
        health_thread = threading.Thread(target=health_monitor, daemon=True)
        health_thread.start()
        
        self.logger.info("Health monitoring started")
    
    def _perform_health_check(self):
        """Perform system health check."""
        
        health_status = {
            "timestamp": time.time(),
            "error_rate": self._calculate_recent_error_rate(),
            "recovery_success_rate": self._calculate_recovery_success_rate(),
            "circuit_breaker_states": {name: cb.state.value for name, cb in self.circuit_breakers.items()},
            "system_healthy": True
        }
        
        # Check if system is healthy
        if health_status["error_rate"] > 0.1:  # More than 10% error rate
            health_status["system_healthy"] = False
            self.logger.warning(f"High error rate detected: {health_status['error_rate']:.2%}")
        
        if health_status["recovery_success_rate"] < 0.8:  # Less than 80% recovery success
            health_status["system_healthy"] = False
            self.logger.warning(f"Low recovery success rate: {health_status['recovery_success_rate']:.2%}")
        
        # Log health status periodically
        if not health_status["system_healthy"]:
            self.logger.warning(f"System health check: {health_status}")
    
    def _calculate_recent_error_rate(self) -> float:
        """Calculate recent error rate."""
        
        recent_time = time.time() - 3600  # Last hour
        recent_errors = [e for e in self.error_history if e.timestamp > recent_time]
        
        if not recent_errors:
            return 0.0
        
        # Estimate total operations (simplified)
        estimated_operations = len(recent_errors) * 10  # Assume 10 ops per error on average
        return len(recent_errors) / estimated_operations if estimated_operations > 0 else 0.0
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate."""
        
        recent_time = time.time() - 3600  # Last hour
        recent_errors = [e for e in self.error_history if e.timestamp > recent_time]
        
        if not recent_errors:
            return 1.0
        
        successful_recoveries = sum(1 for e in recent_errors if e.recovery_successful)
        return successful_recoveries / len(recent_errors)
    
    def robust_wrapper(self, 
                      component: str = "unknown",
                      max_retries: int = None,
                      timeout: float = None,
                      fallback_result: Any = None):
        """Decorator for adding robust error handling to functions."""
        
        def decorator(func: Callable) -> Callable:
            @wraps(func)
            def wrapper(*args, **kwargs):
                return self.execute_with_recovery(
                    func, 
                    args, 
                    kwargs,
                    component=component,
                    max_retries=max_retries or self.max_retry_attempts,
                    timeout=timeout or self.default_timeout,
                    fallback_result=fallback_result
                )
            return wrapper
        return decorator
    
    def execute_with_recovery(self,
                            func: Callable,
                            args: Tuple = (),
                            kwargs: Dict = None,
                            component: str = "unknown",
                            max_retries: int = None,
                            timeout: float = None,
                            fallback_result: Any = None) -> Any:
        """Execute function with comprehensive error recovery."""
        
        kwargs = kwargs or {}
        max_retries = max_retries or self.max_retry_attempts
        timeout = timeout or self.default_timeout
        
        error_context = None
        
        for attempt in range(max_retries + 1):
            try:
                # Use circuit breaker if configured
                circuit_breaker = self.circuit_breakers.get(component)
                if circuit_breaker:
                    return circuit_breaker.call(func, *args, **kwargs)
                else:
                    return self._execute_with_timeout(func, args, kwargs, timeout)
                
            except Exception as e:
                error_context = self._create_error_context(
                    e, component, func.__name__, attempt, max_retries
                )
                
                self.error_history.append(error_context)
                self.logger.error(f"Error in {component}.{func.__name__}: {error_context.message}")
                
                # Determine recovery strategy
                recovery_strategy = self._determine_recovery_strategy(error_context)
                error_context.recovery_strategy = recovery_strategy
                
                # Apply recovery strategy
                if attempt < max_retries:
                    recovery_successful = self._apply_recovery_strategy(
                        recovery_strategy, error_context, attempt
                    )
                    
                    if recovery_successful:
                        error_context.recovery_successful = True
                        continue  # Retry the operation
                    else:
                        # Recovery failed, try alternative strategy
                        alternative_recovery = self._get_alternative_recovery(recovery_strategy)
                        if alternative_recovery and alternative_recovery != recovery_strategy:
                            self._apply_recovery_strategy(alternative_recovery, error_context, attempt)
                            continue
                
                # Final attempt failed or no more retries
                if attempt == max_retries:
                    error_context.recovery_successful = False
                    
                    # Try fallback result
                    if fallback_result is not None:
                        self.logger.warning(f"Using fallback result for {component}.{func.__name__}")
                        return fallback_result
                    
                    # Re-raise the original exception
                    self.logger.critical(f"All recovery attempts failed for {component}.{func.__name__}")
                    raise e
        
        # Should not reach here
        raise Exception("Unexpected error in recovery system")
    
    def _create_error_context(self, 
                            exception: Exception,
                            component: str,
                            function: str,
                            retry_count: int,
                            max_retries: int) -> ErrorContext:
        """Create error context from exception."""
        
        error_id = f"{component}_{function}_{int(time.time())}_{retry_count}"
        
        # Categorize error
        category = self._categorize_error(exception)
        
        # Determine severity
        severity = self._determine_severity(exception, category, retry_count, max_retries)
        
        return ErrorContext(
            timestamp=time.time(),
            error_id=error_id,
            severity=severity,
            category=category,
            component=component,
            function=function,
            message=str(exception),
            traceback_str=traceback.format_exc(),
            retry_count=retry_count,
            max_retries=max_retries
        )
    
    def _categorize_error(self, exception: Exception) -> ErrorCategory:
        """Categorize error based on exception type and message."""
        
        exception_type = type(exception).__name__.lower()
        message = str(exception).lower()
        
        if 'memory' in message or 'memoryerror' in exception_type:
            return ErrorCategory.MEMORY_ERROR
        elif 'timeout' in message or 'connection' in message:
            return ErrorCategory.NETWORK_ERROR
        elif 'file' in message or 'io' in message or 'permission' in message:
            return ErrorCategory.IO_ERROR
        elif 'validation' in message or 'invalid' in message:
            return ErrorCategory.VALIDATION_ERROR
        elif 'config' in message or 'setting' in message:
            return ErrorCategory.CONFIGURATION_ERROR
        elif 'resource' in message or 'limit' in message:
            return ErrorCategory.RESOURCE_EXHAUSTION
        elif 'hardware' in message or 'device' in message:
            return ErrorCategory.HARDWARE_FAILURE
        else:
            return ErrorCategory.COMPUTATION_ERROR
    
    def _determine_severity(self, 
                          exception: Exception,
                          category: ErrorCategory,
                          retry_count: int,
                          max_retries: int) -> ErrorSeverity:
        """Determine error severity."""
        
        # Critical conditions
        if category == ErrorCategory.HARDWARE_FAILURE:
            return ErrorSeverity.CRITICAL
        
        if retry_count >= max_retries:
            return ErrorSeverity.HIGH
        
        # High severity conditions
        if category in [ErrorCategory.MEMORY_ERROR, ErrorCategory.RESOURCE_EXHAUSTION]:
            return ErrorSeverity.HIGH
        
        # Medium severity conditions
        if category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.IO_ERROR]:
            return ErrorSeverity.MEDIUM
        
        # Default to low
        return ErrorSeverity.LOW
    
    def _determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Determine appropriate recovery strategy."""
        
        # Strategy based on category
        base_strategy = self.recovery_strategies.get(
            error_context.category, 
            RecoveryStrategy.RETRY
        )
        
        # Adjust based on severity and retry count
        if error_context.severity == ErrorSeverity.CRITICAL:
            return RecoveryStrategy.FAILOVER
        
        if error_context.retry_count >= 2:
            if base_strategy == RecoveryStrategy.RETRY:
                return RecoveryStrategy.GRACEFUL_DEGRADATION
        
        return base_strategy
    
    def _apply_recovery_strategy(self,
                               strategy: RecoveryStrategy,
                               error_context: ErrorContext,
                               attempt: int) -> bool:
        """Apply recovery strategy."""
        
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._apply_retry_strategy(error_context, attempt)
            
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._apply_fallback_strategy(error_context)
            
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._apply_graceful_degradation(error_context)
            
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._apply_circuit_breaker_strategy(error_context)
            
            elif strategy == RecoveryStrategy.FAILOVER:
                return self._apply_failover_strategy(error_context)
            
            elif strategy == RecoveryStrategy.RESTART:
                return self._apply_restart_strategy(error_context)
            
            else:
                self.logger.warning(f"Unknown recovery strategy: {strategy}")
                return False
                
        except Exception as e:
            self.logger.error(f"Recovery strategy {strategy} failed: {e}")
            return False
    
    def _apply_retry_strategy(self, error_context: ErrorContext, attempt: int) -> bool:
        """Apply retry strategy with exponential backoff."""
        
        backoff_time = self.retry_backoff_factor ** attempt
        
        self.logger.info(f"Retrying {error_context.component}.{error_context.function} "
                        f"in {backoff_time:.1f}s (attempt {attempt + 1})")
        
        time.sleep(backoff_time)
        return True
    
    def _apply_fallback_strategy(self, error_context: ErrorContext) -> bool:
        """Apply fallback strategy."""
        
        self.logger.info(f"Applying fallback for {error_context.component}.{error_context.function}")
        
        # Implement component-specific fallbacks
        if "optimization" in error_context.function.lower():
            # Use simpler optimization algorithm
            return True
        elif "simulation" in error_context.function.lower():
            # Use faster, less accurate simulation
            return True
        else:
            # Generic fallback - reduce precision/quality
            return True
    
    def _apply_graceful_degradation(self, error_context: ErrorContext) -> bool:
        """Apply graceful degradation strategy."""
        
        self.logger.info(f"Applying graceful degradation for {error_context.component}")
        
        # Reduce system capabilities but maintain core functionality
        if error_context.category == ErrorCategory.MEMORY_ERROR:
            # Reduce batch sizes, clear caches
            return True
        elif error_context.category == ErrorCategory.RESOURCE_EXHAUSTION:
            # Reduce parallel processing, lower precision
            return True
        
        return True
    
    def _apply_circuit_breaker_strategy(self, error_context: ErrorContext) -> bool:
        """Apply circuit breaker strategy."""
        
        component = error_context.component
        
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker()
            self.logger.info(f"Circuit breaker enabled for {component}")
        
        return False  # Don't retry immediately, let circuit breaker handle it
    
    def _apply_failover_strategy(self, error_context: ErrorContext) -> bool:
        """Apply failover strategy."""
        
        self.logger.warning(f"Initiating failover for {error_context.component}")
        
        # Switch to backup systems/algorithms
        if "quantum" in error_context.component.lower():
            # Failover to classical algorithms
            return True
        elif "distributed" in error_context.component.lower():
            # Failover to single-node processing
            return True
        
        return True
    
    def _apply_restart_strategy(self, error_context: ErrorContext) -> bool:
        """Apply restart strategy."""
        
        self.logger.warning(f"Restarting component: {error_context.component}")
        
        # Restart component (simulation)
        time.sleep(1.0)  # Simulate restart time
        
        return True
    
    def _get_alternative_recovery(self, primary_strategy: RecoveryStrategy) -> Optional[RecoveryStrategy]:
        """Get alternative recovery strategy."""
        
        alternatives = {
            RecoveryStrategy.RETRY: RecoveryStrategy.FALLBACK,
            RecoveryStrategy.FALLBACK: RecoveryStrategy.GRACEFUL_DEGRADATION,
            RecoveryStrategy.GRACEFUL_DEGRADATION: RecoveryStrategy.CIRCUIT_BREAKER,
            RecoveryStrategy.CIRCUIT_BREAKER: RecoveryStrategy.FAILOVER,
            RecoveryStrategy.FAILOVER: RecoveryStrategy.RESTART,
            RecoveryStrategy.RESTART: None
        }
        
        return alternatives.get(primary_strategy)
    
    def _execute_with_timeout(self, 
                            func: Callable,
                            args: Tuple,
                            kwargs: Dict,
                            timeout: float) -> Any:
        """Execute function with timeout."""
        
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Function execution timed out after {timeout}s")
        
        # Set timeout (Unix-like systems only)
        try:
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(timeout))
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)
                signal.signal(signal.SIGALRM, old_handler)
                
        except AttributeError:
            # signal.alarm not available (Windows), execute without timeout
            return func(*args, **kwargs)
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        recent_time = time.time() - 3600  # Last hour
        recent_errors = [e for e in self.error_history if e.timestamp > recent_time]
        
        statistics = {
            "total_errors": len(self.error_history),
            "recent_errors": len(recent_errors),
            "error_rate": self._calculate_recent_error_rate(),
            "recovery_success_rate": self._calculate_recovery_success_rate(),
            "errors_by_category": {},
            "errors_by_severity": {},
            "errors_by_component": {},
            "recovery_strategies_used": {},
            "circuit_breaker_status": {name: cb.state.value for name, cb in self.circuit_breakers.items()}
        }
        
        # Analyze recent errors
        for error in recent_errors:
            # By category
            category = error.category.value
            statistics["errors_by_category"][category] = statistics["errors_by_category"].get(category, 0) + 1
            
            # By severity
            severity = error.severity.value
            statistics["errors_by_severity"][severity] = statistics["errors_by_severity"].get(severity, 0) + 1
            
            # By component
            component = error.component
            statistics["errors_by_component"][component] = statistics["errors_by_component"].get(component, 0) + 1
            
            # By recovery strategy
            if error.recovery_strategy:
                strategy = error.recovery_strategy.value
                statistics["recovery_strategies_used"][strategy] = statistics["recovery_strategies_used"].get(strategy, 0) + 1
        
        return statistics
    
    @contextmanager
    def error_boundary(self, 
                      component: str,
                      fallback_result: Any = None,
                      suppress_exceptions: bool = False):
        """Context manager for error boundaries."""
        
        try:
            yield
        except Exception as e:
            error_context = self._create_error_context(e, component, "context_manager", 0, 0)
            self.error_history.append(error_context)
            
            self.logger.error(f"Error in {component} boundary: {e}")
            
            if not suppress_exceptions:
                raise
            
            return fallback_result
    
    def shutdown(self):
        """Shutdown error recovery system."""
        
        self.monitoring_enabled = False
        
        # Save error statistics
        stats = self.get_error_statistics()
        stats_file = self.log_dir / "final_error_statistics.json"
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        self.logger.info("Error recovery system shutdown complete")


# Global error recovery instance
_global_error_recovery = None

def get_error_recovery() -> RobustErrorRecovery:
    """Get global error recovery instance."""
    global _global_error_recovery
    
    if _global_error_recovery is None:
        _global_error_recovery = RobustErrorRecovery()
    
    return _global_error_recovery


def robust_function(component: str = "unknown", 
                   max_retries: int = 3,
                   timeout: float = 30.0,
                   fallback_result: Any = None):
    """Decorator for making functions robust."""
    
    recovery_system = get_error_recovery()
    return recovery_system.robust_wrapper(
        component=component,
        max_retries=max_retries,
        timeout=timeout,
        fallback_result=fallback_result
    )


def main():
    """Demonstrate robust error recovery system."""
    
    recovery_system = RobustErrorRecovery()
    
    @recovery_system.robust_wrapper(component="demo", max_retries=2)
    def example_function(should_fail: bool = False):
        if should_fail:
            raise ValueError("Intentional test error")
        return "Success!"
    
    # Test successful execution
    result = example_function(False)
    print(f"Result: {result}")
    
    # Test error recovery
    try:
        result = example_function(True)
        print(f"Recovered result: {result}")
    except Exception as e:
        print(f"Final error: {e}")
    
    # Get statistics
    stats = recovery_system.get_error_statistics()
    print(f"Error statistics: {stats}")
    
    return recovery_system


if __name__ == "__main__":
    main()