"""
Utility modules for SpinTron-NN-Kit.

This package provides comprehensive utilities including:
- Performance monitoring and optimization
- Robust error handling and recovery
- Security validation and compliance
- Structured logging and audit trails
- Memory management and caching
"""

from .performance import (
    PerformanceOptimizer,
    PerformanceConfig,
    ModelCache,
    MemoryManager,
    AutoScaler,
    cached_inference,
    parallel_batch_process,
    memory_efficient
)

from .validation import (
    SecureValidator,
    ValidationError,
    ValidationResult,
    SecurityRisk,
    create_validator
)

from .error_handling import (
    SpintronError,
    ValidationError as ValidationErr,
    HardwareError,
    SimulationError,
    ConversionError,
    TrainingError,
    SecurityError,
    ErrorHandler,
    safe_execute,
    robust_operation,
    handle_error
)

from .logging_config import (
    SpintronLogger,
    get_logger,
    initialize_logging,
    log_performance
)

__all__ = [
    "PerformanceOptimizer",
    "PerformanceConfig", 
    "ModelCache",
    "MemoryManager",
    "AutoScaler",
    "cached_inference",
    "parallel_batch_process",
    "memory_efficient",
    "SecureValidator",
    "ValidationError",
    "ValidationResult",
    "SecurityRisk",
    "create_validator",
    "SpintronError",
    "ValidationErr",
    "HardwareError",
    "SimulationError",
    "ConversionError",
    "TrainingError",
    "SecurityError",
    "ErrorHandler",
    "safe_execute",
    "robust_operation",
    "handle_error",
    "SpintronLogger",
    "get_logger",
    "initialize_logging",
    "log_performance"
]