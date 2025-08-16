"""
Comprehensive validation utilities for SpinTron-NN-Kit.

This module provides robust input validation, error handling,
and security checks for all components.
"""

import re
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import json
import logging


class ValidationLevel(Enum):
    """Validation strictness levels."""
    BASIC = "basic"
    STRICT = "strict"
    PARANOID = "paranoid"


class SecurityRisk(Enum):
    """Security risk levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Structured validation error."""
    code: str
    message: str
    field: str
    risk_level: SecurityRisk = SecurityRisk.LOW
    suggested_fix: str = ""


@dataclass
class ValidationResult:
    """Result of validation operation."""
    is_valid: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    sanitized_data: Optional[Dict] = None


class SecureValidator:
    """Secure validation with comprehensive security checks."""
    
    def __init__(self, level: ValidationLevel = ValidationLevel.STRICT):
        self.level = level
        self.logger = logging.getLogger(__name__)
        
        # Security patterns to detect
        self.dangerous_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'<script[^>]*>',
            r'javascript:',
            r'vbscript:',
            r'file://',
            r'\.\./',
            r'DROP\s+TABLE',
            r'UNION\s+SELECT',
            r'<\s*iframe',
        ]
        
        # Compile patterns for performance
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in self.dangerous_patterns]
    
    def validate_mtj_config(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate MTJ device configuration.
        
        Args:
            config: MTJ configuration dictionary
            
        Returns:
            Validation result with errors/warnings
        """
        errors = []
        warnings = []
        sanitized = {}
        
        # Required fields
        required_fields = ['resistance_high', 'resistance_low', 'switching_voltage', 'cell_area']
        for field in required_fields:
            if field not in config:
                errors.append(ValidationError(
                    code="MISSING_REQUIRED_FIELD",
                    message=f"Required field '{field}' is missing",
                    field=field,
                    risk_level=SecurityRisk.HIGH,
                    suggested_fix=f"Add '{field}' to configuration"
                ))
        
        # Validate resistance values
        if 'resistance_high' in config and 'resistance_low' in config:
            rh = config['resistance_high']
            rl = config['resistance_low']
            
            if not isinstance(rh, (int, float)) or not isinstance(rl, (int, float)):
                errors.append(ValidationError(
                    code="INVALID_RESISTANCE_TYPE",
                    message="Resistance values must be numeric",
                    field="resistance",
                    risk_level=SecurityRisk.MEDIUM,
                    suggested_fix="Use float or int values for resistance"
                ))
            else:
                # Sanitize and validate ranges
                rh = float(rh)
                rl = float(rl)
                
                if rh <= rl:
                    errors.append(ValidationError(
                        code="INVALID_RESISTANCE_RATIO",
                        message="High resistance must be greater than low resistance",
                        field="resistance",
                        risk_level=SecurityRisk.HIGH,
                        suggested_fix="Ensure resistance_high > resistance_low"
                    ))
                
                if not (1e3 <= rh <= 1e6):  # 1kΩ to 1MΩ
                    warnings.append(ValidationError(
                        code="RESISTANCE_OUT_OF_RANGE",
                        message=f"High resistance {rh/1e3:.1f}kΩ outside typical range 1-1000kΩ",
                        field="resistance_high",
                        risk_level=SecurityRisk.LOW,
                        suggested_fix="Consider using values between 1kΩ and 1MΩ"
                    ))
                
                sanitized['resistance_high'] = rh
                sanitized['resistance_low'] = rl
        
        # Validate switching voltage
        if 'switching_voltage' in config:
            vsw = config['switching_voltage']
            if not isinstance(vsw, (int, float)):
                errors.append(ValidationError(
                    code="INVALID_VOLTAGE_TYPE",
                    message="Switching voltage must be numeric",
                    field="switching_voltage",
                    risk_level=SecurityRisk.MEDIUM,
                    suggested_fix="Use float or int value for switching_voltage"
                ))
            else:
                vsw = float(vsw)
                if not (0.1 <= vsw <= 2.0):  # 0.1V to 2V
                    warnings.append(ValidationError(
                        code="VOLTAGE_OUT_OF_RANGE",
                        message=f"Switching voltage {vsw}V outside typical range 0.1-2.0V",
                        field="switching_voltage",
                        risk_level=SecurityRisk.LOW,
                        suggested_fix="Consider using values between 0.1V and 2.0V"
                    ))
                sanitized['switching_voltage'] = vsw
        
        # Validate cell area
        if 'cell_area' in config:
            area = config['cell_area']
            if not isinstance(area, (int, float)):
                errors.append(ValidationError(
                    code="INVALID_AREA_TYPE",
                    message="Cell area must be numeric",
                    field="cell_area",
                    risk_level=SecurityRisk.MEDIUM,
                    suggested_fix="Use float or int value for cell_area"
                ))
            else:
                area = float(area)
                if not (1e-18 <= area <= 1e-12):  # 1 nm² to 1 μm²
                    warnings.append(ValidationError(
                        code="AREA_OUT_OF_RANGE",
                        message=f"Cell area {area*1e18:.1f}nm² outside typical range 1nm²-1μm²",
                        field="cell_area",
                        risk_level=SecurityRisk.LOW,
                        suggested_fix="Consider using values between 1nm² and 1μm²"
                    ))
                sanitized['cell_area'] = area
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_data=sanitized
        )
    
    def validate_tensor_data(self, data: Any, expected_shape: Optional[Tuple] = None) -> ValidationResult:
        """Validate tensor-like data with security checks.
        
        Args:
            data: Data to validate (list, tuple, or tensor-like)
            expected_shape: Expected shape tuple
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check for malicious content in data
        data_str = str(data)
        for pattern in self.compiled_patterns:
            if pattern.search(data_str):
                errors.append(ValidationError(
                    code="MALICIOUS_CONTENT_DETECTED",
                    message=f"Potentially malicious pattern detected: {pattern.pattern}",
                    field="tensor_data",
                    risk_level=SecurityRisk.CRITICAL,
                    suggested_fix="Remove malicious content from input data"
                ))
        
        # Validate data type
        if not isinstance(data, (list, tuple)) and not hasattr(data, 'shape'):
            errors.append(ValidationError(
                code="INVALID_DATA_TYPE",
                message="Data must be list, tuple, or tensor-like object",
                field="tensor_data",
                risk_level=SecurityRisk.MEDIUM,
                suggested_fix="Convert data to list or tensor format"
            ))
        
        # Validate shape if provided
        if expected_shape and hasattr(data, 'shape'):
            if data.shape != expected_shape:
                warnings.append(ValidationError(
                    code="SHAPE_MISMATCH",
                    message=f"Expected shape {expected_shape}, got {data.shape}",
                    field="tensor_shape",
                    risk_level=SecurityRisk.LOW,
                    suggested_fix="Reshape data to match expected dimensions"
                ))
        
        # Check for NaN/Inf values
        if hasattr(data, 'isnan'):
            import torch
            if torch.isnan(data).any():
                errors.append(ValidationError(
                    code="NAN_VALUES_DETECTED",
                    message="NaN values detected in tensor data",
                    field="tensor_data",
                    risk_level=SecurityRisk.HIGH,
                    suggested_fix="Replace NaN values with valid numbers"
                ))
            if torch.isinf(data).any():
                errors.append(ValidationError(
                    code="INF_VALUES_DETECTED",
                    message="Infinite values detected in tensor data",
                    field="tensor_data",
                    risk_level=SecurityRisk.HIGH,
                    suggested_fix="Replace infinite values with valid numbers"
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_file_path(self, path: str, allowed_extensions: Optional[List[str]] = None) -> ValidationResult:
        """Validate file path for security vulnerabilities.
        
        Args:
            path: File path to validate
            allowed_extensions: List of allowed file extensions
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check for path traversal attacks
        if '../' in path or '..\\' in path:
            errors.append(ValidationError(
                code="PATH_TRAVERSAL_DETECTED",
                message="Path traversal attack detected",
                field="file_path",
                risk_level=SecurityRisk.CRITICAL,
                suggested_fix="Remove '../' sequences from path"
            ))
        
        # Check for absolute paths (potential security risk)
        if path.startswith('/') or (len(path) > 1 and path[1] == ':'):
            warnings.append(ValidationError(
                code="ABSOLUTE_PATH_DETECTED",
                message="Absolute path detected - potential security risk",
                field="file_path",
                risk_level=SecurityRisk.MEDIUM,
                suggested_fix="Use relative paths when possible"
            ))
        
        # Validate file extension
        if allowed_extensions:
            ext = path.split('.')[-1].lower() if '.' in path else ''
            if ext not in allowed_extensions:
                errors.append(ValidationError(
                    code="INVALID_FILE_EXTENSION",
                    message=f"File extension '{ext}' not in allowed list: {allowed_extensions}",
                    field="file_path",
                    risk_level=SecurityRisk.MEDIUM,
                    suggested_fix=f"Use files with extensions: {', '.join(allowed_extensions)}"
                ))
        
        # Check for null bytes (security vulnerability)
        if '\x00' in path:
            errors.append(ValidationError(
                code="NULL_BYTE_DETECTED",
                message="Null byte detected in file path",
                field="file_path",
                risk_level=SecurityRisk.CRITICAL,
                suggested_fix="Remove null bytes from file path"
            ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def validate_network_architecture(self, architecture: List[int]) -> ValidationResult:
        """Validate neural network architecture specification.
        
        Args:
            architecture: List of layer sizes
            
        Returns:
            Validation result
        """
        errors = []
        warnings = []
        
        # Check minimum requirements
        if not isinstance(architecture, (list, tuple)):
            errors.append(ValidationError(
                code="INVALID_ARCHITECTURE_TYPE",
                message="Architecture must be a list or tuple",
                field="architecture",
                risk_level=SecurityRisk.MEDIUM,
                suggested_fix="Convert architecture to list format"
            ))
            return ValidationResult(is_valid=False, errors=errors, warnings=warnings)
        
        if len(architecture) < 2:
            errors.append(ValidationError(
                code="INSUFFICIENT_LAYERS",
                message="Architecture must have at least 2 layers (input and output)",
                field="architecture",
                risk_level=SecurityRisk.HIGH,
                suggested_fix="Add more layers to create valid network"
            ))
        
        # Validate layer sizes
        for i, size in enumerate(architecture):
            if not isinstance(size, int):
                errors.append(ValidationError(
                    code="INVALID_LAYER_SIZE_TYPE",
                    message=f"Layer {i} size must be integer, got {type(size)}",
                    field=f"architecture[{i}]",
                    risk_level=SecurityRisk.MEDIUM,
                    suggested_fix="Use integer values for layer sizes"
                ))
            elif size <= 0:
                errors.append(ValidationError(
                    code="INVALID_LAYER_SIZE_VALUE",
                    message=f"Layer {i} size must be positive, got {size}",
                    field=f"architecture[{i}]",
                    risk_level=SecurityRisk.HIGH,
                    suggested_fix="Use positive integer values for layer sizes"
                ))
            elif size > 10000:  # Arbitrary large limit
                warnings.append(ValidationError(
                    code="LARGE_LAYER_SIZE",
                    message=f"Layer {i} size {size} is very large - potential memory issues",
                    field=f"architecture[{i}]",
                    risk_level=SecurityRisk.LOW,
                    suggested_fix="Consider reducing layer size for better performance"
                ))
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def sanitize_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize configuration dictionary.
        
        Args:
            config: Configuration to sanitize
            
        Returns:
            Sanitized configuration
        """
        sanitized = {}
        
        for key, value in config.items():
            # Sanitize key
            safe_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
            
            # Sanitize value based on type
            if isinstance(value, str):
                # Remove potentially dangerous characters
                safe_value = re.sub(r'[<>"\'\x00-\x1f]', '', value)
                # Limit string length
                safe_value = safe_value[:1000]
            elif isinstance(value, (int, float)):
                # Clamp numeric values to reasonable ranges
                safe_value = max(-1e10, min(1e10, float(value)))
            elif isinstance(value, (list, tuple)):
                # Recursively sanitize lists
                safe_value = [self.sanitize_config({0: item})[0] for item in value[:1000]]  # Limit list length
            elif isinstance(value, dict):
                # Recursively sanitize dictionaries
                safe_value = self.sanitize_config(value)
            else:
                safe_value = str(value)[:100]  # Convert to string and limit length
            
            sanitized[safe_key] = safe_value
        
        return sanitized
    
    def compute_security_hash(self, data: Any) -> str:
        """Compute security hash for data integrity verification.
        
        Args:
            data: Data to hash
            
        Returns:
            SHA-256 hash string
        """
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def log_security_event(self, event_type: str, details: Dict[str, Any], risk_level: SecurityRisk):
        """Log security-related events.
        
        Args:
            event_type: Type of security event
            details: Event details
            risk_level: Risk level of the event
        """
        log_entry = {
            'timestamp': '2025-01-01T00:00:00Z',  # Would use real timestamp
            'event_type': event_type,
            'risk_level': risk_level.value,
            'details': details,
            'hash': self.compute_security_hash(details)
        }
        
        if risk_level in [SecurityRisk.HIGH, SecurityRisk.CRITICAL]:
            self.logger.error(f"SECURITY EVENT: {event_type} - {details}")
        elif risk_level == SecurityRisk.MEDIUM:
            self.logger.warning(f"Security warning: {event_type} - {details}")
        else:
            self.logger.info(f"Security info: {event_type} - {details}")


def create_validator(level: str = "strict") -> SecureValidator:
    """Create a validator with specified security level.
    
    Args:
        level: Security level ("basic", "strict", or "paranoid")
        
    Returns:
        Configured validator instance
    """
    validation_level = ValidationLevel(level.lower())
    return SecureValidator(validation_level)