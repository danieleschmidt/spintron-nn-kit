"""
Advanced Security Framework for SpinTron-NN-Kit.

This module provides comprehensive security features including:
- Input validation and sanitization
- Secure model handling
- Anti-tampering measures
- Audit logging
- Access control
"""

import hashlib
import hmac
import json
import os
import secrets
import time
from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

from .logging_config import get_logger
from .error_handling import SecurityError, ErrorSeverity


class SecurityLevel(Enum):
    """Security levels for operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatType(Enum):
    """Types of security threats."""
    MALICIOUS_INPUT = "malicious_input"
    MODEL_TAMPERING = "model_tampering"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_INJECTION = "data_injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    INFORMATION_DISCLOSURE = "information_disclosure"
    DENIAL_OF_SERVICE = "denial_of_service"


@dataclass
class SecurityContext:
    """Security context for operations."""
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    permissions: List[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    audit_enabled: bool = True
    

@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: str
    severity: str
    timestamp: float
    user_id: Optional[str]
    component: str
    details: Dict[str, Any]
    threat_indicators: List[str]


class InputValidator:
    """Secure input validation and sanitization."""
    
    def __init__(self):
        self.logger = get_logger("security.input_validator")
        
        # Dangerous patterns that should be blocked
        self.dangerous_patterns = [
            r'<script[^>]*>.*?</script>',  # Script injection
            r'javascript:',               # JavaScript protocol
            r'data:text/html',           # Data URI with HTML
            r'vbscript:',                # VBScript protocol
            r'on\w+\s*=',                # Event handlers
            r'expression\s*\(',          # CSS expressions
            r'import\s+os',              # Python OS imports
            r'exec\s*\(',                # Python exec
            r'eval\s*\(',                # Python eval
            r'__import__\s*\(',          # Python import
            r'subprocess\.',             # Subprocess calls
        ]
    
    def validate_model_input(self, data: Any, context: SecurityContext) -> bool:
        """
        Validate model input data for security threats.
        
        Args:
            data: Input data to validate
            context: Security context
            
        Returns:
            True if safe, raises SecurityError if unsafe
        """
        try:
            # Check data size limits
            if self._check_size_limits(data, context):
                self.logger.warning("Input size limit exceeded", 
                                  component="input_validator",
                                  size=len(str(data)) if data else 0)
            
            # Check for malicious patterns
            if self._check_malicious_patterns(data):
                raise SecurityError(
                    "Malicious pattern detected in input",
                    threat_type=ThreatType.MALICIOUS_INPUT.value
                )
            
            # Validate data types
            if not self._validate_data_types(data, context):
                raise SecurityError(
                    "Invalid data types in input",
                    threat_type=ThreatType.DATA_INJECTION.value
                )
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(
                f"Input validation failed: {str(e)}",
                threat_type=ThreatType.MALICIOUS_INPUT.value
            )
    
    def _check_size_limits(self, data: Any, context: SecurityContext) -> bool:
        """Check if data exceeds size limits."""
        max_sizes = {
            SecurityLevel.LOW: 100 * 1024 * 1024,      # 100 MB
            SecurityLevel.MEDIUM: 50 * 1024 * 1024,    # 50 MB  
            SecurityLevel.HIGH: 10 * 1024 * 1024,      # 10 MB
            SecurityLevel.CRITICAL: 1 * 1024 * 1024,   # 1 MB
        }
        
        max_size = max_sizes.get(context.security_level, max_sizes[SecurityLevel.MEDIUM])
        data_size = len(str(data)) if data else 0
        
        return data_size > max_size
    
    def _check_malicious_patterns(self, data: Any) -> bool:
        """Check for malicious patterns in data."""
        import re
        
        data_str = str(data).lower()
        
        for pattern in self.dangerous_patterns:
            if re.search(pattern, data_str, re.IGNORECASE):
                self.logger.warning(f"Dangerous pattern detected: {pattern}",
                                  component="input_validator")
                return True
        
        return False
    
    def _validate_data_types(self, data: Any, context: SecurityContext) -> bool:
        """Validate data types are safe."""
        # Allow only safe data types
        safe_types = (int, float, str, bool, list, dict, type(None))
        
        if not isinstance(data, safe_types):
            return False
        
        # Recursively check collections
        if isinstance(data, dict):
            return all(
                isinstance(k, str) and self._validate_data_types(v, context)
                for k, v in data.items()
            )
        elif isinstance(data, list):
            return all(self._validate_data_types(item, context) for item in data)
        
        return True


class ModelSecurityChecker:
    """Security checks for neural network models."""
    
    def __init__(self):
        self.logger = get_logger("security.model_checker")
    
    def verify_model_integrity(self, model_path: Union[str, Path], 
                             expected_hash: Optional[str] = None) -> bool:
        """
        Verify model file integrity.
        
        Args:
            model_path: Path to model file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            True if integrity check passes
        """
        try:
            model_path = Path(model_path)
            
            if not model_path.exists():
                raise SecurityError(
                    f"Model file not found: {model_path}",
                    threat_type=ThreatType.MODEL_TAMPERING.value
                )
            
            # Calculate file hash
            file_hash = self._calculate_file_hash(model_path)
            
            # Check against expected hash if provided
            if expected_hash and not hmac.compare_digest(file_hash, expected_hash):
                raise SecurityError(
                    "Model integrity check failed - hash mismatch",
                    threat_type=ThreatType.MODEL_TAMPERING.value
                )
            
            # Check file permissions
            if not self._check_file_permissions(model_path):
                raise SecurityError(
                    f"Unsafe file permissions for model: {model_path}",
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS.value
                )
            
            self.logger.info(f"Model integrity verified: {model_path}",
                           component="model_checker", file_hash=file_hash)
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(
                f"Model integrity check failed: {str(e)}",
                threat_type=ThreatType.MODEL_TAMPERING.value
            )
    
    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file."""
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _check_file_permissions(self, file_path: Path) -> bool:
        """Check if file has safe permissions."""
        stat_info = file_path.stat()
        
        # Check if file is world-writable (unsafe)
        if stat_info.st_mode & 0o002:
            return False
        
        # Check if file is group-writable and group has excessive permissions
        if (stat_info.st_mode & 0o020) and (stat_info.st_mode & 0o044):
            return False
        
        return True


class AccessController:
    """Role-based access control."""
    
    def __init__(self):
        self.logger = get_logger("security.access_controller")
        self.permissions_cache = {}
        self.session_tokens = {}
    
    def check_permission(self, context: SecurityContext, 
                        operation: str, resource: str = None) -> bool:
        """
        Check if operation is permitted.
        
        Args:
            context: Security context
            operation: Operation being attempted
            resource: Resource being accessed
            
        Returns:
            True if permitted
        """
        try:
            # Check session validity
            if context.session_id and not self._validate_session(context.session_id):
                raise SecurityError(
                    "Invalid or expired session",
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS.value
                )
            
            # Check permissions
            if not self._has_permission(context, operation, resource):
                self.logger.warning(f"Permission denied for {operation}",
                                  component="access_controller",
                                  user_id=context.user_id,
                                  operation=operation,
                                  resource=resource)
                
                raise SecurityError(
                    f"Permission denied for operation: {operation}",
                    threat_type=ThreatType.UNAUTHORIZED_ACCESS.value
                )
            
            return True
            
        except SecurityError:
            raise
        except Exception as e:
            raise SecurityError(
                f"Access control check failed: {str(e)}",
                threat_type=ThreatType.PRIVILEGE_ESCALATION.value
            )
    
    def _validate_session(self, session_id: str) -> bool:
        """Validate session token."""
        if session_id not in self.session_tokens:
            return False
        
        session_data = self.session_tokens[session_id]
        current_time = time.time()
        
        # Check expiration
        if current_time > session_data.get('expires_at', 0):
            del self.session_tokens[session_id]
            return False
        
        return True
    
    def _has_permission(self, context: SecurityContext, 
                       operation: str, resource: str = None) -> bool:
        """Check if context has required permissions."""
        if not context.permissions:
            return False
        
        # Simple permission checking - can be extended
        required_permissions = {
            'read_model': ['model_read', 'admin'],
            'write_model': ['model_write', 'admin'], 
            'convert_model': ['model_convert', 'admin'],
            'train_model': ['training', 'admin'],
            'hardware_access': ['hardware', 'admin'],
            'system_config': ['admin'],
        }
        
        allowed_perms = required_permissions.get(operation, ['admin'])
        return any(perm in context.permissions for perm in allowed_perms)


class SecurityAuditor:
    """Security audit logging and monitoring."""
    
    def __init__(self):
        self.logger = get_logger("security.auditor")
        self.audit_log = []
        self.threat_counters = {}
    
    def log_security_event(self, event_type: str, severity: str,
                          details: Dict[str, Any], component: str,
                          context: Optional[SecurityContext] = None,
                          threat_indicators: List[str] = None) -> None:
        """
        Log security event for audit trail.
        
        Args:
            event_type: Type of security event
            severity: Event severity level
            details: Event details
            component: Component that generated event
            context: Security context
            threat_indicators: List of threat indicators
        """
        event = SecurityEvent(
            event_type=event_type,
            severity=severity,
            timestamp=time.time(),
            user_id=context.user_id if context else None,
            component=component,
            details=details,
            threat_indicators=threat_indicators or []
        )
        
        self.audit_log.append(event)
        
        # Update threat counters
        for indicator in event.threat_indicators:
            self.threat_counters[indicator] = self.threat_counters.get(indicator, 0) + 1
        
        # Log to security logger
        self.logger.critical(f"Security event: {event_type}",
                           component=component,
                           severity=severity,
                           user_id=context.user_id if context else None,
                           details=details,
                           threat_indicators=threat_indicators)
        
        # Alert on high-risk events
        if severity in ['CRITICAL', 'HIGH']:
            self._trigger_security_alert(event)
    
    def _trigger_security_alert(self, event: SecurityEvent) -> None:
        """Trigger security alert for high-risk events."""
        self.logger.critical(f"SECURITY ALERT: {event.event_type}",
                           component="security_alert",
                           event_details=event.details,
                           threat_indicators=event.threat_indicators)
        
        # Here you could integrate with external alerting systems
        # like PagerDuty, Slack, email notifications, etc.
    
    def get_security_summary(self) -> Dict[str, Any]:
        """Get security audit summary."""
        return {
            'total_events': len(self.audit_log),
            'threat_counters': dict(self.threat_counters),
            'recent_events': [
                {
                    'event_type': event.event_type,
                    'severity': event.severity,
                    'timestamp': event.timestamp,
                    'component': event.component
                }
                for event in self.audit_log[-10:]  # Last 10 events
            ]
        }


class SecurityManager:
    """Main security manager coordinating all security components."""
    
    def __init__(self):
        self.logger = get_logger("security.manager")
        self.input_validator = InputValidator()
        self.model_checker = ModelSecurityChecker()
        self.access_controller = AccessController()
        self.auditor = SecurityAuditor()
        
        # Generate secure API keys for internal communication
        self.internal_api_key = secrets.token_urlsafe(32)
        
        self.logger.info("Security manager initialized",
                        component="security_manager")
    
    def validate_operation(self, operation: str, data: Any = None,
                          context: Optional[SecurityContext] = None,
                          model_path: Optional[str] = None) -> bool:
        """
        Comprehensive security validation for operations.
        
        Args:
            operation: Operation being performed
            data: Input data (if applicable)
            context: Security context
            model_path: Model path (if applicable)
            
        Returns:
            True if operation is secure
        """
        if not context:
            context = SecurityContext()
        
        try:
            # Log operation attempt
            self.auditor.log_security_event(
                "operation_attempt",
                "INFO",
                {"operation": operation, "has_data": data is not None},
                "security_manager",
                context
            )
            
            # Check access permissions
            self.access_controller.check_permission(context, operation)
            
            # Validate input data if provided
            if data is not None:
                self.input_validator.validate_model_input(data, context)
            
            # Check model integrity if model path provided
            if model_path:
                self.model_checker.verify_model_integrity(model_path)
            
            # Log successful validation
            self.auditor.log_security_event(
                "operation_validated",
                "INFO", 
                {"operation": operation},
                "security_manager",
                context
            )
            
            return True
            
        except SecurityError as e:
            # Log security violation
            self.auditor.log_security_event(
                "security_violation",
                "HIGH",
                {
                    "operation": operation,
                    "error": str(e),
                    "threat_type": e.threat_type if hasattr(e, 'threat_type') else 'unknown'
                },
                "security_manager",
                context,
                threat_indicators=[e.threat_type if hasattr(e, 'threat_type') else 'unknown_threat']
            )
            raise
        except Exception as e:
            # Log unexpected error
            self.auditor.log_security_event(
                "security_check_error",
                "MEDIUM",
                {"operation": operation, "error": str(e)},
                "security_manager",
                context
            )
            raise SecurityError(
                f"Security validation failed: {str(e)}",
                threat_type=ThreatType.DENIAL_OF_SERVICE.value
            )
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "security_manager_active": True,
            "components": {
                "input_validator": "active",
                "model_checker": "active", 
                "access_controller": "active",
                "auditor": "active"
            },
            "audit_summary": self.auditor.get_security_summary(),
            "timestamp": time.time()
        }


# Global security manager instance
_global_security_manager = SecurityManager()


def validate_operation(operation: str, data: Any = None,
                      context: Optional[SecurityContext] = None,
                      model_path: Optional[str] = None) -> bool:
    """
    Global security validation function.
    
    Args:
        operation: Operation being performed
        data: Input data
        context: Security context
        model_path: Model file path
        
    Returns:
        True if secure
    """
    return _global_security_manager.validate_operation(
        operation, data, context, model_path
    )


def get_security_status() -> Dict[str, Any]:
    """Get global security status."""
    return _global_security_manager.get_security_status()