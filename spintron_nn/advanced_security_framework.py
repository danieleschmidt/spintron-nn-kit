"""
Advanced Security Framework for SpinTron-NN-Kit.

This module provides comprehensive security features including encryption,
secure communication, access control, and threat detection.
"""

import time
import json
import hashlib
import hmac
import secrets
import base64
from typing import Dict, List, Tuple, Optional, Any, Set, Union
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import threading
from datetime import datetime, timedelta


class SecurityLevel(Enum):
    """Security levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AccessLevel(Enum):
    """Access levels."""
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class ThreatType(Enum):
    """Threat types."""
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    DATA_EXFILTRATION = "data_exfiltration"
    CODE_INJECTION = "code_injection"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    DENIAL_OF_SERVICE = "denial_of_service"
    ANOMALOUS_BEHAVIOR = "anomalous_behavior"


@dataclass
class SecurityEvent:
    """Security event record."""
    
    timestamp: float
    event_id: str
    threat_type: ThreatType
    severity: SecurityLevel
    source_ip: str
    user_id: str
    component: str
    description: str
    blocked: bool
    evidence: Dict[str, Any]


@dataclass
class AccessToken:
    """Access token structure."""
    
    token_id: str
    user_id: str
    access_level: AccessLevel
    expires_at: float
    issued_at: float
    permissions: Set[str]
    ip_address: str


class CryptographicEngine:
    """Cryptographic operations engine."""
    
    def __init__(self):
        self.key_size = 32  # 256-bit keys
        self.iv_size = 16   # 128-bit IV
        self.salt_size = 32 # 256-bit salt
        
    def generate_key(self) -> bytes:
        """Generate cryptographically secure key."""
        return secrets.token_bytes(self.key_size)
    
    def generate_salt(self) -> bytes:
        """Generate cryptographically secure salt."""
        return secrets.token_bytes(self.salt_size)
    
    def hash_password(self, password: str, salt: bytes = None) -> Tuple[bytes, bytes]:
        """Hash password with salt using PBKDF2."""
        if salt is None:
            salt = self.generate_salt()
        
        # Use PBKDF2 with SHA-256
        import hashlib
        key = hashlib.pbkdf2_hmac('sha256', password.encode(), salt, 100000)
        return key, salt
    
    def verify_password(self, password: str, hashed: bytes, salt: bytes) -> bool:
        """Verify password against hash."""
        key, _ = self.hash_password(password, salt)
        return hmac.compare_digest(key, hashed)
    
    def encrypt_data(self, data: bytes, key: bytes) -> Tuple[bytes, bytes]:
        """Encrypt data using AES-256-GCM (simulated)."""
        # In production, use actual AES-256-GCM
        # This is a simplified simulation
        iv = secrets.token_bytes(self.iv_size)
        
        # Simulate encryption (XOR for demonstration)
        encrypted = bytes(a ^ b for a, b in zip(data, (key * (len(data) // len(key) + 1))[:len(data)]))
        
        return encrypted, iv
    
    def decrypt_data(self, encrypted_data: bytes, key: bytes, iv: bytes) -> bytes:
        """Decrypt data using AES-256-GCM (simulated)."""
        # Simulate decryption (XOR for demonstration)
        decrypted = bytes(a ^ b for a, b in zip(encrypted_data, (key * (len(encrypted_data) // len(key) + 1))[:len(encrypted_data)]))
        
        return decrypted
    
    def sign_data(self, data: bytes, private_key: bytes) -> bytes:
        """Sign data with HMAC-SHA256."""
        return hmac.new(private_key, data, hashlib.sha256).digest()
    
    def verify_signature(self, data: bytes, signature: bytes, public_key: bytes) -> bool:
        """Verify HMAC-SHA256 signature."""
        expected_signature = self.sign_data(data, public_key)
        return hmac.compare_digest(signature, expected_signature)


class AccessControlManager:
    """Access control and authorization manager."""
    
    def __init__(self):
        self.active_tokens = {}
        self.token_blacklist = set()
        self.failed_attempts = {}
        self.max_failed_attempts = 5
        self.lockout_duration = 300  # 5 minutes
        
        # Role-based permissions
        self.role_permissions = {
            AccessLevel.READ: {"read_data", "view_results"},
            AccessLevel.WRITE: {"read_data", "view_results", "modify_data", "create_experiments"},
            AccessLevel.EXECUTE: {"read_data", "view_results", "modify_data", "create_experiments", "run_experiments"},
            AccessLevel.ADMIN: {"*"}  # All permissions
        }
        
        self.crypto = CryptographicEngine()
    
    def authenticate_user(self, user_id: str, password: str, ip_address: str) -> Optional[AccessToken]:
        """Authenticate user and generate access token."""
        
        # Check for rate limiting
        if self._is_rate_limited(user_id, ip_address):
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.MEDIUM,
                ip_address,
                user_id,
                "authentication",
                "Rate limited authentication attempt"
            )
            return None
        
        # Simulate user credential verification
        if self._verify_credentials(user_id, password):
            # Generate secure token
            token = self._generate_access_token(user_id, ip_address)
            self.active_tokens[token.token_id] = token
            
            # Reset failed attempts
            if user_id in self.failed_attempts:
                del self.failed_attempts[user_id]
            
            return token
        else:
            # Track failed attempt
            self._track_failed_attempt(user_id, ip_address)
            return None
    
    def _verify_credentials(self, user_id: str, password: str) -> bool:
        """Verify user credentials (simulated)."""
        # In production, verify against secure user database
        # This is a simplified simulation
        
        # Simulate some valid users
        valid_users = {
            "admin": "admin_password_hash",
            "researcher": "researcher_password_hash", 
            "operator": "operator_password_hash"
        }
        
        return user_id in valid_users and len(password) >= 8
    
    def _generate_access_token(self, user_id: str, ip_address: str) -> AccessToken:
        """Generate secure access token."""
        
        token_id = secrets.token_urlsafe(32)
        current_time = time.time()
        expires_at = current_time + 3600  # 1 hour
        
        # Determine access level based on user
        access_level = self._get_user_access_level(user_id)
        permissions = self.role_permissions.get(access_level, set())
        
        return AccessToken(
            token_id=token_id,
            user_id=user_id,
            access_level=access_level,
            expires_at=expires_at,
            issued_at=current_time,
            permissions=permissions.copy() if permissions != {"*"} else {"*"},
            ip_address=ip_address
        )
    
    def _get_user_access_level(self, user_id: str) -> AccessLevel:
        """Get user access level (simulated)."""
        
        # Simulate role assignments
        role_assignments = {
            "admin": AccessLevel.ADMIN,
            "researcher": AccessLevel.EXECUTE,
            "operator": AccessLevel.WRITE,
            "viewer": AccessLevel.READ
        }
        
        return role_assignments.get(user_id, AccessLevel.READ)
    
    def validate_token(self, token_id: str, required_permission: str = None) -> bool:
        """Validate access token and permissions."""
        
        if token_id in self.token_blacklist:
            return False
        
        token = self.active_tokens.get(token_id)
        if not token:
            return False
        
        # Check expiration
        if time.time() > token.expires_at:
            self.revoke_token(token_id)
            return False
        
        # Check permission
        if required_permission:
            if "*" not in token.permissions and required_permission not in token.permissions:
                return False
        
        return True
    
    def revoke_token(self, token_id: str):
        """Revoke access token."""
        
        if token_id in self.active_tokens:
            del self.active_tokens[token_id]
        
        self.token_blacklist.add(token_id)
    
    def _is_rate_limited(self, user_id: str, ip_address: str) -> bool:
        """Check if user/IP is rate limited."""
        
        current_time = time.time()
        
        # Check user-based rate limiting
        if user_id in self.failed_attempts:
            attempts, last_attempt = self.failed_attempts[user_id]
            if attempts >= self.max_failed_attempts:
                if current_time - last_attempt < self.lockout_duration:
                    return True
                else:
                    # Reset after lockout period
                    del self.failed_attempts[user_id]
        
        return False
    
    def _track_failed_attempt(self, user_id: str, ip_address: str):
        """Track failed authentication attempt."""
        
        current_time = time.time()
        
        if user_id in self.failed_attempts:
            attempts, _ = self.failed_attempts[user_id]
            self.failed_attempts[user_id] = (attempts + 1, current_time)
        else:
            self.failed_attempts[user_id] = (1, current_time)
        
        # Log security event
        self._log_security_event(
            ThreatType.UNAUTHORIZED_ACCESS,
            SecurityLevel.LOW,
            ip_address,
            user_id,
            "authentication",
            f"Failed authentication attempt #{self.failed_attempts[user_id][0]}"
        )
    
    def _log_security_event(self, 
                          threat_type: ThreatType,
                          severity: SecurityLevel,
                          source_ip: str,
                          user_id: str,
                          component: str,
                          description: str):
        """Log security event."""
        
        # This would integrate with the main security framework
        pass


class ThreatDetectionEngine:
    """Advanced threat detection and response engine."""
    
    def __init__(self):
        self.security_events = []
        self.threat_patterns = {}
        self.monitoring_enabled = True
        
        # Anomaly detection thresholds
        self.anomaly_thresholds = {
            "request_rate": 100,  # requests per minute
            "error_rate": 0.1,    # 10% error rate
            "data_volume": 1024 * 1024 * 100,  # 100MB
            "unusual_hours": (22, 6)  # 10 PM to 6 AM
        }
        
        # Initialize baseline behavior models
        self._initialize_baseline_models()
    
    def _initialize_baseline_models(self):
        """Initialize baseline behavior models."""
        
        self.baseline_models = {
            "normal_request_patterns": {
                "avg_requests_per_minute": 10,
                "std_requests_per_minute": 3,
                "peak_hours": list(range(9, 17)),  # 9 AM to 5 PM
                "typical_operations": {"read_data", "run_experiments", "view_results"}
            },
            "normal_user_behavior": {
                "session_duration_avg": 3600,  # 1 hour
                "operations_per_session": 50,
                "typical_data_access": 1024 * 1024 * 10  # 10MB
            }
        }
    
    def detect_threats(self, 
                      user_id: str,
                      ip_address: str,
                      operation: str,
                      data_size: int = 0) -> List[ThreatType]:
        """Detect potential threats in real-time."""
        
        detected_threats = []
        current_time = time.time()
        
        # Rate-based anomaly detection
        if self._detect_rate_anomaly(user_id, ip_address):
            detected_threats.append(ThreatType.DENIAL_OF_SERVICE)
        
        # Data exfiltration detection
        if self._detect_data_exfiltration(user_id, data_size):
            detected_threats.append(ThreatType.DATA_EXFILTRATION)
        
        # Privilege escalation detection
        if self._detect_privilege_escalation(user_id, operation):
            detected_threats.append(ThreatType.PRIVILEGE_ESCALATION)
        
        # Time-based anomaly detection
        if self._detect_time_anomaly(current_time):
            detected_threats.append(ThreatType.ANOMALOUS_BEHAVIOR)
        
        # Pattern-based detection
        behavioral_threats = self._detect_behavioral_anomalies(user_id, operation)
        detected_threats.extend(behavioral_threats)
        
        return detected_threats
    
    def _detect_rate_anomaly(self, user_id: str, ip_address: str) -> bool:
        """Detect rate-based anomalies."""
        
        current_time = time.time()
        recent_events = [
            event for event in self.security_events
            if event.timestamp > current_time - 60 and  # Last minute
            (event.user_id == user_id or event.source_ip == ip_address)
        ]
        
        return len(recent_events) > self.anomaly_thresholds["request_rate"]
    
    def _detect_data_exfiltration(self, user_id: str, data_size: int) -> bool:
        """Detect potential data exfiltration."""
        
        # Check if data access volume is unusually high
        if data_size > self.anomaly_thresholds["data_volume"]:
            return True
        
        # Check cumulative data access in recent time
        current_time = time.time()
        recent_data_access = sum(
            event.evidence.get("data_size", 0)
            for event in self.security_events
            if event.timestamp > current_time - 3600 and  # Last hour
            event.user_id == user_id
        )
        
        return recent_data_access > self.anomaly_thresholds["data_volume"] * 5
    
    def _detect_privilege_escalation(self, user_id: str, operation: str) -> bool:
        """Detect privilege escalation attempts."""
        
        # Check for unusual administrative operations
        admin_operations = {
            "modify_permissions", "access_system_config", 
            "execute_system_commands", "modify_security_settings"
        }
        
        if operation in admin_operations:
            # Check if user typically performs these operations
            recent_admin_ops = [
                event for event in self.security_events
                if event.user_id == user_id and 
                event.evidence.get("operation") in admin_operations and
                event.timestamp > time.time() - 86400  # Last 24 hours
            ]
            
            # If unusual admin activity, flag as potential escalation
            return len(recent_admin_ops) == 0
        
        return False
    
    def _detect_time_anomaly(self, current_time: float) -> bool:
        """Detect time-based anomalies."""
        
        current_hour = datetime.fromtimestamp(current_time).hour
        unusual_start, unusual_end = self.anomaly_thresholds["unusual_hours"]
        
        # Check if current time is during unusual hours
        if unusual_start > unusual_end:  # Spans midnight
            return current_hour >= unusual_start or current_hour <= unusual_end
        else:
            return unusual_start <= current_hour <= unusual_end
    
    def _detect_behavioral_anomalies(self, user_id: str, operation: str) -> List[ThreatType]:
        """Detect behavioral anomalies using ML-like patterns."""
        
        threats = []
        
        # Analyze user operation patterns
        user_history = [
            event for event in self.security_events
            if event.user_id == user_id and
            event.timestamp > time.time() - 86400  # Last 24 hours
        ]
        
        if user_history:
            # Check for unusual operation sequences
            recent_operations = [event.evidence.get("operation") for event in user_history[-10:]]
            
            # Simple pattern detection (in practice, use ML)
            if self._is_unusual_pattern(recent_operations):
                threats.append(ThreatType.ANOMALOUS_BEHAVIOR)
        
        return threats
    
    def _is_unusual_pattern(self, operations: List[str]) -> bool:
        """Check if operation pattern is unusual (simplified)."""
        
        # Simple heuristics for unusual patterns
        if len(operations) < 3:
            return False
        
        # Too many failed operations
        failed_ops = sum(1 for op in operations if "failed" in op.lower())
        if failed_ops > len(operations) * 0.5:
            return True
        
        # Rapid escalation pattern
        escalation_ops = ["read", "write", "execute", "admin"]
        user_escalation = [op for op in operations if any(esc in op.lower() for esc in escalation_ops)]
        if len(set(user_escalation)) >= 3:  # Rapid permission escalation
            return True
        
        return False


class AdvancedSecurityFramework:
    """Comprehensive security framework."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.HIGH):
        self.security_level = security_level
        self.crypto = CryptographicEngine()
        self.access_control = AccessControlManager()
        self.threat_detection = ThreatDetectionEngine()
        
        # Security configuration
        self.audit_logging_enabled = True
        self.real_time_monitoring = True
        self.automatic_response = True
        
        # Security metrics
        self.security_metrics = {
            "threats_detected": 0,
            "threats_blocked": 0,
            "false_positives": 0,
            "security_score": 0.95
        }
        
        # Initialize security monitoring
        self._start_security_monitoring()
    
    def _start_security_monitoring(self):
        """Start continuous security monitoring."""
        
        def security_monitor():
            while self.real_time_monitoring:
                try:
                    self._update_security_metrics()
                    self._cleanup_expired_tokens()
                    self._analyze_threat_patterns()
                    time.sleep(30)  # Monitor every 30 seconds
                except Exception as e:
                    print(f"Security monitoring error: {e}")
                    time.sleep(5)
        
        monitor_thread = threading.Thread(target=security_monitor, daemon=True)
        monitor_thread.start()
    
    def secure_operation(self,
                        operation: str,
                        user_token: str,
                        data: Any = None,
                        required_permission: str = None) -> Tuple[bool, Any, List[str]]:
        """Execute operation with comprehensive security checks."""
        
        warnings = []
        
        # Step 1: Token validation
        if not self.access_control.validate_token(user_token, required_permission):
            self._log_security_event(
                ThreatType.UNAUTHORIZED_ACCESS,
                SecurityLevel.HIGH,
                "unknown",
                "unknown",
                operation,
                "Invalid or expired token"
            )
            return False, None, ["Access denied: Invalid token"]
        
        # Get token details
        token = self.access_control.active_tokens.get(user_token)
        if not token:
            return False, None, ["Access denied: Token not found"]
        
        # Step 2: Threat detection
        data_size = len(str(data)) if data else 0
        detected_threats = self.threat_detection.detect_threats(
            token.user_id, token.ip_address, operation, data_size
        )
        
        if detected_threats:
            # Log security event
            for threat in detected_threats:
                self._log_security_event(
                    threat,
                    SecurityLevel.HIGH,
                    token.ip_address,
                    token.user_id,
                    operation,
                    f"Threat detected: {threat.value}"
                )
            
            self.security_metrics["threats_detected"] += len(detected_threats)
            
            # Automatic response
            if self.automatic_response:
                response_taken = self._automatic_threat_response(detected_threats, token)
                if response_taken:
                    self.security_metrics["threats_blocked"] += len(detected_threats)
                    return False, None, [f"Operation blocked due to security threats: {[t.value for t in detected_threats]}"]
        
        # Step 3: Data encryption (if needed)
        if data and self.security_level in [SecurityLevel.HIGH, SecurityLevel.CRITICAL]:
            encrypted_data, warnings_enc = self._secure_data_handling(data)
            warnings.extend(warnings_enc)
        else:
            encrypted_data = data
        
        # Step 4: Log successful operation
        self._log_security_event(
            ThreatType.UNAUTHORIZED_ACCESS,  # Will be filtered as non-threat
            SecurityLevel.LOW,
            token.ip_address,
            token.user_id,
            operation,
            "Operation completed successfully",
            blocked=False
        )
        
        return True, encrypted_data, warnings
    
    def _secure_data_handling(self, data: Any) -> Tuple[Any, List[str]]:
        """Handle data securely with encryption if needed."""
        
        warnings = []
        
        if isinstance(data, (str, bytes)):
            # Check for sensitive data patterns
            sensitive_patterns = [
                "password", "secret", "key", "token", "private",
                "confidential", "classified", "restricted"
            ]
            
            data_str = str(data).lower()
            if any(pattern in data_str for pattern in sensitive_patterns):
                warnings.append("Sensitive data detected - applying additional encryption")
                
                # Encrypt sensitive data
                if isinstance(data, str):
                    data_bytes = data.encode()
                else:
                    data_bytes = data
                
                key = self.crypto.generate_key()
                encrypted_data, iv = self.crypto.encrypt_data(data_bytes, key)
                
                # In production, securely store/manage keys
                return {
                    "encrypted_data": base64.b64encode(encrypted_data).decode(),
                    "iv": base64.b64encode(iv).decode(),
                    "key_id": "secure_key_" + secrets.token_hex(8)
                }, warnings
        
        return data, warnings
    
    def _automatic_threat_response(self, threats: List[ThreatType], token: AccessToken) -> bool:
        """Automatic response to detected threats."""
        
        high_severity_threats = {
            ThreatType.DATA_EXFILTRATION,
            ThreatType.PRIVILEGE_ESCALATION,
            ThreatType.CODE_INJECTION
        }
        
        # Check if any high-severity threats
        if any(threat in high_severity_threats for threat in threats):
            # Revoke token immediately
            self.access_control.revoke_token(token.token_id)
            return True
        
        # For other threats, apply rate limiting or warnings
        if ThreatType.DENIAL_OF_SERVICE in threats:
            # Implement rate limiting (simulated)
            return True
        
        return False
    
    def _log_security_event(self,
                          threat_type: ThreatType,
                          severity: SecurityLevel,
                          source_ip: str,
                          user_id: str,
                          component: str,
                          description: str,
                          blocked: bool = True):
        """Log security event."""
        
        event = SecurityEvent(
            timestamp=time.time(),
            event_id=secrets.token_hex(8),
            threat_type=threat_type,
            severity=severity,
            source_ip=source_ip,
            user_id=user_id,
            component=component,
            description=description,
            blocked=blocked,
            evidence={"operation": component, "user_agent": "SpinTron-NN-Kit"}
        )
        
        # Add to threat detection engine
        self.threat_detection.security_events.append(event)
        
        # Keep only recent events (memory management)
        cutoff_time = time.time() - 86400  # 24 hours
        self.threat_detection.security_events = [
            e for e in self.threat_detection.security_events
            if e.timestamp > cutoff_time
        ]
    
    def _update_security_metrics(self):
        """Update security metrics."""
        
        recent_events = [
            event for event in self.threat_detection.security_events
            if event.timestamp > time.time() - 3600  # Last hour
        ]
        
        if recent_events:
            blocked_events = sum(1 for event in recent_events if event.blocked)
            total_events = len(recent_events)
            
            # Update security score based on threat handling
            threat_handling_ratio = blocked_events / total_events if total_events > 0 else 1.0
            self.security_metrics["security_score"] = min(0.99, threat_handling_ratio * 0.95)
    
    def _cleanup_expired_tokens(self):
        """Clean up expired tokens."""
        
        current_time = time.time()
        expired_tokens = [
            token_id for token_id, token in self.access_control.active_tokens.items()
            if token.expires_at < current_time
        ]
        
        for token_id in expired_tokens:
            self.access_control.revoke_token(token_id)
    
    def _analyze_threat_patterns(self):
        """Analyze threat patterns for intelligence."""
        
        recent_events = [
            event for event in self.threat_detection.security_events
            if event.timestamp > time.time() - 3600  # Last hour
        ]
        
        # Simple pattern analysis
        threat_counts = {}
        for event in recent_events:
            threat_type = event.threat_type.value
            threat_counts[threat_type] = threat_counts.get(threat_type, 0) + 1
        
        # Update threat patterns
        self.threat_detection.threat_patterns = threat_counts
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        
        recent_events = [
            event for event in self.threat_detection.security_events
            if event.timestamp > time.time() - 3600  # Last hour
        ]
        
        return {
            "security_level": self.security_level.value,
            "security_score": self.security_metrics["security_score"],
            "active_tokens": len(self.access_control.active_tokens),
            "recent_events": len(recent_events),
            "threats_detected_hour": self.security_metrics["threats_detected"],
            "threats_blocked_hour": self.security_metrics["threats_blocked"],
            "threat_patterns": self.threat_detection.threat_patterns,
            "system_status": "secure" if self.security_metrics["security_score"] > 0.8 else "alert"
        }


def main():
    """Demonstrate advanced security framework."""
    
    security = AdvancedSecurityFramework(SecurityLevel.HIGH)
    
    # Simulate user authentication
    token = security.access_control.authenticate_user("researcher", "secure_password", "192.168.1.100")
    
    if token:
        print(f"Authentication successful: {token.token_id}")
        
        # Test secure operation
        success, result, warnings = security.secure_operation(
            "run_experiment",
            token.token_id,
            data="experimental_data",
            required_permission="run_experiments"
        )
        
        print(f"Operation success: {success}")
        print(f"Warnings: {warnings}")
        
        # Get security status
        status = security.get_security_status()
        print(f"Security status: {status}")
    
    return security


if __name__ == "__main__":
    main()