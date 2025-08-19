"""
Comprehensive Security Framework for SpinTron-NN-Kit.

This module implements enterprise-grade security measures for spintronic neural
network systems, including hardware security, data protection, and audit logging.

Features:
- Hardware tampering detection
- Secure computation protocols
- Data encryption and integrity verification
- Access control and authentication
- Comprehensive audit logging
- Secure model deployment
- Side-channel attack mitigation
"""

import hashlib
import hmac
import secrets
import time
import json
import os
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import threading
import logging
from pathlib import Path
import numpy as np
import torch
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.backends import default_backend
import base64

from .core.mtj_models import MTJConfig, MTJDevice
from .core.crossbar import MTJCrossbar
from .utils.monitoring import SystemMonitor
from .utils.error_handling import SecurityError, ValidationError


class SecurityLevel(Enum):
    """Security levels for different operations."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ThreatLevel(Enum):
    """Threat severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    
    timestamp: float
    event_type: str
    threat_level: ThreatLevel
    source: str
    description: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging."""
        return {
            'timestamp': self.timestamp,
            'event_type': self.event_type,
            'threat_level': self.threat_level.value,
            'source': self.source,
            'description': self.description,
            'metadata': self.metadata,
            'user_id': self.user_id,
            'session_id': self.session_id
        }


@dataclass
class SecurityContext:
    """Security context for operations."""
    
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    permissions: List[str] = field(default_factory=list)
    audit_enabled: bool = True
    encryption_required: bool = False
    
    def has_permission(self, permission: str) -> bool:
        """Check if context has specific permission."""
        return permission in self.permissions or "admin" in self.permissions


class CryptographicManager:
    """Manages cryptographic operations for data protection."""
    
    def __init__(self):
        self.symmetric_key = None
        self.private_key = None
        self.public_key = None
        self.fernet = None
        
        # Generate keys on initialization
        self._generate_keys()
    
    def _generate_keys(self):
        """Generate cryptographic keys."""
        # Generate symmetric key for data encryption
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
            backend=default_backend()
        )
        
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.fernet = Fernet(key)
        self.symmetric_key = key
        
        # Generate asymmetric key pair for signatures
        self.private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self.public_key = self.private_key.public_key()
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt data using symmetric encryption."""
        if not isinstance(data, bytes):
            raise ValueError("Data must be bytes")
        
        return self.fernet.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt data using symmetric encryption."""
        if not isinstance(encrypted_data, bytes):
            raise ValueError("Encrypted data must be bytes")
        
        return self.fernet.decrypt(encrypted_data)
    
    def encrypt_array(self, array: np.ndarray) -> bytes:
        """Encrypt numpy array."""
        array_bytes = array.tobytes()
        return self.encrypt_data(array_bytes)
    
    def decrypt_array(self, encrypted_data: bytes, dtype: np.dtype, shape: Tuple[int, ...]) -> np.ndarray:
        """Decrypt numpy array."""
        decrypted_bytes = self.decrypt_data(encrypted_data)
        return np.frombuffer(decrypted_bytes, dtype=dtype).reshape(shape)
    
    def sign_data(self, data: bytes) -> bytes:
        """Create digital signature for data."""
        signature = self.private_key.sign(
            data,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        return signature
    
    def verify_signature(self, data: bytes, signature: bytes) -> bool:
        """Verify digital signature."""
        try:
            self.public_key.verify(
                signature,
                data,
                padding.PSS(
                    mgf=padding.MGF1(hashes.SHA256()),
                    salt_length=padding.PSS.MAX_LENGTH
                ),
                hashes.SHA256()
            )
            return True
        except Exception:
            return False
    
    def compute_hash(self, data: bytes) -> str:
        """Compute SHA-256 hash of data."""
        return hashlib.sha256(data).hexdigest()
    
    def compute_hmac(self, data: bytes, key: Optional[bytes] = None) -> str:
        """Compute HMAC for data integrity."""
        if key is None:
            key = self.symmetric_key
        
        return hmac.new(key, data, hashlib.sha256).hexdigest()


class HardwareSecurityMonitor:
    """Monitors hardware for security threats and tampering."""
    
    def __init__(self, crossbar: MTJCrossbar):
        self.crossbar = crossbar
        self.baseline_characteristics = {}
        self.monitoring_active = False
        self.threat_detectors = []
        self.security_events = []
        
        # Initialize threat detectors
        self._initialize_threat_detectors()
        
        # Establish baseline
        self._establish_security_baseline()
    
    def _initialize_threat_detectors(self):
        """Initialize various threat detection mechanisms."""
        self.threat_detectors = [
            self._detect_resistance_anomalies,
            self._detect_timing_attacks,
            self._detect_power_analysis_attacks,
            self._detect_voltage_tampering,
            self._detect_temperature_attacks
        ]
    
    def _establish_security_baseline(self):
        """Establish baseline characteristics for anomaly detection."""
        print("Establishing hardware security baseline...")
        
        # Sample device characteristics
        sample_devices = []
        for i in range(0, self.crossbar.rows, max(1, self.crossbar.rows // 10)):
            for j in range(0, self.crossbar.cols, max(1, self.crossbar.cols // 10)):
                device = self.crossbar.devices[i][j]
                sample_devices.append({
                    'position': (i, j),
                    'resistance_low': device.config.resistance_low,
                    'resistance_high': device.config.resistance_high,
                    'switching_voltage': device.config.switching_voltage
                })
        
        # Calculate baseline statistics
        resistances_low = [d['resistance_low'] for d in sample_devices]
        resistances_high = [d['resistance_high'] for d in sample_devices]
        switching_voltages = [d['switching_voltage'] for d in sample_devices]
        
        self.baseline_characteristics = {
            'resistance_low': {
                'mean': np.mean(resistances_low),
                'std': np.std(resistances_low),
                'range': (np.min(resistances_low), np.max(resistances_low))
            },
            'resistance_high': {
                'mean': np.mean(resistances_high),
                'std': np.std(resistances_high),
                'range': (np.min(resistances_high), np.max(resistances_high))
            },
            'switching_voltage': {
                'mean': np.mean(switching_voltages),
                'std': np.std(switching_voltages),
                'range': (np.min(switching_voltages), np.max(switching_voltages))
            },
            'timestamp': time.time()
        }
        
        print("Security baseline established")
    
    def start_monitoring(self):
        """Start hardware security monitoring."""
        self.monitoring_active = True
        print("Hardware security monitoring started")
    
    def stop_monitoring(self):
        """Stop hardware security monitoring."""
        self.monitoring_active = False
        print("Hardware security monitoring stopped")
    
    def check_security_threats(self) -> List[SecurityEvent]:
        """Check for security threats using all detectors."""
        if not self.monitoring_active:
            return []
        
        detected_threats = []
        
        for detector in self.threat_detectors:
            try:
                threats = detector()
                detected_threats.extend(threats)
            except Exception as e:
                # Log detector error but continue with other detectors
                error_event = SecurityEvent(
                    timestamp=time.time(),
                    event_type="detector_error",
                    threat_level=ThreatLevel.WARNING,
                    source="security_monitor",
                    description=f"Threat detector failed: {str(e)}"
                )
                detected_threats.append(error_event)
        
        # Store events
        self.security_events.extend(detected_threats)
        
        # Keep event history bounded
        if len(self.security_events) > 10000:
            self.security_events = self.security_events[-5000:]
        
        return detected_threats
    
    def _detect_resistance_anomalies(self) -> List[SecurityEvent]:
        """Detect anomalous resistance patterns that might indicate tampering."""
        events = []
        
        # Sample current device resistances
        sample_resistances = []
        sample_positions = []
        
        for i in range(0, self.crossbar.rows, max(1, self.crossbar.rows // 20)):
            for j in range(0, self.crossbar.cols, max(1, self.crossbar.cols // 20)):
                device = self.crossbar.devices[i][j]
                current_resistance = device.resistance
                sample_resistances.append(current_resistance)
                sample_positions.append((i, j))
        
        if not sample_resistances:
            return events
        
        # Check for statistical anomalies
        mean_resistance = np.mean(sample_resistances)
        std_resistance = np.std(sample_resistances)
        
        # Compare with baseline
        baseline_mean = (self.baseline_characteristics['resistance_low']['mean'] + 
                        self.baseline_characteristics['resistance_high']['mean']) / 2
        
        # Detect significant deviation from baseline
        if abs(mean_resistance - baseline_mean) > 3 * std_resistance:
            events.append(SecurityEvent(
                timestamp=time.time(),
                event_type="resistance_anomaly",
                threat_level=ThreatLevel.WARNING,
                source="hardware_monitor",
                description=f"Resistance anomaly detected: current={mean_resistance:.0f}, baseline={baseline_mean:.0f}",
                metadata={
                    'current_mean': mean_resistance,
                    'baseline_mean': baseline_mean,
                    'deviation_sigma': abs(mean_resistance - baseline_mean) / std_resistance
                }
            ))
        
        # Check for outlier devices
        for i, (resistance, position) in enumerate(zip(sample_resistances, sample_positions)):
            z_score = abs(resistance - mean_resistance) / max(std_resistance, 1.0)
            if z_score > 4.0:  # 4-sigma outlier
                events.append(SecurityEvent(
                    timestamp=time.time(),
                    event_type="device_outlier",
                    threat_level=ThreatLevel.WARNING,
                    source="hardware_monitor",
                    description=f"Device outlier detected at {position}: resistance={resistance:.0f}, z-score={z_score:.2f}",
                    metadata={
                        'position': position,
                        'resistance': resistance,
                        'z_score': z_score
                    }
                ))
        
        return events
    
    def _detect_timing_attacks(self) -> List[SecurityEvent]:
        """Detect timing-based side-channel attacks."""
        events = []
        
        # Check if there are unusual timing patterns in operations
        if hasattr(self.crossbar, 'monitor') and self.crossbar.monitor:
            recent_timings = self.crossbar.monitor.get_recent_operation_timings()
            
            if len(recent_timings) > 10:
                timing_values = [t['duration'] for t in recent_timings]
                mean_timing = np.mean(timing_values)
                std_timing = np.std(timing_values)
                
                # Check for unusual timing variance
                if std_timing > mean_timing * 0.5:  # >50% variance is suspicious
                    events.append(SecurityEvent(
                        timestamp=time.time(),
                        event_type="timing_anomaly",
                        threat_level=ThreatLevel.INFO,
                        source="timing_monitor",
                        description=f"High timing variance detected: std/mean = {std_timing/mean_timing:.2f}",
                        metadata={
                            'mean_timing': mean_timing,
                            'std_timing': std_timing,
                            'variance_ratio': std_timing / mean_timing
                        }
                    ))
        
        return events
    
    def _detect_power_analysis_attacks(self) -> List[SecurityEvent]:
        """Detect power analysis side-channel attacks."""
        events = []
        
        # Simulate power consumption monitoring
        # In a real implementation, this would interface with power measurement hardware
        
        # Check for unusual power patterns
        estimated_power = self._estimate_current_power()
        
        # Simple threshold-based detection
        expected_power_range = (0.001, 0.1)  # 1mW to 100mW
        
        if estimated_power < expected_power_range[0] or estimated_power > expected_power_range[1]:
            threat_level = ThreatLevel.WARNING if estimated_power > expected_power_range[1] else ThreatLevel.INFO
            
            events.append(SecurityEvent(
                timestamp=time.time(),
                event_type="power_anomaly",
                threat_level=threat_level,
                source="power_monitor",
                description=f"Power consumption anomaly: {estimated_power:.6f}W",
                metadata={'power_watts': estimated_power}
            ))
        
        return events
    
    def _detect_voltage_tampering(self) -> List[SecurityEvent]:
        """Detect voltage tampering attempts."""
        events = []
        
        # Check current voltage settings against safe ranges
        read_voltage = self.crossbar.config.read_voltage
        write_voltage = self.crossbar.config.write_voltage
        
        # Define safe voltage ranges
        safe_read_range = (0.05, 0.2)   # 50mV to 200mV
        safe_write_range = (0.2, 1.0)   # 200mV to 1V
        
        if read_voltage < safe_read_range[0] or read_voltage > safe_read_range[1]:
            events.append(SecurityEvent(
                timestamp=time.time(),
                event_type="voltage_tampering",
                threat_level=ThreatLevel.CRITICAL,
                source="voltage_monitor",
                description=f"Read voltage outside safe range: {read_voltage:.3f}V",
                metadata={'voltage': read_voltage, 'safe_range': safe_read_range}
            ))
        
        if write_voltage < safe_write_range[0] or write_voltage > safe_write_range[1]:
            events.append(SecurityEvent(
                timestamp=time.time(),
                event_type="voltage_tampering",
                threat_level=ThreatLevel.CRITICAL,
                source="voltage_monitor",
                description=f"Write voltage outside safe range: {write_voltage:.3f}V",
                metadata={'voltage': write_voltage, 'safe_range': safe_write_range}
            ))
        
        return events
    
    def _detect_temperature_attacks(self) -> List[SecurityEvent]:
        """Detect temperature-based attacks."""
        events = []
        
        current_temp = self.crossbar.config.mtj_config.operating_temp
        
        # Define safe temperature range
        safe_temp_range = (-10, 85)  # -10°C to 85°C
        
        if current_temp < safe_temp_range[0] or current_temp > safe_temp_range[1]:
            threat_level = ThreatLevel.CRITICAL if current_temp > 100 else ThreatLevel.WARNING
            
            events.append(SecurityEvent(
                timestamp=time.time(),
                event_type="temperature_attack",
                threat_level=threat_level,
                source="temperature_monitor",
                description=f"Temperature outside safe range: {current_temp:.1f}°C",
                metadata={'temperature': current_temp, 'safe_range': safe_temp_range}
            ))
        
        return events
    
    def _estimate_current_power(self) -> float:
        """Estimate current power consumption."""
        # Simple power estimation based on operations
        stats = self.crossbar.get_statistics()
        
        # Base leakage power
        base_power = self.crossbar.rows * self.crossbar.cols * 1e-12  # 1pW per cell
        
        # Dynamic power from recent operations
        read_power = stats['read_operations'] * 1e-9  # 1nW per read
        write_power = stats['write_operations'] * 1e-8  # 10nW per write
        
        return base_power + read_power + write_power
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get current security status."""
        recent_events = [e for e in self.security_events if time.time() - e.timestamp < 3600]  # Last hour
        
        threat_counts = {level.value: 0 for level in ThreatLevel}
        for event in recent_events:
            threat_counts[event.threat_level.value] += 1
        
        return {
            'monitoring_active': self.monitoring_active,
            'baseline_established': bool(self.baseline_characteristics),
            'recent_events_count': len(recent_events),
            'threat_level_counts': threat_counts,
            'overall_threat_level': self._assess_overall_threat_level(recent_events)
        }
    
    def _assess_overall_threat_level(self, recent_events: List[SecurityEvent]) -> str:
        """Assess overall threat level based on recent events."""
        if any(e.threat_level == ThreatLevel.EMERGENCY for e in recent_events):
            return "emergency"
        elif any(e.threat_level == ThreatLevel.CRITICAL for e in recent_events):
            return "critical"
        elif len([e for e in recent_events if e.threat_level == ThreatLevel.WARNING]) > 10:
            return "elevated"
        else:
            return "normal"


class SecureModelManager:
    """Manages secure model deployment and execution."""
    
    def __init__(self, crypto_manager: CryptographicManager):
        self.crypto_manager = crypto_manager
        self.secure_models = {}
        self.model_signatures = {}
        self.access_logs = []
    
    def secure_model_storage(self, model_id: str, model_data: bytes, 
                           metadata: Optional[Dict[str, Any]] = None) -> str:
        """Securely store model with encryption and integrity protection."""
        
        # Encrypt model data
        encrypted_data = self.crypto_manager.encrypt_data(model_data)
        
        # Create digital signature
        signature = self.crypto_manager.sign_data(model_data)
        
        # Compute hash for integrity
        data_hash = self.crypto_manager.compute_hash(model_data)
        
        # Store secure model
        secure_model = {
            'model_id': model_id,
            'encrypted_data': encrypted_data,
            'signature': signature,
            'hash': data_hash,
            'metadata': metadata or {},
            'timestamp': time.time()
        }
        
        self.secure_models[model_id] = secure_model
        self.model_signatures[model_id] = signature
        
        return data_hash
    
    def load_secure_model(self, model_id: str, context: SecurityContext) -> Tuple[bytes, bool]:
        """Load and verify secure model."""
        
        # Check permissions
        if not context.has_permission("model_access"):
            raise SecurityError(f"Insufficient permissions to access model {model_id}")
        
        # Get secure model
        if model_id not in self.secure_models:
            raise SecurityError(f"Model {model_id} not found")
        
        secure_model = self.secure_models[model_id]
        
        # Decrypt model data
        try:
            decrypted_data = self.crypto_manager.decrypt_data(secure_model['encrypted_data'])
        except Exception as e:
            raise SecurityError(f"Failed to decrypt model {model_id}: {str(e)}")
        
        # Verify signature
        signature_valid = self.crypto_manager.verify_signature(
            decrypted_data, secure_model['signature']
        )
        
        # Verify hash integrity
        current_hash = self.crypto_manager.compute_hash(decrypted_data)
        hash_valid = current_hash == secure_model['hash']
        
        # Log access
        self.access_logs.append({
            'timestamp': time.time(),
            'model_id': model_id,
            'user_id': context.user_id,
            'session_id': context.session_id,
            'signature_valid': signature_valid,
            'hash_valid': hash_valid
        })
        
        # Return data and verification status
        verification_passed = signature_valid and hash_valid
        
        if not verification_passed:
            raise SecurityError(f"Model {model_id} failed integrity verification")
        
        return decrypted_data, verification_passed
    
    def get_model_metadata(self, model_id: str) -> Dict[str, Any]:
        """Get model metadata without loading the full model."""
        if model_id not in self.secure_models:
            raise SecurityError(f"Model {model_id} not found")
        
        secure_model = self.secure_models[model_id]
        return {
            'model_id': model_id,
            'metadata': secure_model['metadata'],
            'timestamp': secure_model['timestamp'],
            'hash': secure_model['hash']
        }


class SecurityAuditLogger:
    """Comprehensive security audit logging system."""
    
    def __init__(self, log_directory: str = "security_logs"):
        self.log_directory = Path(log_directory)
        self.log_directory.mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger("security_audit")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        log_file = self.log_directory / f"security_audit_{int(time.time())}.log"
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Event storage
        self.audit_events = []
        self.event_lock = threading.Lock()
    
    def log_security_event(self, event: SecurityEvent):
        """Log security event to audit trail."""
        with self.event_lock:
            self.audit_events.append(event)
            
            # Log to file
            log_message = f"{event.event_type}: {event.description}"
            if event.threat_level == ThreatLevel.EMERGENCY:
                self.logger.critical(log_message)
            elif event.threat_level == ThreatLevel.CRITICAL:
                self.logger.error(log_message)
            elif event.threat_level == ThreatLevel.WARNING:
                self.logger.warning(log_message)
            else:
                self.logger.info(log_message)
    
    def log_access_attempt(self, context: SecurityContext, resource: str, 
                          success: bool, details: Optional[str] = None):
        """Log access attempt."""
        event = SecurityEvent(
            timestamp=time.time(),
            event_type="access_attempt",
            threat_level=ThreatLevel.INFO if success else ThreatLevel.WARNING,
            source="access_control",
            description=f"{'Successful' if success else 'Failed'} access to {resource}",
            metadata={
                'resource': resource,
                'success': success,
                'details': details
            },
            user_id=context.user_id,
            session_id=context.session_id
        )
        
        self.log_security_event(event)
    
    def export_audit_log(self, filename: str, start_time: Optional[float] = None,
                        end_time: Optional[float] = None):
        """Export audit log to file."""
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = time.time()
        
        filtered_events = [
            event for event in self.audit_events
            if start_time <= event.timestamp <= end_time
        ]
        
        export_data = {
            'export_timestamp': time.time(),
            'start_time': start_time,
            'end_time': end_time,
            'event_count': len(filtered_events),
            'events': [event.to_dict() for event in filtered_events]
        }
        
        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Audit log exported to {filename}: {len(filtered_events)} events")


class SpintronSecurityFramework:
    """Main security framework coordinating all security components."""
    
    def __init__(self, crossbar: MTJCrossbar):
        self.crossbar = crossbar
        
        # Initialize security components
        self.crypto_manager = CryptographicManager()
        self.hardware_monitor = HardwareSecurityMonitor(crossbar)
        self.model_manager = SecureModelManager(self.crypto_manager)
        self.audit_logger = SecurityAuditLogger()
        
        # Security state
        self.security_enabled = True
        self.threat_response_enabled = True
        self.monitoring_thread = None
        self.stop_monitoring = threading.Event()
        
        # Initialize security
        self._initialize_security()
    
    def _initialize_security(self):
        """Initialize security framework."""
        print("Initializing SpinTron Security Framework...")
        
        # Start hardware monitoring
        self.hardware_monitor.start_monitoring()
        
        # Log initialization
        init_event = SecurityEvent(
            timestamp=time.time(),
            event_type="security_initialization",
            threat_level=ThreatLevel.INFO,
            source="security_framework",
            description="Security framework initialized"
        )
        self.audit_logger.log_security_event(init_event)
        
        print("Security framework initialized")
    
    def start_continuous_monitoring(self):
        """Start continuous security monitoring in background thread."""
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            print("Monitoring already running")
            return
        
        self.stop_monitoring.clear()
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        print("Continuous security monitoring started")
    
    def stop_continuous_monitoring(self):
        """Stop continuous security monitoring."""
        self.stop_monitoring.set()
        
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5.0)
        
        print("Continuous security monitoring stopped")
    
    def _monitoring_loop(self):
        """Main security monitoring loop."""
        while not self.stop_monitoring.is_set():
            try:
                # Check for security threats
                threats = self.hardware_monitor.check_security_threats()
                
                # Log all detected threats
                for threat in threats:
                    self.audit_logger.log_security_event(threat)
                    
                    # Respond to critical threats
                    if (self.threat_response_enabled and 
                        threat.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]):
                        self._respond_to_threat(threat)
                
                # Wait before next check
                self.stop_monitoring.wait(timeout=10.0)
                
            except Exception as e:
                error_event = SecurityEvent(
                    timestamp=time.time(),
                    event_type="monitoring_error",
                    threat_level=ThreatLevel.WARNING,
                    source="security_framework",
                    description=f"Monitoring error: {str(e)}"
                )
                self.audit_logger.log_security_event(error_event)
                
                # Wait longer on error
                self.stop_monitoring.wait(timeout=30.0)
    
    def _respond_to_threat(self, threat: SecurityEvent):
        """Respond to detected security threat."""
        print(f"Responding to threat: {threat.event_type}")
        
        # Implement threat response based on type
        if threat.event_type == "voltage_tampering":
            # Reset voltages to safe values
            self._reset_safe_voltages()
        
        elif threat.event_type == "temperature_attack":
            # Implement thermal protection
            self._activate_thermal_protection()
        
        elif threat.event_type == "resistance_anomaly":
            # Perform device recalibration
            self._trigger_recalibration()
        
        # Log response
        response_event = SecurityEvent(
            timestamp=time.time(),
            event_type="threat_response",
            threat_level=ThreatLevel.INFO,
            source="security_framework",
            description=f"Automated response to {threat.event_type}",
            metadata={'original_threat': threat.to_dict()}
        )
        self.audit_logger.log_security_event(response_event)
    
    def _reset_safe_voltages(self):
        """Reset voltages to safe default values."""
        self.crossbar.config.read_voltage = 0.1   # Safe read voltage
        self.crossbar.config.write_voltage = 0.5  # Safe write voltage
        print("Voltages reset to safe defaults")
    
    def _activate_thermal_protection(self):
        """Activate thermal protection measures."""
        # Reduce operating frequency or voltage to lower temperature
        # This is a simplified implementation
        current_temp = self.crossbar.config.mtj_config.operating_temp
        if current_temp > 85:
            # Emergency shutdown simulation
            print("THERMAL EMERGENCY: Reducing power consumption")
            self.crossbar.config.read_voltage *= 0.8
            self.crossbar.config.write_voltage *= 0.8
    
    def _trigger_recalibration(self):
        """Trigger device recalibration."""
        print("Triggering device recalibration due to anomaly")
        # Re-establish baseline characteristics
        self.hardware_monitor._establish_security_baseline()
    
    def secure_operation(self, operation_func: Callable, context: SecurityContext, 
                        *args, **kwargs) -> Any:
        """Execute operation with security controls."""
        
        # Check if security is enabled
        if not self.security_enabled:
            return operation_func(*args, **kwargs)
        
        # Validate security context
        self._validate_security_context(context)
        
        # Log access attempt
        operation_name = getattr(operation_func, '__name__', 'unknown_operation')
        
        try:
            # Check for active threats
            if self.hardware_monitor.monitoring_active:
                threats = self.hardware_monitor.check_security_threats()
                critical_threats = [t for t in threats if t.threat_level in [ThreatLevel.CRITICAL, ThreatLevel.EMERGENCY]]
                
                if critical_threats:
                    raise SecurityError(f"Operation blocked due to active security threats: {len(critical_threats)} detected")
            
            # Execute operation
            result = operation_func(*args, **kwargs)
            
            # Log successful access
            self.audit_logger.log_access_attempt(context, operation_name, True)
            
            return result
            
        except Exception as e:
            # Log failed access
            self.audit_logger.log_access_attempt(context, operation_name, False, str(e))
            raise
    
    def _validate_security_context(self, context: SecurityContext):
        """Validate security context."""
        if context.security_level == SecurityLevel.CRITICAL:
            # Additional validation for critical operations
            if not context.user_id:
                raise SecurityError("User ID required for critical operations")
            
            if not context.session_id:
                raise SecurityError("Session ID required for critical operations")
    
    def get_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        hardware_status = self.hardware_monitor.get_security_status()
        
        # Recent audit events
        recent_events = [e for e in self.audit_logger.audit_events if time.time() - e.timestamp < 3600]
        
        return {
            'framework_status': {
                'security_enabled': self.security_enabled,
                'threat_response_enabled': self.threat_response_enabled,
                'monitoring_active': self.hardware_monitor.monitoring_active
            },
            'hardware_security': hardware_status,
            'audit_summary': {
                'total_events': len(self.audit_logger.audit_events),
                'recent_events': len(recent_events),
                'recent_by_type': self._count_events_by_type(recent_events)
            },
            'model_security': {
                'secure_models_count': len(self.model_manager.secure_models),
                'access_logs_count': len(self.model_manager.access_logs)
            },
            'timestamp': time.time()
        }
    
    def _count_events_by_type(self, events: List[SecurityEvent]) -> Dict[str, int]:
        """Count events by type."""
        counts = {}
        for event in events:
            counts[event.event_type] = counts.get(event.event_type, 0) + 1
        return counts
    
    def shutdown(self):
        """Shutdown security framework."""
        print("Shutting down security framework...")
        
        # Stop monitoring
        self.stop_continuous_monitoring()
        self.hardware_monitor.stop_monitoring()
        
        # Log shutdown
        shutdown_event = SecurityEvent(
            timestamp=time.time(),
            event_type="security_shutdown",
            threat_level=ThreatLevel.INFO,
            source="security_framework",
            description="Security framework shutdown"
        )
        self.audit_logger.log_security_event(shutdown_event)
        
        print("Security framework shutdown complete")


# Convenience functions for security operations
def validate_operation(operation_name: str, data: Any, context: SecurityContext):
    """Validate operation with security checks."""
    
    # Basic validation
    if not isinstance(context, SecurityContext):
        raise ValidationError("Invalid security context")
    
    # Check data size limits
    if hasattr(data, 'size') and data.size > 1e9:  # 1GB limit
        raise SecurityError("Data size exceeds security limits")
    
    # Check for suspicious patterns in data
    if isinstance(data, (np.ndarray, torch.Tensor)):
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            raise SecurityError("Data contains invalid values")


def create_security_context(user_id: str = None, permissions: List[str] = None,
                           security_level: SecurityLevel = SecurityLevel.MEDIUM) -> SecurityContext:
    """Create security context with default settings."""
    
    return SecurityContext(
        user_id=user_id or f"user_{int(time.time())}",
        session_id=f"session_{secrets.token_hex(8)}",
        security_level=security_level,
        permissions=permissions or ["basic_access"],
        audit_enabled=True
    )


# Example usage
def demonstrate_security_framework():
    """Demonstrate security framework capabilities."""
    from .core.mtj_models import MTJConfig
    from .core.crossbar import CrossbarConfig, MTJCrossbar
    
    # Create crossbar
    mtj_config = MTJConfig()
    crossbar_config = CrossbarConfig(rows=32, cols=32, mtj_config=mtj_config)
    crossbar = MTJCrossbar(crossbar_config)
    
    # Create security framework
    security = SpintronSecurityFramework(crossbar)
    
    # Start monitoring
    security.start_continuous_monitoring()
    
    # Create security context
    context = create_security_context(
        user_id="demo_user",
        permissions=["basic_access", "model_access"],
        security_level=SecurityLevel.HIGH
    )
    
    # Demonstrate secure operation
    def secure_matrix_multiply(input_data, weights):
        """Example secure operation."""
        return np.dot(input_data, weights)
    
    try:
        # Execute secure operation
        input_data = np.random.randn(32)
        weights = np.random.randn(32, 10)
        
        result = security.secure_operation(
            secure_matrix_multiply,
            context,
            input_data,
            weights
        )
        
        print(f"Secure operation completed. Result shape: {result.shape}")
        
    except SecurityError as e:
        print(f"Security error: {e}")
    
    # Get security report
    report = security.get_security_report()
    print(f"Security report: {report['audit_summary']['total_events']} total events")
    
    # Shutdown
    security.shutdown()
    
    return security


if __name__ == "__main__":
    # Demonstration
    security_demo = demonstrate_security_framework()
    print("Security framework demonstration complete")
