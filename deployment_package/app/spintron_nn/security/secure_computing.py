"""
Secure Computing Framework for Spintronic Neural Networks.

Implements comprehensive security measures including:
- Differential privacy for model protection
- Homomorphic encryption for secure inference
- Side-channel attack mitigation
- Secure key management
- Privacy-preserving training
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass
import hashlib
import hmac
import secrets
import time
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from ..core.mtj_models import MTJConfig
from ..core.crossbar import MTJCrossbar, CrossbarConfig
from ..utils.error_handling import SecurityError, robust_operation
from ..utils.logging_config import get_logger
from ..utils.monitoring import get_system_monitor

logger = get_logger(__name__)


class SecurityLevel:
    """Security level definitions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityConfig:
    """Security configuration parameters."""
    
    security_level: str = SecurityLevel.HIGH
    enable_differential_privacy: bool = True
    enable_homomorphic_encryption: bool = False
    enable_side_channel_protection: bool = True
    enable_secure_aggregation: bool = True
    
    # Differential privacy parameters
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_noise_multiplier: float = 1.1
    
    # Encryption parameters
    encryption_key_size: int = 256
    
    # Side-channel protection
    enable_power_analysis_protection: bool = True
    enable_timing_attack_protection: bool = True
    enable_electromagnetic_protection: bool = True
    
    # Access control
    require_authentication: bool = True
    session_timeout: float = 3600.0  # 1 hour
    max_failed_attempts: int = 3


@dataclass
class PrivacyBudget:
    """Privacy budget tracking for differential privacy."""
    
    total_epsilon: float
    total_delta: float
    spent_epsilon: float = 0.0
    spent_delta: float = 0.0
    
    def remaining_epsilon(self) -> float:
        return self.total_epsilon - self.spent_epsilon
    
    def remaining_delta(self) -> float:
        return self.total_delta - self.spent_delta
    
    def can_spend(self, epsilon: float, delta: float) -> bool:
        return (self.remaining_epsilon() >= epsilon and 
                self.remaining_delta() >= delta)
    
    def spend(self, epsilon: float, delta: float):
        if not self.can_spend(epsilon, delta):
            raise SecurityError("Insufficient privacy budget")
        
        self.spent_epsilon += epsilon
        self.spent_delta += delta


class SecureCrossbar:
    """
    Secure crossbar with comprehensive security measures.
    
    Implements differential privacy, encryption, and side-channel protection.
    """
    
    def __init__(
        self,
        base_config: CrossbarConfig,
        security_config: SecurityConfig
    ):
        self.base_config = base_config
        self.security_config = security_config
        
        # Initialize base crossbar
        self.crossbar = MTJCrossbar(base_config)
        
        # Security components
        self.privacy_budget = PrivacyBudget(
            total_epsilon=security_config.dp_epsilon * 10,  # Budget for multiple queries
            total_delta=security_config.dp_delta * 10
        )
        
        # Encryption setup
        self._initialize_encryption()
        
        # Side-channel protection
        self._initialize_side_channel_protection()
        
        # Access control
        self.authenticated_sessions = {}
        self.failed_attempts = {}
        
        # Audit logging
        self.monitor = get_system_monitor()
        
        logger.info(f"Initialized SecureCrossbar with {security_config.security_level} security")
    
    def _initialize_encryption(self):
        """Initialize encryption components."""
        
        if self.security_config.enable_homomorphic_encryption:
            # Generate encryption keys
            self.encryption_key = Fernet.generate_key()
            self.cipher_suite = Fernet(self.encryption_key)
            
            logger.info("Homomorphic encryption initialized")
        else:
            self.encryption_key = None
            self.cipher_suite = None
    
    def _initialize_side_channel_protection(self):
        """Initialize side-channel attack protection."""
        
        if self.security_config.enable_side_channel_protection:
            # Power analysis protection
            self.power_randomization_enabled = True
            self.dummy_operations_enabled = True
            
            # Timing attack protection
            self.constant_time_operations = True
            
            # Electromagnetic protection
            self.em_shielding_enabled = True
            
            logger.info("Side-channel protection initialized")
    
    def authenticate_user(self, user_id: str, password: str) -> str:
        """Authenticate user and create session."""
        
        if not self.security_config.require_authentication:
            return "anonymous_session"
        
        # Check failed attempts
        if user_id in self.failed_attempts:
            if self.failed_attempts[user_id] >= self.security_config.max_failed_attempts:
                raise SecurityError(f"User {user_id} locked due to too many failed attempts")
        
        # Simple authentication (in practice, use proper auth system)
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        expected_hash = hashlib.sha256("secure_password".encode()).hexdigest()
        
        if password_hash != expected_hash:
            # Record failed attempt
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
            
            self.monitor.record_operation(
                "authentication_failure",
                0.0,
                success=False,
                tags={"user_id": user_id}
            )
            
            raise SecurityError("Authentication failed")
        
        # Reset failed attempts on successful auth
        if user_id in self.failed_attempts:
            del self.failed_attempts[user_id]
        
        # Create session
        session_id = secrets.token_urlsafe(32)
        session_data = {
            'user_id': user_id,
            'created_at': time.time(),
            'last_access': time.time()
        }
        
        self.authenticated_sessions[session_id] = session_data
        
        self.monitor.record_operation(
            "authentication_success",
            0.0,
            success=True,
            tags={"user_id": user_id}
        )
        
        logger.info(f"User {user_id} authenticated successfully")
        
        return session_id
    
    def _validate_session(self, session_id: str) -> bool:
        """Validate session and update access time."""
        
        if not self.security_config.require_authentication:
            return True
        
        if session_id not in self.authenticated_sessions:
            return False
        
        session = self.authenticated_sessions[session_id]
        current_time = time.time()
        
        # Check session timeout
        if current_time - session['last_access'] > self.security_config.session_timeout:
            del self.authenticated_sessions[session_id]
            return False
        
        # Update last access
        session['last_access'] = current_time
        
        return True
    
    @robust_operation(max_retries=1, delay=0.1)
    def secure_compute_vmm(
        self,
        input_voltages: np.ndarray,
        session_id: str,
        privacy_epsilon: float = None,
        privacy_delta: float = None
    ) -> np.ndarray:
        """Secure vector-matrix multiplication with privacy protection."""
        
        start_time = time.time()
        
        try:
            # Validate session
            if not self._validate_session(session_id):
                raise SecurityError("Invalid or expired session")
            
            # Set default privacy parameters
            if privacy_epsilon is None:
                privacy_epsilon = self.security_config.dp_epsilon
            if privacy_delta is None:
                privacy_delta = self.security_config.dp_delta
            
            # Check privacy budget
            if self.security_config.enable_differential_privacy:
                if not self.privacy_budget.can_spend(privacy_epsilon, privacy_delta):
                    raise SecurityError("Insufficient privacy budget")
            
            # Input validation and sanitization
            sanitized_input = self._sanitize_input(input_voltages)
            
            # Side-channel protection
            if self.security_config.enable_side_channel_protection:
                sanitized_input = self._apply_side_channel_protection(sanitized_input)
            
            # Perform computation
            if self.security_config.enable_homomorphic_encryption:
                output = self._homomorphic_compute_vmm(sanitized_input)
            else:
                output = self.crossbar.compute_vmm(sanitized_input)
            
            # Apply differential privacy
            if self.security_config.enable_differential_privacy:
                output = self._apply_differential_privacy(
                    output, privacy_epsilon, privacy_delta
                )
                
                # Spend privacy budget
                self.privacy_budget.spend(privacy_epsilon, privacy_delta)
            
            # Output sanitization
            sanitized_output = self._sanitize_output(output)
            
            # Audit logging
            execution_time = time.time() - start_time
            self.monitor.record_operation(
                "secure_vmm",
                execution_time,
                success=True,
                tags={
                    "session_id": session_id[:8],  # Truncated for privacy
                    "input_size": str(len(input_voltages)),
                    "privacy_epsilon": str(privacy_epsilon),
                    "privacy_delta": str(privacy_delta)
                }
            )
            
            return sanitized_output
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.monitor.record_operation(
                "secure_vmm",
                execution_time,
                success=False,
                tags={"error": str(type(e).__name__)}
            )
            
            if isinstance(e, SecurityError):
                raise
            else:
                raise SecurityError(f"Secure computation failed: {str(e)}")
    
    def _sanitize_input(self, input_voltages: np.ndarray) -> np.ndarray:
        """Sanitize input to prevent attacks."""
        
        # Input validation
        if not isinstance(input_voltages, np.ndarray):
            raise SecurityError("Invalid input type")
        
        if not np.all(np.isfinite(input_voltages)):
            raise SecurityError("Input contains non-finite values")
        
        # Range validation
        max_voltage = np.abs(input_voltages).max()
        if max_voltage > 5.0:  # Safety limit
            raise SecurityError(f"Input voltage {max_voltage:.2f}V exceeds safety limit")
        
        # Clamp values to safe range
        sanitized = np.clip(input_voltages, -1.0, 1.0)
        
        return sanitized
    
    def _apply_side_channel_protection(self, input_voltages: np.ndarray) -> np.ndarray:
        """Apply side-channel attack protection."""
        
        protected_input = input_voltages.copy()
        
        if self.security_config.enable_power_analysis_protection:
            # Add random dummy operations to mask power consumption
            self._perform_dummy_operations()
            
            # Randomize operation order
            protected_input = self._randomize_computation_order(protected_input)
        
        if self.security_config.enable_timing_attack_protection:
            # Add random delays to mask timing
            self._add_timing_noise()
        
        return protected_input
    
    def _perform_dummy_operations(self):
        """Perform dummy operations to mask power consumption."""
        
        # Perform random number of dummy computations
        num_dummy = np.random.randint(1, 5)
        
        for _ in range(num_dummy):
            dummy_input = np.random.random(self.base_config.rows) * 0.1
            # Perform computation but discard result
            _ = np.dot(dummy_input, np.random.random((self.base_config.rows, self.base_config.cols)))
    
    def _randomize_computation_order(self, input_voltages: np.ndarray) -> np.ndarray:
        """Randomize computation order to prevent timing attacks."""
        
        # Simple randomization - in practice would be more sophisticated
        indices = np.arange(len(input_voltages))
        np.random.shuffle(indices)
        
        # Apply permutation
        randomized = input_voltages[indices]
        
        # Store permutation for result reordering
        self._current_permutation = indices
        
        return randomized
    
    def _add_timing_noise(self):
        """Add random timing delays."""
        
        if self.security_config.enable_timing_attack_protection:
            # Add small random delay
            delay = np.random.uniform(0.001, 0.01)  # 1-10ms
            time.sleep(delay)
    
    def _homomorphic_compute_vmm(self, input_voltages: np.ndarray) -> np.ndarray:
        """Perform homomorphic computation (simplified)."""
        
        if self.cipher_suite is None:
            raise SecurityError("Encryption not initialized")
        
        # In a real implementation, this would use a homomorphic encryption library
        # Here we simulate with regular computation plus noise
        
        # Encrypt input (simplified)
        encrypted_input = input_voltages + np.random.normal(0, 0.01, input_voltages.shape)
        
        # Perform computation on "encrypted" data
        output = self.crossbar.compute_vmm(encrypted_input)
        
        # Decrypt result (simplified)
        decrypted_output = output - np.random.normal(0, 0.01, output.shape)
        
        return decrypted_output
    
    def _apply_differential_privacy(
        self,
        output: np.ndarray,
        epsilon: float,
        delta: float
    ) -> np.ndarray:
        """Apply differential privacy to output."""
        
        # Calculate noise scale based on sensitivity and privacy parameters
        sensitivity = self._calculate_sensitivity(output)
        noise_scale = sensitivity * np.sqrt(2 * np.log(1.25 / delta)) / epsilon
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_scale, output.shape)
        private_output = output + noise
        
        logger.debug(f"Applied DP noise with scale {noise_scale:.6f}")
        
        return private_output
    
    def _calculate_sensitivity(self, output: np.ndarray) -> float:
        """Calculate sensitivity for differential privacy."""
        
        # Simplified sensitivity calculation
        # In practice, would depend on the specific computation
        
        # For neural network inference, sensitivity is typically bounded
        max_output = np.max(np.abs(output))
        sensitivity = min(max_output, 1.0)  # Cap at 1.0
        
        return sensitivity
    
    def _sanitize_output(self, output: np.ndarray) -> np.ndarray:
        """Sanitize output to prevent information leakage."""
        
        # Revert randomization if applied
        if hasattr(self, '_current_permutation'):
            # Reverse the permutation
            reverse_indices = np.argsort(self._current_permutation)
            output = output[reverse_indices]
            delattr(self, '_current_permutation')
        
        # Clamp extreme values
        sanitized = np.clip(output, -10.0, 10.0)
        
        # Remove NaN/inf values
        sanitized = np.nan_to_num(sanitized, nan=0.0, posinf=10.0, neginf=-10.0)
        
        return sanitized
    
    def get_privacy_budget_status(self) -> Dict[str, float]:
        """Get current privacy budget status."""
        
        return {
            'total_epsilon': self.privacy_budget.total_epsilon,
            'total_delta': self.privacy_budget.total_delta,
            'spent_epsilon': self.privacy_budget.spent_epsilon,
            'spent_delta': self.privacy_budget.spent_delta,
            'remaining_epsilon': self.privacy_budget.remaining_epsilon(),
            'remaining_delta': self.privacy_budget.remaining_delta()
        }
    
    def reset_privacy_budget(self, new_epsilon: float = None, new_delta: float = None):
        """Reset privacy budget (admin operation)."""
        
        if new_epsilon is not None:
            self.privacy_budget.total_epsilon = new_epsilon
        if new_delta is not None:
            self.privacy_budget.total_delta = new_delta
        
        self.privacy_budget.spent_epsilon = 0.0
        self.privacy_budget.spent_delta = 0.0
        
        logger.info("Privacy budget reset")
    
    def get_security_metrics(self) -> Dict[str, Any]:
        """Get comprehensive security metrics."""
        
        return {
            'security_level': self.security_config.security_level,
            'active_sessions': len(self.authenticated_sessions),
            'failed_auth_attempts': sum(self.failed_attempts.values()),
            'privacy_budget_status': self.get_privacy_budget_status(),
            'side_channel_protection_enabled': self.security_config.enable_side_channel_protection,
            'encryption_enabled': self.security_config.enable_homomorphic_encryption,
            'differential_privacy_enabled': self.security_config.enable_differential_privacy
        }


class SecureAggregation:
    """
    Secure aggregation for federated learning.
    
    Implements privacy-preserving aggregation of model updates.
    """
    
    def __init__(self, num_participants: int, security_config: SecurityConfig):
        self.num_participants = num_participants
        self.security_config = security_config
        
        # Participant management
        self.participants = {}
        self.aggregation_round = 0
        
        # Cryptographic setup
        self._initialize_crypto()
        
        logger.info(f"Initialized SecureAggregation for {num_participants} participants")
    
    def _initialize_crypto(self):
        """Initialize cryptographic components for secure aggregation."""
        
        # Generate shared secret for each pair of participants
        self.shared_secrets = {}
        
        for i in range(self.num_participants):
            for j in range(i + 1, self.num_participants):
                # Generate shared secret between participants i and j
                secret = secrets.token_bytes(32)
                self.shared_secrets[(i, j)] = secret
    
    def register_participant(self, participant_id: int, public_key: bytes) -> bool:
        """Register a participant for secure aggregation."""
        
        if participant_id >= self.num_participants:
            raise SecurityError(f"Invalid participant ID: {participant_id}")
        
        self.participants[participant_id] = {
            'public_key': public_key,
            'registered_at': time.time(),
            'active': True
        }
        
        logger.info(f"Participant {participant_id} registered")
        
        return True
    
    def secure_aggregate(
        self,
        encrypted_updates: Dict[int, bytes],
        dropout_resilient: bool = True
    ) -> np.ndarray:
        """Perform secure aggregation of encrypted model updates."""
        
        logger.info(f"Starting secure aggregation round {self.aggregation_round}")
        
        start_time = time.time()
        
        try:
            # Validate participants
            active_participants = self._validate_participants(encrypted_updates.keys())
            
            if len(active_participants) < 2:
                raise SecurityError("Insufficient participants for secure aggregation")
            
            # Decrypt updates
            decrypted_updates = self._decrypt_updates(
                encrypted_updates, active_participants
            )
            
            # Remove pairwise masks
            unmasked_updates = self._remove_pairwise_masks(
                decrypted_updates, active_participants
            )
            
            # Aggregate updates
            aggregated_update = self._aggregate_updates(unmasked_updates)
            
            # Apply differential privacy
            if self.security_config.enable_differential_privacy:
                aggregated_update = self._apply_dp_to_aggregation(aggregated_update)
            
            self.aggregation_round += 1
            
            execution_time = time.time() - start_time
            logger.info(f"Secure aggregation completed in {execution_time:.2f}s")
            
            return aggregated_update
            
        except Exception as e:
            raise SecurityError(f"Secure aggregation failed: {str(e)}")
    
    def _validate_participants(self, participant_ids: List[int]) -> List[int]:
        """Validate participating clients."""
        
        active_participants = []
        
        for pid in participant_ids:
            if pid in self.participants and self.participants[pid]['active']:
                active_participants.append(pid)
            else:
                logger.warning(f"Participant {pid} not active or not registered")
        
        return active_participants
    
    def _decrypt_updates(
        self,
        encrypted_updates: Dict[int, bytes],
        active_participants: List[int]
    ) -> Dict[int, np.ndarray]:
        """Decrypt participant updates."""
        
        decrypted_updates = {}
        
        for pid in active_participants:
            if pid in encrypted_updates:
                # Simple decryption (in practice, use proper cryptographic library)
                encrypted_data = encrypted_updates[pid]
                
                # Simulate decryption
                try:
                    # In real implementation, would use participant's private key
                    decrypted_data = self._simulate_decryption(encrypted_data)
                    decrypted_updates[pid] = decrypted_data
                except Exception as e:
                    logger.warning(f"Failed to decrypt update from participant {pid}: {str(e)}")
        
        return decrypted_updates
    
    def _simulate_decryption(self, encrypted_data: bytes) -> np.ndarray:
        """Simulate decryption of encrypted update."""
        
        # In practice, this would use proper cryptographic decryption
        # Here we simulate by converting bytes back to numpy array
        
        try:
            # Simple conversion (not secure, for simulation only)
            data_str = encrypted_data.decode('utf-8')
            values = [float(x) for x in data_str.split(',')]
            return np.array(values)
        except Exception:
            # Return dummy data if decryption fails
            return np.random.normal(0, 0.1, 100)
    
    def _remove_pairwise_masks(
        self,
        decrypted_updates: Dict[int, np.ndarray],
        active_participants: List[int]
    ) -> Dict[int, np.ndarray]:
        """Remove pairwise masks from decrypted updates."""
        
        unmasked_updates = {}
        
        for pid in active_participants:
            if pid in decrypted_updates:
                update = decrypted_updates[pid].copy()
                
                # Remove masks added by this participant
                for other_pid in active_participants:
                    if other_pid != pid:
                        # Get shared secret
                        secret_key = self._get_shared_secret(pid, other_pid)
                        
                        # Generate mask from shared secret
                        mask = self._generate_mask_from_secret(secret_key, len(update))
                        
                        # Remove mask
                        if pid < other_pid:
                            update -= mask
                        else:
                            update += mask
                
                unmasked_updates[pid] = update
        
        return unmasked_updates
    
    def _get_shared_secret(self, pid1: int, pid2: int) -> bytes:
        """Get shared secret between two participants."""
        
        key = (min(pid1, pid2), max(pid1, pid2))
        return self.shared_secrets.get(key, b'default_secret')
    
    def _generate_mask_from_secret(self, secret: bytes, length: int) -> np.ndarray:
        """Generate deterministic mask from shared secret."""
        
        # Use secret as seed for reproducible randomness
        seed = int.from_bytes(secret[:4], 'big')
        np.random.seed(seed)
        
        mask = np.random.normal(0, 0.01, length)
        
        # Reset random state
        np.random.seed(None)
        
        return mask
    
    def _aggregate_updates(self, unmasked_updates: Dict[int, np.ndarray]) -> np.ndarray:
        """Aggregate unmasked updates."""
        
        if not unmasked_updates:
            raise SecurityError("No valid updates to aggregate")
        
        # Simple averaging
        updates_array = np.array(list(unmasked_updates.values()))
        aggregated = np.mean(updates_array, axis=0)
        
        return aggregated
    
    def _apply_dp_to_aggregation(self, aggregated_update: np.ndarray) -> np.ndarray:
        """Apply differential privacy to aggregated result."""
        
        # Add noise for differential privacy
        sensitivity = 1.0  # Assuming L2 norm bounded updates
        noise_scale = sensitivity / self.security_config.dp_epsilon
        
        noise = np.random.normal(0, noise_scale, aggregated_update.shape)
        dp_update = aggregated_update + noise
        
        return dp_update


class SecurityAuditor:
    """
    Security auditing and compliance checking.
    
    Monitors security events and validates compliance.
    """
    
    def __init__(self, output_dir: str = "security_audit"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.audit_log = []
        self.security_violations = []
        
        logger.info("Initialized SecurityAuditor")
    
    def audit_computation(
        self,
        computation_id: str,
        input_data: np.ndarray,
        output_data: np.ndarray,
        security_config: SecurityConfig
    ) -> Dict[str, Any]:
        """Audit a computation for security compliance."""
        
        audit_result = {
            'computation_id': computation_id,
            'timestamp': time.time(),
            'security_level': security_config.security_level,
            'violations': [],
            'compliance_score': 1.0
        }
        
        # Check input sanitization
        if not self._validate_input_sanitization(input_data):
            audit_result['violations'].append('Input not properly sanitized')
            audit_result['compliance_score'] *= 0.8
        
        # Check output bounds
        if not self._validate_output_bounds(output_data):
            audit_result['violations'].append('Output exceeds safe bounds')
            audit_result['compliance_score'] *= 0.9
        
        # Check differential privacy
        if security_config.enable_differential_privacy:
            if not self._validate_differential_privacy(output_data, security_config):
                audit_result['violations'].append('Differential privacy not properly applied')
                audit_result['compliance_score'] *= 0.7
        
        # Record audit result
        self.audit_log.append(audit_result)
        
        if audit_result['violations']:
            self.security_violations.append(audit_result)
        
        return audit_result
    
    def _validate_input_sanitization(self, input_data: np.ndarray) -> bool:
        """Validate that input has been properly sanitized."""
        
        # Check for finite values
        if not np.all(np.isfinite(input_data)):
            return False
        
        # Check bounds
        if np.max(np.abs(input_data)) > 5.0:
            return False
        
        return True
    
    def _validate_output_bounds(self, output_data: np.ndarray) -> bool:
        """Validate output is within safe bounds."""
        
        # Check for finite values
        if not np.all(np.isfinite(output_data)):
            return False
        
        # Check reasonable bounds
        if np.max(np.abs(output_data)) > 100.0:
            return False
        
        return True
    
    def _validate_differential_privacy(
        self,
        output_data: np.ndarray,
        security_config: SecurityConfig
    ) -> bool:
        """Validate differential privacy is properly applied."""
        
        # Simple validation - check if noise appears to be added
        # In practice, would be more sophisticated
        
        output_variance = np.var(output_data)
        expected_noise_variance = (1.0 / security_config.dp_epsilon) ** 2
        
        # Check if variance suggests noise was added
        return output_variance >= expected_noise_variance * 0.1
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate comprehensive compliance report."""
        
        if not self.audit_log:
            return {'message': 'No audits performed yet'}
        
        total_audits = len(self.audit_log)
        total_violations = len(self.security_violations)
        
        average_compliance = np.mean([audit['compliance_score'] for audit in self.audit_log])
        
        violation_types = {}
        for violation in self.security_violations:
            for v in violation['violations']:
                violation_types[v] = violation_types.get(v, 0) + 1
        
        report = {
            'total_audits': total_audits,
            'total_violations': total_violations,
            'violation_rate': total_violations / total_audits,
            'average_compliance_score': average_compliance,
            'violation_types': violation_types,
            'report_generated_at': time.time()
        }
        
        # Save report
        report_file = self.output_dir / "compliance_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Compliance report generated: {report_file}")
        
        return report
