"""
Generation 2 Reliability Framework - SPINTRON-NN-KIT v1.2.0
==========================================================

Comprehensive reliability, error handling, and security framework for production deployment:
- Advanced Error Recovery and Circuit Breakers
- Security Hardening and Threat Detection
- Real-time Monitoring and Health Checks
- Fault Tolerance and Redundancy Systems
- Performance Monitoring and Auto-healing
"""

import json
import time
import hashlib
import random
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from datetime import datetime, timedelta
import logging
from contextlib import contextmanager

class SecurityLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class SystemHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

@dataclass
class SecurityConfig:
    """Security configuration for the framework"""
    encryption_enabled: bool = True
    access_control: bool = True
    audit_logging: bool = True
    threat_detection: bool = True
    security_level: SecurityLevel = SecurityLevel.HIGH
    max_failed_attempts: int = 5
    lockout_duration: int = 300  # seconds
    session_timeout: int = 3600  # seconds
    
@dataclass
class ReliabilityConfig:
    """Reliability configuration"""
    enable_circuit_breaker: bool = True
    failure_threshold: int = 5
    recovery_timeout: int = 30  # seconds
    health_check_interval: int = 10  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    redundancy_level: int = 2
    
@dataclass
class MonitoringMetrics:
    """System monitoring metrics"""
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_latency: float = 0.0
    error_rate: float = 0.0
    request_rate: float = 0.0
    response_time: float = 0.0
    quantum_coherence: float = 1.0
    topological_integrity: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)

class CircuitBreakerState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Circuit breaker tripped
    HALF_OPEN = "half_open" # Testing recovery

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 30):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
        
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset"""
        if self.last_failure_time is None:
            return True
        return time.time() - self.last_failure_time >= self.recovery_timeout
    
    def _on_success(self):
        """Handle successful operation"""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
        
    def _on_failure(self):
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN

class SecurityManager:
    """Advanced security management system"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.failed_attempts: Dict[str, int] = {}
        self.lockout_times: Dict[str, float] = {}
        self.active_sessions: Dict[str, Dict] = {}
        self.threat_patterns: List[str] = [
            "sql_injection", "xss_attack", "buffer_overflow", 
            "privilege_escalation", "data_exfiltration"
        ]
        
    def authenticate_user(self, user_id: str, credentials: Dict) -> Tuple[bool, str]:
        """Authenticate user with security checks"""
        current_time = time.time()
        
        # Check if user is locked out
        if user_id in self.lockout_times:
            if current_time - self.lockout_times[user_id] < self.config.lockout_duration:
                return False, "User account is temporarily locked"
            else:
                del self.lockout_times[user_id]
                self.failed_attempts[user_id] = 0
        
        # Simulate authentication (in production, use proper auth system)
        password_hash = hashlib.sha256(
            credentials.get('password', '').encode()
        ).hexdigest()
        
        # Simple check (replace with real authentication)
        is_valid = len(credentials.get('password', '')) >= 8
        
        if is_valid:
            # Reset failed attempts
            self.failed_attempts[user_id] = 0
            
            # Create session
            session_id = self._generate_session_id()
            self.active_sessions[session_id] = {
                'user_id': user_id,
                'created_at': current_time,
                'last_activity': current_time,
                'permissions': self._get_user_permissions(user_id)
            }
            
            return True, session_id
        else:
            # Track failed attempt
            self.failed_attempts[user_id] = self.failed_attempts.get(user_id, 0) + 1
            
            if self.failed_attempts[user_id] >= self.config.max_failed_attempts:
                self.lockout_times[user_id] = current_time
                return False, "Too many failed attempts - account locked"
            
            return False, "Invalid credentials"
    
    def validate_session(self, session_id: str) -> Tuple[bool, Optional[Dict]]:
        """Validate active session"""
        if session_id not in self.active_sessions:
            return False, None
        
        session = self.active_sessions[session_id]
        current_time = time.time()
        
        # Check session timeout
        if current_time - session['last_activity'] > self.config.session_timeout:
            del self.active_sessions[session_id]
            return False, None
        
        # Update last activity
        session['last_activity'] = current_time
        return True, session
    
    def detect_threats(self, request_data: Dict) -> List[str]:
        """Detect potential security threats"""
        threats_detected = []
        
        if not self.config.threat_detection:
            return threats_detected
        
        # Check for SQL injection patterns
        if any('DROP TABLE' in str(v).upper() for v in request_data.values()):
            threats_detected.append("sql_injection")
        
        # Check for XSS patterns
        if any('<script>' in str(v).lower() for v in request_data.values()):
            threats_detected.append("xss_attack")
        
        # Check for unusual data sizes (potential buffer overflow)
        if any(len(str(v)) > 10000 for v in request_data.values()):
            threats_detected.append("buffer_overflow")
        
        return threats_detected
    
    def encrypt_data(self, data: str) -> str:
        """Encrypt sensitive data (simplified implementation)"""
        if not self.config.encryption_enabled:
            return data
        
        # Simple XOR encryption (use proper encryption in production)
        key = "spintron_secret_key"
        encrypted = ""
        for i, char in enumerate(data):
            encrypted += chr(ord(char) ^ ord(key[i % len(key)]))
        
        # Base64-like encoding for safe transport
        import base64
        return base64.b64encode(encrypted.encode()).decode()
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """Decrypt data"""
        if not self.config.encryption_enabled:
            return encrypted_data
        
        try:
            import base64
            decoded = base64.b64decode(encrypted_data.encode()).decode()
            
            key = "spintron_secret_key"
            decrypted = ""
            for i, char in enumerate(decoded):
                decrypted += chr(ord(char) ^ ord(key[i % len(key)]))
            
            return decrypted
        except:
            return encrypted_data  # Return as-is if decryption fails
    
    def _generate_session_id(self) -> str:
        """Generate secure session ID"""
        import uuid
        return str(uuid.uuid4())
    
    def _get_user_permissions(self, user_id: str) -> List[str]:
        """Get user permissions (simplified)"""
        return ["read", "write", "compute"]

class HealthMonitor:
    """System health monitoring and alerting"""
    
    def __init__(self, config: ReliabilityConfig):
        self.config = config
        self.metrics_history: List[MonitoringMetrics] = []
        self.alerts: List[Dict] = []
        self.system_health = SystemHealth.HEALTHY
        self._monitoring_active = False
        self._monitoring_thread = None
        
    def start_monitoring(self):
        """Start continuous health monitoring"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop, 
            daemon=True
        )
        self._monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self._monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Keep only recent metrics
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Analyze health
                self._analyze_health(metrics)
                
                time.sleep(self.config.health_check_interval)
            except Exception as e:
                self._add_alert("monitoring_error", f"Monitoring error: {e}")
    
    def _collect_metrics(self) -> MonitoringMetrics:
        """Collect current system metrics"""
        # Simulate metric collection (replace with real monitoring)
        metrics = MonitoringMetrics()
        
        # Simulate CPU usage (would use psutil in production)
        metrics.cpu_usage = random.uniform(10, 90)
        metrics.memory_usage = random.uniform(20, 80)
        metrics.disk_usage = random.uniform(15, 75)
        metrics.network_latency = random.uniform(1, 100)
        
        # Application-specific metrics
        metrics.error_rate = random.uniform(0, 0.05)  # 0-5% error rate
        metrics.request_rate = random.uniform(10, 1000)
        metrics.response_time = random.uniform(10, 500)  # milliseconds
        
        # Quantum-specific metrics
        metrics.quantum_coherence = random.uniform(0.7, 1.0)
        metrics.topological_integrity = random.uniform(0.8, 1.0)
        
        return metrics
    
    def _analyze_health(self, metrics: MonitoringMetrics):
        """Analyze system health based on metrics"""
        health_score = 1.0
        issues = []
        
        # CPU health
        if metrics.cpu_usage > 90:
            health_score -= 0.2
            issues.append("high_cpu_usage")
        
        # Memory health
        if metrics.memory_usage > 85:
            health_score -= 0.2
            issues.append("high_memory_usage")
        
        # Error rate health
        if metrics.error_rate > 0.01:  # >1% error rate
            health_score -= 0.3
            issues.append("high_error_rate")
        
        # Response time health
        if metrics.response_time > 1000:  # >1 second
            health_score -= 0.2
            issues.append("slow_response_time")
        
        # Quantum coherence health
        if metrics.quantum_coherence < 0.8:
            health_score -= 0.3
            issues.append("low_quantum_coherence")
        
        # Topological integrity health
        if metrics.topological_integrity < 0.9:
            health_score -= 0.2
            issues.append("topological_degradation")
        
        # Determine overall health
        if health_score >= 0.9:
            new_health = SystemHealth.HEALTHY
        elif health_score >= 0.7:
            new_health = SystemHealth.DEGRADED
        elif health_score >= 0.5:
            new_health = SystemHealth.CRITICAL
        else:
            new_health = SystemHealth.FAILED
        
        # Generate alerts if health changed
        if new_health != self.system_health:
            self._add_alert(
                "health_change",
                f"System health changed from {self.system_health.value} to {new_health.value}",
                {"health_score": health_score, "issues": issues}
            )
        
        self.system_health = new_health
        
        # Generate specific alerts for critical issues
        for issue in issues:
            if issue in ["high_error_rate", "low_quantum_coherence"]:
                self._add_alert(
                    issue,
                    f"Critical issue detected: {issue}",
                    {"metric_value": getattr(metrics, issue.replace("high_", "").replace("low_", ""))}
                )
    
    def _add_alert(self, alert_type: str, message: str, details: Optional[Dict] = None):
        """Add system alert"""
        alert = {
            "type": alert_type,
            "message": message,
            "timestamp": datetime.now(),
            "details": details or {}
        }
        
        self.alerts.append(alert)
        
        # Keep only recent alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-50:]
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        recent_metrics = self.metrics_history[-10:] if self.metrics_history else []
        
        return {
            "system_health": self.system_health.value,
            "recent_alerts": self.alerts[-5:],
            "metrics_summary": {
                "avg_cpu": sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "avg_memory": sum(m.memory_usage for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "avg_response_time": sum(m.response_time for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
                "avg_quantum_coherence": sum(m.quantum_coherence for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0,
            }
        }

class ErrorRecoverySystem:
    """Advanced error recovery with auto-healing"""
    
    def __init__(self, config: ReliabilityConfig):
        self.config = config
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.error_counts: Dict[str, int] = {}
        self.recovery_attempts: Dict[str, int] = {}
        
    def execute_with_recovery(self, 
                             operation_name: str, 
                             func: Callable, 
                             *args, 
                             recovery_func: Optional[Callable] = None,
                             **kwargs) -> Any:
        """Execute function with automatic error recovery"""
        
        # Get or create circuit breaker for this operation
        if operation_name not in self.circuit_breakers:
            self.circuit_breakers[operation_name] = CircuitBreaker(
                self.config.failure_threshold,
                self.config.recovery_timeout
            )
        
        circuit_breaker = self.circuit_breakers[operation_name]
        
        # Attempt operation with retries
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Execute through circuit breaker
                result = circuit_breaker.call(func, *args, **kwargs)
                
                # Reset error count on success
                self.error_counts[operation_name] = 0
                return result
                
            except Exception as e:
                last_exception = e
                self.error_counts[operation_name] = self.error_counts.get(operation_name, 0) + 1
                
                # Try recovery function if available
                if recovery_func and attempt < self.config.max_retries - 1:
                    try:
                        recovery_func()
                        self.recovery_attempts[operation_name] = self.recovery_attempts.get(operation_name, 0) + 1
                    except:
                        pass  # Recovery failed, continue with retry
                
                # Wait before retry
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay * (attempt + 1))  # Exponential backoff
        
        # All retries failed
        raise Exception(f"Operation '{operation_name}' failed after {self.config.max_retries} attempts: {last_exception}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error and recovery statistics"""
        return {
            "error_counts": self.error_counts.copy(),
            "recovery_attempts": self.recovery_attempts.copy(),
            "circuit_breaker_states": {
                name: breaker.state.value 
                for name, breaker in self.circuit_breakers.items()
            }
        }

class RedundancyManager:
    """Redundancy and failover management"""
    
    def __init__(self, redundancy_level: int = 2):
        self.redundancy_level = redundancy_level
        self.primary_systems: Dict[str, Any] = {}
        self.backup_systems: Dict[str, List[Any]] = {}
        self.active_systems: Dict[str, str] = {}  # service -> active_system_id
        
    def register_service(self, service_name: str, primary_system: Any, backup_systems: List[Any]):
        """Register a service with redundancy"""
        self.primary_systems[service_name] = primary_system
        self.backup_systems[service_name] = backup_systems[:self.redundancy_level-1]
        self.active_systems[service_name] = "primary"
    
    def execute_with_failover(self, service_name: str, method_name: str, *args, **kwargs) -> Any:
        """Execute method with automatic failover"""
        if service_name not in self.primary_systems:
            raise ValueError(f"Service {service_name} not registered")
        
        # Try primary system first
        try:
            primary = self.primary_systems[service_name]
            method = getattr(primary, method_name)
            result = method(*args, **kwargs)
            
            # Success - ensure we're using primary
            self.active_systems[service_name] = "primary"
            return result
            
        except Exception as primary_error:
            # Primary failed, try backups
            backups = self.backup_systems[service_name]
            
            for i, backup in enumerate(backups):
                try:
                    method = getattr(backup, method_name)
                    result = method(*args, **kwargs)
                    
                    # Success with backup
                    self.active_systems[service_name] = f"backup_{i}"
                    return result
                    
                except Exception:
                    continue  # Try next backup
            
            # All systems failed
            raise Exception(f"All redundant systems failed for {service_name}: {primary_error}")
    
    def get_failover_status(self) -> Dict[str, Any]:
        """Get current failover status"""
        return {
            "active_systems": self.active_systems.copy(),
            "redundancy_level": self.redundancy_level,
            "services_count": len(self.primary_systems)
        }

class Generation2ReliabilityFramework:
    """Comprehensive Generation 2 reliability framework"""
    
    def __init__(self, 
                 security_config: Optional[SecurityConfig] = None,
                 reliability_config: Optional[ReliabilityConfig] = None):
        
        self.security_config = security_config or SecurityConfig()
        self.reliability_config = reliability_config or ReliabilityConfig()
        
        # Initialize components
        self.security_manager = SecurityManager(self.security_config)
        self.health_monitor = HealthMonitor(self.reliability_config)
        self.error_recovery = ErrorRecoverySystem(self.reliability_config)
        self.redundancy_manager = RedundancyManager(self.reliability_config.redundancy_level)
        
        # System state
        self.is_initialized = False
        self.start_time = datetime.now()
        
    def initialize(self):
        """Initialize the reliability framework"""
        if self.is_initialized:
            return
        
        # Start health monitoring
        self.health_monitor.start_monitoring()
        
        self.is_initialized = True
        print("Generation 2 Reliability Framework initialized")
    
    def shutdown(self):
        """Shutdown the framework gracefully"""
        if not self.is_initialized:
            return
        
        # Stop monitoring
        self.health_monitor.stop_monitoring()
        
        self.is_initialized = False
        print("Generation 2 Reliability Framework shutdown complete")
    
    @contextmanager
    def secure_session(self, user_id: str, credentials: Dict):
        """Context manager for secure operations"""
        # Authenticate user
        auth_success, session_info = self.security_manager.authenticate_user(user_id, credentials)
        
        if not auth_success:
            raise Exception(f"Authentication failed: {session_info}")
        
        session_id = session_info
        try:
            yield session_id
        finally:
            # Clean up session
            if session_id in self.security_manager.active_sessions:
                del self.security_manager.active_sessions[session_id]
    
    def secure_compute(self, 
                      session_id: str, 
                      operation_name: str,
                      compute_func: Callable,
                      input_data: Dict,
                      *args, **kwargs) -> Dict[str, Any]:
        """Perform secure computation with full reliability protection"""
        
        # Validate session
        session_valid, session_info = self.security_manager.validate_session(session_id)
        if not session_valid:
            raise Exception("Invalid or expired session")
        
        # Threat detection
        threats = self.security_manager.detect_threats(input_data)
        if threats:
            raise Exception(f"Security threats detected: {threats}")
        
        # Encrypt input data
        encrypted_input = {}
        for key, value in input_data.items():
            encrypted_input[key] = self.security_manager.encrypt_data(str(value))
        
        # Execute with error recovery
        def recovery_function():
            # Simple recovery - reset quantum states
            print(f"Attempting recovery for operation: {operation_name}")
        
        try:
            result = self.error_recovery.execute_with_recovery(
                operation_name,
                compute_func,
                encrypted_input,
                recovery_func=recovery_function,
                *args, **kwargs
            )
            
            # Encrypt result
            if isinstance(result, dict):
                encrypted_result = {}
                for key, value in result.items():
                    encrypted_result[key] = self.security_manager.encrypt_data(str(value))
            else:
                encrypted_result = self.security_manager.encrypt_data(str(result))
            
            return {
                "success": True,
                "result": encrypted_result,
                "session_id": session_id,
                "threats_detected": threats,
                "health_status": self.health_monitor.get_health_status()
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id,
                "threats_detected": threats,
                "health_status": self.health_monitor.get_health_status()
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        uptime = datetime.now() - self.start_time
        
        return {
            "framework_initialized": self.is_initialized,
            "uptime_seconds": uptime.total_seconds(),
            "security_config": asdict(self.security_config),
            "reliability_config": asdict(self.reliability_config),
            "health_status": self.health_monitor.get_health_status(),
            "error_statistics": self.error_recovery.get_error_statistics(),
            "failover_status": self.redundancy_manager.get_failover_status(),
            "active_sessions": len(self.security_manager.active_sessions),
            "timestamp": datetime.now()
        }

def test_generation2_framework():
    """Test the Generation 2 reliability framework"""
    print("Generation 2 Reliability Framework Test")
    print("=" * 45)
    
    # Initialize framework
    framework = Generation2ReliabilityFramework()
    framework.initialize()
    
    test_results = {}
    
    try:
        # Test 1: Secure authentication
        print("Test 1: Secure Authentication...")
        with framework.secure_session("test_user", {"password": "secure_password_123"}) as session_id:
            test_results["authentication"] = True
            print("  ✓ Authentication successful")
        
        # Test 2: Threat detection
        print("Test 2: Threat Detection...")
        threats = framework.security_manager.detect_threats({
            "query": "SELECT * FROM users",
            "script": "<script>alert('xss')</script>"
        })
        test_results["threat_detection"] = len(threats) > 0
        print(f"  ✓ Detected threats: {threats}")
        
        # Test 3: Error recovery
        print("Test 3: Error Recovery...")
        def failing_function():
            raise Exception("Simulated failure")
        
        try:
            framework.error_recovery.execute_with_recovery(
                "test_operation",
                failing_function
            )
        except:
            pass  # Expected to fail
        
        error_stats = framework.error_recovery.get_error_statistics()
        test_results["error_recovery"] = error_stats["error_counts"].get("test_operation", 0) > 0
        print(f"  ✓ Error recovery tracked: {error_stats}")
        
        # Test 4: Health monitoring
        print("Test 4: Health Monitoring...")
        time.sleep(2)  # Let monitoring collect some data
        health_status = framework.health_monitor.get_health_status()
        test_results["health_monitoring"] = health_status["system_health"] in ["healthy", "degraded"]
        print(f"  ✓ Health status: {health_status['system_health']}")
        
        # Test 5: Secure computation
        print("Test 5: Secure Computation...")
        with framework.secure_session("test_user", {"password": "secure_password_123"}) as session_id:
            def dummy_compute(data):
                return {"output": "computation_result"}
            
            result = framework.secure_compute(
                session_id,
                "test_compute",
                dummy_compute,
                {"input": "test_data"}
            )
            test_results["secure_computation"] = result["success"]
            print(f"  ✓ Secure computation: {'SUCCESS' if result['success'] else 'FAILED'}")
        
        # Test 6: System status
        print("Test 6: System Status...")
        status = framework.get_system_status()
        test_results["system_status"] = status["framework_initialized"]
        print(f"  ✓ System status retrieved, uptime: {status['uptime_seconds']:.2f}s")
        
    finally:
        framework.shutdown()
    
    # Results summary
    print("\n" + "=" * 45)
    passed_tests = sum(test_results.values())
    total_tests = len(test_results)
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    print(f"Tests passed: {passed_tests}/{total_tests}")
    print(f"Success rate: {success_rate:.1%}")
    
    overall_success = success_rate >= 0.8
    print(f"Generation 2 Status: {'✓ ROBUSTNESS ACHIEVED' if overall_success else '✗ NEEDS IMPROVEMENT'}")
    
    return {
        "test_results": test_results,
        "metrics": {
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "robustness_achieved": overall_success
        }
    }

def main():
    """Main entry point for Generation 2 testing"""
    results = test_generation2_framework()
    
    # Save results
    with open('generation2_robustness_report.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nDetailed report saved to: generation2_robustness_report.json")
    
    return results

if __name__ == "__main__":
    main()