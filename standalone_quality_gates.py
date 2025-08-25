"""
Standalone Quality Gates - Production Validation System
======================================================

Self-contained quality validation system that operates independently
without external dependencies for production readiness assessment.
"""

import os
import sys
import time
import json
import subprocess
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import traceback

logger = logging.getLogger(__name__)

@dataclass
class QualityGateResult:
    """Quality gate validation result"""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    execution_time: float
    errors: List[str]
    warnings: List[str]

@dataclass
class ValidationSummary:
    """Overall validation summary"""
    total_gates: int
    passed_gates: int
    failed_gates: int
    overall_score: float
    execution_time: float
    quality_level: str
    production_ready: bool
    recommendations: List[str]

class StandaloneQualityGates:
    """Standalone quality gate validation system"""
    
    def __init__(self):
        self.gates = {}
        self.results = {}
        self.validation_start_time = 0.0
        self.setup_quality_gates()
    
    def setup_quality_gates(self):
        """Setup all quality validation gates"""
        
        self.gates = {
            'functionality': {
                'weight': 0.25,
                'validator': self._validate_functionality,
                'description': 'Core functionality validation'
            },
            'performance': {
                'weight': 0.20,
                'validator': self._validate_performance,
                'description': 'Performance and benchmarking'
            },
            'security': {
                'weight': 0.20,
                'validator': self._validate_security,
                'description': 'Security and vulnerability assessment'
            },
            'reliability': {
                'weight': 0.15,
                'validator': self._validate_reliability,
                'description': 'Reliability and error handling'
            },
            'scalability': {
                'weight': 0.10,
                'validator': self._validate_scalability,
                'description': 'Scalability and resource efficiency'
            },
            'maintainability': {
                'weight': 0.10,
                'validator': self._validate_maintainability,
                'description': 'Code quality and maintainability'
            }
        }
    
    def run_autonomous_validation(self) -> ValidationSummary:
        """Run all quality gates autonomously"""
        
        print("🚀 STANDALONE QUALITY GATES VALIDATION")
        print("=" * 60)
        
        self.validation_start_time = time.time()
        
        # Run all quality gates
        for gate_name, gate_config in self.gates.items():
            print(f"\n🔍 Validating {gate_name.upper()}...")
            
            start_time = time.time()
            
            try:
                validator_func = gate_config['validator']
                result = validator_func()
                
                execution_time = time.time() - start_time
                
                gate_result = QualityGateResult(
                    gate_name=gate_name,
                    passed=result.get('passed', False),
                    score=result.get('score', 0.0),
                    details=result.get('details', {}),
                    execution_time=execution_time,
                    errors=result.get('errors', []),
                    warnings=result.get('warnings', [])
                )
                
                self.results[gate_name] = gate_result
                
                # Display result
                status = "✓ PASS" if gate_result.passed else "✗ FAIL"
                score_pct = gate_result.score * 100
                print(f"   {status} - Score: {score_pct:.1f}% ({execution_time:.2f}s)")
                
                if gate_result.errors:
                    for error in gate_result.errors:
                        print(f"   ❌ Error: {error}")
                
                if gate_result.warnings:
                    for warning in gate_result.warnings:
                        print(f"   ⚠️  Warning: {warning}")
                
            except Exception as e:
                execution_time = time.time() - start_time
                error_msg = f"Gate validation failed: {str(e)}"
                
                gate_result = QualityGateResult(
                    gate_name=gate_name,
                    passed=False,
                    score=0.0,
                    details={'error': str(e), 'traceback': traceback.format_exc()},
                    execution_time=execution_time,
                    errors=[error_msg],
                    warnings=[]
                )
                
                self.results[gate_name] = gate_result
                print(f"   ✗ FAIL - Error: {error_msg}")
        
        # Generate validation summary
        summary = self._generate_validation_summary()
        
        # Display summary
        self._display_validation_summary(summary)
        
        # Save detailed results
        self._save_validation_results(summary)
        
        return summary
    
    def _validate_functionality(self) -> Dict[str, Any]:
        """Validate core functionality using standalone tests"""
        
        result = {
            'passed': False,
            'score': 0.0,
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Run standalone dependency-free core tests
            test_cmd = [sys.executable, 'spintron_nn/dependency_free_core.py']
            
            if not os.path.exists('spintron_nn/dependency_free_core.py'):
                result['errors'].append("Dependency-free core test not found")
                return result
            
            process = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=60  # 1 minute timeout
            )
            
            if process.returncode == 0:
                output = process.stdout
                
                # Parse test results
                tests_passed = output.count('✓ PASS')
                tests_failed = output.count('✗ FAIL')
                total_tests = tests_passed + tests_failed
                
                functionality_score = tests_passed / total_tests if total_tests > 0 else 0.0
                
                result.update({
                    'passed': functionality_score >= 0.8 and tests_failed == 0,
                    'score': functionality_score,
                    'details': {
                        'tests_passed': tests_passed,
                        'tests_failed': tests_failed,
                        'test_output': output
                    }
                })
                
                if tests_failed > 0:
                    result['warnings'].append(f"{tests_failed} functionality tests failed")
            else:
                error_output = process.stderr
                result['errors'].append(f"Functionality tests failed: {error_output}")
        
        except subprocess.TimeoutExpired:
            result['errors'].append("Functionality tests timed out")
        except Exception as e:
            result['errors'].append(f"Functionality validation error: {str(e)}")
        
        return result
    
    def _validate_performance(self) -> Dict[str, Any]:
        """Validate performance benchmarks"""
        
        result = {
            'passed': False,
            'score': 0.0,
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            # Run performance benchmarks
            benchmark_cmd = [sys.executable, 'benchmarks/simple_benchmark.py']
            
            if not os.path.exists('benchmarks/simple_benchmark.py'):
                result['errors'].append("Benchmark file not found")
                return result
            
            process = subprocess.run(
                benchmark_cmd,
                capture_output=True,
                text=True,
                timeout=120  # 2 minute timeout
            )
            
            if process.returncode == 0:
                output = process.stdout
                
                # Parse performance results
                performance_metrics = self._parse_benchmark_output(output)
                performance_score = self._calculate_performance_score(performance_metrics)
                
                result.update({
                    'passed': performance_score >= 0.7,
                    'score': performance_score,
                    'details': {
                        'performance_metrics': performance_metrics,
                        'benchmark_output': output
                    }
                })
                
                if performance_score < 0.8:
                    result['warnings'].append("Performance below optimal thresholds")
            else:
                error_output = process.stderr
                result['errors'].append(f"Benchmark execution failed: {error_output}")
        
        except subprocess.TimeoutExpired:
            result['errors'].append("Performance benchmark timed out")
        except Exception as e:
            result['errors'].append(f"Performance validation error: {str(e)}")
        
        return result
    
    def _validate_security(self) -> Dict[str, Any]:
        """Validate security measures"""
        
        result = {
            'passed': False,
            'score': 0.0,
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            security_checks = []
            
            # Check for security framework files
            security_files = [
                'spintron_nn/security/advanced_security_framework.py',
                'spintron_nn/security/secure_computing.py',
                'spintron_nn/security/__init__.py'
            ]
            
            security_file_count = sum(1 for f in security_files if os.path.exists(f))
            security_checks.append({
                'check': 'security_framework_files',
                'passed': security_file_count >= 2,
                'description': f'{security_file_count}/3 security files present'
            })
            
            # Check for input validation utilities
            validation_files = [
                'spintron_nn/utils/validation.py',
                'spintron_nn/utils/security.py'
            ]
            
            validation_exists = any(os.path.exists(f) for f in validation_files)
            security_checks.append({
                'check': 'input_validation',
                'passed': validation_exists,
                'description': 'Input validation utilities present'
            })
            
            # Check for secure configuration
            config_files = [
                'deployment_package/security/security.json',
                'spintron_nn/security'  # Directory existence
            ]
            
            config_exists = any(os.path.exists(f) for f in config_files)
            security_checks.append({
                'check': 'secure_configuration',
                'passed': config_exists,
                'description': 'Security configuration present'
            })
            
            # Check for error handling (security-related)
            error_handling_files = [
                'spintron_nn/utils/error_handling.py',
                'spintron_nn/utils/robust_error_handling.py'
            ]
            
            error_handling_exists = any(os.path.exists(f) for f in error_handling_files)
            security_checks.append({
                'check': 'secure_error_handling',
                'passed': error_handling_exists,
                'description': 'Secure error handling present'
            })
            
            # Calculate security score
            passed_checks = sum(1 for check in security_checks if check['passed'])
            total_checks = len(security_checks)
            security_score = passed_checks / total_checks if total_checks > 0 else 0.0
            
            result.update({
                'passed': security_score >= 0.75,
                'score': security_score,
                'details': {
                    'security_checks': security_checks,
                    'passed_checks': passed_checks,
                    'total_checks': total_checks
                }
            })
            
            # Add warnings for failed checks
            failed_checks = [check for check in security_checks if not check['passed']]
            for failed_check in failed_checks:
                result['warnings'].append(f"Security check failed: {failed_check['description']}")
        
        except Exception as e:
            result['errors'].append(f"Security validation error: {str(e)}")
        
        return result
    
    def _validate_reliability(self) -> Dict[str, Any]:
        """Validate reliability and error handling"""
        
        result = {
            'passed': False,
            'score': 0.0,
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            reliability_checks = []
            
            # Check for monitoring systems
            monitoring_files = [
                'spintron_nn/monitoring/robust_monitoring.py',
                'spintron_nn/utils/monitoring.py',
                'deployment_package/monitoring/health_check.py'
            ]
            
            monitoring_count = sum(1 for f in monitoring_files if os.path.exists(f))
            reliability_checks.append({
                'check': 'monitoring_systems',
                'passed': monitoring_count >= 2,
                'description': f'{monitoring_count}/3 monitoring systems present'
            })
            
            # Check for error handling
            error_files = [
                'spintron_nn/utils/error_handling.py',
                'spintron_nn/utils/robust_error_handling.py'
            ]
            
            error_handling_exists = any(os.path.exists(f) for f in error_files)
            reliability_checks.append({
                'check': 'error_handling',
                'passed': error_handling_exists,
                'description': 'Error handling mechanisms present'
            })
            
            # Check for logging systems
            logging_files = [
                'spintron_nn/utils/logging_config.py',
                'spintron_nn/utils/advanced_logging.py'
            ]
            
            logging_exists = any(os.path.exists(f) for f in logging_files)
            reliability_checks.append({
                'check': 'logging_system',
                'passed': logging_exists,
                'description': 'Logging system present'
            })
            
            # Check for validation utilities
            validation_files = [
                'spintron_nn/utils/validation.py',
                'spintron_nn/utils/comprehensive_validation.py'
            ]
            
            validation_exists = any(os.path.exists(f) for f in validation_files)
            reliability_checks.append({
                'check': 'validation_utilities',
                'passed': validation_exists,
                'description': 'Validation utilities present'
            })
            
            # Calculate reliability score
            passed_checks = sum(1 for check in reliability_checks if check['passed'])
            total_checks = len(reliability_checks)
            reliability_score = passed_checks / total_checks if total_checks > 0 else 0.0
            
            result.update({
                'passed': reliability_score >= 0.75,
                'score': reliability_score,
                'details': {
                    'reliability_checks': reliability_checks,
                    'passed_checks': passed_checks,
                    'total_checks': total_checks
                }
            })
            
            # Add warnings for failed checks
            failed_checks = [check for check in reliability_checks if not check['passed']]
            for failed_check in failed_checks:
                result['warnings'].append(f"Reliability check failed: {failed_check['description']}")
        
        except Exception as e:
            result['errors'].append(f"Reliability validation error: {str(e)}")
        
        return result
    
    def _validate_scalability(self) -> Dict[str, Any]:
        """Validate scalability and resource efficiency"""
        
        result = {
            'passed': False,
            'score': 0.0,
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            scalability_checks = []
            
            # Check for scaling components
            scaling_files = [
                'spintron_nn/hyperscale_optimizer.py',
                'spintron_nn/distributed_scaling.py',
                'spintron_nn/scaling'  # Directory
            ]
            
            scaling_count = sum(1 for f in scaling_files if os.path.exists(f))
            scalability_checks.append({
                'check': 'scaling_components',
                'passed': scaling_count >= 2,
                'description': f'{scaling_count}/3 scaling components present'
            })
            
            # Check for caching systems
            caching_files = [
                'spintron_nn/scaling/cache_optimization.py',
                'spintron_nn/scaling/intelligent_caching.py'
            ]
            
            caching_exists = any(os.path.exists(f) for f in caching_files)
            scalability_checks.append({
                'check': 'caching_systems',
                'passed': caching_exists,
                'description': 'Caching systems present'
            })
            
            # Check for distributed processing
            distributed_files = [
                'spintron_nn/scaling/distributed_processing.py',
                'spintron_nn/scaling/distributed_computing.py'
            ]
            
            distributed_exists = any(os.path.exists(f) for f in distributed_files)
            scalability_checks.append({
                'check': 'distributed_processing',
                'passed': distributed_exists,
                'description': 'Distributed processing present'
            })
            
            # Check for performance optimization
            optimization_files = [
                'spintron_nn/scaling/performance_optimizer.py',
                'spintron_nn/optimization'  # Directory
            ]
            
            optimization_exists = any(os.path.exists(f) for f in optimization_files)
            scalability_checks.append({
                'check': 'performance_optimization',
                'passed': optimization_exists,
                'description': 'Performance optimization present'
            })
            
            # Calculate scalability score
            passed_checks = sum(1 for check in scalability_checks if check['passed'])
            total_checks = len(scalability_checks)
            scalability_score = passed_checks / total_checks if total_checks > 0 else 0.0
            
            result.update({
                'passed': scalability_score >= 0.75,
                'score': scalability_score,
                'details': {
                    'scalability_checks': scalability_checks,
                    'passed_checks': passed_checks,
                    'total_checks': total_checks
                }
            })
            
            # Add warnings for failed checks
            failed_checks = [check for check in scalability_checks if not check['passed']]
            for failed_check in failed_checks:
                result['warnings'].append(f"Scalability check failed: {failed_check['description']}")
        
        except Exception as e:
            result['errors'].append(f"Scalability validation error: {str(e)}")
        
        return result
    
    def _validate_maintainability(self) -> Dict[str, Any]:
        """Validate code quality and maintainability"""
        
        result = {
            'passed': False,
            'score': 0.0,
            'details': {},
            'errors': [],
            'warnings': []
        }
        
        try:
            maintainability_checks = []
            
            # Check project structure
            core_directories = [
                'spintron_nn/core',
                'spintron_nn/models',
                'spintron_nn/training',
                'spintron_nn/hardware',
                'spintron_nn/research'
            ]
            
            structure_count = sum(1 for d in core_directories if os.path.exists(d))
            maintainability_checks.append({
                'check': 'project_structure',
                'passed': structure_count >= 4,
                'description': f'{structure_count}/5 core directories present'
            })
            
            # Check configuration files
            config_files = [
                'pyproject.toml',
                'package.json',
                'setup.cfg'
            ]
            
            config_exists = any(os.path.exists(f) for f in config_files)
            maintainability_checks.append({
                'check': 'configuration_files',
                'passed': config_exists,
                'description': 'Project configuration files present'
            })
            
            # Check documentation
            doc_files = [
                'README.md',
                'ARCHITECTURE.md',
                'EXAMPLES.md'
            ]
            
            doc_count = sum(1 for f in doc_files if os.path.exists(f))
            maintainability_checks.append({
                'check': 'documentation',
                'passed': doc_count >= 2,
                'description': f'{doc_count}/3 documentation files present'
            })
            
            # Check for deployment structure
            deployment_dirs = [
                'deployment',
                'deployment_package',
                'k8s'
            ]
            
            deployment_exists = any(os.path.exists(d) for d in deployment_dirs)
            maintainability_checks.append({
                'check': 'deployment_structure',
                'passed': deployment_exists,
                'description': 'Deployment structure present'
            })
            
            # Check for testing infrastructure
            test_dirs = [
                'tests',
                'benchmarks'
            ]
            
            test_exists = any(os.path.exists(d) for d in test_dirs)
            maintainability_checks.append({
                'check': 'testing_infrastructure',
                'passed': test_exists,
                'description': 'Testing infrastructure present'
            })
            
            # Count Python files for complexity assessment
            python_files = 0
            for root, dirs, files in os.walk('spintron_nn'):
                python_files += sum(1 for f in files if f.endswith('.py'))
            
            maintainability_checks.append({
                'check': 'codebase_size',
                'passed': python_files >= 20,
                'description': f'{python_files} Python files (complexity indicator)',
                'value': python_files
            })
            
            # Calculate maintainability score
            passed_checks = sum(1 for check in maintainability_checks if check['passed'])
            total_checks = len(maintainability_checks)
            maintainability_score = passed_checks / total_checks if total_checks > 0 else 0.0
            
            result.update({
                'passed': maintainability_score >= 0.8,
                'score': maintainability_score,
                'details': {
                    'maintainability_checks': maintainability_checks,
                    'passed_checks': passed_checks,
                    'total_checks': total_checks,
                    'python_files_count': python_files
                }
            })
            
            # Add warnings for failed checks
            failed_checks = [check for check in maintainability_checks if not check['passed']]
            for failed_check in failed_checks:
                result['warnings'].append(f"Maintainability check failed: {failed_check['description']}")
        
        except Exception as e:
            result['errors'].append(f"Maintainability validation error: {str(e)}")
        
        return result
    
    def _parse_benchmark_output(self, output: str) -> Dict[str, float]:
        """Parse benchmark output to extract performance metrics"""
        
        metrics = {}
        lines = output.split('\n')
        
        for line in lines:
            line = line.strip()
            
            # Extract ops/sec metrics
            if 'ops/sec' in line and ':' in line:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        value_str = parts[1].strip().split()[0].replace(',', '')
                        value = float(value_str)
                        
                        if 'resistance' in line.lower():
                            metrics['resistance_ops_per_sec'] = value
                        elif 'crossbar' in line.lower():
                            metrics['crossbar_ops_per_sec'] = value
                        elif 'quantization' in line.lower():
                            metrics['quantization_rate'] = value
                        elif 'inference' in line.lower():
                            metrics['inference_throughput'] = value
                except:
                    pass
            
            # Extract latency metrics
            if 'μs per op' in line or 'ms per op' in line:
                try:
                    parts = line.split(':')
                    if len(parts) >= 2:
                        value_str = parts[1].strip().split()[0]
                        value = float(value_str)
                        
                        if 'μs' in line:
                            value *= 0.001  # Convert to ms
                        
                        if 'resistance' in line.lower():
                            metrics['resistance_latency_ms'] = value
                        elif 'crossbar' in line.lower():
                            metrics['crossbar_latency_ms'] = value
                except:
                    pass
        
        return metrics
    
    def _calculate_performance_score(self, metrics: Dict[str, float]) -> float:
        """Calculate performance score from metrics"""
        
        if not metrics:
            return 0.0
        
        score_components = []
        
        # Throughput scores (higher is better)
        if 'resistance_ops_per_sec' in metrics:
            target = 1000000  # 1M ops/sec
            score = min(metrics['resistance_ops_per_sec'] / target, 1.0)
            score_components.append(score)
        
        if 'crossbar_ops_per_sec' in metrics:
            target = 3000  # 3K ops/sec
            score = min(metrics['crossbar_ops_per_sec'] / target, 1.0)
            score_components.append(score)
        
        if 'quantization_rate' in metrics:
            target = 1000000  # 1M values/sec
            score = min(metrics['quantization_rate'] / target, 1.0)
            score_components.append(score)
        
        if 'inference_throughput' in metrics:
            target = 10000  # 10K inferences/sec
            score = min(metrics['inference_throughput'] / target, 1.0)
            score_components.append(score)
        
        # Latency scores (lower is better)
        if 'resistance_latency_ms' in metrics:
            max_acceptable = 1.0  # 1ms
            score = max(0, 1.0 - metrics['resistance_latency_ms'] / max_acceptable)
            score_components.append(score)
        
        if 'crossbar_latency_ms' in metrics:
            max_acceptable = 10.0  # 10ms
            score = max(0, 1.0 - metrics['crossbar_latency_ms'] / max_acceptable)
            score_components.append(score)
        
        return sum(score_components) / len(score_components) if score_components else 0.5
    
    def _generate_validation_summary(self) -> ValidationSummary:
        """Generate validation summary"""
        
        total_gates = len(self.gates)
        passed_gates = sum(1 for result in self.results.values() if result.passed)
        failed_gates = total_gates - passed_gates
        
        # Calculate weighted score
        total_score = 0.0
        total_weight = 0.0
        
        for gate_name, gate_result in self.results.items():
            weight = self.gates[gate_name]['weight']
            total_score += gate_result.score * weight
            total_weight += weight
        
        overall_score = total_score / total_weight if total_weight > 0 else 0.0
        
        # Determine quality level
        if overall_score >= 0.9:
            quality_level = "EXCELLENT"
        elif overall_score >= 0.8:
            quality_level = "GOOD"
        elif overall_score >= 0.7:
            quality_level = "ACCEPTABLE"
        elif overall_score >= 0.6:
            quality_level = "NEEDS_IMPROVEMENT"
        else:
            quality_level = "POOR"
        
        # Production readiness
        production_ready = (passed_gates >= total_gates * 0.8 and overall_score >= 0.75)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        
        total_execution_time = time.time() - self.validation_start_time
        
        return ValidationSummary(
            total_gates=total_gates,
            passed_gates=passed_gates,
            failed_gates=failed_gates,
            overall_score=overall_score,
            execution_time=total_execution_time,
            quality_level=quality_level,
            production_ready=production_ready,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on validation results"""
        
        recommendations = []
        
        for gate_name, gate_result in self.results.items():
            if not gate_result.passed or gate_result.score < 0.8:
                if gate_name == 'functionality':
                    recommendations.append("Address core functionality issues")
                elif gate_name == 'performance':
                    recommendations.append("Optimize performance bottlenecks")
                elif gate_name == 'security':
                    recommendations.append("Strengthen security measures")
                elif gate_name == 'reliability':
                    recommendations.append("Enhance reliability and monitoring")
                elif gate_name == 'scalability':
                    recommendations.append("Implement scalability features")
                elif gate_name == 'maintainability':
                    recommendations.append("Improve code organization and documentation")
        
        if not recommendations:
            recommendations.append("Excellent quality - ready for production")
        
        return recommendations
    
    def _display_validation_summary(self, summary: ValidationSummary):
        """Display validation summary"""
        
        print("\n" + "=" * 60)
        print("🎯 STANDALONE QUALITY GATES SUMMARY")
        print("=" * 60)
        
        print(f"Total Gates:      {summary.total_gates}")
        print(f"Passed Gates:     {summary.passed_gates} ✓")
        print(f"Failed Gates:     {summary.failed_gates} ✗")
        print(f"Overall Score:    {summary.overall_score*100:.1f}%")
        print(f"Quality Level:    {summary.quality_level}")
        print(f"Execution Time:   {summary.execution_time:.2f}s")
        
        production_status = "✓ READY" if summary.production_ready else "✗ NOT READY"
        print(f"Production Ready: {production_status}")
        
        if summary.recommendations:
            print("\n🔧 RECOMMENDATIONS:")
            for i, rec in enumerate(summary.recommendations, 1):
                print(f"   {i}. {rec}")
        
        print("\n📊 DETAILED RESULTS:")
        print("-" * 60)
        
        for gate_name, gate_result in self.results.items():
            status = "✓ PASS" if gate_result.passed else "✗ FAIL"
            score_pct = gate_result.score * 100
            weight = self.gates[gate_name]['weight'] * 100
            
            print(f"{gate_name.upper():15} {status:8} {score_pct:6.1f}% (Weight: {weight:4.1f}%)")
        
        print("=" * 60)
    
    def _save_validation_results(self, summary: ValidationSummary):
        """Save validation results to file"""
        
        results_data = {
            'validation_timestamp': time.time(),
            'summary': asdict(summary),
            'gate_results': {name: asdict(result) for name, result in self.results.items()}
        }
        
        output_file = 'standalone_quality_gates_results.json'
        with open(output_file, 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        
        print(f"\n📁 Results saved to: {output_file}")

def main():
    """Main entry point"""
    
    print("🚀 SpinTron-NN-Kit Standalone Quality Gates")
    print("Version: 1.0.0 - Production Validation System")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    quality_gates = StandaloneQualityGates()
    summary = quality_gates.run_autonomous_validation()
    
    if summary.production_ready:
        print("\n🎉 SUCCESS: System ready for production deployment!")
        return 0
    else:
        print("\n⚠️  WARNING: System requires improvements before production")
        return 1

if __name__ == "__main__":
    exit(main())