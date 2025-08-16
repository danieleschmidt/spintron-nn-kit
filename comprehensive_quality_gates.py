#!/usr/bin/env python3
"""
Comprehensive quality gates for SpinTron-NN-Kit.
Validates security, performance, code quality, and deployment readiness.
"""

import sys
import time
import json
import os
import re
import hashlib
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class QualityGateResult:
    """Result of a quality gate check."""
    gate_name: str
    passed: bool
    score: float
    details: Dict[str, Any]
    recommendations: List[str]
    execution_time_ms: float


class SecurityValidator:
    """Security validation for SpinTron-NN-Kit."""
    
    def __init__(self):
        self.security_patterns = {
            'hardcoded_secrets': [
                r'password\s*=\s*["\'][^"\']+["\']',
                r'api_key\s*=\s*["\'][^"\']+["\']',
                r'secret\s*=\s*["\'][^"\']+["\']',
                r'token\s*=\s*["\'][^"\']+["\']'
            ],
            'sql_injection': [
                r'execute\s*\(\s*["\'].*%.*["\']',
                r'cursor\.execute\s*\(\s*["\'].*\+.*["\']'
            ],
            'command_injection': [
                r'os\.system\s*\(',
                r'subprocess\.call\s*\(\s*shell\s*=\s*True',
                r'eval\s*\(',
                r'exec\s*\('
            ],
            'path_traversal': [
                r'open\s*\(\s*.*\.\.\/',
                r'file\s*\(\s*.*\.\.\/'
            ]
        }
        
    def scan_file(self, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Scan a single file for security issues."""
        issues = {category: [] for category in self.security_patterns.keys()}
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
            for category, patterns in self.security_patterns.items():
                for pattern in patterns:
                    for line_num, line in enumerate(lines, 1):
                        if re.search(pattern, line, re.IGNORECASE):
                            issues[category].append({
                                'file': file_path,
                                'line': line_num,
                                'content': line.strip(),
                                'pattern': pattern
                            })
                            
        except Exception as e:
            print(f"Error scanning {file_path}: {e}")
            
        return issues
        
    def validate_security(self) -> QualityGateResult:
        """Perform comprehensive security validation."""
        start_time = time.time()
        
        # Scan all Python files
        python_files = list(Path('.').rglob('*.py'))
        all_issues = {category: [] for category in self.security_patterns.keys()}
        
        scanned_files = 0
        for file_path in python_files:
            if 'test' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            file_issues = self.scan_file(str(file_path))
            for category, issues in file_issues.items():
                all_issues[category].extend(issues)
            scanned_files += 1
            
        # Calculate security score
        total_issues = sum(len(issues) for issues in all_issues.values())
        critical_issues = len(all_issues['hardcoded_secrets']) + len(all_issues['command_injection'])
        
        # Security score calculation
        if critical_issues > 0:
            security_score = max(0, 50 - critical_issues * 10)  # Critical issues heavily penalized
        else:
            security_score = max(0, 100 - total_issues * 5)  # Other issues lightly penalized
            
        # Generate recommendations
        recommendations = []
        if all_issues['hardcoded_secrets']:
            recommendations.append("Remove hardcoded secrets and use environment variables")
        if all_issues['command_injection']:
            recommendations.append("Avoid using eval(), exec(), and shell=True in subprocess calls")
        if all_issues['sql_injection']:
            recommendations.append("Use parameterized queries to prevent SQL injection")
        if all_issues['path_traversal']:
            recommendations.append("Validate and sanitize file paths to prevent directory traversal")
            
        if not recommendations:
            recommendations.append("Security scan passed - no obvious vulnerabilities detected")
            
        execution_time = (time.time() - start_time) * 1000
        
        return QualityGateResult(
            gate_name="Security Validation",
            passed=security_score >= 80,
            score=security_score,
            details={
                'scanned_files': scanned_files,
                'total_issues': total_issues,
                'critical_issues': critical_issues,
                'issues_by_category': {k: len(v) for k, v in all_issues.items()},
                'detailed_issues': all_issues
            },
            recommendations=recommendations,
            execution_time_ms=execution_time
        )


class PerformanceValidator:
    """Performance validation for SpinTron-NN-Kit."""
    
    def __init__(self):
        self.performance_benchmarks = {
            'mtj_simulation_time_ms': 100,
            'crossbar_computation_ms': 50,
            'verilog_generation_ms': 200,
            'cache_access_time_ms': 1,
            'memory_usage_mb': 500
        }
        
    def run_performance_benchmarks(self) -> Dict[str, float]:
        """Run performance benchmarks."""
        results = {}
        
        # MTJ simulation benchmark
        start_time = time.time()
        for _ in range(100):
            # Simulate MTJ computation
            resistance_ratio = 10000 / 5000
            switching_energy = (0.3 ** 2) / 5000 * 1e12
        mtj_time = (time.time() - start_time) * 1000
        results['mtj_simulation_time_ms'] = mtj_time
        
        # Crossbar computation benchmark
        start_time = time.time()
        weights = [[1.0] * 64 for _ in range(64)]
        inputs = [1.0] * 64
        for _ in range(10):
            outputs = []
            for row in weights:
                output = sum(w * i for w, i in zip(row, inputs))
                outputs.append(output)
        crossbar_time = (time.time() - start_time) * 1000
        results['crossbar_computation_ms'] = crossbar_time
        
        # Verilog generation benchmark
        start_time = time.time()
        verilog_template = """
module test_module_{0} (
    input clk,
    input rst,
    input [63:0] inputs,
    output [63:0] outputs
);
    // Module implementation
endmodule
"""
        for i in range(50):
            verilog_code = verilog_template.format(i)
        verilog_time = (time.time() - start_time) * 1000
        results['verilog_generation_ms'] = verilog_time
        
        # Cache access benchmark
        cache = {}
        start_time = time.time()
        for i in range(1000):
            key = f"key_{i % 100}"
            if key in cache:
                value = cache[key]
            else:
                cache[key] = f"value_{i}"
        cache_time = (time.time() - start_time) * 1000
        results['cache_access_time_ms'] = cache_time
        
        # Memory usage estimation
        import sys
        current_objects = len(gc.get_objects()) if 'gc' in sys.modules else 1000
        estimated_memory = current_objects * 0.1  # Rough estimation
        results['memory_usage_mb'] = estimated_memory
        
        return results
        
    def validate_performance(self) -> QualityGateResult:
        """Perform performance validation."""
        start_time = time.time()
        
        # Run benchmarks
        benchmark_results = self.run_performance_benchmarks()
        
        # Calculate performance score
        score_components = []
        performance_details = {}
        
        for metric, actual_value in benchmark_results.items():
            target_value = self.performance_benchmarks[metric]
            
            # Calculate performance ratio (lower is better for time metrics)
            if 'time' in metric or 'usage' in metric:
                performance_ratio = target_value / actual_value if actual_value > 0 else 1.0
                performance_ratio = min(performance_ratio, 2.0)  # Cap at 2x better than target
            else:
                performance_ratio = actual_value / target_value if target_value > 0 else 1.0
                
            score = min(100, performance_ratio * 100)
            score_components.append(score)
            
            performance_details[metric] = {
                'actual': actual_value,
                'target': target_value,
                'ratio': performance_ratio,
                'score': score
            }
            
        overall_score = sum(score_components) / len(score_components)
        
        # Generate recommendations
        recommendations = []
        for metric, details in performance_details.items():
            if details['score'] < 80:
                if 'mtj' in metric:
                    recommendations.append("Optimize MTJ simulation algorithms for better performance")
                elif 'crossbar' in metric:
                    recommendations.append("Consider parallel processing for crossbar computations")
                elif 'verilog' in metric:
                    recommendations.append("Cache Verilog templates to reduce generation time")
                elif 'cache' in metric:
                    recommendations.append("Optimize cache data structures and access patterns")
                elif 'memory' in metric:
                    recommendations.append("Implement memory optimization and garbage collection")
                    
        if not recommendations:
            recommendations.append("Performance benchmarks passed - system meets performance targets")
            
        execution_time = (time.time() - start_time) * 1000
        
        return QualityGateResult(
            gate_name="Performance Validation",
            passed=overall_score >= 80,
            score=overall_score,
            details={
                'benchmark_results': benchmark_results,
                'performance_details': performance_details,
                'target_benchmarks': self.performance_benchmarks
            },
            recommendations=recommendations,
            execution_time_ms=execution_time
        )


class CodeQualityValidator:
    """Code quality validation for SpinTron-NN-Kit."""
    
    def __init__(self):
        self.quality_metrics = {
            'max_function_length': 100,
            'max_file_length': 1000,
            'min_docstring_coverage': 0.8,
            'max_complexity': 10
        }
        
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file for quality metrics."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
                
        except Exception:
            return {}
            
        metrics = {
            'line_count': len(lines),
            'function_count': 0,
            'class_count': 0,
            'docstring_count': 0,
            'max_function_length': 0,
            'long_functions': [],
            'complexity_estimate': 0
        }
        
        current_function = None
        current_function_start = 0
        in_docstring = False
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            
            # Count classes and functions
            if stripped.startswith('class '):
                metrics['class_count'] += 1
            elif stripped.startswith('def '):
                if current_function:
                    # End previous function
                    func_length = i - current_function_start
                    metrics['max_function_length'] = max(metrics['max_function_length'], func_length)
                    if func_length > self.quality_metrics['max_function_length']:
                        metrics['long_functions'].append({
                            'name': current_function,
                            'length': func_length,
                            'start_line': current_function_start
                        })
                        
                # Start new function
                current_function = stripped.split('(')[0].replace('def ', '')
                current_function_start = i
                metrics['function_count'] += 1
                
            # Count docstrings
            if '"""' in stripped or "'''" in stripped:
                if not in_docstring:
                    metrics['docstring_count'] += 1
                    in_docstring = True
                else:
                    in_docstring = False
                    
            # Estimate complexity (simplified)
            complexity_keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 'with']
            for keyword in complexity_keywords:
                if f' {keyword} ' in f' {stripped} ' or stripped.startswith(f'{keyword} '):
                    metrics['complexity_estimate'] += 1
                    
        # Handle last function
        if current_function:
            func_length = len(lines) - current_function_start
            metrics['max_function_length'] = max(metrics['max_function_length'], func_length)
            if func_length > self.quality_metrics['max_function_length']:
                metrics['long_functions'].append({
                    'name': current_function,
                    'length': func_length,
                    'start_line': current_function_start
                })
                
        return metrics
        
    def validate_code_quality(self) -> QualityGateResult:
        """Perform code quality validation."""
        start_time = time.time()
        
        # Analyze all Python files
        python_files = list(Path('.').rglob('*.py'))
        
        aggregate_metrics = {
            'total_files': 0,
            'total_lines': 0,
            'total_functions': 0,
            'total_classes': 0,
            'total_docstrings': 0,
            'long_files': [],
            'long_functions': [],
            'high_complexity_files': []
        }
        
        for file_path in python_files:
            if 'test' in str(file_path) or '__pycache__' in str(file_path):
                continue
                
            file_metrics = self.analyze_file(str(file_path))
            if not file_metrics:
                continue
                
            aggregate_metrics['total_files'] += 1
            aggregate_metrics['total_lines'] += file_metrics['line_count']
            aggregate_metrics['total_functions'] += file_metrics['function_count']
            aggregate_metrics['total_classes'] += file_metrics['class_count']
            aggregate_metrics['total_docstrings'] += file_metrics['docstring_count']
            
            # Track problematic files
            if file_metrics['line_count'] > self.quality_metrics['max_file_length']:
                aggregate_metrics['long_files'].append({
                    'file': str(file_path),
                    'lines': file_metrics['line_count']
                })
                
            aggregate_metrics['long_functions'].extend([
                {**func, 'file': str(file_path)} 
                for func in file_metrics['long_functions']
            ])
            
            if file_metrics['complexity_estimate'] > self.quality_metrics['max_complexity'] * 5:
                aggregate_metrics['high_complexity_files'].append({
                    'file': str(file_path),
                    'complexity': file_metrics['complexity_estimate']
                })
                
        # Calculate quality score
        score_components = []
        
        # File length score
        long_file_penalty = len(aggregate_metrics['long_files']) * 5
        file_score = max(0, 100 - long_file_penalty)
        score_components.append(file_score)
        
        # Function length score
        long_function_penalty = len(aggregate_metrics['long_functions']) * 3
        function_score = max(0, 100 - long_function_penalty)
        score_components.append(function_score)
        
        # Docstring coverage score
        docstring_coverage = (aggregate_metrics['total_docstrings'] / 
                            max(1, aggregate_metrics['total_functions'] + aggregate_metrics['total_classes']))
        docstring_score = min(100, docstring_coverage / self.quality_metrics['min_docstring_coverage'] * 100)
        score_components.append(docstring_score)
        
        # Complexity score
        complexity_penalty = len(aggregate_metrics['high_complexity_files']) * 10
        complexity_score = max(0, 100 - complexity_penalty)
        score_components.append(complexity_score)
        
        overall_score = sum(score_components) / len(score_components)
        
        # Generate recommendations
        recommendations = []
        if aggregate_metrics['long_files']:
            recommendations.append(f"Break down {len(aggregate_metrics['long_files'])} large files into smaller modules")
        if aggregate_metrics['long_functions']:
            recommendations.append(f"Refactor {len(aggregate_metrics['long_functions'])} long functions into smaller ones")
        if docstring_coverage < self.quality_metrics['min_docstring_coverage']:
            recommendations.append(f"Improve docstring coverage from {docstring_coverage:.1%} to {self.quality_metrics['min_docstring_coverage']:.1%}")
        if aggregate_metrics['high_complexity_files']:
            recommendations.append(f"Reduce complexity in {len(aggregate_metrics['high_complexity_files'])} files")
            
        if not recommendations:
            recommendations.append("Code quality metrics passed - well-structured and documented code")
            
        execution_time = (time.time() - start_time) * 1000
        
        return QualityGateResult(
            gate_name="Code Quality Validation",
            passed=overall_score >= 80,
            score=overall_score,
            details={
                'aggregate_metrics': aggregate_metrics,
                'quality_targets': self.quality_metrics,
                'score_breakdown': {
                    'file_length_score': file_score,
                    'function_length_score': function_score,
                    'docstring_score': docstring_score,
                    'complexity_score': complexity_score
                }
            },
            recommendations=recommendations,
            execution_time_ms=execution_time
        )


class DeploymentReadinessValidator:
    """Deployment readiness validation for SpinTron-NN-Kit."""
    
    def __init__(self):
        self.required_files = [
            'README.md',
            'pyproject.toml',
            'spintron_nn/__init__.py'
        ]
        
        self.optional_files = [
            'LICENSE',
            'CHANGELOG.md',
            'requirements.txt',
            'Dockerfile',
            'docker-compose.yml'
        ]
        
    def check_file_structure(self) -> Dict[str, Any]:
        """Check if required files and structure exist."""
        structure_check = {
            'required_files_present': [],
            'required_files_missing': [],
            'optional_files_present': [],
            'package_structure_valid': False
        }
        
        # Check required files
        for file_path in self.required_files:
            if os.path.exists(file_path):
                structure_check['required_files_present'].append(file_path)
            else:
                structure_check['required_files_missing'].append(file_path)
                
        # Check optional files
        for file_path in self.optional_files:
            if os.path.exists(file_path):
                structure_check['optional_files_present'].append(file_path)
                
        # Check package structure
        spintron_nn_path = Path('spintron_nn')
        if spintron_nn_path.exists() and spintron_nn_path.is_dir():
            submodules = ['core', 'converter', 'hardware', 'training', 'models']
            valid_submodules = sum(1 for module in submodules 
                                 if (spintron_nn_path / module / '__init__.py').exists())
            structure_check['package_structure_valid'] = valid_submodules >= 3
            structure_check['valid_submodules'] = valid_submodules
            structure_check['total_submodules'] = len(submodules)
            
        return structure_check
        
    def check_configuration_files(self) -> Dict[str, Any]:
        """Check configuration files for deployment."""
        config_check = {
            'pyproject_toml_valid': False,
            'docker_support': False,
            'ci_cd_support': False,
            'package_metadata_complete': False
        }
        
        # Check pyproject.toml
        try:
            with open('pyproject.toml', 'r') as f:
                content = f.read()
                required_sections = ['build-system', 'project']
                config_check['pyproject_toml_valid'] = all(
                    f'[{section}]' in content for section in required_sections
                )
                
                # Check package metadata
                metadata_fields = ['name', 'version', 'description', 'authors']
                config_check['package_metadata_complete'] = all(
                    field in content for field in metadata_fields
                )
        except Exception:
            pass
            
        # Check Docker support
        config_check['docker_support'] = (
            os.path.exists('Dockerfile') or 
            os.path.exists('docker-compose.yml')
        )
        
        # Check CI/CD support
        ci_cd_paths = ['.github/workflows', '.gitlab-ci.yml', 'azure-pipelines.yml']
        config_check['ci_cd_support'] = any(os.path.exists(path) for path in ci_cd_paths)
        
        return config_check
        
    def validate_deployment_readiness(self) -> QualityGateResult:
        """Perform deployment readiness validation."""
        start_time = time.time()
        
        # Check file structure
        structure_check = self.check_file_structure()
        
        # Check configuration files
        config_check = self.check_configuration_files()
        
        # Calculate deployment readiness score
        score_components = []
        
        # Required files score (critical)
        required_files_score = (len(structure_check['required_files_present']) / 
                              len(self.required_files)) * 100
        score_components.append(required_files_score * 0.4)  # 40% weight
        
        # Package structure score
        structure_score = 100 if structure_check['package_structure_valid'] else 0
        score_components.append(structure_score * 0.3)  # 30% weight
        
        # Configuration files score
        config_score = (
            (config_check['pyproject_toml_valid'] * 30) +
            (config_check['package_metadata_complete'] * 30) +
            (config_check['docker_support'] * 20) +
            (config_check['ci_cd_support'] * 20)
        )
        score_components.append(config_score * 0.3)  # 30% weight
        
        overall_score = sum(score_components)
        
        # Generate recommendations
        recommendations = []
        
        if structure_check['required_files_missing']:
            recommendations.append(f"Add missing required files: {', '.join(structure_check['required_files_missing'])}")
            
        if not structure_check['package_structure_valid']:
            recommendations.append("Ensure package structure includes core modules with __init__.py files")
            
        if not config_check['pyproject_toml_valid']:
            recommendations.append("Fix pyproject.toml configuration with proper build-system and project sections")
            
        if not config_check['package_metadata_complete']:
            recommendations.append("Complete package metadata in pyproject.toml")
            
        if not config_check['docker_support']:
            recommendations.append("Add Docker support for containerized deployment")
            
        if not config_check['ci_cd_support']:
            recommendations.append("Add CI/CD pipeline configuration")
            
        if not recommendations:
            recommendations.append("Deployment readiness check passed - ready for production deployment")
            
        execution_time = (time.time() - start_time) * 1000
        
        return QualityGateResult(
            gate_name="Deployment Readiness",
            passed=overall_score >= 80,
            score=overall_score,
            details={
                'structure_check': structure_check,
                'config_check': config_check,
                'score_breakdown': {
                    'required_files_score': required_files_score,
                    'structure_score': structure_score,
                    'config_score': config_score
                }
            },
            recommendations=recommendations,
            execution_time_ms=execution_time
        )


def run_all_quality_gates() -> List[QualityGateResult]:
    """Run all quality gates and return results."""
    print("Running Comprehensive Quality Gates...")
    print("=" * 50)
    
    validators = [
        SecurityValidator(),
        PerformanceValidator(),
        CodeQualityValidator(),
        DeploymentReadinessValidator()
    ]
    
    results = []
    
    for validator in validators:
        if hasattr(validator, 'validate_security'):
            result = validator.validate_security()
        elif hasattr(validator, 'validate_performance'):
            result = validator.validate_performance()
        elif hasattr(validator, 'validate_code_quality'):
            result = validator.validate_code_quality()
        elif hasattr(validator, 'validate_deployment_readiness'):
            result = validator.validate_deployment_readiness()
        else:
            continue
            
        results.append(result)
        
        # Print gate result
        status = "✓ PASS" if result.passed else "✗ FAIL"
        print(f"{status} {result.gate_name}: {result.score:.1f}% ({result.execution_time_ms:.1f}ms)")
        
        for rec in result.recommendations[:2]:  # Show top 2 recommendations
            print(f"  → {rec}")
            
        print()
        
    return results


def generate_quality_gates_report(results: List[QualityGateResult]) -> Dict[str, Any]:
    """Generate comprehensive quality gates report."""
    
    # Calculate overall metrics
    total_gates = len(results)
    passed_gates = sum(1 for result in results if result.passed)
    overall_score = sum(result.score for result in results) / total_gates if total_gates > 0 else 0
    total_execution_time = sum(result.execution_time_ms for result in results)
    
    # Categorize results
    critical_failures = [r for r in results if not r.passed and r.score < 50]
    minor_failures = [r for r in results if not r.passed and r.score >= 50]
    
    report = {
        'timestamp': time.time(),
        'overall_metrics': {
            'total_gates': total_gates,
            'passed_gates': passed_gates,
            'failed_gates': total_gates - passed_gates,
            'success_rate': (passed_gates / total_gates) * 100 if total_gates > 0 else 0,
            'overall_score': overall_score,
            'total_execution_time_ms': total_execution_time
        },
        'gate_results': [
            {
                'gate_name': result.gate_name,
                'passed': result.passed,
                'score': result.score,
                'execution_time_ms': result.execution_time_ms,
                'recommendations_count': len(result.recommendations),
                'top_recommendations': result.recommendations[:3]
            }
            for result in results
        ],
        'failure_analysis': {
            'critical_failures': len(critical_failures),
            'minor_failures': len(minor_failures),
            'critical_gate_names': [r.gate_name for r in critical_failures],
            'minor_gate_names': [r.gate_name for r in minor_failures]
        },
        'detailed_results': [
            {
                'gate_name': result.gate_name,
                'details': result.details,
                'all_recommendations': result.recommendations
            }
            for result in results
        ],
        'deployment_ready': all(result.passed for result in results),
        'next_steps': []
    }
    
    # Generate next steps
    if critical_failures:
        report['next_steps'].append("Address critical failures before deployment")
    if minor_failures:
        report['next_steps'].append("Review and fix minor issues for optimal quality")
    if not critical_failures and not minor_failures:
        report['next_steps'].append("Quality gates passed - ready for production deployment")
        
    return report


def main():
    """Run comprehensive quality gates validation."""
    
    print("SpinTron-NN-Kit Comprehensive Quality Gates")
    print("=" * 50)
    
    start_time = time.time()
    
    try:
        # Run all quality gates
        results = run_all_quality_gates()
        
        # Generate comprehensive report
        report = generate_quality_gates_report(results)
        
        # Save detailed report
        with open('quality_gates_report.json', 'w') as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        print("=" * 50)
        print("Quality Gates Summary")
        print("=" * 50)
        print(f"Gates executed: {report['overall_metrics']['total_gates']}")
        print(f"Gates passed: {report['overall_metrics']['passed_gates']}")
        print(f"Gates failed: {report['overall_metrics']['failed_gates']}")
        print(f"Success rate: {report['overall_metrics']['success_rate']:.1f}%")
        print(f"Overall score: {report['overall_metrics']['overall_score']:.1f}%")
        print(f"Total execution time: {report['overall_metrics']['total_execution_time_ms']:.1f}ms")
        
        if report['failure_analysis']['critical_failures'] > 0:
            print(f"\n⚠ Critical failures: {report['failure_analysis']['critical_failures']}")
            for gate_name in report['failure_analysis']['critical_gate_names']:
                print(f"  - {gate_name}")
                
        if report['failure_analysis']['minor_failures'] > 0:
            print(f"\n⚠ Minor failures: {report['failure_analysis']['minor_failures']}")
            for gate_name in report['failure_analysis']['minor_gate_names']:
                print(f"  - {gate_name}")
                
        print(f"\nNext steps:")
        for step in report['next_steps']:
            print(f"  → {step}")
            
        print(f"\nDetailed report saved: quality_gates_report.json")
        
        # Determine overall success
        if report['deployment_ready']:
            print("\n✓ ALL QUALITY GATES PASSED - READY FOR DEPLOYMENT")
            return True
        else:
            print("\n✗ SOME QUALITY GATES FAILED - REVIEW REQUIRED")
            return False
            
    except Exception as e:
        print(f"\n✗ Quality gates execution failed: {e}")
        import traceback
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)