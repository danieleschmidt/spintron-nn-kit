"""
Standalone Quality Validator for SpinTron-NN-Kit.

This module provides comprehensive quality validation without external dependencies,
focusing on code quality, architecture validation, and system readiness.
"""

import time
import json
import os
import re
import subprocess
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class QualityGate(Enum):
    """Quality gate types."""
    ARCHITECTURE = "architecture"
    CODE_QUALITY = "code_quality"
    PERFORMANCE = "performance"
    SECURITY = "security"
    DOCUMENTATION = "documentation"
    DEPLOYMENT = "deployment"


@dataclass
class QualityMetric:
    """Quality metric result."""
    
    name: str
    value: float
    target: float
    passed: bool
    details: str


@dataclass
class QualityGateResult:
    """Quality gate validation result."""
    
    gate: QualityGate
    score: float
    passed: bool
    metrics: List[QualityMetric]
    execution_time: float
    recommendations: List[str]


@dataclass
class StandaloneQualityReport:
    """Standalone quality report."""
    
    timestamp: float
    overall_score: float
    gates_passed: int
    gates_total: int
    deployment_ready: bool
    gate_results: List[QualityGateResult]
    summary: str


class StandaloneQualityValidator:
    """Standalone quality validation without external dependencies."""
    
    def __init__(self, repo_path: str = "/root/repo"):
        self.repo_path = Path(repo_path)
        self.results = []
        
        # Quality thresholds
        self.thresholds = {
            QualityGate.ARCHITECTURE: 0.85,
            QualityGate.CODE_QUALITY: 0.80,
            QualityGate.PERFORMANCE: 0.75,
            QualityGate.SECURITY: 0.80,
            QualityGate.DOCUMENTATION: 0.70,
            QualityGate.DEPLOYMENT: 0.85
        }
    
    def run_validation(self) -> StandaloneQualityReport:
        """Run comprehensive standalone validation."""
        
        print("🎯 SpinTron-NN-Kit Standalone Quality Validation")
        print("=" * 60)
        
        start_time = time.time()
        
        # Run all quality gates
        validators = [
            (QualityGate.ARCHITECTURE, self._validate_architecture),
            (QualityGate.CODE_QUALITY, self._validate_code_quality),
            (QualityGate.PERFORMANCE, self._validate_performance),
            (QualityGate.SECURITY, self._validate_security),
            (QualityGate.DOCUMENTATION, self._validate_documentation),
            (QualityGate.DEPLOYMENT, self._validate_deployment)
        ]
        
        for gate, validator_func in validators:
            print(f"\n🔍 {gate.value.upper()} Quality Gate")
            print("-" * 40)
            
            gate_start = time.time()
            result = validator_func()
            result.execution_time = time.time() - gate_start
            
            self.results.append(result)
            
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"{status} Score: {result.score:.1%} ({result.execution_time:.2f}s)")
            
            # Show key metrics
            for metric in result.metrics[:3]:  # Show top 3 metrics
                icon = "✅" if metric.passed else "❌"
                print(f"  {icon} {metric.name}: {metric.value:.2f} (target: {metric.target:.2f})")
        
        # Generate report
        report = self._generate_report()
        total_time = time.time() - start_time
        
        print(f"\n🏁 Validation Complete ({total_time:.2f}s)")
        print(f"📊 Overall Score: {report.overall_score:.1%}")
        print(f"🎯 Gates Passed: {report.gates_passed}/{report.gates_total}")
        print(f"🚀 Deployment Ready: {'YES' if report.deployment_ready else 'NO'}")
        
        return report
    
    def _validate_architecture(self) -> QualityGateResult:
        """Validate system architecture and design."""
        
        metrics = []
        recommendations = []
        
        # Module structure analysis
        py_files = list(self.repo_path.rglob("*.py"))
        total_files = len(py_files)
        
        if total_files > 0:
            # Check for proper package structure
            init_files = list(self.repo_path.rglob("__init__.py"))
            package_score = len(init_files) / max(1, total_files // 10)  # Expect 1 __init__ per ~10 files
            package_score = min(1.0, package_score)
            
            metrics.append(QualityMetric(
                name="Package Structure",
                value=package_score,
                target=0.8,
                passed=package_score >= 0.8,
                details=f"{len(init_files)} __init__.py files for {total_files} Python files"
            ))
            
            # Check for modular design
            main_modules = [
                "core", "converter", "hardware", "training", "models",
                "utils", "research", "scaling", "security"
            ]
            
            existing_modules = []
            for module in main_modules:
                module_path = self.repo_path / "spintron_nn" / module
                if module_path.exists():
                    existing_modules.append(module)
            
            modularity_score = len(existing_modules) / len(main_modules)
            metrics.append(QualityMetric(
                name="Modular Design",
                value=modularity_score,
                target=0.7,
                passed=modularity_score >= 0.7,
                details=f"{len(existing_modules)}/{len(main_modules)} expected modules present"
            ))
            
            # Check file size distribution (avoid monolithic files)
            large_files = []
            total_lines = 0
            
            for py_file in py_files[:20]:  # Sample first 20 files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = len(f.readlines())
                        total_lines += lines
                        if lines > 1000:
                            large_files.append((py_file.name, lines))
                except:
                    continue
            
            avg_file_size = total_lines / len(py_files) if py_files else 0
            size_score = 1.0 if avg_file_size < 500 else max(0.3, 1.0 - (avg_file_size - 500) / 1000)
            
            metrics.append(QualityMetric(
                name="File Size Distribution",
                value=size_score,
                target=0.7,
                passed=size_score >= 0.7,
                details=f"Average {avg_file_size:.0f} lines per file, {len(large_files)} large files"
            ))
            
            if len(large_files) > 3:
                recommendations.append("Consider breaking down large files into smaller modules")
        
        # Calculate overall architecture score
        if metrics:
            score = sum(m.value for m in metrics) / len(metrics)
        else:
            score = 0.0
            metrics.append(QualityMetric(
                name="Architecture",
                value=0.0,
                target=0.8,
                passed=False,
                details="No Python files found"
            ))
            recommendations.append("Ensure proper Python package structure")
        
        return QualityGateResult(
            gate=QualityGate.ARCHITECTURE,
            score=score,
            passed=score >= self.thresholds[QualityGate.ARCHITECTURE],
            metrics=metrics,
            execution_time=0,
            recommendations=recommendations
        )
    
    def _validate_code_quality(self) -> QualityGateResult:
        """Validate code quality metrics."""
        
        metrics = []
        recommendations = []
        
        py_files = list(self.repo_path.rglob("*.py"))
        
        if py_files:
            # Documentation coverage
            documented_files = 0
            total_functions = 0
            documented_functions = 0
            
            for py_file in py_files[:15]:  # Sample files to avoid timeout
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for file-level docstring
                        if content.strip().startswith('"""') or content.strip().startswith("'''"):
                            documented_files += 1
                        
                        # Count functions and their documentation
                        function_matches = re.findall(r'def\s+\w+\s*\(', content)
                        total_functions += len(function_matches)
                        
                        # Simple docstring detection
                        docstring_matches = re.findall(r'def\s+\w+\s*\([^)]*\):[^"\']*["\'\s]*"""', content)
                        documented_functions += len(docstring_matches)
                        
                except:
                    continue
            
            doc_coverage = documented_files / len(py_files) if py_files else 0
            func_doc_coverage = documented_functions / total_functions if total_functions > 0 else 0
            
            metrics.append(QualityMetric(
                name="Documentation Coverage",
                value=doc_coverage,
                target=0.8,
                passed=doc_coverage >= 0.8,
                details=f"{documented_files}/{len(py_files)} files documented"
            ))
            
            metrics.append(QualityMetric(
                name="Function Documentation",
                value=func_doc_coverage,
                target=0.6,
                passed=func_doc_coverage >= 0.6,
                details=f"{documented_functions}/{total_functions} functions documented"
            ))
            
            # Code complexity (simplified)
            complex_files = 0
            total_complexity = 0
            
            for py_file in py_files[:10]:  # Sample files
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Simple complexity metrics
                        nested_blocks = content.count('    if ') + content.count('    for ') + content.count('    while ')
                        file_complexity = nested_blocks / max(1, content.count('\n') // 10)
                        total_complexity += file_complexity
                        
                        if file_complexity > 0.5:
                            complex_files += 1
                            
                except:
                    continue
            
            avg_complexity = total_complexity / len(py_files) if py_files else 0
            complexity_score = max(0.0, 1.0 - avg_complexity)
            
            metrics.append(QualityMetric(
                name="Code Complexity",
                value=complexity_score,
                target=0.7,
                passed=complexity_score >= 0.7,
                details=f"Average complexity {avg_complexity:.2f}, {complex_files} complex files"
            ))
            
            # Import quality
            import_issues = 0
            for py_file in py_files[:10]:
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for problematic imports
                        if 'import *' in content:
                            import_issues += 1
                        if 'from __future__' not in content and 'print(' in content:
                            # Modern Python style
                            pass
                            
                except:
                    continue
            
            import_score = max(0.0, 1.0 - import_issues / len(py_files)) if py_files else 0
            
            metrics.append(QualityMetric(
                name="Import Quality",
                value=import_score,
                target=0.9,
                passed=import_score >= 0.9,
                details=f"{import_issues} import issues found"
            ))
            
            if import_issues > 0:
                recommendations.append("Avoid wildcard imports and improve import structure")
            if doc_coverage < 0.8:
                recommendations.append("Improve documentation coverage")
            if complex_files > 2:
                recommendations.append("Reduce code complexity in complex files")
        
        # Calculate overall score
        if metrics:
            score = sum(m.value for m in metrics) / len(metrics)
        else:
            score = 0.0
            recommendations.append("No Python files found for quality analysis")
        
        return QualityGateResult(
            gate=QualityGate.CODE_QUALITY,
            score=score,
            passed=score >= self.thresholds[QualityGate.CODE_QUALITY],
            metrics=metrics,
            execution_time=0,
            recommendations=recommendations
        )
    
    def _validate_performance(self) -> QualityGateResult:
        """Validate performance characteristics."""
        
        metrics = []
        recommendations = []
        
        # Run built-in benchmark if available
        benchmark_score = 0.0
        benchmark_passed = False
        
        benchmark_file = self.repo_path / "benchmarks" / "simple_benchmark.py"
        if benchmark_file.exists():
            try:
                result = subprocess.run(
                    ["python3", str(benchmark_file)],
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=str(self.repo_path)
                )
                
                if result.returncode == 0:
                    output = result.stdout
                    if "Overall performance validation: ✓ PASSED" in output:
                        benchmark_score = 1.0
                        benchmark_passed = True
                    elif "Overall performance validation:" in output:
                        benchmark_score = 0.7  # Partial pass
                    else:
                        benchmark_score = 0.3
                else:
                    benchmark_score = 0.1
                    
            except subprocess.TimeoutExpired:
                benchmark_score = 0.2
                recommendations.append("Performance benchmark timed out - optimize performance")
            except Exception:
                benchmark_score = 0.0
                recommendations.append("Fix performance benchmark execution")
        else:
            benchmark_score = 0.0
            recommendations.append("Add performance benchmarks")
        
        metrics.append(QualityMetric(
            name="Benchmark Execution",
            value=benchmark_score,
            target=0.8,
            passed=benchmark_passed,
            details="Built-in benchmark test results"
        ))
        
        # Algorithm efficiency analysis
        algorithm_files = [
            "spintron_nn/research/quantum_enhanced_crossbar_optimization.py",
            "spintron_nn/adaptive_performance_optimizer.py",
            "spintron_nn/quantum_distributed_accelerator.py"
        ]
        
        efficient_algorithms = 0
        for algo_file in algorithm_files:
            file_path = self.repo_path / algo_file
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        
                        # Check for optimization patterns
                        optimization_patterns = [
                            'optimization', 'parallel', 'concurrent', 'async',
                            'cache', 'memory', 'efficiency', 'performance'
                        ]
                        
                        pattern_count = sum(1 for pattern in optimization_patterns if pattern in content.lower())
                        if pattern_count >= 3:
                            efficient_algorithms += 1
                            
                except:
                    continue
        
        algorithm_score = efficient_algorithms / len(algorithm_files) if algorithm_files else 0
        
        metrics.append(QualityMetric(
            name="Algorithm Efficiency",
            value=algorithm_score,
            target=0.7,
            passed=algorithm_score >= 0.7,
            details=f"{efficient_algorithms}/{len(algorithm_files)} algorithms show optimization patterns"
        ))
        
        # Memory usage patterns
        memory_score = 1.0  # Default good score
        large_structures = 0
        
        for py_file in list(self.repo_path.rglob("*.py"))[:10]:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for potential memory issues
                    if 'list(' in content and 'range(' in content:
                        # Potential large list creation
                        large_structures += 1
                    
            except:
                continue
        
        if large_structures > 5:
            memory_score = 0.6
            recommendations.append("Review memory usage patterns in large data structures")
        
        metrics.append(QualityMetric(
            name="Memory Efficiency",
            value=memory_score,
            target=0.8,
            passed=memory_score >= 0.8,
            details=f"Memory usage patterns analysis"
        ))
        
        # Calculate overall score
        score = sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
        
        return QualityGateResult(
            gate=QualityGate.PERFORMANCE,
            score=score,
            passed=score >= self.thresholds[QualityGate.PERFORMANCE],
            metrics=metrics,
            execution_time=0,
            recommendations=recommendations
        )
    
    def _validate_security(self) -> QualityGateResult:
        """Validate security practices."""
        
        metrics = []
        recommendations = []
        
        # Security pattern analysis
        py_files = list(self.repo_path.rglob("*.py"))
        
        security_issues = 0
        security_features = 0
        
        security_antipatterns = [
            'eval(', 'exec(', 'input(', '__import__',
            'password = ', 'secret = ', 'key = "'
        ]
        
        security_patterns = [
            'hashlib', 'hmac', 'secrets', 'ssl',
            'authentication', 'authorization', 'encryption'
        ]
        
        for py_file in py_files[:15]:  # Sample files
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                    # Check for security anti-patterns
                    for antipattern in security_antipatterns:
                        if antipattern in content:
                            security_issues += 1
                            break
                    
                    # Check for security features
                    for pattern in security_patterns:
                        if pattern in content.lower():
                            security_features += 1
                            break
                            
            except:
                continue
        
        security_score = max(0.0, 1.0 - security_issues / max(1, len(py_files)))
        security_feature_score = min(1.0, security_features / max(1, len(py_files) // 5))
        
        metrics.append(QualityMetric(
            name="Security Anti-patterns",
            value=security_score,
            target=0.9,
            passed=security_score >= 0.9,
            details=f"{security_issues} potential security issues found"
        ))
        
        metrics.append(QualityMetric(
            name="Security Features",
            value=security_feature_score,
            target=0.5,
            passed=security_feature_score >= 0.5,
            details=f"{security_features} security features detected"
        ))
        
        # Check for security framework
        security_framework_file = self.repo_path / "spintron_nn" / "advanced_security_framework.py"
        if security_framework_file.exists():
            framework_score = 1.0
            metrics.append(QualityMetric(
                name="Security Framework",
                value=framework_score,
                target=1.0,
                passed=True,
                details="Advanced security framework present"
            ))
        else:
            recommendations.append("Implement comprehensive security framework")
        
        if security_issues > 0:
            recommendations.append("Address security anti-patterns in code")
        if security_features < 2:
            recommendations.append("Implement additional security features")
        
        # Calculate overall score
        score = sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
        
        return QualityGateResult(
            gate=QualityGate.SECURITY,
            score=score,
            passed=score >= self.thresholds[QualityGate.SECURITY],
            metrics=metrics,
            execution_time=0,
            recommendations=recommendations
        )
    
    def _validate_documentation(self) -> QualityGateResult:
        """Validate documentation quality and completeness."""
        
        metrics = []
        recommendations = []
        
        # README analysis
        readme_score = 0.0
        readme_file = self.repo_path / "README.md"
        
        if readme_file.exists():
            try:
                with open(readme_file, 'r', encoding='utf-8') as f:
                    readme_content = f.read()
                
                required_sections = [
                    "installation", "usage", "example", "api", "overview"
                ]
                
                sections_found = sum(1 for section in required_sections 
                                   if section in readme_content.lower())
                readme_score = sections_found / len(required_sections)
                
                # Check README length (should be comprehensive)
                if len(readme_content) > 1000:
                    readme_score = min(1.0, readme_score + 0.2)
                    
            except:
                readme_score = 0.0
        else:
            recommendations.append("Create comprehensive README.md file")
        
        metrics.append(QualityMetric(
            name="README Quality",
            value=readme_score,
            target=0.8,
            passed=readme_score >= 0.8,
            details=f"README completeness and structure"
        ))
        
        # API documentation
        api_doc_score = 0.0
        
        # Check for docstrings in main modules
        main_modules = [
            "spintron_nn/__init__.py",
            "spintron_nn/core/__init__.py",
            "spintron_nn/autonomous_research_executor.py"
        ]
        
        documented_modules = 0
        for module_path in main_modules:
            file_path = self.repo_path / module_path
            if file_path.exists():
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                        if '"""' in content or "'''" in content:
                            documented_modules += 1
                except:
                    continue
        
        api_doc_score = documented_modules / len(main_modules) if main_modules else 0
        
        metrics.append(QualityMetric(
            name="API Documentation",
            value=api_doc_score,
            target=0.7,
            passed=api_doc_score >= 0.7,
            details=f"{documented_modules}/{len(main_modules)} core modules documented"
        ))
        
        # Examples and tutorials
        examples_score = 0.0
        examples_dir = self.repo_path / "examples"
        
        if examples_dir.exists():
            example_files = list(examples_dir.glob("*.py"))
            if len(example_files) >= 2:
                examples_score = 1.0
            elif len(example_files) == 1:
                examples_score = 0.7
            else:
                examples_score = 0.3
        else:
            recommendations.append("Add usage examples and tutorials")
        
        metrics.append(QualityMetric(
            name="Examples & Tutorials",
            value=examples_score,
            target=0.6,
            passed=examples_score >= 0.6,
            details=f"Example files and tutorial content"
        ))
        
        if readme_score < 0.8:
            recommendations.append("Improve README.md with more comprehensive documentation")
        if api_doc_score < 0.7:
            recommendations.append("Add API documentation to core modules")
        
        # Calculate overall score
        score = sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
        
        return QualityGateResult(
            gate=QualityGate.DOCUMENTATION,
            score=score,
            passed=score >= self.thresholds[QualityGate.DOCUMENTATION],
            metrics=metrics,
            execution_time=0,
            recommendations=recommendations
        )
    
    def _validate_deployment(self) -> QualityGateResult:
        """Validate deployment readiness."""
        
        metrics = []
        recommendations = []
        
        # Configuration files
        config_score = 0.0
        required_config_files = [
            "pyproject.toml", "requirements.txt", "setup.py", "Dockerfile"
        ]
        
        config_files_present = 0
        for config_file in required_config_files:
            if (self.repo_path / config_file).exists():
                config_files_present += 1
        
        config_score = config_files_present / len(required_config_files)
        
        metrics.append(QualityMetric(
            name="Configuration Files",
            value=config_score,
            target=0.5,
            passed=config_score >= 0.5,
            details=f"{config_files_present}/{len(required_config_files)} config files present"
        ))
        
        # Deployment scripts
        deployment_score = 0.0
        deployment_indicators = [
            "deployment/", "docker-compose.yml", "k8s/", 
            "scripts/deploy", "Makefile"
        ]
        
        deployment_features = 0
        for indicator in deployment_indicators:
            if (self.repo_path / indicator).exists():
                deployment_features += 1
        
        deployment_score = min(1.0, deployment_features / 3)  # Need at least 3 for full score
        
        metrics.append(QualityMetric(
            name="Deployment Infrastructure",
            value=deployment_score,
            target=0.6,
            passed=deployment_score >= 0.6,
            details=f"{deployment_features} deployment features found"
        ))
        
        # Testing infrastructure
        testing_score = 0.0
        testing_files = list(self.repo_path.rglob("test_*.py"))
        benchmark_files = list(self.repo_path.rglob("benchmark*.py"))
        
        if len(testing_files) > 0 or len(benchmark_files) > 0:
            testing_score = min(1.0, (len(testing_files) + len(benchmark_files)) / 5)
        
        metrics.append(QualityMetric(
            name="Testing Infrastructure",
            value=testing_score,
            target=0.5,
            passed=testing_score >= 0.5,
            details=f"{len(testing_files)} test files, {len(benchmark_files)} benchmark files"
        ))
        
        # License and legal
        license_score = 1.0 if (self.repo_path / "LICENSE").exists() else 0.0
        
        metrics.append(QualityMetric(
            name="License & Legal",
            value=license_score,
            target=1.0,
            passed=license_score >= 1.0,
            details="License file presence"
        ))
        
        if config_score < 0.5:
            recommendations.append("Add essential configuration files (pyproject.toml, etc.)")
        if deployment_score < 0.6:
            recommendations.append("Implement deployment infrastructure (Docker, scripts)")
        if testing_score < 0.5:
            recommendations.append("Add comprehensive testing infrastructure")
        if license_score < 1.0:
            recommendations.append("Add LICENSE file")
        
        # Calculate overall score
        score = sum(m.value for m in metrics) / len(metrics) if metrics else 0.0
        
        return QualityGateResult(
            gate=QualityGate.DEPLOYMENT,
            score=score,
            passed=score >= self.thresholds[QualityGate.DEPLOYMENT],
            metrics=metrics,
            execution_time=0,
            recommendations=recommendations
        )
    
    def _generate_report(self) -> StandaloneQualityReport:
        """Generate comprehensive quality report."""
        
        gates_passed = sum(1 for result in self.results if result.passed)
        gates_total = len(self.results)
        
        # Calculate weighted overall score
        weights = {
            QualityGate.ARCHITECTURE: 0.20,
            QualityGate.CODE_QUALITY: 0.20,
            QualityGate.PERFORMANCE: 0.20,
            QualityGate.SECURITY: 0.15,
            QualityGate.DOCUMENTATION: 0.15,
            QualityGate.DEPLOYMENT: 0.10
        }
        
        overall_score = sum(
            result.score * weights.get(result.gate, 0.1)
            for result in self.results
        )
        
        # Determine deployment readiness
        critical_gates_passed = sum(1 for result in self.results 
                                  if result.passed and result.gate in 
                                  [QualityGate.ARCHITECTURE, QualityGate.PERFORMANCE, QualityGate.SECURITY])
        
        deployment_ready = (
            overall_score >= 0.75 and
            critical_gates_passed >= 2 and
            gates_passed >= 4
        )
        
        # Generate summary
        summary = self._generate_summary(overall_score, gates_passed, gates_total, deployment_ready)
        
        return StandaloneQualityReport(
            timestamp=time.time(),
            overall_score=overall_score,
            gates_passed=gates_passed,
            gates_total=gates_total,
            deployment_ready=deployment_ready,
            gate_results=self.results,
            summary=summary
        )
    
    def _generate_summary(self, overall_score: float, gates_passed: int, 
                         gates_total: int, deployment_ready: bool) -> str:
        """Generate executive summary."""
        
        summary = f"""
SPINTRON-NN-KIT QUALITY ASSESSMENT SUMMARY

Overall Quality Score: {overall_score:.1%}
Quality Gates Passed: {gates_passed}/{gates_total}
Deployment Ready: {'✅ YES' if deployment_ready else '❌ NO'}

Gate Results:"""
        
        for result in self.results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            summary += f"\n  {status} {result.gate.value.upper()}: {result.score:.1%}"
        
        summary += f"\n\nKey Strengths:"
        strengths = [result for result in self.results if result.passed]
        for strength in strengths[:3]:
            summary += f"\n  ✅ {strength.gate.value.title()}: Strong implementation"
        
        if not strengths:
            summary += "\n  ⚠️  Focus needed on core quality improvements"
        
        summary += f"\n\nPriority Improvements:"
        failures = [result for result in self.results if not result.passed]
        for failure in failures[:3]:
            if failure.recommendations:
                summary += f"\n  🔧 {failure.gate.value.title()}: {failure.recommendations[0]}"
        
        if deployment_ready:
            summary += f"\n\n✅ READY FOR DEPLOYMENT"
        else:
            summary += f"\n\n⚠️  ADDRESS QUALITY ISSUES BEFORE DEPLOYMENT"
        
        return summary
    
    def save_report(self, report: StandaloneQualityReport, filename: str = "quality_assessment.json"):
        """Save quality report to file."""
        
        # Convert to JSON-serializable format
        report_dict = asdict(report)
        
        # Handle enum serialization
        for gate_result in report_dict["gate_results"]:
            if isinstance(gate_result["gate"], dict):
                gate_result["gate"] = gate_result["gate"]["value"]
            else:
                gate_result["gate"] = str(gate_result["gate"])
        
        output_path = self.repo_path / filename
        with open(output_path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        print(f"\n📄 Quality report saved: {output_path}")
        return str(output_path)


def main():
    """Run standalone quality validation."""
    
    validator = StandaloneQualityValidator()
    report = validator.run_validation()
    
    # Save report
    validator.save_report(report)
    
    # Print summary
    print(f"\n{report.summary}")
    
    return report


if __name__ == "__main__":
    main()