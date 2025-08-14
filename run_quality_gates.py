#!/usr/bin/env python3
"""
Quality Gates Validation Script for Spintronic Neural Network Framework.

This script runs comprehensive quality checks and validation tests
to ensure the framework meets production readiness standards.
"""

import sys
import subprocess
import time
import json
from pathlib import Path
from typing import Dict, List, Any
import traceback

def run_command(cmd: List[str], description: str = "") -> Dict[str, Any]:
    """Run a command and return result."""
    print(f"\n🔍 {description or ' '.join(cmd)}")
    print("=" * 60)
    
    start_time = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        execution_time = time.time() - start_time
        
        return {
            "command": " ".join(cmd),
            "description": description,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "execution_time": execution_time,
            "success": result.returncode == 0
        }
    except subprocess.TimeoutExpired:
        return {
            "command": " ".join(cmd),
            "description": description,
            "returncode": -1,
            "stdout": "",
            "stderr": "Command timed out after 300 seconds",
            "execution_time": 300,
            "success": False
        }
    except Exception as e:
        return {
            "command": " ".join(cmd),
            "description": description,
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "execution_time": time.time() - start_time,
            "success": False
        }

def validate_framework_structure() -> Dict[str, Any]:
    """Validate framework structure and key files."""
    print("\n📁 Validating Framework Structure")
    print("=" * 60)
    
    required_files = [
        "spintron_nn/__init__.py",
        "spintron_nn/core/__init__.py", 
        "spintron_nn/core/mtj_models.py",
        "spintron_nn/core/crossbar.py",
        "spintron_nn/research/__init__.py",
        "spintron_nn/research/algorithms.py",
        "spintron_nn/research/validation.py",
        "spintron_nn/research/autonomous_optimization.py",
        "spintron_nn/reliability/__init__.py",
        "spintron_nn/reliability/fault_tolerance.py",
        "spintron_nn/security/__init__.py", 
        "spintron_nn/security/secure_computing.py",
        "spintron_nn/scaling/__init__.py",
        "spintron_nn/scaling/quantum_acceleration.py",
        "spintron_nn/scaling/cloud_orchestration.py",
        "tests/__init__.py",
        "tests/comprehensive_integration_test.py",
        "pyproject.toml",
        "README.md"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        if Path(file_path).exists():
            existing_files.append(file_path)
            print(f"✅ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"❌ {file_path}")
    
    return {
        "test_name": "Framework Structure Validation",
        "success": len(missing_files) == 0,
        "required_files": len(required_files),
        "existing_files": len(existing_files),
        "missing_files": missing_files,
        "coverage": len(existing_files) / len(required_files) * 100
    }

def run_basic_import_tests() -> Dict[str, Any]:
    """Test basic imports work."""
    print("\n🐍 Testing Basic Imports")
    print("=" * 60)
    
    import_tests = [
        "import numpy as np; print('NumPy:', np.__version__)",
        "import torch; print('PyTorch:', torch.__version__)",
        "from spintron_nn.core.mtj_models import MTJConfig, MTJDevice; print('MTJ models imported successfully')",
        "from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig; print('Crossbar imported successfully')",
    ]
    
    results = []
    for test in import_tests:
        result = run_command([sys.executable, "-c", test], f"Import test: {test[:50]}...")
        results.append(result)
        if result["success"]:
            print(f"✅ {result['stdout'].strip()}")
        else:
            print(f"❌ Import failed: {result['stderr']}")
    
    success_count = sum(1 for r in results if r["success"])
    
    return {
        "test_name": "Basic Import Tests",
        "success": success_count == len(import_tests),
        "total_tests": len(import_tests),
        "passed_tests": success_count,
        "failed_tests": len(import_tests) - success_count,
        "results": results
    }

def run_core_functionality_tests() -> Dict[str, Any]:
    """Test core framework functionality."""
    print("\n⚙️ Testing Core Functionality")
    print("=" * 60)
    
    core_test_script = '''
import numpy as np
import torch
from spintron_nn.core.mtj_models import MTJConfig, MTJDevice
from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig

print("Testing MTJ Device...")
config = MTJConfig(resistance_high=10e3, resistance_low=5e3, switching_voltage=0.3)
device = MTJDevice(config)
print(f"✅ MTJ Device created: resistance = {device.resistance:.0f} Ohm")

print("Testing MTJ Crossbar...")
crossbar_config = CrossbarConfig(rows=16, cols=16, mtj_config=config)
crossbar = MTJCrossbar(crossbar_config)
print(f"✅ Crossbar created: {crossbar.rows}x{crossbar.cols}")

print("Testing weight programming...")
weights = np.random.uniform(-1, 1, (16, 16))
conductances = crossbar.set_weights(weights)
print(f"✅ Weights programmed: shape {conductances.shape}")

print("Testing VMM computation...")
input_voltages = np.random.uniform(-0.5, 0.5, 16)
output = crossbar.compute_vmm(input_voltages)
print(f"✅ VMM computed: output shape {output.shape}")

print("Testing statistics...")
stats = crossbar.get_statistics()
print(f"✅ Statistics: {stats['total_cells']} cells, {stats['read_operations']} reads")

print("🎉 Core functionality tests PASSED")
'''
    
    result = run_command([sys.executable, "-c", core_test_script], "Core Functionality Test")
    
    return {
        "test_name": "Core Functionality Tests",
        "success": result["success"] and "PASSED" in result["stdout"],
        "execution_time": result["execution_time"],
        "output": result["stdout"],
        "error": result["stderr"] if not result["success"] else None
    }

def run_performance_benchmarks() -> Dict[str, Any]:
    """Run performance benchmarks."""
    print("\n🏃 Running Performance Benchmarks")
    print("=" * 60)
    
    benchmark_script = '''
import time
import numpy as np
from spintron_nn.core.mtj_models import MTJConfig
from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig

def benchmark_crossbar_performance():
    config = CrossbarConfig(rows=64, cols=64, mtj_config=MTJConfig())
    crossbar = MTJCrossbar(config)
    
    # Benchmark weight programming
    weights = np.random.uniform(-1, 1, (64, 64))
    start_time = time.time()
    conductances = crossbar.set_weights(weights)
    weight_time = time.time() - start_time
    
    # Benchmark VMM computation
    input_voltages = np.random.uniform(-0.5, 0.5, 64)
    start_time = time.time()
    output = crossbar.compute_vmm(input_voltages)
    vmm_time = time.time() - start_time
    
    # Benchmark multiple VMM operations
    start_time = time.time()
    for _ in range(100):
        input_voltages = np.random.uniform(-0.5, 0.5, 64)
        output = crossbar.compute_vmm(input_voltages)
    batch_time = time.time() - start_time
    
    print(f"Weight Programming: {weight_time*1000:.2f} ms")
    print(f"Single VMM: {vmm_time*1000:.2f} ms") 
    print(f"100 VMM Operations: {batch_time*1000:.2f} ms")
    print(f"VMM Throughput: {100/batch_time:.1f} ops/sec")
    
    # Performance criteria
    assert weight_time < 1.0, f"Weight programming too slow: {weight_time:.3f}s"
    assert vmm_time < 0.1, f"VMM computation too slow: {vmm_time:.3f}s"
    assert batch_time < 5.0, f"Batch processing too slow: {batch_time:.3f}s"
    
    print("🎉 Performance benchmarks PASSED")
    
    return {
        "weight_programming_ms": weight_time * 1000,
        "single_vmm_ms": vmm_time * 1000,
        "batch_vmm_ms": batch_time * 1000,
        "vmm_throughput_ops_per_sec": 100 / batch_time
    }

if __name__ == "__main__":
    results = benchmark_crossbar_performance()
    print(f"BENCHMARK_RESULTS: {results}")
'''
    
    result = run_command([sys.executable, "-c", benchmark_script], "Performance Benchmarks")
    
    # Extract benchmark results
    benchmark_results = {}
    if result["success"] and "BENCHMARK_RESULTS:" in result["stdout"]:
        try:
            import ast
            results_line = [line for line in result["stdout"].split('\n') if line.startswith("BENCHMARK_RESULTS:")][0]
            benchmark_data = results_line.split("BENCHMARK_RESULTS:")[1].strip()
            benchmark_results = ast.literal_eval(benchmark_data)
        except:
            pass
    
    return {
        "test_name": "Performance Benchmarks",
        "success": result["success"] and "PASSED" in result["stdout"],
        "execution_time": result["execution_time"],
        "benchmark_results": benchmark_results,
        "output": result["stdout"],
        "error": result["stderr"] if not result["success"] else None
    }

def run_memory_safety_tests() -> Dict[str, Any]:
    """Test memory safety and resource management."""
    print("\n🧠 Testing Memory Safety")
    print("=" * 60)
    
    memory_test_script = '''
import gc
import psutil
import numpy as np
from spintron_nn.core.mtj_models import MTJConfig
from spintron_nn.core.crossbar import MTJCrossbar, CrossbarConfig

def test_memory_safety():
    process = psutil.Process()
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Initial memory: {initial_memory:.1f} MB")
    
    # Create many crossbars
    crossbars = []
    for i in range(20):
        config = CrossbarConfig(rows=32, cols=32, mtj_config=MTJConfig())
        crossbar = MTJCrossbar(config)
        weights = np.random.uniform(-1, 1, (32, 32))
        crossbar.set_weights(weights)
        crossbars.append(crossbar)
    
    peak_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_used = peak_memory - initial_memory
    print(f"Peak memory: {peak_memory:.1f} MB (used: {memory_used:.1f} MB)")
    
    # Clean up
    del crossbars
    gc.collect()
    
    final_memory = process.memory_info().rss / 1024 / 1024  # MB
    memory_recovered = peak_memory - final_memory
    print(f"Final memory: {final_memory:.1f} MB (recovered: {memory_recovered:.1f} MB)")
    
    # Memory usage criteria
    assert memory_used < 200, f"Excessive memory usage: {memory_used:.1f} MB"
    assert memory_recovered > memory_used * 0.5, f"Poor memory recovery: {memory_recovered:.1f} MB"
    
    print("🎉 Memory safety tests PASSED")
    
    return {
        "initial_memory_mb": initial_memory,
        "peak_memory_mb": peak_memory,
        "final_memory_mb": final_memory,
        "memory_used_mb": memory_used,
        "memory_recovered_mb": memory_recovered
    }

if __name__ == "__main__":
    results = test_memory_safety()
    print(f"MEMORY_RESULTS: {results}")
'''
    
    result = run_command([sys.executable, "-c", memory_test_script], "Memory Safety Tests")
    
    # Extract memory results
    memory_results = {}
    if result["success"] and "MEMORY_RESULTS:" in result["stdout"]:
        try:
            import ast
            results_line = [line for line in result["stdout"].split('\n') if line.startswith("MEMORY_RESULTS:")][0]
            memory_data = results_line.split("MEMORY_RESULTS:")[1].strip()
            memory_results = ast.literal_eval(memory_data)
        except:
            pass
    
    return {
        "test_name": "Memory Safety Tests",
        "success": result["success"] and "PASSED" in result["stdout"],
        "execution_time": result["execution_time"],
        "memory_results": memory_results,
        "output": result["stdout"],
        "error": result["stderr"] if not result["success"] else None
    }

def run_security_validation() -> Dict[str, Any]:
    """Test security features."""
    print("\n🔒 Testing Security Features")
    print("=" * 60)
    
    security_test_script = '''
import numpy as np
from spintron_nn.core.crossbar import CrossbarConfig
from spintron_nn.core.mtj_models import MTJConfig
from spintron_nn.security.secure_computing import SecureCrossbar, SecurityConfig, SecurityLevel

def test_security_features():
    print("Testing security configuration...")
    security_config = SecurityConfig(
        security_level=SecurityLevel.HIGH,
        enable_differential_privacy=True,
        dp_epsilon=1.0
    )
    print("✅ Security config created")
    
    print("Testing secure crossbar...")
    crossbar_config = CrossbarConfig(rows=16, cols=16, mtj_config=MTJConfig())
    secure_crossbar = SecureCrossbar(crossbar_config, security_config)
    print("✅ Secure crossbar created")
    
    print("Testing authentication...")
    session_id = secure_crossbar.authenticate_user("test_user", "secure_password")
    assert session_id is not None
    print("✅ Authentication successful")
    
    print("Testing secure computation...")
    input_voltages = np.random.uniform(-0.5, 0.5, 16)
    output = secure_crossbar.secure_compute_vmm(input_voltages, session_id)
    assert len(output) == 16
    assert np.all(np.isfinite(output))
    print("✅ Secure computation successful")
    
    print("Testing privacy budget...")
    budget_status = secure_crossbar.get_privacy_budget_status()
    assert budget_status['spent_epsilon'] > 0
    print("✅ Privacy budget tracking working")
    
    print("🎉 Security validation PASSED")

if __name__ == "__main__":
    test_security_features()
'''
    
    result = run_command([sys.executable, "-c", security_test_script], "Security Validation")
    
    return {
        "test_name": "Security Validation",
        "success": result["success"] and "PASSED" in result["stdout"],
        "execution_time": result["execution_time"],
        "output": result["stdout"],
        "error": result["stderr"] if not result["success"] else None
    }

def generate_quality_report(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate comprehensive quality report."""
    
    total_tests = len(all_results)
    passed_tests = sum(1 for r in all_results if r.get("success", False))
    failed_tests = total_tests - passed_tests
    
    overall_success = failed_tests == 0
    quality_score = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
    
    report = {
        "timestamp": time.time(),
        "overall_success": overall_success,
        "quality_score": quality_score,
        "total_tests": total_tests,
        "passed_tests": passed_tests,
        "failed_tests": failed_tests,
        "test_results": all_results,
        "recommendations": []
    }
    
    # Add recommendations based on results
    if failed_tests > 0:
        report["recommendations"].append("Address failing tests before production deployment")
    
    if quality_score < 80:
        report["recommendations"].append("Quality score below 80% - additional testing recommended")
    
    # Check specific criteria
    performance_test = next((r for r in all_results if r.get("test_name") == "Performance Benchmarks"), None)
    if performance_test and performance_test.get("success"):
        benchmark_results = performance_test.get("benchmark_results", {})
        if benchmark_results.get("vmm_throughput_ops_per_sec", 0) < 50:
            report["recommendations"].append("VMM throughput below optimal - consider performance optimization")
    
    memory_test = next((r for r in all_results if r.get("test_name") == "Memory Safety Tests"), None)
    if memory_test and memory_test.get("success"):
        memory_results = memory_test.get("memory_results", {})
        if memory_results.get("memory_used_mb", 0) > 100:
            report["recommendations"].append("High memory usage detected - monitor in production")
    
    if not report["recommendations"]:
        report["recommendations"].append("All quality gates passed - framework ready for production")
    
    return report

def main():
    """Run all quality gates."""
    print("🚀 SPINTRONIC NEURAL NETWORK FRAMEWORK - QUALITY GATES")
    print("=" * 80)
    print("Running comprehensive quality validation...")
    
    all_results = []
    
    try:
        # 1. Framework Structure Validation
        result = validate_framework_structure()
        all_results.append(result)
        
        # 2. Basic Import Tests
        result = run_basic_import_tests()
        all_results.append(result)
        
        # 3. Core Functionality Tests
        result = run_core_functionality_tests()
        all_results.append(result)
        
        # 4. Performance Benchmarks
        result = run_performance_benchmarks()
        all_results.append(result)
        
        # 5. Memory Safety Tests
        result = run_memory_safety_tests()
        all_results.append(result)
        
        # 6. Security Validation
        result = run_security_validation()
        all_results.append(result)
        
    except Exception as e:
        print(f"\n❌ Quality gates execution failed: {str(e)}")
        traceback.print_exc()
        return 1
    
    # Generate comprehensive report
    quality_report = generate_quality_report(all_results)
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏁 QUALITY GATES SUMMARY")
    print("=" * 80)
    
    if quality_report["overall_success"]:
        print("🎉 ALL QUALITY GATES PASSED!")
        status_emoji = "✅"
    else:
        print("❌ SOME QUALITY GATES FAILED")
        status_emoji = "❌"
    
    print(f"\n{status_emoji} Overall Success: {quality_report['overall_success']}")
    print(f"📊 Quality Score: {quality_report['quality_score']:.1f}%")
    print(f"✅ Passed Tests: {quality_report['passed_tests']}/{quality_report['total_tests']}")
    print(f"❌ Failed Tests: {quality_report['failed_tests']}")
    
    print("\n📋 TEST RESULTS:")
    for result in all_results:
        status = "✅" if result.get("success", False) else "❌"
        test_name = result.get("test_name", "Unknown Test")
        print(f"  {status} {test_name}")
    
    print("\n💡 RECOMMENDATIONS:")
    for rec in quality_report["recommendations"]:
        print(f"  • {rec}")
    
    # Save detailed report
    report_file = "quality_gates_report.json"
    with open(report_file, 'w') as f:
        json.dump(quality_report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Return exit code
    return 0 if quality_report["overall_success"] else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)