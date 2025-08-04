#!/usr/bin/env python3
"""
Basic Validation Suite for SpinTron-NN-Kit.

Validates core functionality without external dependencies.
"""

import time
import sys
import os
import json
import math
from pathlib import Path

# Add spintron_nn to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_imports():
    """Test that all modules can be imported."""
    print("=== Import Validation ===")
    
    modules_to_test = [
        'spintron_nn',
        'spintron_nn.core',
        'spintron_nn.core.mtj_models',
        'spintron_nn.core.crossbar',
        'spintron_nn.converter',
        'spintron_nn.hardware',
        'spintron_nn.training',
        'spintron_nn.simulation',
        'spintron_nn.models',
        'spintron_nn.cli',
        'spintron_nn.utils'
    ]
    
    passed = 0
    failed = 0
    
    for module in modules_to_test:
        try:
            __import__(module)
            print(f"  ✓ {module}")
            passed += 1
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            failed += 1
    
    print(f"\nImport results: {passed} passed, {failed} failed")
    return failed == 0


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\n=== Basic Functionality Tests ===")
    
    try:
        # Test MTJ configuration
        from spintron_nn.core.mtj_models import MTJConfig
        config = MTJConfig()
        assert config.resistance_high > config.resistance_low
        print("  ✓ MTJConfig creation and validation")
        
        # Test crossbar configuration  
        from spintron_nn.core.crossbar import CrossbarConfig
        crossbar_config = CrossbarConfig(rows=64, cols=64)
        assert crossbar_config.rows == 64
        assert crossbar_config.cols == 64
        print("  ✓ CrossbarConfig creation")
        
        # Test performance configuration
        from spintron_nn.utils.performance import PerformanceConfig
        perf_config = PerformanceConfig()
        assert perf_config.cache_size_mb > 0
        print("  ✓ PerformanceConfig creation")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Basic functionality test failed: {e}")
        return False


def test_file_structure():
    """Validate file structure and organization."""
    print("\n=== File Structure Validation ===")
    
    required_files = [
        'spintron_nn/__init__.py',
        'spintron_nn/core/__init__.py',
        'spintron_nn/core/mtj_models.py',
        'spintron_nn/core/crossbar.py',
        'spintron_nn/converter/__init__.py',
        'spintron_nn/converter/pytorch_parser.py',
        'spintron_nn/hardware/__init__.py',
        'spintron_nn/hardware/verilog_gen.py',
        'spintron_nn/training/__init__.py',
        'spintron_nn/simulation/__init__.py',
        'spintron_nn/models/__init__.py',
        'spintron_nn/cli/__init__.py',
        'spintron_nn/utils/__init__.py',
        'README.md',
        'pyproject.toml',
        'package.json'
    ]
    
    missing_files = []
    present_files = []
    
    for file_path in required_files:
        if os.path.exists(file_path):
            present_files.append(file_path)
            print(f"  ✓ {file_path}")
        else:
            missing_files.append(file_path)
            print(f"  ✗ {file_path} (missing)")
    
    print(f"\nFile structure: {len(present_files)} present, {len(missing_files)} missing")
    return len(missing_files) == 0


def test_code_quality():
    """Test code quality metrics."""
    print("\n=== Code Quality Tests ===")
    
    # Count Python files
    python_files = []
    total_lines = 0
    
    for root, dirs, files in os.walk('spintron_nn'):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                python_files.append(file_path)
                
                with open(file_path, 'r') as f:
                    lines = len(f.readlines())
                    total_lines += lines
    
    # Check for basic documentation
    documented_files = 0
    for file_path in python_files:
        with open(file_path, 'r') as f:
            content = f.read()
            if '"""' in content or "'''" in content:
                documented_files += 1
    
    documentation_ratio = documented_files / len(python_files) if python_files else 0
    
    print(f"  Python files: {len(python_files)}")
    print(f"  Total lines of code: {total_lines}")
    print(f"  Documented files: {documented_files}/{len(python_files)} ({documentation_ratio:.1%})")
    print(f"  Average lines per file: {total_lines/len(python_files):.0f}")
    
    # Quality metrics
    quality_metrics = {
        'total_files': len(python_files),
        'total_lines': total_lines,
        'documentation_ratio': documentation_ratio,
        'avg_lines_per_file': total_lines / len(python_files) if python_files else 0
    }
    
    # Basic quality thresholds
    quality_passed = (
        quality_metrics['total_files'] >= 20 and  # At least 20 Python files
        quality_metrics['documentation_ratio'] >= 0.8 and  # 80% documented
        quality_metrics['avg_lines_per_file'] >= 50  # Meaningful file sizes
    )
    
    print(f"  Quality metrics: {'✓ PASS' if quality_passed else '✗ FAIL'}")
    
    return quality_metrics, quality_passed


def performance_smoke_test():
    """Basic performance smoke test."""
    print("\n=== Performance Smoke Test ===")
    
    try:
        from spintron_nn.core.mtj_models import MTJConfig, MTJDevice
        
        # Test MTJ device creation performance
        start_time = time.time()
        devices = []
        for _ in range(100):
            config = MTJConfig()
            device = MTJDevice(config)
            devices.append(device)
        creation_time = time.time() - start_time
        
        # Test resistance calculation performance
        start_time = time.time()
        for device in devices:
            resistance = device.resistance
        calc_time = time.time() - start_time
        
        print(f"  Device creation: {creation_time*1000:.2f} ms (100 devices)")
        print(f"  Resistance calculation: {calc_time*1000:.2f} ms (100 calls)")
        print(f"  Creation rate: {100/creation_time:.0f} devices/sec")
        print(f"  Calculation rate: {100/calc_time:.0f} calls/sec")
        
        # Basic performance thresholds
        performance_ok = (
            creation_time < 1.0 and  # Less than 1 second for 100 devices
            calc_time < 0.1  # Less than 100ms for 100 calculations
        )
        
        print(f"  Performance: {'✓ PASS' if performance_ok else '✗ FAIL'}")
        
        return {
            'device_creation_time_ms': creation_time * 1000,
            'resistance_calc_time_ms': calc_time * 1000,
            'performance_ok': performance_ok
        }
        
    except Exception as e:
        print(f"  ✗ Performance test failed: {e}")
        return {'performance_ok': False, 'error': str(e)}


def main():
    """Run complete validation suite."""
    print("SpinTron-NN-Kit Basic Validation Suite")
    print("=" * 50)
    
    start_time = time.time()
    
    # Run all tests
    test_results = {}
    
    test_results['imports'] = test_imports()
    test_results['functionality'] = test_basic_functionality()
    test_results['file_structure'] = test_file_structure()
    
    quality_metrics, quality_passed = test_code_quality()
    test_results['code_quality'] = quality_passed
    
    perf_results = performance_smoke_test()
    test_results['performance'] = perf_results.get('performance_ok', False)
    
    total_time = time.time() - start_time
    
    # Overall results
    all_passed = all(test_results.values())
    
    print(f"\n=== Validation Summary ===")
    print(f"Total validation time: {total_time:.2f} seconds")
    print(f"Test results:")
    for test_name, passed in test_results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} {test_name}")
    
    print(f"\nOverall result: {'✓ ALL TESTS PASSED' if all_passed else '✗ SOME TESTS FAILED'}")
    
    # Save results
    results = {
        'timestamp': time.time(),
        'total_time_seconds': total_time,
        'test_results': test_results,
        'all_tests_passed': all_passed,
        'quality_metrics': quality_metrics,
        'performance_results': perf_results
    }
    
    output_file = Path(__file__).parent / 'validation_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    
    # Exit code
    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())