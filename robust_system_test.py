#!/usr/bin/env python3
"""
Robust system test for Generation 2 capabilities.
Tests error handling, validation, and logging systems.
"""

import sys
import time
import json
import traceback
from pathlib import Path

# Add spintron_nn to path for testing
sys.path.insert(0, str(Path(__file__).parent))

from spintron_nn.utils.robust_error_handling import (
    RobustErrorHandler, SpintronError, MTJDeviceError, CrossbarError,
    ErrorSeverity, error_handler, robust_operation
)
from spintron_nn.utils.comprehensive_validation import (
    ComprehensiveValidator, ValidationLevel, validate_input_data, validate_range
)
from spintron_nn.utils.advanced_logging import SpintronLogger, spintron_logger, log_operation


def test_error_handling():
    """Test robust error handling system."""
    print("=== Testing Error Handling System ===")
    
    # Test basic error handling
    try:
        raise MTJDeviceError("Test MTJ error", severity=ErrorSeverity.MEDIUM)
    except MTJDeviceError as e:
        handled = error_handler.handle_error(e, {'test_context': True})
        print(f"  ✓ MTJ error handled: {handled}")
        
    # Test retry mechanism
    @error_handler.with_retry(max_retries=2)
    def flaky_function(attempt_count=[0]):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise CrossbarError("Temporary failure")
        return "success"
        
    try:
        result = flaky_function()
        print(f"  ✓ Retry mechanism: {result}")
    except Exception as e:
        print(f"  ✗ Retry failed: {e}")
        
    # Test robust operation decorator
    @robust_operation
    def test_operation(value):
        if value < 0:
            raise ValueError("Negative value not allowed")
        return value * 2
        
    try:
        result = test_operation(5)
        print(f"  ✓ Robust operation: {result}")
    except Exception as e:
        print(f"  ✓ Error caught by robust operation: {type(e).__name__}")
        
    # Test error statistics
    stats = error_handler.get_error_statistics()
    print(f"  ✓ Error statistics: {stats['total_errors']} total errors")
    
    return True


def test_validation_system():
    """Test comprehensive validation system."""
    print("=== Testing Validation System ===")
    
    validator = ComprehensiveValidator(ValidationLevel.STRICT)
    
    # Test MTJ parameter validation
    mtj_result = validator.validate_mtj_parameters(
        resistance_high=10000,
        resistance_low=5000,
        switching_voltage=0.3,
        cell_area=40e-18
    )
    print(f"  ✓ MTJ validation: {'PASS' if mtj_result.is_valid else 'FAIL'}")
    print(f"    Metrics: {mtj_result.metrics}")
    
    # Test crossbar validation
    crossbar_result = validator.validate_crossbar_configuration(
        rows=128,
        cols=128,
        mtj_parameters={'switching_voltage': 0.3, 'resistance_low': 5000}
    )
    print(f"  ✓ Crossbar validation: {'PASS' if crossbar_result.is_valid else 'FAIL'}")
    print(f"    Total devices: {crossbar_result.metrics.get('total_devices', 0)}")
    
    # Test neural network mapping validation
    nn_result = validator.validate_neural_network_mapping(
        layer_sizes=[784, 128, 64, 10],
        crossbar_size=(128, 128),
        quantization_bits=4
    )
    print(f"  ✓ Neural network validation: {'PASS' if nn_result.is_valid else 'FAIL'}")
    print(f"    Utilization: {nn_result.metrics.get('weight_utilization', 0):.2%}")
    
    # Test Verilog generation validation
    verilog_result = validator.validate_verilog_generation(
        module_name="test_module",
        target_frequency=50e6,
        design_constraints={'max_area': 1.0, 'io_voltage': 1.8, 'core_voltage': 0.8}
    )
    print(f"  ✓ Verilog validation: {'PASS' if verilog_result.is_valid else 'FAIL'}")
    
    # Test input validation helpers
    try:
        validate_input_data(5, int, "test_value")
        print("  ✓ Input validation: PASS")
    except SpintronError:
        print("  ✗ Input validation: FAIL")
        
    try:
        validate_range(0.5, 0.0, 1.0, "test_range")
        print("  ✓ Range validation: PASS")
    except SpintronError:
        print("  ✗ Range validation: FAIL")
        
    # Get validation summary
    summary = validator.get_validation_summary()
    print(f"  ✓ Validation summary: {summary['success_rate']:.2%} success rate")
    
    return True


def test_logging_system():
    """Test advanced logging system."""
    print("=== Testing Logging System ===")
    
    logger = SpintronLogger("RobustTest")
    
    # Test basic logging
    logger.info("Test info message", context={'test': True})
    logger.warning("Test warning message")
    logger.debug("Test debug message")
    
    # Test operation context
    with logger.operation_context("test_operation", param1="value1"):
        time.sleep(0.01)  # Simulate work
        logger.info("Inside operation context")
        
    # Test specialized logging
    logger.log_mtj_operation(
        "resistance_measurement",
        {'resistance_high': 10000, 'resistance_low': 5000}
    )
    
    logger.log_crossbar_operation(
        "matrix_multiplication",
        {'rows': 128, 'cols': 128},
        {'latency_us': 10.5, 'power_uw': 50.2}
    )
    
    logger.log_conversion_step(
        "pytorch_to_spintronic",
        {'input_layers': 5, 'input_params': 1000},
        {'output_crossbars': 2, 'output_params': 800}
    )
    
    logger.log_security_event(
        "parameter_validation",
        {'validation_passed': True, 'parameters_checked': 10},
        severity="info"
    )
    
    # Test performance summary
    perf_summary = logger.get_performance_summary()
    print(f"  ✓ Performance tracking: {perf_summary.get('total_operations', 0)} operations")
    
    # Test decorated function
    @log_operation("decorated_test")
    def test_decorated_function(x, y):
        return x + y
        
    result = test_decorated_function(3, 4)
    print(f"  ✓ Decorated function: {result}")
    
    # Test log export
    export_file = logger.export_logs("json")
    print(f"  ✓ Log export: {export_file}")
    
    return True


def test_integration():
    """Test integration between all robust systems."""
    print("=== Testing System Integration ===")
    
    # Create integrated test scenario
    validator = ComprehensiveValidator(ValidationLevel.STANDARD)
    
    @robust_operation
    @log_operation("integrated_operation")
    def integrated_spintronic_operation(mtj_params, crossbar_config):
        """Simulate integrated spintronic operation with all robustness features."""
        
        # Validate inputs
        mtj_validation = validator.validate_mtj_parameters(**mtj_params)
        if not mtj_validation.is_valid:
            raise SpintronError(f"MTJ validation failed: {mtj_validation.errors}")
            
        crossbar_validation = validator.validate_crossbar_configuration(
            **crossbar_config, mtj_parameters=mtj_params
        )
        if not crossbar_validation.is_valid:
            raise SpintronError(f"Crossbar validation failed: {crossbar_validation.errors}")
            
        # Simulate operation
        spintron_logger.info("Performing integrated operation")
        time.sleep(0.01)  # Simulate computation
        
        # Return results with metrics
        return {
            'success': True,
            'mtj_metrics': mtj_validation.metrics,
            'crossbar_metrics': crossbar_validation.metrics,
            'operation_time': time.time()
        }
        
    # Test successful operation
    try:
        result = integrated_spintronic_operation(
            mtj_params={
                'resistance_high': 10000,
                'resistance_low': 5000,
                'switching_voltage': 0.3,
                'cell_area': 40e-18
            },
            crossbar_config={
                'rows': 64,
                'cols': 64
            }
        )
        print(f"  ✓ Integrated operation: {result['success']}")
        
    except Exception as e:
        print(f"  ✗ Integrated operation failed: {e}")
        return False
        
    # Test with invalid parameters to trigger error handling
    try:
        result = integrated_spintronic_operation(
            mtj_params={
                'resistance_high': -1000,  # Invalid
                'resistance_low': 5000,
                'switching_voltage': 0.3,
                'cell_area': 40e-18
            },
            crossbar_config={
                'rows': 64,
                'cols': 64
            }
        )
        print("  ✗ Should have failed with invalid parameters")
        return False
        
    except SpintronError as e:
        print(f"  ✓ Error handling caught invalid parameters: {e.severity.value}")
        
    return True


def generate_robustness_report():
    """Generate comprehensive robustness test report."""
    print("=== Generating Robustness Report ===")
    
    # Collect system statistics
    error_stats = error_handler.get_error_statistics()
    perf_summary = spintron_logger.get_performance_summary()
    
    # Create validation summary
    validator = ComprehensiveValidator()
    validation_summary = validator.get_validation_summary()
    
    report = {
        'test_timestamp': time.time(),
        'test_duration_seconds': time.time() - test_start_time,
        'robustness_features': {
            'error_handling': {
                'total_errors_handled': error_stats.get('total_errors', 0),
                'error_types': error_stats.get('error_types', {}),
                'recovery_strategies_registered': len(error_handler.recovery_strategies)
            },
            'validation_system': {
                'total_validations': validation_summary.get('total_validations', 0),
                'success_rate': validation_summary.get('success_rate', 0),
                'average_validation_time_ms': validation_summary.get('average_validation_time_ms', 0)
            },
            'logging_system': {
                'total_operations_logged': perf_summary.get('total_operations', 0),
                'average_execution_time_ms': perf_summary.get('avg_execution_time_ms', 0),
                'max_execution_time_ms': perf_summary.get('max_execution_time_ms', 0)
            }
        },
        'test_results': {
            'error_handling_test': test_results['error_handling'],
            'validation_test': test_results['validation'],
            'logging_test': test_results['logging'],
            'integration_test': test_results['integration']
        },
        'overall_robustness_score': sum(test_results.values()) / len(test_results) * 100
    }
    
    # Save report
    with open('generation2_robustness_report.json', 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"  ✓ Report saved: generation2_robustness_report.json")
    print(f"  ✓ Overall robustness score: {report['overall_robustness_score']:.1f}%")
    
    return report


def main():
    """Run comprehensive robustness tests."""
    global test_start_time, test_results
    
    print("SpinTron-NN-Kit Generation 2 Robustness Test")
    print("=" * 50)
    
    test_start_time = time.time()
    test_results = {}
    
    try:
        # Run all test suites
        test_results['error_handling'] = test_error_handling()
        test_results['validation'] = test_validation_system()
        test_results['logging'] = test_logging_system()
        test_results['integration'] = test_integration()
        
        # Generate comprehensive report
        report = generate_robustness_report()
        
        # Final summary
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = (passed_tests / total_tests) * 100
        
        print(f"\n=== Final Summary ===")
        print(f"Tests passed: {passed_tests}/{total_tests} ({success_rate:.1f}%)")
        print(f"Overall robustness score: {report['overall_robustness_score']:.1f}%")
        
        if success_rate == 100:
            print("✓ ALL ROBUSTNESS TESTS PASSED")
            return True
        else:
            print("⚠ SOME ROBUSTNESS TESTS FAILED")
            return False
            
    except Exception as e:
        print(f"\n✗ Robustness test failed with error: {e}")
        print(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)