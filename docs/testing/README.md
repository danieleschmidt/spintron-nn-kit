# Testing Guide

## Overview

SpinTron-NN-Kit uses a comprehensive testing strategy to ensure reliability, performance, and correctness across all components from PyTorch models to synthesizable hardware.

## Test Structure

```
tests/
├── unit/                    # Unit tests for individual components
├── integration/             # Integration tests for component interactions
├── e2e/                     # End-to-end workflow tests
├── performance/             # Performance and benchmark tests
├── fixtures/                # Test data and sample models
├── conftest.py             # PyTest configuration and fixtures
└── README.md               # This file
```

## Test Categories

### Unit Tests
- **MTJ Device Models**: Test individual device physics and characteristics
- **Converter Components**: Test PyTorch to spintronic conversion logic
- **Hardware Generation**: Test Verilog generation components
- **Training Components**: Test quantization-aware training and optimization

### Integration Tests
- **Hardware Generation**: Test complete Verilog generation pipeline
- **Simulation Interface**: Test SPICE and behavioral simulation integration
- **Cross-platform Compatibility**: Test across different OS and hardware

### End-to-End Tests
- **Complete Workflows**: Test full PyTorch → Hardware workflows
- **Application Examples**: Test keyword spotting, vision, and other applications
- **Performance Validation**: Test energy, latency, and accuracy targets

### Performance Tests
- **Benchmarking**: Energy, latency, memory, and throughput benchmarks
- **Regression Testing**: Ensure performance doesn't degrade
- **Hardware Verification**: Validate against physical hardware when available

## Running Tests

### Quick Start
```bash
# Run all fast tests
make test-fast

# Run specific test categories
make test-unit
make test-integration
make test-e2e

# Run with coverage
make test-coverage
```

### Advanced Usage
```bash
# Run specific test files
pytest tests/unit/test_mtj_models.py -v

# Run tests with specific markers
pytest -m "not slow" -v                    # Skip slow tests
pytest -m "hardware" -v                    # Only hardware tests
pytest -m "integration or e2e" -v          # Multiple markers

# Run tests in parallel
pytest -n auto tests/                      # Auto-detect CPU cores
pytest -n 4 tests/                         # Use 4 processes

# Run with detailed output
pytest tests/ -v --tb=long --capture=no
```

## Test Markers

We use custom pytest markers to categorize tests:

- `@pytest.mark.slow`: Tests that take significant time (>5 seconds)
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.e2e`: End-to-end tests
- `@pytest.mark.hardware`: Tests requiring hardware simulation
- `@pytest.mark.gpu`: Tests requiring GPU (CUDA)
- `@pytest.mark.requires_tools`: Tests requiring external EDA tools

## Test Fixtures

### Common Fixtures (conftest.py)
- `sample_model`: Basic PyTorch model for testing
- `sample_input`: Compatible input tensor
- `mtj_config`: MTJ device configuration
- `crossbar_config`: Crossbar array configuration
- `verilog_constraints`: Hardware generation constraints
- `mock_spice_simulator`: Mock SPICE simulation
- `temp_dir`: Temporary directory for test files

### Model Fixtures (fixtures/sample_models.py)
- `KeywordSpottingNet`: Realistic keyword spotting model
- `TinyConvNet`: Small vision model
- `MicroNet`: Minimal test model
- Test data generators for different model types

## Writing New Tests

### Unit Test Example
```python
def test_mtj_resistance_calculation(mtj_config):
    """Test MTJ resistance calculation."""
    r_high = mtj_config['resistance_high']
    r_low = mtj_config['resistance_low']
    
    assert r_high > r_low
    tmr_ratio = (r_high - r_low) / r_low
    assert tmr_ratio > 0.5
```

### Integration Test Example
```python
@pytest.mark.integration
def test_conversion_pipeline(sample_model, mtj_config):
    """Test complete conversion pipeline."""
    converter = SpintronicConverter(mtj_config)
    spintronic_model = converter.convert(sample_model)
    
    assert spintronic_model is not None
    assert spintronic_model.crossbar_count > 0
```

### Hardware Test Example
```python
@pytest.mark.hardware
def test_verilog_synthesis(spintronic_model, temp_dir):
    """Test Verilog synthesis."""
    verilog_gen = VerilogGenerator()
    files = verilog_gen.generate(spintronic_model, temp_dir)
    
    for file_path in files:
        assert file_path.exists()
        assert file_path.stat().st_size > 0
```

## Performance Testing

### Energy Benchmarks
```python
def test_energy_efficiency():
    """Test energy efficiency targets."""
    energy_per_mac = measure_energy_per_mac()
    assert energy_per_mac < 10.0  # pJ target
```

### Latency Benchmarks
```python
@pytest.mark.slow
def test_inference_latency():
    """Test inference latency."""
    latency = benchmark_inference_latency()
    assert latency < 1.0  # ms target
```

## Continuous Integration

Tests are automatically run on:
- Every pull request
- Main branch commits
- Nightly builds (including slow tests)

### CI Test Matrix
- Python versions: 3.8, 3.9, 3.10, 3.11
- Operating systems: Ubuntu, macOS, Windows
- PyTorch versions: Latest stable, Previous stable
- Hardware configurations: CPU-only, GPU (when available)

## Hardware-in-the-Loop Testing

When physical hardware is available:

```python
@pytest.mark.requires_hardware
def test_fpga_validation():
    """Test against FPGA prototype."""
    # Test implementation against real hardware
    pass
```

## Test Data Management

### Large Test Data
- Use `tests/fixtures/` for small test data
- Large datasets stored in external repositories
- Use `pytest-benchmark` for performance regression tracking

### Mock Data Generation
```python
def generate_realistic_test_data():
    """Generate realistic test data."""
    # Use domain-specific distributions
    # Ensure reproducibility with fixed seeds
    return test_data
```

## Coverage Targets

- **Unit Tests**: >95% code coverage
- **Integration Tests**: >80% feature coverage  
- **E2E Tests**: 100% critical path coverage

## Test Performance Optimization

### Parallelization
```bash
# Run tests in parallel
pytest -n auto

# Distribute across multiple machines
pytest --dist=loadscope --tx=popen//python=python3.8
```

### Test Selection
```bash
# Run only changed tests
pytest --lf  # Last failed
pytest --ff  # Failed first

# Custom test selection
pytest -k "mtj and not slow"
```

## Debugging Tests

### Verbose Output
```bash
pytest -v --tb=long --capture=no
```

### Interactive Debugging
```bash
pytest --pdb              # Drop into debugger on failure
pytest --pdb-trace        # Drop into debugger at start
```

### Log Capture
```bash
pytest --log-cli-level=DEBUG --log-cli-format='%(asctime)s [%(levelname)8s] %(name)s: %(message)s'
```

## Best Practices

1. **Test Naming**: Use descriptive names that explain what is being tested
2. **Test Isolation**: Each test should be independent
3. **Deterministic Tests**: Use fixed seeds for reproducibility
4. **Fast Tests**: Keep unit tests fast (<1s each)
5. **Mock External Dependencies**: Use mocks for hardware, networks, etc.
6. **Clear Assertions**: Use descriptive assertion messages
7. **Test Documentation**: Document complex test scenarios

## Hardware Simulation Testing

### SPICE Integration
```python
@pytest.mark.requires_tools
def test_spice_accuracy():
    """Test SPICE simulation accuracy."""
    spice_sim = SPICESimulator("ngspice")
    results = spice_sim.simulate(circuit, test_vectors)
    verify_timing_accuracy(results)
```

### Behavioral Simulation
```python
def test_behavioral_accuracy():
    """Test behavioral simulation vs reference."""
    behavioral_output = run_behavioral_sim(model, inputs)
    reference_output = run_reference_model(inputs)
    assert_close(behavioral_output, reference_output, rtol=1e-3)
```

## Contributing Tests

When contributing new features:

1. Add unit tests for all new functions/classes
2. Add integration tests for new workflows
3. Update e2e tests for major features
4. Include performance tests for optimization features
5. Update this documentation for new test patterns

## Troubleshooting

### Common Issues
- **Import Errors**: Ensure `PYTHONPATH` includes project root
- **Fixture Not Found**: Check fixture scope and conftest.py location
- **Slow Tests**: Use `@pytest.mark.slow` and exclude with `-m "not slow"`
- **Hardware Tests**: Ensure required tools are installed and licensed

### Performance Issues
- Use `pytest-profiling` to identify slow tests
- Consider test parallelization for large test suites
- Mock expensive operations in unit tests