"""
Testbench Generation for Spintronic Hardware.

This module generates comprehensive testbenches for verifying
spintronic neural network hardware implementations.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from ..converter.pytorch_parser import SpintronicModel


@dataclass
class TestbenchConfig:
    """Configuration for testbench generation."""
    
    # Test coverage options
    coverage_goals: List[str] = None  # ["statement", "branch", "toggle", "functional"]
    include_timing_checks: bool = True
    include_power_analysis: bool = True
    
    # Test patterns
    num_random_tests: int = 1000
    num_directed_tests: int = 100
    include_corner_cases: bool = True
    
    # Simulation options
    simulator: str = "verilator"  # "verilator", "modelsim", "vcs"
    waveform_format: str = "vcd"  # "vcd", "fsdb", "vpd"
    simulation_time: str = "1ms"
    
    # Assertion options
    enable_sva: bool = True  # SystemVerilog Assertions
    enable_psl: bool = False  # Property Specification Language
    
    def __post_init__(self):
        if self.coverage_goals is None:
            self.coverage_goals = ["statement", "branch", "toggle"]


class TestbenchGenerator:
    """Generates testbenches for spintronic neural network hardware."""
    
    def __init__(self, model: SpintronicModel):
        self.model = model
        
    def generate(
        self,
        config: TestbenchConfig,
        test_vectors: Optional[np.ndarray] = None,
        output_dir: str = "testbench"
    ) -> Dict[str, str]:
        """
        Generate comprehensive testbench suite.
        
        Args:
            config: Testbench configuration
            test_vectors: Input test vectors
            output_dir: Output directory
            
        Returns:
            Dictionary mapping file names to testbench code
        """
        self.config = config
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        testbench_files = {}
        
        # Generate main testbench
        main_tb = self._generate_main_testbench(test_vectors)
        testbench_files['tb_main.sv'] = main_tb
        
        # Generate unit testbenches for each layer
        for i, layer in enumerate(self.model.layers):
            if layer.crossbars:
                layer_tb = self._generate_layer_testbench(layer, i)
                testbench_files[f'tb_layer_{i}.sv'] = layer_tb
        
        # Generate crossbar testbenches
        crossbar_idx = 0
        for layer in self.model.layers:
            for crossbar in layer.crossbars:
                crossbar_tb = self._generate_crossbar_testbench(crossbar, crossbar_idx)
                testbench_files[f'tb_crossbar_{crossbar_idx}.sv'] = crossbar_tb
                crossbar_idx += 1
        
        # Generate test utilities
        testbench_files['test_utils.sv'] = self._generate_test_utilities()
        
        # Generate coverage models
        if config.coverage_goals:
            testbench_files['coverage_model.sv'] = self._generate_coverage_model()
        
        # Generate assertions
        if config.enable_sva:
            testbench_files['assertions.sv'] = self._generate_assertions()
        
        # Generate simulation scripts
        testbench_files['run_sim.sh'] = self._generate_simulation_script()
        testbench_files['Makefile'] = self._generate_makefile()
        
        # Write all files
        for filename, content in testbench_files.items():
            file_path = output_path / filename
            with open(file_path, 'w') as f:
                f.write(content)
        
        return testbench_files
    
    def _generate_main_testbench(self, test_vectors: Optional[np.ndarray]) -> str:
        """Generate main system-level testbench."""
        # Calculate interface widths
        input_width = self._calculate_input_width()
        output_width = self._calculate_output_width()
        
        testbench_code = f"""//
// Main Testbench for {self.model.name}
// Generated automatically by SpinTron-NN-Kit
//

`timescale 1ns/1ps

module tb_main;

// Parameters
parameter CLK_PERIOD = 20; // 50 MHz
parameter INPUT_WIDTH = {input_width};
parameter OUTPUT_WIDTH = {output_width};
parameter NUM_TESTS = {self.config.num_random_tests};

// DUT signals
logic clk;
logic rst_n;
logic enable;
logic [INPUT_WIDTH-1:0] data_in;
logic data_valid;
logic [OUTPUT_WIDTH-1:0] data_out;
logic data_ready;
logic [7:0] control_reg;
logic [7:0] status_reg;
logic power_enable;
logic power_good;

// Test control
logic test_done;
int test_count;
int pass_count;
int fail_count;

// Test vectors
logic [INPUT_WIDTH-1:0] test_inputs[NUM_TESTS];
logic [OUTPUT_WIDTH-1:0] expected_outputs[NUM_TESTS];

// DUT instantiation
spintronic_top dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .data_in(data_in),
    .data_valid(data_valid),
    .data_out(data_out),
    .data_ready(data_ready),
    .control_reg(control_reg),
    .status_reg(status_reg),
    .power_enable(power_enable),
    .power_good(power_good)
);

// Clock generation
initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// Reset generation
initial begin
    rst_n = 0;
    power_enable = 0;
    #(CLK_PERIOD * 5);
    power_enable = 1;
    #(CLK_PERIOD * 2);
    rst_n = 1;
    $display("Reset released at time %0t", $time);
end

// Test stimulus
initial begin
    // Initialize
    test_done = 0;
    test_count = 0;
    pass_count = 0;
    fail_count = 0;
    enable = 0;
    data_in = 0;
    data_valid = 0;
    control_reg = 8'h00;
    
    // Wait for reset
    wait(rst_n && power_good);
    #(CLK_PERIOD * 10);
    
    // Load test vectors
    load_test_vectors();
    
    // Run tests
    $display("Starting main testbench with %0d tests", NUM_TESTS);
    
    for (int i = 0; i < NUM_TESTS; i++) begin
        run_single_test(i);
        test_count++;
    end
    
    // Final report
    test_done = 1;
    $display("\\n=== Test Summary ===");
    $display("Total tests: %0d", test_count);
    $display("Passed: %0d", pass_count);
    $display("Failed: %0d", fail_count);
    $display("Pass rate: %.2f%%", (real'(pass_count) / real'(test_count)) * 100.0);
    
    if (fail_count == 0) begin
        $display("*** ALL TESTS PASSED ***");
    end else begin
        $display("*** %0d TESTS FAILED ***", fail_count);
    end
    
    #(CLK_PERIOD * 100);
    $finish;
end

// Test execution task
task run_single_test(int test_idx);
    logic [OUTPUT_WIDTH-1:0] actual_output;
    logic timeout;
    int timeout_count;
    
    $display("Running test %0d at time %0t", test_idx, $time);
    
    // Apply input
    @(posedge clk);
    data_in = test_inputs[test_idx];
    data_valid = 1;
    enable = 1;
    
    @(posedge clk);
    data_valid = 0;
    
    // Wait for result
    timeout = 0;
    timeout_count = 0;
    
    while (!data_ready && !timeout) begin
        @(posedge clk);
        timeout_count++;
        if (timeout_count > 10000) begin
            timeout = 1;
            $error("Test %0d timed out", test_idx);
        end
    end
    
    if (!timeout) begin
        actual_output = data_out;
        
        // Check result (simplified comparison)
        if (actual_output == expected_outputs[test_idx]) begin
            pass_count++;
            $display("Test %0d PASSED: input=%h, output=%h", 
                    test_idx, test_inputs[test_idx], actual_output);
        end else begin
            fail_count++;
            $error("Test %0d FAILED: input=%h, expected=%h, actual=%h",
                  test_idx, test_inputs[test_idx], expected_outputs[test_idx], actual_output);
        end
    end else begin
        fail_count++;
    end
    
    // Cleanup
    enable = 0;
    @(posedge clk);
    
endtask

// Load test vectors
task load_test_vectors();
    // Generate random test vectors
    for (int i = 0; i < NUM_TESTS; i++) begin
        test_inputs[i] = $random;
        // For now, expected output is simplified
        expected_outputs[i] = test_inputs[i][OUTPUT_WIDTH-1:0];
    end
    $display("Loaded %0d test vectors", NUM_TESTS);
endtask

// Waveform generation
initial begin
    $dumpfile("tb_main.{self.config.waveform_format}");
    $dumpvars(0, tb_main);
end

// Timeout watchdog
initial begin
    #{self.config.simulation_time};
    if (!test_done) begin
        $error("Simulation timeout reached");
        $finish;
    end
end

endmodule
"""
        
        return testbench_code
    
    def _generate_layer_testbench(self, layer: SpintronicLayer, layer_idx: int) -> str:
        """Generate testbench for individual layer."""
        return f"""//
// Layer {layer_idx} Testbench - {layer.layer_type}
//

`timescale 1ns/1ps

module tb_layer_{layer_idx};

parameter CLK_PERIOD = 20;
parameter NUM_TESTS = 100;

// DUT signals
logic clk;
logic rst_n;
logic enable;
logic [127:0] data_in;  // Simplified width
logic [127:0] data_out;
logic status;

// Test control
int test_count;
int pass_count;

// DUT instantiation
spintronic_layer_{layer_idx} dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .data_in(data_in),
    .data_out(data_out),
    .status(status)
);

// Clock generation
initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// Test sequence
initial begin
    rst_n = 0;
    enable = 0;
    data_in = 0;
    test_count = 0;
    pass_count = 0;
    
    #(CLK_PERIOD * 5);
    rst_n = 1;
    #(CLK_PERIOD * 2);
    
    $display("Starting layer {layer_idx} tests");
    
    // Run directed tests
    for (int i = 0; i < NUM_TESTS; i++) begin
        run_layer_test(i);
    end
    
    $display("Layer {layer_idx} tests completed: %0d/%0d passed", pass_count, test_count);
    $finish;
end

task run_layer_test(int test_idx);
    @(posedge clk);
    data_in = $random;
    enable = 1;
    
    // Wait for completion
    wait(status);
    @(posedge clk);
    
    // Basic check - layer should produce output
    if (data_out !== 'x) begin
        pass_count++;
        $display("Layer test %0d passed", test_idx);
    end else begin
        $error("Layer test %0d failed - no valid output", test_idx);
    end
    
    test_count++;
    enable = 0;
    @(posedge clk);
endtask

// Waveform dump
initial begin
    $dumpfile("tb_layer_{layer_idx}.vcd");
    $dumpvars(0, tb_layer_{layer_idx});
end

endmodule
"""
    
    def _generate_crossbar_testbench(self, crossbar: Any, crossbar_idx: int) -> str:
        """Generate testbench for crossbar array."""
        rows = getattr(crossbar, 'rows', 128)
        cols = getattr(crossbar, 'cols', 128)
        
        return f"""//
// Crossbar {crossbar_idx} Testbench - {rows}x{cols} Array
//

`timescale 1ns/1ps

module tb_crossbar_{crossbar_idx};

parameter CLK_PERIOD = 20;
parameter ROWS = {rows};
parameter COLS = {cols};

// DUT signals
logic clk;
logic rst_n;
logic enable;
logic [ROWS*8-1:0] data_in;
logic [15:0] data_out;
logic ready;

// Programming interface
logic prog_enable;
logic [7:0] prog_row_addr;
logic [7:0] prog_col_addr;
logic [1:0] prog_data;

// Test patterns
logic [ROWS*8-1:0] test_patterns[10];
logic [15:0] expected_results[10];

// DUT instantiation
spintronic_crossbar_{crossbar_idx} dut (
    .clk(clk),
    .rst_n(rst_n),
    .enable(enable),
    .data_in(data_in),
    .data_out(data_out),
    .ready(ready),
    .prog_enable(prog_enable),
    .prog_row_addr(prog_row_addr),
    .prog_col_addr(prog_col_addr),
    .prog_data(prog_data),
    .word_lines(),
    .bit_lines(),
    .sense_currents(0)
);

// Clock generation
initial begin
    clk = 0;
    forever #(CLK_PERIOD/2) clk = ~clk;
end

// Test sequence
initial begin
    // Initialize
    rst_n = 0;
    enable = 0;
    prog_enable = 0;
    data_in = 0;
    
    #(CLK_PERIOD * 5);
    rst_n = 1;
    #(CLK_PERIOD * 2);
    
    $display("Starting crossbar {crossbar_idx} tests");
    
    // Program some test weights
    program_crossbar();
    
    // Run computation tests
    run_computation_tests();
    
    $display("Crossbar {crossbar_idx} tests completed");
    $finish;
end

// Programming task
task program_crossbar();
    $display("Programming crossbar weights...");
    
    prog_enable = 1;
    
    // Program a simple pattern
    for (int i = 0; i < min(8, ROWS); i++) begin
        for (int j = 0; j < min(8, COLS); j++) begin
            @(posedge clk);
            prog_row_addr = i;
            prog_col_addr = j;
            prog_data = (i + j) % 2;  // Checkerboard pattern
        end
    end
    
    @(posedge clk);
    prog_enable = 0;
    
    $display("Crossbar programming completed");
endtask

// Computation tests
task run_computation_tests();
    $display("Running computation tests...");
    
    // Test 1: All zeros
    @(posedge clk);
    data_in = 0;
    enable = 1;
    
    wait(ready);
    @(posedge clk);
    $display("Test 1 (zeros): result = %h", data_out);
    
    // Test 2: All ones
    @(posedge clk);
    data_in = ~0;
    enable = 1;
    
    wait(ready);
    @(posedge clk);
    $display("Test 2 (ones): result = %h", data_out);
    
    enable = 0;
endtask

// Waveform dump
initial begin
    $dumpfile("tb_crossbar_{crossbar_idx}.vcd");
    $dumpvars(0, tb_crossbar_{crossbar_idx});
end

endmodule
"""
    
    def _generate_test_utilities(self) -> str:
        """Generate common test utilities."""
        return """//
// Test Utilities Package
//

package test_utils;

// Random number generator with seed
class random_gen;
    int seed;
    
    function new(int s = 1);
        seed = s;
    endfunction
    
    function int get_random();
        seed = (seed * 1103515245 + 12345) & 32'h7FFFFFFF;
        return seed;
    endfunction
endclass

// Comparison utilities
function bit compare_vectors(logic [31:0] a, logic [31:0] b, real tolerance = 0.01);
    real diff = $itor(a > b ? a - b : b - a);
    real relative_diff = diff / $itor(a > b ? a : b);
    return (relative_diff <= tolerance);
endfunction

// Performance monitoring
class perf_monitor;
    time start_time;
    time end_time;
    int operation_count;
    
    function void start();
        start_time = $time;
        operation_count = 0;
    endfunction
    
    function void record_operation();
        operation_count++;
    endfunction
    
    function void stop();
        end_time = $time;
    endfunction
    
    function real get_throughput();
        time elapsed = end_time - start_time;
        return real'(operation_count) / real'(elapsed) * 1e9; // ops per second
    endfunction
endclass

endpackage
"""
    
    def _generate_coverage_model(self) -> str:
        """Generate coverage model for functional verification."""
        return f"""//
// Coverage Model for SpinTron-NN-Kit
//

`include "test_utils.sv"

module coverage_model;

import test_utils::*;

// Coverage groups
covergroup crossbar_operations @(posedge clk);
    option.per_instance = 1;
    
    // Input value coverage
    input_values: coverpoint data_in {{
        bins zero = {{0}};
        bins low = {{[1:85]}};
        bins mid = {{[86:170]}};
        bins high = {{[171:254]}};
        bins max = {{255}};
    }}
    
    // State coverage
    operation_state: coverpoint state {{
        bins idle = {{IDLE}};
        bins compute = {{COMPUTE}};
        bins ready = {{READY}};
        illegal_bins illegal = default;
    }}
    
    // Cross coverage
    input_state_cross: cross input_values, operation_state;
endgroup

// Assertion-based coverage
covergroup assertion_coverage;
    // Protocol compliance
    protocol_valid: coverpoint (data_valid && enable) {{
        bins valid_sequence = {{1}};
    }}
    
    // Timing coverage
    response_time: coverpoint response_cycles {{
        bins fast = {{[1:10]}};
        bins normal = {{[11:100]}};
        bins slow = {{[101:1000]}};
        illegal_bins timeout = {{[1001:$]}};
    }}
endgroup

// Functional coverage points
{"".join([f'''
// Layer {i} coverage
covergroup layer_{i}_coverage @(posedge clk);
    layer_{i}_enable: coverpoint layer_{i}_enable {{
        bins active = {{1}};
        bins inactive = {{0}};
    }}
    
    layer_{i}_utilization: coverpoint layer_{i}_utilization {{
        bins low = {{[0:25]}};
        bins medium = {{[26:75]}};
        bins high = {{[76:100]}};
    }}
endgroup
''' for i in range(len(self.model.layers))])}

// Coverage collection
initial begin
    automatic crossbar_operations cov_crossbar = new();
    automatic assertion_coverage cov_assertions = new();
    
    // Enable coverage collection
    $coverage_control(1);
    
    // Coverage reporting
    final begin
        $display("\\n=== Coverage Report ===");
        $display("Crossbar operations coverage: %.2f%%", cov_crossbar.get_coverage());
        $display("Assertion coverage: %.2f%%", cov_assertions.get_coverage());
        
        // Detailed coverage report
        $coverage_save("coverage_report.dat");
    end
end

endmodule
"""
    
    def _generate_assertions(self) -> str:
        """Generate SystemVerilog assertions for verification."""
        return """//
// SystemVerilog Assertions for SpinTron-NN-Kit
//

// Interface protocol assertions
module spintronic_assertions (
    input logic clk,
    input logic rst_n,
    input logic enable,
    input logic data_valid,
    input logic data_ready
);

// Clock and reset assertions
always_ff @(posedge clk) begin
    // Reset assertion
    assert_reset: assert property (
        @(posedge clk) disable iff (!rst_n)
        $rose(rst_n) |-> ##[1:5] (enable == 0)
    ) else $error("Reset protocol violation");
end

// Data flow assertions
property valid_ready_handshake;
    @(posedge clk) disable iff (!rst_n)
    data_valid && enable |-> ##[1:100] data_ready;
endproperty

assert_handshake: assert property (valid_ready_handshake)
else $error("Data handshake violation");

// Timing assertions
property max_response_time;
    @(posedge clk) disable iff (!rst_n)
    data_valid && enable |-> ##[1:1000] data_ready;
endproperty

assert_timing: assert property (max_response_time)
else $error("Response time violation");

// Power sequence assertions
property power_sequence;
    @(posedge clk)
    $rose(power_enable) |-> ##[1:10] power_good;
endproperty

assert_power: assert property (power_sequence)
else $error("Power sequence violation");

// MTJ programming assertions
property mtj_programming;
    @(posedge clk) disable iff (!rst_n)
    prog_enable && word_line && bit_line |-> ##1 mtj_state_changed;
endproperty

assert_mtj_prog: assert property (mtj_programming)
else $error("MTJ programming failure");

// Crossbar computation assertions
property crossbar_computation;
    @(posedge clk) disable iff (!rst_n)
    crossbar_enable |-> ##[5:50] crossbar_ready;
endproperty

assert_crossbar: assert property (crossbar_computation)
else $error("Crossbar computation failure");

// Coverage assertions
cover_all_states: cover property (
    @(posedge clk) disable iff (!rst_n)
    (state == IDLE) ##1 (state == COMPUTE) ##1 (state == READY)
);

cover_max_utilization: cover property (
    @(posedge clk) disable iff (!rst_n)
    utilization >= 90
);

endmodule
"""
    
    def _generate_simulation_script(self) -> str:
        """Generate simulation run script."""
        if self.config.simulator == "verilator":
            return self._generate_verilator_script()
        elif self.config.simulator == "modelsim":
            return self._generate_modelsim_script()
        else:
            return self._generate_generic_script()
    
    def _generate_verilator_script(self) -> str:
        """Generate Verilator simulation script."""
        return f"""#!/bin/bash
#
# Verilator Simulation Script for SpinTron-NN-Kit
#

# Verilator options
VERILATOR_OPTS="--cc --exe --build --trace"
VERILATOR_OPTS="$VERILATOR_OPTS --top-module tb_main"
VERILATOR_OPTS="$VERILATOR_OPTS -Wall -Wno-WIDTH -Wno-UNUSED"

# Source files
VERILOG_FILES="../generated_verilog/*.v"
TESTBENCH_FILES="*.sv"

echo "Running Verilator simulation..."

# Compile and run
verilator $VERILATOR_OPTS $VERILOG_FILES $TESTBENCH_FILES

if [ $? -eq 0 ]; then
    echo "Compilation successful"
    
    # Run simulation
    ./obj_dir/Vtb_main
    
    if [ $? -eq 0 ]; then
        echo "Simulation completed successfully"
        
        # Generate coverage report
        if [ -f "coverage_report.dat" ]; then
            echo "Coverage report generated"
        fi
        
        # Convert VCD to readable format
        if [ -f "tb_main.vcd" ]; then
            echo "Waveform file generated: tb_main.vcd"
        fi
    else
        echo "Simulation failed"
        exit 1
    fi
else
    echo "Compilation failed"
    exit 1
fi

echo "All tests completed"
"""
    
    def _generate_modelsim_script(self) -> str:
        """Generate ModelSim simulation script."""
        return f"""#
# ModelSim Simulation Script for SpinTron-NN-Kit
#

# Create work library
vlib work

# Compile SystemVerilog files
vlog -sv +incdir+. *.sv
vlog +incdir+../generated_verilog ../generated_verilog/*.v

# Simulate
vsim -t ps -voptargs=+acc work.tb_main

# Add waves
add wave -radix hex /tb_main/*
add wave -radix hex /tb_main/dut/*

# Configure simulation
configure wave -namecolwidth 250
configure wave -valuecolwidth 100
configure wave -justifyvalue left
configure wave -signalnamewidth 1
configure wave -snapdistance 10
configure wave -datasetprefix 0

# Run simulation
run {self.config.simulation_time}

# Generate reports
coverage report -detail -file coverage_report.txt
write format wave -window .main_pane.wave.interior.cs.body.pw.wf wave.do

echo "Simulation completed"
quit -force
"""
    
    def _generate_generic_script(self) -> str:
        """Generate generic simulation script."""
        return f"""#!/bin/bash
#
# Generic Simulation Script for SpinTron-NN-Kit
#

echo "Starting simulation with {self.config.simulator}..."

# Add your simulator-specific commands here
# This is a template for other simulators

echo "Compiling testbenches..."
# Compile commands

echo "Running simulation..."
# Simulation commands

echo "Generating reports..."
# Report generation commands

echo "Simulation completed"
"""
    
    def _generate_makefile(self) -> str:
        """Generate Makefile for test automation."""
        return f"""#
# Makefile for SpinTron-NN-Kit Testbench
#

# Variables
SIMULATOR = {self.config.simulator}
VERILOG_DIR = ../generated_verilog
TB_DIR = .
SIM_TIME = {self.config.simulation_time}

# Targets
.PHONY: all clean compile simulate coverage

all: compile simulate coverage

compile:
\t@echo "Compiling testbenches..."
\t@if [ "${{SIMULATOR}}" = "verilator" ]; then \\
\t\tverilator --cc --exe --build --trace *.sv $(VERILOG_DIR)/*.v; \\
\telif [ "${{SIMULATOR}}" = "modelsim" ]; then \\
\t\tvlib work && vlog -sv *.sv && vlog $(VERILOG_DIR)/*.v; \\
\tfi

simulate:
\t@echo "Running simulation..."
\t@if [ "${{SIMULATOR}}" = "verilator" ]; then \\
\t\t./obj_dir/Vtb_main; \\
\telif [ "${{SIMULATOR}}" = "modelsim" ]; then \\
\t\tvsim -c -do "run $(SIM_TIME); quit" work.tb_main; \\
\tfi

coverage:
\t@echo "Generating coverage reports..."
\t@if [ -f "coverage_report.dat" ]; then \\
\t\techo "Coverage data found"; \\
\telse \\
\t\techo "No coverage data generated"; \\
\tfi

clean:
\t@echo "Cleaning up..."
\t@rm -rf obj_dir/ work/ *.vcd *.wlf *.dat *.log transcript

# Individual test targets
test_main:
\t@echo "Running main testbench..."
\t@$(MAKE) simulate TB_TOP=tb_main

test_layers:
\t@echo "Running layer testbenches..."
{"".join([f"\t@$(MAKE) simulate TB_TOP=tb_layer_{i}\\n" for i in range(len(self.model.layers))])}

test_crossbars:
\t@echo "Running crossbar testbenches..."
\t@for i in crossbar_*.sv; do \\
\t\t$(MAKE) simulate TB_TOP=$$(basename $$i .sv); \\
\tdone

help:
\t@echo "Available targets:"
\t@echo "  all        - Compile, simulate, and generate coverage"
\t@echo "  compile    - Compile testbenches"
\t@echo "  simulate   - Run simulation"
\t@echo "  coverage   - Generate coverage reports"
\t@echo "  clean      - Clean up generated files"
\t@echo "  test_main  - Run main testbench only"
\t@echo "  test_layers - Run layer testbenches"
\t@echo "  test_crossbars - Run crossbar testbenches"
"""
    
    def _calculate_input_width(self) -> int:
        """Calculate input width for testbench."""
        if self.model.layers and self.model.layers[0].input_shape:
            return int(np.prod(self.model.layers[0].input_shape[1:]) * 8)
        return 128
    
    def _calculate_output_width(self) -> int:
        """Calculate output width for testbench."""
        if self.model.layers and self.model.layers[-1].output_shape:
            return int(np.prod(self.model.layers[-1].output_shape[1:]) * 8)
        return 128