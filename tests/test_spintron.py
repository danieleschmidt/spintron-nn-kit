"""
Tests for spintron-nn-kit.

Covers SOTDevice, CrossbarArray, VerilogGenerator, and EnergyEstimator.
"""

import math
import os
import tempfile

import numpy as np
import pytest

from spintron import CrossbarArray, EnergyEstimator, SOTDevice, VerilogGenerator


# ==========================================================================
# SOTDevice tests
# ==========================================================================


class TestSOTDevice:

    def test_sot_device_initial_state(self):
        """Device initialises in parallel state by default."""
        dev = SOTDevice()
        assert dev.state == "parallel"
        assert dev.is_parallel is True

    def test_sot_device_initial_state_antiparallel(self):
        """Device can be initialised in antiparallel state."""
        dev = SOTDevice(state="antiparallel")
        assert dev.state == "antiparallel"
        assert dev.is_parallel is False

    def test_sot_device_invalid_state(self):
        """Invalid state string raises ValueError."""
        with pytest.raises(ValueError):
            SOTDevice(state="invalid")

    def test_sot_device_conductance_parallel(self):
        """Parallel state → 1/1000 S = 1 mS."""
        dev = SOTDevice(state="parallel")
        expected = 1.0 / 1_000  # 1 mS
        assert math.isclose(dev.conductance(), expected, rel_tol=1e-9)

    def test_sot_device_conductance_antiparallel(self):
        """Antiparallel state → 1/2000 S = 0.5 mS."""
        dev = SOTDevice(state="antiparallel")
        expected = 1.0 / 2_000  # 0.5 mS
        assert math.isclose(dev.conductance(), expected, rel_tol=1e-9)

    def test_sot_device_switching_probability_high_current(self):
        """Current above Ic0 → probability ≥ 1 (clipped to 1.0)."""
        dev = SOTDevice()
        # 200 µA >> Ic0=100 µA → should deterministically switch
        p = dev.switching_probability(current_ua=200.0, pulse_ns=1.0)
        assert p == pytest.approx(1.0)

    def test_sot_device_switching_probability_low_current(self):
        """Very low current → very low switching probability."""
        dev = SOTDevice()
        # 1 µA << Ic0=100 µA → extremely low probability
        p = dev.switching_probability(current_ua=1.0, pulse_ns=1.0)
        assert p < 1e-10

    def test_sot_device_switching_probability_zero_current(self):
        """Zero current → probability = 0."""
        dev = SOTDevice()
        p = dev.switching_probability(current_ua=0.0)
        assert p == 0.0

    def test_sot_device_switching_probability_range(self):
        """Switching probability is always in [0, 1]."""
        dev = SOTDevice()
        for i_ua in [0, 10, 50, 100, 150, 300]:
            p = dev.switching_probability(current_ua=float(i_ua), pulse_ns=1.0)
            assert 0.0 <= p <= 1.0, f"Probability out of range for I={i_ua} µA: {p}"

    def test_sot_device_set_state(self):
        """set_state() correctly changes device state."""
        dev = SOTDevice(state="parallel")
        dev.set_state("antiparallel")
        assert dev.state == "antiparallel"
        dev.set_state("parallel")
        assert dev.state == "parallel"


# ==========================================================================
# CrossbarArray tests
# ==========================================================================


class TestCrossbarArray:

    def test_crossbar_creation(self):
        """CrossbarArray initialises with correct dimensions."""
        cb = CrossbarArray(rows=4, cols=4)
        assert cb.rows == 4
        assert cb.cols == 4

    def test_crossbar_creation_non_square(self):
        """CrossbarArray supports non-square shapes."""
        cb = CrossbarArray(rows=3, cols=5)
        assert cb.rows == 3
        assert cb.cols == 5

    def test_crossbar_invalid_dims(self):
        """Zero or negative dimensions raise ValueError."""
        with pytest.raises(ValueError):
            CrossbarArray(rows=0, cols=4)
        with pytest.raises(ValueError):
            CrossbarArray(rows=4, cols=-1)

    def test_crossbar_program_weights(self):
        """Positive weights → parallel; negative → antiparallel."""
        cb = CrossbarArray(rows=2, cols=2)
        wm = np.array([[1.0, -1.0], [-0.5, 0.5]])
        cb.program_weights(wm)

        assert cb.device(0, 0).state == "parallel"       # positive
        assert cb.device(0, 1).state == "antiparallel"   # negative
        assert cb.device(1, 0).state == "antiparallel"   # negative
        assert cb.device(1, 1).state == "parallel"       # positive

    def test_crossbar_program_weights_shape_mismatch(self):
        """Weight matrix of wrong shape raises ValueError."""
        cb = CrossbarArray(rows=2, cols=2)
        with pytest.raises(ValueError):
            cb.program_weights(np.ones((3, 3)))

    def test_crossbar_vmm_identity(self):
        """All-parallel crossbar with identity-ish input returns expected."""
        cb = CrossbarArray(rows=2, cols=2)
        # All devices in parallel → G = 1/1000 S for all cells
        cb.program_weights(np.ones((2, 2)))
        v = np.array([1.0, 0.0])
        result = cb.vmm(v)
        # Only row 0 contributes: result[j] = 1.0 * G_parallel for each col
        G_p = 1.0 / 1_000
        expected = np.array([G_p, G_p])
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_crossbar_vmm_correctness(self):
        """VMM output matches manual conductance calculation."""
        cb = CrossbarArray(rows=3, cols=2)
        wm = np.array([[1, -1], [-1, 1], [1, 1]], dtype=float)
        cb.program_weights(wm)
        v = np.array([1.0, 1.0, 1.0])

        G = cb.conductance_matrix()
        expected = v @ G
        result = cb.vmm(v)
        np.testing.assert_allclose(result, expected, rtol=1e-9)

    def test_crossbar_vmm_input_length_mismatch(self):
        """Input vector of wrong length raises ValueError."""
        cb = CrossbarArray(rows=4, cols=4)
        with pytest.raises(ValueError):
            cb.vmm(np.ones(3))

    def test_crossbar_energy_pj_positive(self):
        """Energy estimate is non-negative for any input."""
        cb = CrossbarArray(rows=4, cols=4)
        cb.program_weights(np.random.randn(4, 4))
        energy = cb.energy_pj(np.ones(4))
        assert energy >= 0.0


# ==========================================================================
# VerilogGenerator tests
# ==========================================================================


class TestVerilogGenerator:

    @pytest.fixture
    def small_crossbar(self):
        cb = CrossbarArray(rows=4, cols=4)
        wm = np.array([
            [ 1, -1,  1, -1],
            [-1,  1, -1,  1],
            [ 1,  1, -1, -1],
            [-1, -1,  1,  1],
        ], dtype=float)
        cb.program_weights(wm)
        return cb

    def test_verilog_generator_module_header(self, small_crossbar):
        """Generated Verilog starts with a module declaration."""
        vgen = VerilogGenerator(small_crossbar)
        verilog = vgen.generate_module("sot_test")
        assert "module sot_test" in verilog

    def test_verilog_generator_has_ports(self, small_crossbar):
        """Generated Verilog includes clk, rst_n, in_vec, and out_vec ports."""
        vgen = VerilogGenerator(small_crossbar)
        verilog = vgen.generate_module()
        assert "clk" in verilog
        assert "rst_n" in verilog
        assert "in_vec" in verilog
        assert "out_vec" in verilog

    def test_verilog_generator_endmodule(self, small_crossbar):
        """Generated Verilog ends with endmodule."""
        vgen = VerilogGenerator(small_crossbar)
        verilog = vgen.generate_module()
        assert "endmodule" in verilog

    def test_verilog_generator_weight_params(self, small_crossbar):
        """Generated Verilog contains weight initialisation."""
        vgen = VerilogGenerator(small_crossbar)
        verilog = vgen.generate_module()
        assert "weight" in verilog

    def test_verilog_generator_save(self, small_crossbar):
        """save() writes a file to the given path."""
        vgen = VerilogGenerator(small_crossbar)
        with tempfile.NamedTemporaryFile(suffix=".v", delete=False) as f:
            tmp_path = f.name
        try:
            vgen.save(tmp_path)
            assert os.path.exists(tmp_path)
            with open(tmp_path) as f:
                content = f.read()
            assert "module" in content
        finally:
            os.unlink(tmp_path)

    def test_verilog_generator_rows_cols_params(self, small_crossbar):
        """Generated Verilog contains ROWS and COLS parameters."""
        vgen = VerilogGenerator(small_crossbar)
        verilog = vgen.generate_module()
        assert "ROWS" in verilog
        assert "COLS" in verilog


# ==========================================================================
# EnergyEstimator tests
# ==========================================================================


class TestEnergyEstimator:

    def test_energy_estimator_mac(self):
        """MAC energy = n_macs * READ_ENERGY_PJ."""
        est = EnergyEstimator()
        energy = est.mac_energy_pj(100)
        assert energy == pytest.approx(100 * EnergyEstimator.READ_ENERGY_PJ)

    def test_energy_estimator_mac_zero(self):
        """Zero MACs → zero energy."""
        est = EnergyEstimator()
        assert est.mac_energy_pj(0) == 0.0

    def test_energy_estimator_mac_negative(self):
        """Negative MACs raises ValueError."""
        est = EnergyEstimator()
        with pytest.raises(ValueError):
            est.mac_energy_pj(-1)

    def test_energy_estimator_layer(self):
        """Layer energy = rows*cols * READ_ENERGY_PJ."""
        est = EnergyEstimator()
        wm = np.ones((4, 4))
        energy = est.layer_energy_pj(wm)
        assert energy == pytest.approx(16 * EnergyEstimator.READ_ENERGY_PJ)

    def test_energy_estimator_layer_1d_raises(self):
        """1-D weight array raises ValueError."""
        est = EnergyEstimator()
        with pytest.raises(ValueError):
            est.layer_energy_pj(np.ones(4))

    def test_energy_estimator_programming(self):
        """Programming energy counts only negative-weight switches."""
        est = EnergyEstimator()
        wm = np.array([[1, -1], [-1, 1]], dtype=float)
        prog_energy = est.programming_energy_pj(wm)
        # 2 negative weights → 2 switches
        assert prog_energy == pytest.approx(2 * EnergyEstimator.MTJ_SWITCH_ENERGY_PJ)

    def test_energy_estimator_total(self):
        """Total energy = programming + n_inferences * read energy."""
        est = EnergyEstimator()
        wm = np.array([[1, -1], [-1, 1]], dtype=float)
        n = 10
        total = est.total_energy_pj(wm, n_inferences=n)
        expected = est.programming_energy_pj(wm) + est.layer_energy_pj(wm) * n
        assert total == pytest.approx(expected)

    def test_energy_estimator_constants(self):
        """Default physical constants are correct."""
        est = EnergyEstimator()
        assert est.MTJ_SWITCH_ENERGY_PJ == 10.0
        assert est.READ_ENERGY_PJ == 0.1
