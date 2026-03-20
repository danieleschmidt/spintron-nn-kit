"""
spintron-nn-kit: Ultra-low-power neural inference via spin-orbit-torque (SOT) hardware.

Provides physics-accurate models of MTJ/SOT devices, crossbar array VMM,
Verilog generation for synthesis, and energy estimation tools.
"""

from .sot_device import SOTDevice
from .crossbar import CrossbarArray
from .verilog_gen import VerilogGenerator
from .energy import EnergyEstimator

__version__ = "1.0.0"
__all__ = ["SOTDevice", "CrossbarArray", "VerilogGenerator", "EnergyEstimator"]
