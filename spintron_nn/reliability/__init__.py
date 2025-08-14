"""
Reliability and Fault Tolerance Module for Spintronic Neural Networks.

This module provides comprehensive reliability frameworks including:
- Fault tolerance mechanisms
- Self-healing systems
- Reliability analysis
- Error detection and correction
"""

from .fault_tolerance import (
    FaultTolerantCrossbar,
    SelfHealingSystem,
    FaultType,
    RedundancyType,
    FaultModel,
    ReliabilityMetrics
)

__all__ = [
    'FaultTolerantCrossbar',
    'SelfHealingSystem', 
    'FaultType',
    'RedundancyType',
    'FaultModel',
    'ReliabilityMetrics'
]

__version__ = "0.1.0"
