"""
Security Module for Spintronic Neural Networks.

This module provides comprehensive security frameworks including:
- Differential privacy
- Homomorphic encryption
- Secure aggregation
- Side-channel attack protection
- Access control and authentication
"""

from .secure_computing import (
    SecureCrossbar,
    SecurityConfig,
    SecurityLevel,
    SecureAggregation,
    SecurityAuditor,
    PrivacyBudget
)

__all__ = [
    'SecureCrossbar',
    'SecurityConfig',
    'SecurityLevel', 
    'SecureAggregation',
    'SecurityAuditor',
    'PrivacyBudget'
]

__version__ = "0.1.0"
