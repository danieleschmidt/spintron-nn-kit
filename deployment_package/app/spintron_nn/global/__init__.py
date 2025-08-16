"""
Global-first implementation package for SpinTron-NN-Kit.

This package provides:
- Multi-region deployment capabilities
- Internationalization (i18n) support
- Compliance frameworks (GDPR, CCPA, PDPA)
- Cross-platform compatibility
- Regulatory audit trails
- Global performance optimization
"""

from .i18n import (
    InternationalizationManager,
    LocalizationConfig,
    TranslationEngine,
    get_localized_message
)

from .compliance import (
    ComplianceManager,
    GDPRCompliance,
    CCPACompliance,
    PDPACompliance,
    DataGovernance,
    AuditLogger
)

from .deployment import (
    GlobalDeploymentManager,
    RegionConfig,
    CrossPlatformSupport,
    PerformanceOptimizer
)

__all__ = [
    "InternationalizationManager",
    "LocalizationConfig", 
    "TranslationEngine",
    "get_localized_message",
    "ComplianceManager",
    "GDPRCompliance",
    "CCPACompliance", 
    "PDPACompliance",
    "DataGovernance",
    "AuditLogger",
    "GlobalDeploymentManager",
    "RegionConfig",
    "CrossPlatformSupport",
    "PerformanceOptimizer"
]