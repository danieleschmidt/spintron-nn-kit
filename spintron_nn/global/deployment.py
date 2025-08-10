"""
Global deployment and multi-region support for SpinTron-NN-Kit.

This module provides:
- Multi-region deployment capabilities
- Cross-platform compatibility layers
- Regional performance optimization
- Regulatory compliance by region
- Global load balancing and failover
"""

import os
import platform
import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path


class DeploymentRegion(Enum):
    """Supported deployment regions with regulatory compliance."""
    US_EAST = "us-east-1"           # Virginia, USA (CCPA compliant)
    US_WEST = "us-west-2"           # Oregon, USA (CCPA compliant) 
    EU_WEST = "eu-west-1"           # Ireland, EU (GDPR compliant)
    EU_CENTRAL = "eu-central-1"     # Frankfurt, Germany (GDPR compliant)
    ASIA_PACIFIC = "ap-southeast-1" # Singapore (PDPA compliant)
    ASIA_NORTHEAST = "ap-northeast-1" # Tokyo, Japan
    CANADA_CENTRAL = "ca-central-1" # Canada (PIPEDA compliant)
    AUSTRALIA = "ap-southeast-2"    # Sydney, Australia


class Platform(Enum):
    """Supported platforms."""
    LINUX_X64 = "linux_x64"
    LINUX_ARM64 = "linux_arm64"
    WINDOWS_X64 = "windows_x64"
    MACOS_X64 = "macos_x64"
    MACOS_ARM64 = "macos_arm64"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"


@dataclass
class RegionConfig:
    """Configuration for a deployment region."""
    region: DeploymentRegion
    name: str
    timezone: str
    regulatory_framework: List[str]
    data_residency_required: bool
    latency_target_ms: float
    energy_efficiency_target: float  # pJ/MAC
    supported_platforms: List[Platform]
    compliance_contacts: Dict[str, str]
    
    def __post_init__(self):
        # Set default regulatory frameworks by region
        if not self.regulatory_framework:
            region_regulations = {
                DeploymentRegion.US_EAST: ["CCPA", "SOX", "HIPAA"],
                DeploymentRegion.US_WEST: ["CCPA", "SOX", "HIPAA"],
                DeploymentRegion.EU_WEST: ["GDPR", "NIS2"],
                DeploymentRegion.EU_CENTRAL: ["GDPR", "NIS2", "BDSG"],
                DeploymentRegion.ASIA_PACIFIC: ["PDPA", "CYBERSECURITY_ACT"],
                DeploymentRegion.ASIA_NORTHEAST: ["APPI", "CYBERSECURITY_BASIC_ACT"],
                DeploymentRegion.CANADA_CENTRAL: ["PIPEDA", "PRIVACY_ACT"],
                DeploymentRegion.AUSTRALIA: ["PRIVACY_ACT", "NOTIFIABLE_DATA_BREACHES"]
            }
            self.regulatory_framework = region_regulations.get(self.region, [])


class CrossPlatformSupport:
    """Cross-platform compatibility and optimization."""
    
    def __init__(self):
        self.current_platform = self._detect_platform()
        self.optimization_profiles = self._load_optimization_profiles()
        
    def _detect_platform(self) -> Platform:
        """Detect current platform."""
        system = platform.system().lower()
        machine = platform.machine().lower()
        
        # Check for containerized environments first
        if os.path.exists("/.dockerenv"):
            return Platform.DOCKER
        
        if os.getenv("KUBERNETES_SERVICE_HOST"):
            return Platform.KUBERNETES
        
        # Detect host platform
        if system == "linux":
            if "arm" in machine or "aarch64" in machine:
                return Platform.LINUX_ARM64
            else:
                return Platform.LINUX_X64
        elif system == "windows":
            return Platform.WINDOWS_X64
        elif system == "darwin":  # macOS
            if "arm" in machine or "m1" in machine.lower():
                return Platform.MACOS_ARM64
            else:
                return Platform.MACOS_X64
        
        # Default fallback
        return Platform.LINUX_X64
    
    def _load_optimization_profiles(self) -> Dict[Platform, Dict[str, Any]]:
        """Load platform-specific optimization profiles."""
        return {
            Platform.LINUX_X64: {
                "compiler_flags": ["-O3", "-march=native", "-mtune=native"],
                "parallel_workers": os.cpu_count(),
                "memory_optimization": "aggressive",
                "energy_profile": "balanced",
                "inference_backend": "cpu_optimized"
            },
            Platform.LINUX_ARM64: {
                "compiler_flags": ["-O3", "-march=armv8-a"],
                "parallel_workers": os.cpu_count(),
                "memory_optimization": "conservative",
                "energy_profile": "low_power",
                "inference_backend": "arm_neon"
            },
            Platform.WINDOWS_X64: {
                "compiler_flags": ["/O2", "/arch:AVX2"],
                "parallel_workers": os.cpu_count(),
                "memory_optimization": "balanced",
                "energy_profile": "balanced",
                "inference_backend": "cpu_optimized"
            },
            Platform.MACOS_X64: {
                "compiler_flags": ["-O3", "-march=native"],
                "parallel_workers": os.cpu_count(),
                "memory_optimization": "balanced",
                "energy_profile": "balanced",
                "inference_backend": "accelerate_framework"
            },
            Platform.MACOS_ARM64: {
                "compiler_flags": ["-O3", "-mcpu=apple-m1"],
                "parallel_workers": os.cpu_count(),
                "memory_optimization": "aggressive",
                "energy_profile": "ultra_low_power",
                "inference_backend": "metal_performance_shaders"
            },
            Platform.DOCKER: {
                "compiler_flags": ["-O3"],
                "parallel_workers": min(4, os.cpu_count()),
                "memory_optimization": "conservative",
                "energy_profile": "balanced",
                "inference_backend": "portable"
            },
            Platform.KUBERNETES: {
                "compiler_flags": ["-O3"],
                "parallel_workers": min(8, os.cpu_count()),
                "memory_optimization": "aggressive",
                "energy_profile": "high_performance",
                "inference_backend": "distributed"
            }
        }
    
    def get_current_platform(self) -> Platform:
        """Get current platform."""
        return self.current_platform
    
    def get_optimization_profile(self, platform: Optional[Platform] = None) -> Dict[str, Any]:
        """Get optimization profile for platform.
        
        Args:
            platform: Target platform (current if None)
            
        Returns:
            Optimization profile
        """
        target_platform = platform or self.current_platform
        return self.optimization_profiles.get(target_platform, {})
    
    def is_platform_supported(self, platform: Platform) -> bool:
        """Check if platform is supported."""
        return platform in self.optimization_profiles
    
    def get_platform_capabilities(self, platform: Optional[Platform] = None) -> Dict[str, Any]:
        """Get platform capabilities.
        
        Args:
            platform: Target platform
            
        Returns:
            Platform capabilities
        """
        target_platform = platform or self.current_platform
        
        capabilities = {
            "simd_support": self._check_simd_support(target_platform),
            "gpu_acceleration": self._check_gpu_support(target_platform),
            "memory_mapping": self._check_memory_mapping(target_platform),
            "high_precision_timers": self._check_timer_support(target_platform),
            "parallel_processing": self._check_parallel_support(target_platform)
        }
        
        return capabilities
    
    def _check_simd_support(self, platform: Platform) -> List[str]:
        """Check SIMD instruction support."""
        simd_support = {
            Platform.LINUX_X64: ["SSE", "SSE2", "AVX", "AVX2"],
            Platform.LINUX_ARM64: ["NEON", "SVE"],
            Platform.WINDOWS_X64: ["SSE", "SSE2", "AVX", "AVX2"],
            Platform.MACOS_X64: ["SSE", "SSE2", "AVX", "AVX2"],
            Platform.MACOS_ARM64: ["NEON", "AMX"],
            Platform.DOCKER: ["SSE", "SSE2"],
            Platform.KUBERNETES: ["SSE", "SSE2", "AVX"]
        }
        
        return simd_support.get(platform, [])
    
    def _check_gpu_support(self, platform: Platform) -> List[str]:
        """Check GPU acceleration support."""
        gpu_support = {
            Platform.LINUX_X64: ["CUDA", "OpenCL", "ROCm"],
            Platform.LINUX_ARM64: ["OpenCL", "Mali GPU"],
            Platform.WINDOWS_X64: ["CUDA", "DirectML", "OpenCL"],
            Platform.MACOS_X64: ["Metal", "OpenCL"],
            Platform.MACOS_ARM64: ["Metal", "Neural Engine"],
            Platform.DOCKER: ["CUDA (if host supports)"],
            Platform.KUBERNETES: ["CUDA", "GPU Operator"]
        }
        
        return gpu_support.get(platform, [])
    
    def _check_memory_mapping(self, platform: Platform) -> bool:
        """Check memory mapping support."""
        return platform not in [Platform.DOCKER]  # Most platforms support mmap
    
    def _check_timer_support(self, platform: Platform) -> bool:
        """Check high precision timer support."""
        return True  # All modern platforms support high-precision timers
    
    def _check_parallel_support(self, platform: Platform) -> Dict[str, Any]:
        """Check parallel processing support."""
        return {
            "threading": True,
            "multiprocessing": platform != Platform.DOCKER,  # Limited in containers
            "async_io": True,
            "max_workers": self.optimization_profiles[platform]["parallel_workers"]
        }


class RegionalPerformanceOptimizer:
    """Regional performance optimization based on local conditions."""
    
    def __init__(self, region_config: RegionConfig):
        self.region_config = region_config
        self.performance_profiles = self._load_regional_profiles()
        
    def _load_regional_profiles(self) -> Dict[str, Any]:
        """Load regional performance optimization profiles."""
        # Regional optimizations based on typical infrastructure and requirements
        regional_profiles = {
            DeploymentRegion.US_EAST: {
                "network_optimization": "high_bandwidth",
                "cache_strategy": "aggressive_caching",
                "energy_priority": "performance_first",
                "preferred_compute": "gpu_accelerated",
                "data_locality": "multi_az"
            },
            DeploymentRegion.US_WEST: {
                "network_optimization": "low_latency",
                "cache_strategy": "intelligent_prefetch",
                "energy_priority": "balanced",
                "preferred_compute": "hybrid_cpu_gpu",
                "data_locality": "edge_distributed"
            },
            DeploymentRegion.EU_WEST: {
                "network_optimization": "gdpr_compliant",
                "cache_strategy": "privacy_aware_caching",
                "energy_priority": "green_computing",
                "preferred_compute": "energy_efficient",
                "data_locality": "eu_only"
            },
            DeploymentRegion.EU_CENTRAL: {
                "network_optimization": "high_security",
                "cache_strategy": "encrypted_caching",
                "energy_priority": "renewable_focused",
                "preferred_compute": "certified_hardware",
                "data_locality": "national_boundaries"
            },
            DeploymentRegion.ASIA_PACIFIC: {
                "network_optimization": "multi_region_sync",
                "cache_strategy": "distributed_caching",
                "energy_priority": "cost_optimized",
                "preferred_compute": "scalable_cpu",
                "data_locality": "regional_compliance"
            },
            DeploymentRegion.ASIA_NORTHEAST: {
                "network_optimization": "earthquake_resilient",
                "cache_strategy": "disaster_recovery",
                "energy_priority": "ultra_low_power",
                "preferred_compute": "fault_tolerant",
                "data_locality": "multi_site_backup"
            }
        }
        
        return regional_profiles.get(self.region_config.region, {})
    
    def get_optimized_config(self) -> Dict[str, Any]:
        """Get region-optimized configuration."""
        base_config = {
            "region": self.region_config.region.value,
            "latency_target": self.region_config.latency_target_ms,
            "energy_target": self.region_config.energy_efficiency_target,
            "compliance_requirements": self.region_config.regulatory_framework
        }
        
        base_config.update(self.performance_profiles)
        return base_config
    
    def optimize_for_region(self, base_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Apply regional optimizations to base performance configuration."""
        optimized = base_performance.copy()
        
        # Apply regional energy optimization
        if self.performance_profiles.get("energy_priority") == "green_computing":
            optimized["energy_weight"] = 0.8  # Prioritize energy efficiency
            optimized["renewable_energy_preferred"] = True
        elif self.performance_profiles.get("energy_priority") == "performance_first":
            optimized["energy_weight"] = 0.2  # Prioritize performance
            optimized["power_scaling_enabled"] = True
        
        # Apply network optimizations
        network_opt = self.performance_profiles.get("network_optimization")
        if network_opt == "low_latency":
            optimized["tcp_nodelay"] = True
            optimized["connection_pooling"] = True
            optimized["request_batching"] = False
        elif network_opt == "high_bandwidth":
            optimized["compression_enabled"] = True
            optimized["request_batching"] = True
            optimized["pipeline_depth"] = 8
        
        # Apply caching strategy
        cache_strategy = self.performance_profiles.get("cache_strategy")
        if cache_strategy == "privacy_aware_caching":
            optimized["cache_encryption"] = True
            optimized["cache_ttl_short"] = True
            optimized["cache_personal_data"] = False
        elif cache_strategy == "aggressive_caching":
            optimized["cache_size_multiplier"] = 2.0
            optimized["prefetch_enabled"] = True
            optimized["cache_warmup"] = True
        
        return optimized


class GlobalDeploymentManager:
    """Main global deployment coordinator."""
    
    def __init__(self):
        self.cross_platform = CrossPlatformSupport()
        self.region_configs = self._initialize_region_configs()
        self.active_deployments = {}
        
    def _initialize_region_configs(self) -> Dict[DeploymentRegion, RegionConfig]:
        """Initialize regional configurations."""
        configs = {}
        
        # US East (Virginia) - High performance, CCPA compliant
        configs[DeploymentRegion.US_EAST] = RegionConfig(
            region=DeploymentRegion.US_EAST,
            name="US East (Virginia)",
            timezone="America/New_York",
            regulatory_framework=["CCPA", "SOX", "HIPAA"],
            data_residency_required=False,
            latency_target_ms=50.0,
            energy_efficiency_target=12.0,  # pJ/MAC
            supported_platforms=[
                Platform.LINUX_X64, Platform.LINUX_ARM64, 
                Platform.DOCKER, Platform.KUBERNETES
            ],
            compliance_contacts={
                "dpo": "dpo.us@spintron-nn-kit.org",
                "legal": "legal.us@spintron-nn-kit.org"
            }
        )
        
        # EU West (Ireland) - GDPR compliant, green computing
        configs[DeploymentRegion.EU_WEST] = RegionConfig(
            region=DeploymentRegion.EU_WEST,
            name="EU West (Ireland)",
            timezone="Europe/Dublin",
            regulatory_framework=["GDPR", "NIS2"],
            data_residency_required=True,
            latency_target_ms=100.0,
            energy_efficiency_target=8.0,  # Ultra-low power for green computing
            supported_platforms=[
                Platform.LINUX_X64, Platform.LINUX_ARM64,
                Platform.DOCKER, Platform.KUBERNETES
            ],
            compliance_contacts={
                "dpo": "dpo.eu@spintron-nn-kit.org",
                "gdpr_officer": "gdpr@spintron-nn-kit.org"
            }
        )
        
        # Asia Pacific (Singapore) - PDPA compliant
        configs[DeploymentRegion.ASIA_PACIFIC] = RegionConfig(
            region=DeploymentRegion.ASIA_PACIFIC,
            name="Asia Pacific (Singapore)",
            timezone="Asia/Singapore",
            regulatory_framework=["PDPA", "CYBERSECURITY_ACT"],
            data_residency_required=True,
            latency_target_ms=150.0,
            energy_efficiency_target=10.0,
            supported_platforms=[
                Platform.LINUX_X64, Platform.LINUX_ARM64,
                Platform.DOCKER, Platform.KUBERNETES
            ],
            compliance_contacts={
                "dpo": "dpo.apac@spintron-nn-kit.org",
                "pdpc_contact": "pdpc@spintron-nn-kit.org"
            }
        )
        
        return configs
    
    def deploy_to_region(self, region: DeploymentRegion,
                        platform: Optional[Platform] = None,
                        custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Deploy SpinTron-NN-Kit to specific region.
        
        Args:
            region: Target deployment region
            platform: Target platform (auto-detect if None)
            custom_config: Custom configuration overrides
            
        Returns:
            Deployment result
        """
        if region not in self.region_configs:
            return {"error": f"Unsupported region: {region.value}"}
        
        region_config = self.region_configs[region]
        target_platform = platform or self.cross_platform.get_current_platform()
        
        # Check platform compatibility
        if target_platform not in region_config.supported_platforms:
            return {
                "error": f"Platform {target_platform.value} not supported in region {region.value}",
                "supported_platforms": [p.value for p in region_config.supported_platforms]
            }
        
        # Generate deployment configuration
        deployment_config = self._generate_deployment_config(
            region_config, target_platform, custom_config
        )
        
        deployment_id = f"spintron-{region.value}-{target_platform.value}-{int(time.time())}"
        
        # Record deployment
        self.active_deployments[deployment_id] = {
            "deployment_id": deployment_id,
            "region": region,
            "platform": target_platform,
            "config": deployment_config,
            "deployment_time": time.time(),
            "status": "active"
        }
        
        return {
            "deployment_id": deployment_id,
            "region": region.value,
            "platform": target_platform.value,
            "configuration": deployment_config,
            "status": "deployed",
            "endpoints": self._generate_regional_endpoints(region, deployment_id)
        }
    
    def _generate_deployment_config(self, region_config: RegionConfig,
                                  platform: Platform,
                                  custom_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive deployment configuration."""
        # Base configuration
        base_config = {
            "region": asdict(region_config),
            "platform": {
                "type": platform.value,
                "optimization_profile": self.cross_platform.get_optimization_profile(platform),
                "capabilities": self.cross_platform.get_platform_capabilities(platform)
            },
            "performance": {
                "latency_target_ms": region_config.latency_target_ms,
                "energy_target_pj_per_mac": region_config.energy_efficiency_target,
                "throughput_scaling": "auto",
                "resource_allocation": "dynamic"
            },
            "compliance": {
                "enabled_frameworks": region_config.regulatory_framework,
                "data_residency": region_config.data_residency_required,
                "audit_logging": True,
                "encryption_at_rest": True,
                "encryption_in_transit": True
            },
            "networking": {
                "load_balancing": "regional",
                "health_checks": True,
                "auto_scaling": True,
                "cdn_enabled": True
            }
        }
        
        # Apply regional optimizations
        optimizer = RegionalPerformanceOptimizer(region_config)
        base_config["performance"] = optimizer.optimize_for_region(base_config["performance"])
        
        # Apply custom configuration overrides
        if custom_config:
            self._deep_merge_config(base_config, custom_config)
        
        return base_config
    
    def _deep_merge_config(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Deep merge configuration dictionaries."""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge_config(base[key], value)
            else:
                base[key] = value
    
    def _generate_regional_endpoints(self, region: DeploymentRegion,
                                   deployment_id: str) -> Dict[str, str]:
        """Generate regional API endpoints."""
        region_code = region.value
        
        return {
            "inference_api": f"https://api.{region_code}.spintron-nn-kit.org/v1/inference",
            "management_api": f"https://api.{region_code}.spintron-nn-kit.org/v1/management",
            "monitoring": f"https://monitor.{region_code}.spintron-nn-kit.org",
            "health_check": f"https://health.{region_code}.spintron-nn-kit.org/status",
            "deployment_specific": f"https://{deployment_id}.{region_code}.spintron-nn-kit.org"
        }
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific deployment."""
        return self.active_deployments.get(deployment_id)
    
    def list_active_deployments(self) -> List[Dict[str, Any]]:
        """List all active deployments."""
        return list(self.active_deployments.values())
    
    def get_supported_regions(self) -> List[Dict[str, Any]]:
        """Get list of supported deployment regions."""
        return [
            {
                "region_code": region.value,
                "region_name": config.name,
                "timezone": config.timezone,
                "regulatory_frameworks": config.regulatory_framework,
                "data_residency_required": config.data_residency_required,
                "supported_platforms": [p.value for p in config.supported_platforms]
            }
            for region, config in self.region_configs.items()
        ]
    
    def get_platform_support_matrix(self) -> Dict[str, List[str]]:
        """Get platform support matrix across all regions."""
        matrix = {}
        
        for platform in Platform:
            supported_regions = [
                region.value for region, config in self.region_configs.items()
                if platform in config.supported_platforms
            ]
            matrix[platform.value] = supported_regions
        
        return matrix
    
    def recommend_optimal_region(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend optimal deployment region based on requirements.
        
        Args:
            requirements: Deployment requirements
            
        Returns:
            Recommendation with reasoning
        """
        user_location = requirements.get("user_location", "unknown")
        latency_requirement = requirements.get("max_latency_ms", 200)
        compliance_needs = requirements.get("regulatory_compliance", [])
        energy_priority = requirements.get("energy_efficiency", False)
        
        scored_regions = []
        
        for region, config in self.region_configs.items():
            score = 0
            reasons = []
            
            # Latency scoring
            if config.latency_target_ms <= latency_requirement:
                latency_score = (latency_requirement - config.latency_target_ms) / latency_requirement
                score += latency_score * 30
                reasons.append(f"Low latency: {config.latency_target_ms}ms")
            
            # Compliance scoring
            compliance_matches = len(set(compliance_needs) & set(config.regulatory_framework))
            if compliance_matches > 0:
                score += compliance_matches * 25
                reasons.append(f"Compliance match: {compliance_matches} frameworks")
            
            # Energy efficiency scoring
            if energy_priority:
                energy_score = 20 / config.energy_efficiency_target  # Lower is better
                score += energy_score * 20
                reasons.append(f"Energy efficient: {config.energy_efficiency_target} pJ/MAC")
            
            # Geographic proximity (simplified)
            geo_bonus = self._calculate_geographic_bonus(user_location, region)
            score += geo_bonus
            if geo_bonus > 0:
                reasons.append(f"Geographic proximity bonus: +{geo_bonus}")
            
            scored_regions.append({
                "region": region.value,
                "region_name": config.name,
                "score": score,
                "reasons": reasons,
                "config": config
            })
        
        # Sort by score
        scored_regions.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "recommended_region": scored_regions[0] if scored_regions else None,
            "all_options": scored_regions,
            "user_requirements": requirements
        }
    
    def _calculate_geographic_bonus(self, user_location: str,
                                  region: DeploymentRegion) -> float:
        """Calculate geographic proximity bonus."""
        proximity_map = {
            "north_america": {
                DeploymentRegion.US_EAST: 15,
                DeploymentRegion.US_WEST: 15,
                DeploymentRegion.CANADA_CENTRAL: 12
            },
            "europe": {
                DeploymentRegion.EU_WEST: 15,
                DeploymentRegion.EU_CENTRAL: 15
            },
            "asia": {
                DeploymentRegion.ASIA_PACIFIC: 15,
                DeploymentRegion.ASIA_NORTHEAST: 15
            },
            "oceania": {
                DeploymentRegion.AUSTRALIA: 15,
                DeploymentRegion.ASIA_PACIFIC: 10
            }
        }
        
        return proximity_map.get(user_location, {}).get(region, 0)


# Global deployment manager instance
_global_deployment_manager = None


def get_deployment_manager() -> GlobalDeploymentManager:
    """Get global deployment manager instance."""
    global _global_deployment_manager
    
    if _global_deployment_manager is None:
        _global_deployment_manager = GlobalDeploymentManager()
    
    return _global_deployment_manager


def deploy_globally(region: DeploymentRegion,
                   platform: Optional[Platform] = None,
                   config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Deploy SpinTron-NN-Kit globally.
    
    Args:
        region: Target region
        platform: Target platform
        config: Custom configuration
        
    Returns:
        Deployment result
    """
    return get_deployment_manager().deploy_to_region(region, platform, config)