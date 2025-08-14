"""
Cloud Orchestration and Auto-Scaling for Spintronic Neural Networks.

Implements intelligent cloud resource management with:
- Multi-cloud deployment strategies
- Auto-scaling based on workload patterns
- Edge-cloud hybrid architectures
- Cost optimization algorithms
- Global load balancing
"""

import numpy as np
import torch
import asyncio
import aiohttp
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import time
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import threading

from ..core.mtj_models import MTJConfig
from ..core.crossbar import MTJCrossbar, CrossbarConfig
from ..utils.error_handling import SpintronError, robust_operation
from ..utils.logging_config import get_logger
from ..utils.monitoring import get_system_monitor

logger = get_logger(__name__)


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    AZURE = "azure"
    GCP = "gcp"
    EDGE = "edge"
    HYBRID = "hybrid"


class ResourceType(Enum):
    """Types of compute resources."""
    CPU = "cpu"
    GPU = "gpu"
    TPU = "tpu"
    QUANTUM = "quantum"
    SPINTRONIC = "spintronic"
    NEUROMORPHIC = "neuromorphic"


class ScalingPolicy(Enum):
    """Auto-scaling policies."""
    REACTIVE = "reactive"  # Scale based on current load
    PREDICTIVE = "predictive"  # Scale based on predicted load
    ADAPTIVE = "adaptive"  # Learn optimal scaling patterns
    COST_OPTIMIZED = "cost_optimized"  # Minimize cost while meeting SLA


@dataclass
class CloudResource:
    """Cloud resource specification."""
    
    provider: CloudProvider
    resource_type: ResourceType
    instance_type: str
    region: str
    availability_zone: str
    
    # Performance characteristics
    compute_units: float
    memory_gb: float
    network_gbps: float
    storage_gb: float = 0
    
    # Cost information
    cost_per_hour: float = 0.0
    spot_price: float = 0.0
    preemptible: bool = False
    
    # Status
    active: bool = False
    utilization: float = 0.0
    last_heartbeat: float = 0.0
    
    # Metadata
    tags: Dict[str, str] = field(default_factory=dict)
    
    def performance_score(self) -> float:
        """Calculate overall performance score."""
        return (self.compute_units * 0.4 + 
                self.memory_gb * 0.3 + 
                self.network_gbps * 0.3)
    
    def cost_efficiency(self) -> float:
        """Calculate cost efficiency (performance per dollar)."""
        if self.cost_per_hour <= 0:
            return float('inf')
        return self.performance_score() / self.cost_per_hour


@dataclass
class WorkloadMetrics:
    """Workload performance metrics."""
    
    timestamp: float
    requests_per_second: float
    average_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    error_rate: float
    cpu_utilization: float
    memory_utilization: float
    network_utilization: float
    queue_length: int = 0
    
    def sla_compliance(self, target_latency_ms: float = 100.0, target_error_rate: float = 0.01) -> float:
        """Calculate SLA compliance score."""
        latency_compliance = 1.0 if self.p95_latency_ms <= target_latency_ms else 0.0
        error_compliance = 1.0 if self.error_rate <= target_error_rate else 0.0
        return (latency_compliance + error_compliance) / 2.0


@dataclass
class ScalingDecision:
    """Auto-scaling decision result."""
    
    action: str  # 'scale_up', 'scale_down', 'no_change'
    target_instances: int
    reasoning: List[str]
    confidence: float
    estimated_cost_impact: float
    estimated_performance_impact: float
    

class CloudOrchestrator:
    """
    Intelligent cloud orchestration with multi-cloud support.
    
    Manages resources across multiple cloud providers, implements
    cost-effective auto-scaling, and optimizes for performance and cost.
    """
    
    def __init__(
        self,
        supported_providers: List[CloudProvider],
        scaling_policy: ScalingPolicy = ScalingPolicy.ADAPTIVE,
        cost_budget_per_hour: float = 100.0
    ):
        self.supported_providers = supported_providers
        self.scaling_policy = scaling_policy
        self.cost_budget_per_hour = cost_budget_per_hour
        
        # Resource management
        self.available_resources: Dict[str, CloudResource] = {}
        self.active_resources: Dict[str, CloudResource] = {}
        self.resource_pool_lock = threading.Lock()
        
        # Workload monitoring
        self.workload_history: List[WorkloadMetrics] = []
        self.current_workload: Optional[WorkloadMetrics] = None
        
        # Scaling history and learning
        self.scaling_history: List[ScalingDecision] = []
        self.scaling_model = None  # Will be initialized for adaptive scaling
        
        # Performance tracking
        self.monitor = get_system_monitor()
        self.orchestration_metrics = {
            'total_requests_served': 0,
            'average_response_time': 0.0,
            'cost_savings_achieved': 0.0,
            'uptime_percentage': 100.0
        }
        
        # Load balancing
        self.load_balancer = GlobalLoadBalancer(self)
        
        # Background tasks
        self._monitoring_task = None
        self._scaling_task = None
        self._cost_optimization_task = None
        
        logger.info(f"Initialized CloudOrchestrator with {len(supported_providers)} providers")
    
    async def initialize(self):
        """Initialize cloud orchestrator and discover resources."""
        
        # Discover available resources
        await self._discover_cloud_resources()
        
        # Initialize adaptive scaling model
        if self.scaling_policy == ScalingPolicy.ADAPTIVE:
            self._initialize_adaptive_scaling_model()
        
        # Start background monitoring tasks
        self._start_background_tasks()
        
        logger.info("Cloud orchestrator initialized successfully")
    
    async def _discover_cloud_resources(self):
        """Discover available cloud resources across providers."""
        
        discovery_tasks = []
        
        for provider in self.supported_providers:
            task = self._discover_provider_resources(provider)
            discovery_tasks.append(task)
        
        # Run discovery in parallel
        results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
        
        total_discovered = 0
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Resource discovery failed for {self.supported_providers[i]}: {str(result)}")
            else:
                total_discovered += result
        
        logger.info(f"Discovered {total_discovered} cloud resources across {len(self.supported_providers)} providers")
    
    async def _discover_provider_resources(self, provider: CloudProvider) -> int:
        """Discover resources for a specific cloud provider."""
        
        # Simulate resource discovery (in practice, would use cloud APIs)
        simulated_resources = self._generate_simulated_resources(provider)
        
        with self.resource_pool_lock:
            for resource in simulated_resources:
                resource_id = f"{provider.value}_{resource.region}_{resource.instance_type}_{int(time.time())}"
                self.available_resources[resource_id] = resource
        
        return len(simulated_resources)
    
    def _generate_simulated_resources(self, provider: CloudProvider) -> List[CloudResource]:
        """Generate simulated resources for demonstration."""
        
        resources = []
        
        # Define resource templates by provider
        if provider == CloudProvider.AWS:
            templates = [
                {'instance_type': 'm5.large', 'compute_units': 2.0, 'memory_gb': 8.0, 'cost_per_hour': 0.096},
                {'instance_type': 'm5.xlarge', 'compute_units': 4.0, 'memory_gb': 16.0, 'cost_per_hour': 0.192},
                {'instance_type': 'c5.2xlarge', 'compute_units': 8.0, 'memory_gb': 16.0, 'cost_per_hour': 0.34},
                {'instance_type': 'p3.2xlarge', 'compute_units': 8.0, 'memory_gb': 61.0, 'cost_per_hour': 3.06},
            ]
            regions = ['us-east-1', 'us-west-2', 'eu-west-1']
            
        elif provider == CloudProvider.GCP:
            templates = [
                {'instance_type': 'n1-standard-2', 'compute_units': 2.0, 'memory_gb': 7.5, 'cost_per_hour': 0.095},
                {'instance_type': 'n1-standard-4', 'compute_units': 4.0, 'memory_gb': 15.0, 'cost_per_hour': 0.19},
                {'instance_type': 'n1-highmem-4', 'compute_units': 4.0, 'memory_gb': 26.0, 'cost_per_hour': 0.236},
            ]
            regions = ['us-central1', 'us-east1', 'europe-west1']
            
        elif provider == CloudProvider.AZURE:
            templates = [
                {'instance_type': 'Standard_D2s_v3', 'compute_units': 2.0, 'memory_gb': 8.0, 'cost_per_hour': 0.096},
                {'instance_type': 'Standard_D4s_v3', 'compute_units': 4.0, 'memory_gb': 16.0, 'cost_per_hour': 0.192},
                {'instance_type': 'Standard_D8s_v3', 'compute_units': 8.0, 'memory_gb': 32.0, 'cost_per_hour': 0.384},
            ]
            regions = ['East US', 'West US 2', 'West Europe']
            
        elif provider == CloudProvider.EDGE:
            templates = [
                {'instance_type': 'edge-small', 'compute_units': 1.0, 'memory_gb': 4.0, 'cost_per_hour': 0.05},
                {'instance_type': 'edge-medium', 'compute_units': 2.0, 'memory_gb': 8.0, 'cost_per_hour': 0.10},
            ]
            regions = ['edge-location-1', 'edge-location-2']
        
        else:
            return []  # Unknown provider
        
        # Generate resources from templates
        for template in templates:
            for region in regions:
                for az_suffix in ['a', 'b']:
                    resource = CloudResource(
                        provider=provider,
                        resource_type=ResourceType.CPU,  # Default to CPU
                        instance_type=template['instance_type'],
                        region=region,
                        availability_zone=f"{region}{az_suffix}",
                        compute_units=template['compute_units'],
                        memory_gb=template['memory_gb'],
                        network_gbps=1.0,  # Default network bandwidth
                        cost_per_hour=template['cost_per_hour'],
                        spot_price=template['cost_per_hour'] * 0.7,  # 30% discount for spot
                        tags={'environment': 'production', 'orchestrator': 'spintron'}
                    )
                    resources.append(resource)
        
        return resources
    
    async def deploy_workload(
        self,
        workload_config: Dict[str, Any],
        performance_requirements: Dict[str, float],
        cost_constraints: Dict[str, float]
    ) -> Dict[str, Any]:
        """Deploy workload with optimal resource allocation."""
        
        logger.info(f"Deploying workload with requirements: {performance_requirements}")
        
        try:
            # Find optimal resource allocation
            allocation = await self._optimize_resource_allocation(
                workload_config, performance_requirements, cost_constraints
            )
            
            # Provision resources
            provisioned_resources = await self._provision_resources(allocation)
            
            # Deploy workload to resources
            deployment_result = await self._deploy_to_resources(
                workload_config, provisioned_resources
            )
            
            # Configure load balancing
            await self.load_balancer.configure_for_deployment(deployment_result)
            
            return {
                'status': 'success',
                'deployment_id': deployment_result['deployment_id'],
                'allocated_resources': len(provisioned_resources),
                'estimated_cost_per_hour': sum(r.cost_per_hour for r in provisioned_resources),
                'endpoints': deployment_result['endpoints']
            }
            
        except Exception as e:
            logger.error(f"Workload deployment failed: {str(e)}")
            raise SpintronError(f"Deployment failed: {str(e)}")
    
    async def _optimize_resource_allocation(
        self,
        workload_config: Dict[str, Any],
        performance_requirements: Dict[str, float],
        cost_constraints: Dict[str, float]
    ) -> List[CloudResource]:
        """Optimize resource allocation using multi-objective optimization."""
        
        # Extract requirements
        min_compute_units = performance_requirements.get('min_compute_units', 2.0)
        max_latency_ms = performance_requirements.get('max_latency_ms', 100.0)
        min_availability = performance_requirements.get('min_availability', 0.99)
        
        max_cost_per_hour = cost_constraints.get('max_cost_per_hour', self.cost_budget_per_hour)
        prefer_spot_instances = cost_constraints.get('prefer_spot_instances', False)
        
        # Filter available resources based on constraints
        candidate_resources = []
        with self.resource_pool_lock:
            for resource in self.available_resources.values():
                if (resource.compute_units >= min_compute_units * 0.5 and  # Allow some flexibility
                    resource.cost_per_hour <= max_cost_per_hour):
                    candidate_resources.append(resource)
        
        if not candidate_resources:
            raise SpintronError("No suitable resources available")
        
        # Multi-objective optimization
        optimal_allocation = self._solve_resource_allocation_problem(
            candidate_resources,
            performance_requirements,
            cost_constraints
        )
        
        return optimal_allocation
    
    def _solve_resource_allocation_problem(
        self,
        candidate_resources: List[CloudResource],
        performance_requirements: Dict[str, float],
        cost_constraints: Dict[str, float]
    ) -> List[CloudResource]:
        """Solve multi-objective resource allocation problem."""
        
        # Simple greedy algorithm (in practice, would use sophisticated optimization)
        selected_resources = []
        total_compute = 0.0
        total_cost = 0.0
        
        # Sort by cost efficiency
        sorted_resources = sorted(candidate_resources, key=lambda r: r.cost_efficiency(), reverse=True)
        
        min_compute_units = performance_requirements.get('min_compute_units', 2.0)
        max_cost_per_hour = cost_constraints.get('max_cost_per_hour', self.cost_budget_per_hour)
        
        for resource in sorted_resources:
            if (total_compute < min_compute_units and 
                total_cost + resource.cost_per_hour <= max_cost_per_hour):
                
                selected_resources.append(resource)
                total_compute += resource.compute_units
                total_cost += resource.cost_per_hour
                
                # Add redundancy for high availability
                if performance_requirements.get('min_availability', 0.99) > 0.95:
                    if len(selected_resources) < 2:  # Minimum 2 instances for HA
                        continue
        
        # Ensure minimum requirements are met
        if total_compute < min_compute_units:
            raise SpintronError(f"Cannot meet compute requirements: need {min_compute_units}, got {total_compute}")
        
        return selected_resources
    
    async def _provision_resources(self, allocation: List[CloudResource]) -> List[CloudResource]:
        """Provision the allocated resources."""
        
        provisioned = []
        
        for resource in allocation:
            try:
                # Simulate resource provisioning
                provisioned_resource = await self._provision_single_resource(resource)
                provisioned.append(provisioned_resource)
                
                # Move to active resources
                with self.resource_pool_lock:
                    resource_id = f"{resource.provider.value}_{resource.region}_{int(time.time())}"
                    self.active_resources[resource_id] = provisioned_resource
                
            except Exception as e:
                logger.warning(f"Failed to provision resource {resource.instance_type}: {str(e)}")
        
        if not provisioned:
            raise SpintronError("Failed to provision any resources")
        
        logger.info(f"Provisioned {len(provisioned)} resources")
        return provisioned
    
    async def _provision_single_resource(self, resource: CloudResource) -> CloudResource:
        """Provision a single cloud resource."""
        
        # Simulate provisioning delay
        await asyncio.sleep(0.1)
        
        # Create provisioned resource copy
        provisioned = CloudResource(
            provider=resource.provider,
            resource_type=resource.resource_type,
            instance_type=resource.instance_type,
            region=resource.region,
            availability_zone=resource.availability_zone,
            compute_units=resource.compute_units,
            memory_gb=resource.memory_gb,
            network_gbps=resource.network_gbps,
            storage_gb=resource.storage_gb,
            cost_per_hour=resource.cost_per_hour,
            spot_price=resource.spot_price,
            preemptible=resource.preemptible,
            active=True,
            utilization=0.0,
            last_heartbeat=time.time(),
            tags=resource.tags.copy()
        )
        
        return provisioned
    
    async def _deploy_to_resources(
        self,
        workload_config: Dict[str, Any],
        resources: List[CloudResource]
    ) -> Dict[str, Any]:
        """Deploy workload to provisioned resources."""
        
        deployment_id = f"deployment_{int(time.time())}"
        endpoints = []
        
        # Deploy to each resource
        for i, resource in enumerate(resources):
            endpoint = await self._deploy_to_single_resource(workload_config, resource, i)
            endpoints.append(endpoint)
        
        return {
            'deployment_id': deployment_id,
            'endpoints': endpoints,
            'resources': resources
        }
    
    async def _deploy_to_single_resource(
        self,
        workload_config: Dict[str, Any],
        resource: CloudResource,
        instance_index: int
    ) -> str:
        """Deploy workload to a single resource."""
        
        # Simulate deployment
        await asyncio.sleep(0.2)
        
        # Generate endpoint URL
        endpoint = f"https://{resource.provider.value}-{resource.region}-{instance_index}.spintron.ai"
        
        return endpoint
    
    async def auto_scale(self) -> ScalingDecision:
        """Perform auto-scaling based on current workload and policy."""
        
        if not self.current_workload:
            return ScalingDecision(
                action='no_change',
                target_instances=len(self.active_resources),
                reasoning=['No workload metrics available'],
                confidence=0.0,
                estimated_cost_impact=0.0,
                estimated_performance_impact=0.0
            )
        
        if self.scaling_policy == ScalingPolicy.REACTIVE:
            decision = await self._reactive_scaling()
        elif self.scaling_policy == ScalingPolicy.PREDICTIVE:
            decision = await self._predictive_scaling()
        elif self.scaling_policy == ScalingPolicy.ADAPTIVE:
            decision = await self._adaptive_scaling()
        elif self.scaling_policy == ScalingPolicy.COST_OPTIMIZED:
            decision = await self._cost_optimized_scaling()
        else:
            decision = ScalingDecision(
                action='no_change',
                target_instances=len(self.active_resources),
                reasoning=['Unknown scaling policy'],
                confidence=0.0,
                estimated_cost_impact=0.0,
                estimated_performance_impact=0.0
            )
        
        # Execute scaling decision
        if decision.action != 'no_change':
            await self._execute_scaling_decision(decision)
        
        # Record scaling decision
        self.scaling_history.append(decision)
        
        return decision
    
    async def _reactive_scaling(self) -> ScalingDecision:
        """Reactive scaling based on current metrics."""
        
        current_instances = len(self.active_resources)
        metrics = self.current_workload
        
        reasoning = []
        action = 'no_change'
        target_instances = current_instances
        confidence = 0.8
        
        # Check if scaling up is needed
        if (metrics.cpu_utilization > 0.8 or 
            metrics.memory_utilization > 0.8 or
            metrics.p95_latency_ms > 100.0):
            
            action = 'scale_up'
            target_instances = min(current_instances + 2, 10)  # Cap at 10 instances
            reasoning.append(f"High utilization: CPU {metrics.cpu_utilization:.1%}, Memory {metrics.memory_utilization:.1%}")
            reasoning.append(f"Latency P95: {metrics.p95_latency_ms:.1f}ms")
        
        # Check if scaling down is possible
        elif (metrics.cpu_utilization < 0.3 and 
              metrics.memory_utilization < 0.3 and
              current_instances > 2):  # Keep minimum 2 instances
            
            action = 'scale_down'
            target_instances = max(current_instances - 1, 2)
            reasoning.append(f"Low utilization: CPU {metrics.cpu_utilization:.1%}, Memory {metrics.memory_utilization:.1%}")
        
        # Estimate impact
        cost_impact = self._estimate_cost_impact(current_instances, target_instances)
        performance_impact = self._estimate_performance_impact(current_instances, target_instances)
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            reasoning=reasoning,
            confidence=confidence,
            estimated_cost_impact=cost_impact,
            estimated_performance_impact=performance_impact
        )
    
    async def _predictive_scaling(self) -> ScalingDecision:
        """Predictive scaling based on forecasted load."""
        
        # Analyze historical patterns
        forecast = self._forecast_workload()
        
        current_instances = len(self.active_resources)
        reasoning = []
        action = 'no_change'
        target_instances = current_instances
        confidence = 0.6  # Lower confidence for predictions
        
        if forecast['predicted_peak_load'] > current_instances * 0.8:
            action = 'scale_up'
            target_instances = int(forecast['predicted_peak_load'] / 0.7)  # Target 70% utilization
            reasoning.append(f"Predicted peak load: {forecast['predicted_peak_load']:.1f}")
            reasoning.append(f"Time to peak: {forecast['time_to_peak_minutes']:.0f} minutes")
        
        elif forecast['predicted_min_load'] < current_instances * 0.3:
            action = 'scale_down'
            target_instances = max(int(forecast['predicted_min_load'] / 0.5), 2)
            reasoning.append(f"Predicted low load period: {forecast['predicted_min_load']:.1f}")
        
        cost_impact = self._estimate_cost_impact(current_instances, target_instances)
        performance_impact = self._estimate_performance_impact(current_instances, target_instances)
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            reasoning=reasoning,
            confidence=confidence,
            estimated_cost_impact=cost_impact,
            estimated_performance_impact=performance_impact
        )
    
    def _forecast_workload(self) -> Dict[str, float]:
        """Simple workload forecasting based on historical data."""
        
        if len(self.workload_history) < 10:
            return {
                'predicted_peak_load': len(self.active_resources),
                'predicted_min_load': len(self.active_resources),
                'time_to_peak_minutes': 60.0
            }
        
        # Simple moving average forecast
        recent_loads = [m.requests_per_second for m in self.workload_history[-10:]]
        avg_load = np.mean(recent_loads)
        load_std = np.std(recent_loads)
        
        # Predict peak and min loads
        predicted_peak = avg_load + 2 * load_std
        predicted_min = max(avg_load - 2 * load_std, avg_load * 0.3)
        
        return {
            'predicted_peak_load': predicted_peak / 10.0,  # Convert to instance count
            'predicted_min_load': predicted_min / 10.0,
            'time_to_peak_minutes': 30.0  # Assume 30-minute horizon
        }
    
    async def _adaptive_scaling(self) -> ScalingDecision:
        """Adaptive scaling using learned patterns."""
        
        # Use machine learning model to predict optimal scaling
        if self.scaling_model is None:
            # Fallback to reactive scaling
            return await self._reactive_scaling()
        
        # Prepare features for ML model
        features = self._extract_scaling_features()
        
        # Predict optimal action
        predicted_action = self.scaling_model.predict(features)
        
        current_instances = len(self.active_resources)
        confidence = 0.7
        
        if predicted_action > 0:  # Scale up
            action = 'scale_up'
            target_instances = current_instances + int(predicted_action)
            reasoning = ['ML model predicts scale up needed']
        elif predicted_action < 0:  # Scale down
            action = 'scale_down'
            target_instances = max(current_instances + int(predicted_action), 2)
            reasoning = ['ML model predicts scale down opportunity']
        else:
            action = 'no_change'
            target_instances = current_instances
            reasoning = ['ML model predicts no scaling needed']
        
        cost_impact = self._estimate_cost_impact(current_instances, target_instances)
        performance_impact = self._estimate_performance_impact(current_instances, target_instances)
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            reasoning=reasoning,
            confidence=confidence,
            estimated_cost_impact=cost_impact,
            estimated_performance_impact=performance_impact
        )
    
    async def _cost_optimized_scaling(self) -> ScalingDecision:
        """Cost-optimized scaling that minimizes cost while meeting SLA."""
        
        current_instances = len(self.active_resources)
        metrics = self.current_workload
        
        # Calculate current cost
        current_cost = sum(r.cost_per_hour for r in self.active_resources.values())
        
        # Check if we can reduce cost while maintaining SLA
        sla_compliance = metrics.sla_compliance()
        
        reasoning = []
        action = 'no_change'
        target_instances = current_instances
        confidence = 0.8
        
        if sla_compliance > 0.99 and current_instances > 2:
            # Try to reduce cost
            action = 'scale_down'
            target_instances = current_instances - 1
            reasoning.append(f"SLA compliance excellent ({sla_compliance:.1%}), optimizing cost")
            reasoning.append(f"Current cost: ${current_cost:.2f}/hour")
        
        elif sla_compliance < 0.95:
            # Need to improve performance
            action = 'scale_up'
            target_instances = current_instances + 1
            reasoning.append(f"SLA compliance poor ({sla_compliance:.1%}), adding capacity")
        
        cost_impact = self._estimate_cost_impact(current_instances, target_instances)
        performance_impact = self._estimate_performance_impact(current_instances, target_instances)
        
        return ScalingDecision(
            action=action,
            target_instances=target_instances,
            reasoning=reasoning,
            confidence=confidence,
            estimated_cost_impact=cost_impact,
            estimated_performance_impact=performance_impact
        )
    
    def _estimate_cost_impact(self, current_instances: int, target_instances: int) -> float:
        """Estimate cost impact of scaling decision."""
        
        if not self.active_resources:
            return 0.0
        
        avg_cost_per_instance = np.mean([r.cost_per_hour for r in self.active_resources.values()])
        cost_change = (target_instances - current_instances) * avg_cost_per_instance
        
        return cost_change
    
    def _estimate_performance_impact(self, current_instances: int, target_instances: int) -> float:
        """Estimate performance impact of scaling decision."""
        
        if current_instances == 0:
            return 0.0
        
        # Simple linear relationship between instances and performance
        performance_ratio = target_instances / current_instances
        
        # Performance improvement diminishes with more instances
        performance_impact = (performance_ratio - 1.0) * 0.8
        
        return performance_impact
    
    async def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute the scaling decision."""
        
        current_instances = len(self.active_resources)
        
        if decision.action == 'scale_up':
            instances_to_add = decision.target_instances - current_instances
            await self._scale_up(instances_to_add)
        
        elif decision.action == 'scale_down':
            instances_to_remove = current_instances - decision.target_instances
            await self._scale_down(instances_to_remove)
        
        logger.info(f"Executed scaling decision: {decision.action} to {decision.target_instances} instances")
    
    async def _scale_up(self, instances_to_add: int):
        """Scale up by adding instances."""
        
        # Find best available resources
        with self.resource_pool_lock:
            available = [r for r in self.available_resources.values() if not r.active]
            best_resources = sorted(available, key=lambda r: r.cost_efficiency(), reverse=True)
            
            resources_to_provision = best_resources[:instances_to_add]
        
        # Provision new resources
        if resources_to_provision:
            await self._provision_resources(resources_to_provision)
    
    async def _scale_down(self, instances_to_remove: int):
        """Scale down by removing instances."""
        
        with self.resource_pool_lock:
            # Select least efficient resources to remove
            active_resources = list(self.active_resources.values())
            resources_to_remove = sorted(active_resources, key=lambda r: r.cost_efficiency())[:instances_to_remove]
            
            # Remove from active pool
            for resource in resources_to_remove:
                for resource_id, active_resource in list(self.active_resources.items()):
                    if active_resource == resource:
                        del self.active_resources[resource_id]
                        break
        
        # Simulate resource termination
        for resource in resources_to_remove:
            await self._terminate_resource(resource)
    
    async def _terminate_resource(self, resource: CloudResource):
        """Terminate a cloud resource."""
        
        # Simulate termination delay
        await asyncio.sleep(0.1)
        resource.active = False
        
        logger.info(f"Terminated resource: {resource.instance_type} in {resource.region}")
    
    def _initialize_adaptive_scaling_model(self):
        """Initialize machine learning model for adaptive scaling."""
        
        # Simplified ML model (in practice, would use sklearn or similar)
        class SimpleScalingModel:
            def predict(self, features):
                # Simple rule-based prediction
                cpu_util, mem_util, latency = features[:3]
                
                if cpu_util > 0.8 or mem_util > 0.8 or latency > 100:
                    return 1  # Scale up
                elif cpu_util < 0.3 and mem_util < 0.3:
                    return -1  # Scale down
                else:
                    return 0  # No change
        
        self.scaling_model = SimpleScalingModel()
        logger.info("Adaptive scaling model initialized")
    
    def _extract_scaling_features(self) -> List[float]:
        """Extract features for ML model."""
        
        if not self.current_workload:
            return [0.0] * 8
        
        m = self.current_workload
        
        features = [
            m.cpu_utilization,
            m.memory_utilization,
            m.average_latency_ms,
            m.requests_per_second,
            m.error_rate,
            len(self.active_resources),
            time.time() % (24 * 3600) / (24 * 3600),  # Time of day (normalized)
            len(self.workload_history)
        ]
        
        return features
    
    def _start_background_tasks(self):
        """Start background monitoring and optimization tasks."""
        
        async def monitoring_loop():
            while True:
                try:
                    await self._collect_workload_metrics()
                    await asyncio.sleep(30)  # Collect metrics every 30 seconds
                except Exception as e:
                    logger.warning(f"Monitoring loop error: {str(e)}")
                    await asyncio.sleep(60)
        
        async def scaling_loop():
            while True:
                try:
                    await self.auto_scale()
                    await asyncio.sleep(60)  # Scale check every minute
                except Exception as e:
                    logger.warning(f"Scaling loop error: {str(e)}")
                    await asyncio.sleep(120)
        
        async def cost_optimization_loop():
            while True:
                try:
                    await self._optimize_costs()
                    await asyncio.sleep(3600)  # Optimize costs every hour
                except Exception as e:
                    logger.warning(f"Cost optimization error: {str(e)}")
                    await asyncio.sleep(1800)
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(monitoring_loop())
        self._scaling_task = asyncio.create_task(scaling_loop())
        self._cost_optimization_task = asyncio.create_task(cost_optimization_loop())
    
    async def _collect_workload_metrics(self):
        """Collect current workload metrics."""
        
        # Simulate metric collection (in practice, would query monitoring systems)
        if self.active_resources:
            self.current_workload = WorkloadMetrics(
                timestamp=time.time(),
                requests_per_second=np.random.uniform(10, 100),
                average_latency_ms=np.random.uniform(20, 150),
                p95_latency_ms=np.random.uniform(50, 200),
                p99_latency_ms=np.random.uniform(100, 300),
                error_rate=np.random.uniform(0, 0.05),
                cpu_utilization=np.random.uniform(0.2, 0.9),
                memory_utilization=np.random.uniform(0.3, 0.8),
                network_utilization=np.random.uniform(0.1, 0.6),
                queue_length=int(np.random.uniform(0, 10))
            )
            
            self.workload_history.append(self.current_workload)
            
            # Keep only recent history
            if len(self.workload_history) > 1000:
                self.workload_history = self.workload_history[-1000:]
    
    async def _optimize_costs(self):
        """Periodic cost optimization."""
        
        logger.info("Running cost optimization")
        
        # Analyze cost patterns
        total_cost = sum(r.cost_per_hour for r in self.active_resources.values())
        
        if total_cost > self.cost_budget_per_hour:
            logger.warning(f"Cost ${total_cost:.2f}/hour exceeds budget ${self.cost_budget_per_hour:.2f}/hour")
            
            # Find cost reduction opportunities
            await self._implement_cost_reductions()
    
    async def _implement_cost_reductions(self):
        """Implement cost reduction strategies."""
        
        with self.resource_pool_lock:
            # Replace expensive instances with cheaper alternatives
            expensive_resources = [r for r in self.active_resources.values() if r.cost_per_hour > 1.0]
            
            for resource in expensive_resources:
                # Look for cheaper alternative
                cheaper_alternatives = [
                    r for r in self.available_resources.values()
                    if (r.cost_per_hour < resource.cost_per_hour * 0.8 and
                        r.compute_units >= resource.compute_units * 0.8)
                ]
                
                if cheaper_alternatives:
                    # Replace with cheaper resource
                    await self._replace_resource(resource, cheaper_alternatives[0])
    
    async def _replace_resource(self, old_resource: CloudResource, new_resource: CloudResource):
        """Replace one resource with another."""
        
        # Provision new resource
        provisioned_new = await self._provision_single_resource(new_resource)
        
        # Add to active pool
        with self.resource_pool_lock:
            new_resource_id = f"{new_resource.provider.value}_{new_resource.region}_{int(time.time())}"
            self.active_resources[new_resource_id] = provisioned_new
            
            # Remove old resource
            for resource_id, active_resource in list(self.active_resources.items()):
                if active_resource == old_resource:
                    del self.active_resources[resource_id]
                    break
        
        # Terminate old resource
        await self._terminate_resource(old_resource)
        
        logger.info(f"Replaced {old_resource.instance_type} with {new_resource.instance_type}")
    
    def get_orchestration_status(self) -> Dict[str, Any]:
        """Get current orchestration status."""
        
        total_cost = sum(r.cost_per_hour for r in self.active_resources.values())
        total_compute = sum(r.compute_units for r in self.active_resources.values())
        
        return {
            'active_resources': len(self.active_resources),
            'available_resources': len(self.available_resources),
            'total_cost_per_hour': total_cost,
            'total_compute_units': total_compute,
            'cost_budget_utilization': total_cost / self.cost_budget_per_hour,
            'scaling_policy': self.scaling_policy.value,
            'recent_scaling_actions': len([s for s in self.scaling_history[-10:] if s.action != 'no_change']),
            'current_workload': {
                'requests_per_second': self.current_workload.requests_per_second if self.current_workload else 0,
                'average_latency_ms': self.current_workload.average_latency_ms if self.current_workload else 0,
                'sla_compliance': self.current_workload.sla_compliance() if self.current_workload else 1.0
            }
        }
    
    async def shutdown(self):
        """Shutdown orchestrator and clean up resources."""
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._scaling_task:
            self._scaling_task.cancel()
        if self._cost_optimization_task:
            self._cost_optimization_task.cancel()
        
        # Terminate all active resources
        for resource in list(self.active_resources.values()):
            await self._terminate_resource(resource)
        
        self.active_resources.clear()
        
        logger.info("CloudOrchestrator shut down successfully")


class GlobalLoadBalancer:
    """
    Global load balancer for multi-region deployments.
    
    Implements intelligent traffic routing based on latency,
    cost, and resource utilization.
    """
    
    def __init__(self, orchestrator: CloudOrchestrator):
        self.orchestrator = orchestrator
        self.routing_table: Dict[str, List[str]] = {}
        self.health_checks: Dict[str, bool] = {}
        
        logger.info("Initialized GlobalLoadBalancer")
    
    async def configure_for_deployment(self, deployment_result: Dict[str, Any]):
        """Configure load balancer for new deployment."""
        
        endpoints = deployment_result['endpoints']
        deployment_id = deployment_result['deployment_id']
        
        # Add endpoints to routing table
        self.routing_table[deployment_id] = endpoints
        
        # Initialize health checks
        for endpoint in endpoints:
            self.health_checks[endpoint] = True
        
        logger.info(f"Configured load balancer for deployment {deployment_id} with {len(endpoints)} endpoints")
    
    async def route_request(self, request_metadata: Dict[str, Any]) -> str:
        """Route request to optimal endpoint."""
        
        # Select deployment (for simplicity, use first available)
        if not self.routing_table:
            raise SpintronError("No deployments available")
        
        deployment_id = list(self.routing_table.keys())[0]
        endpoints = self.routing_table[deployment_id]
        
        # Filter healthy endpoints
        healthy_endpoints = [ep for ep in endpoints if self.health_checks.get(ep, False)]
        
        if not healthy_endpoints:
            raise SpintronError("No healthy endpoints available")
        
        # Simple round-robin selection (in practice, would use sophisticated routing)
        selected_endpoint = healthy_endpoints[int(time.time()) % len(healthy_endpoints)]
        
        return selected_endpoint
