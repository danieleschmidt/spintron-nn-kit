"""
Distributed computing framework for spintronic neural networks.
Enables scaling across multiple nodes with intelligent load balancing.
"""

import time
import json
import hashlib
import threading
from typing import Dict, List, Any, Optional, Tuple, Callable, Set
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
from enum import Enum
import queue
import concurrent.futures


class NodeRole(Enum):
    """Roles for distributed nodes."""
    COORDINATOR = "coordinator"
    WORKER = "worker"
    STORAGE = "storage"
    HYBRID = "hybrid"


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ComputeNode:
    """Compute node in distributed system."""
    node_id: str
    role: NodeRole
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    max_capacity: float = 100.0
    available_memory_mb: float = 1000.0
    processing_units: int = 4
    network_bandwidth_mbps: float = 1000.0
    last_heartbeat: float = 0.0
    status: str = "active"
    tasks_completed: int = 0
    tasks_failed: int = 0


@dataclass
class DistributedTask:
    """Task for distributed execution."""
    task_id: str
    task_type: str
    priority: TaskPriority
    data: Dict[str, Any]
    required_memory_mb: float
    estimated_duration_ms: float
    dependencies: List[str]
    created_time: float
    assigned_node: Optional[str] = None
    started_time: Optional[float] = None
    completed_time: Optional[float] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class LoadBalancingMetrics:
    """Metrics for load balancing decisions."""
    cpu_utilization: float
    memory_utilization: float
    network_utilization: float
    task_queue_length: int
    average_response_time_ms: float
    throughput_tasks_per_sec: float


class DistributedSpintronicsFramework:
    """Distributed framework for spintronic neural network processing."""
    
    def __init__(self, node_id: str, role: NodeRole = NodeRole.HYBRID):
        self.node_id = node_id
        self.role = role
        
        # Node management
        self.nodes: Dict[str, ComputeNode] = {}
        self.local_node = ComputeNode(
            node_id=node_id,
            role=role,
            capabilities={
                "mtj_simulation": True,
                "crossbar_computation": True,
                "verilog_generation": True,
                "neural_inference": True
            }
        )
        self.nodes[node_id] = self.local_node
        
        # Task management
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: Dict[str, DistributedTask] = {}
        self.task_history: List[DistributedTask] = []
        
        # Load balancing
        self.load_balancer = IntelligentLoadBalancer()
        self.resource_monitor = ResourceMonitor()
        
        # Threading
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.worker_threads: List[threading.Thread] = []
        
        # Network simulation (in real implementation, this would be actual network layer)
        self.message_queue = queue.Queue()
        self.network_latency_ms = 1.0
        
        # Performance tracking
        self.performance_metrics: List[Dict[str, Any]] = []
        self.start_time = time.time()
        
    def register_node(self, node: ComputeNode) -> bool:
        """Register a new compute node."""
        if node.node_id in self.nodes:
            return False
            
        self.nodes[node.node_id] = node
        node.last_heartbeat = time.time()
        
        print(f"Node {node.node_id} registered with role {node.role.value}")
        return True
        
    def unregister_node(self, node_id: str) -> bool:
        """Unregister a compute node."""
        if node_id not in self.nodes or node_id == self.node_id:
            return False
            
        # Reassign tasks from the removed node
        self._reassign_node_tasks(node_id)
        
        del self.nodes[node_id]
        print(f"Node {node_id} unregistered")
        return True
        
    def _reassign_node_tasks(self, failed_node_id: str) -> None:
        """Reassign tasks from a failed node."""
        tasks_to_reassign = []
        
        for task in self.active_tasks.values():
            if task.assigned_node == failed_node_id:
                task.assigned_node = None
                task.retry_count += 1
                if task.retry_count <= task.max_retries:
                    tasks_to_reassign.append(task)
                else:
                    task.error = f"Max retries exceeded after node {failed_node_id} failure"
                    self.completed_tasks[task.task_id] = task
                    
        for task in tasks_to_reassign:
            self.submit_task(task)
            
    def submit_task(self, task: DistributedTask) -> str:
        """Submit task for distributed execution."""
        if task.task_id in self.active_tasks or task.task_id in self.completed_tasks:
            return task.task_id
            
        task.created_time = time.time()
        
        # Select best node for task
        best_node = self.load_balancer.select_node(task, self.nodes)
        if best_node:
            task.assigned_node = best_node.node_id
            
        # Add to queue with priority
        priority_value = (5 - task.priority.value, task.created_time)
        self.task_queue.put((priority_value, task))
        self.active_tasks[task.task_id] = task
        
        print(f"Task {task.task_id} submitted (priority: {task.priority.value})")
        return task.task_id
        
    def create_spintronic_task(self, 
                              task_type: str,
                              data: Dict[str, Any],
                              priority: TaskPriority = TaskPriority.NORMAL,
                              estimated_duration_ms: float = 1000.0) -> DistributedTask:
        """Create a spintronic-specific task."""
        task_id = hashlib.md5(f"{task_type}_{time.time()}_{self.node_id}".encode()).hexdigest()[:12]
        
        # Estimate resource requirements based on task type
        memory_requirements = {
            "mtj_simulation": 50.0,
            "crossbar_computation": 100.0,
            "verilog_generation": 200.0,
            "neural_inference": 150.0,
            "weight_mapping": 75.0,
            "performance_analysis": 25.0
        }
        
        required_memory = memory_requirements.get(task_type, 50.0)
        
        return DistributedTask(
            task_id=task_id,
            task_type=task_type,
            priority=priority,
            data=data,
            required_memory_mb=required_memory,
            estimated_duration_ms=estimated_duration_ms,
            dependencies=[],
            created_time=time.time()
        )
        
    def execute_task(self, task: DistributedTask) -> Any:
        """Execute a specific task based on its type."""
        task.started_time = time.time()
        
        try:
            if task.task_type == "mtj_simulation":
                result = self._execute_mtj_simulation(task.data)
            elif task.task_type == "crossbar_computation":
                result = self._execute_crossbar_computation(task.data)
            elif task.task_type == "verilog_generation":
                result = self._execute_verilog_generation(task.data)
            elif task.task_type == "neural_inference":
                result = self._execute_neural_inference(task.data)
            elif task.task_type == "weight_mapping":
                result = self._execute_weight_mapping(task.data)
            elif task.task_type == "performance_analysis":
                result = self._execute_performance_analysis(task.data)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
            task.completed_time = time.time()
            task.result = result
            
            # Update node statistics
            if task.assigned_node in self.nodes:
                self.nodes[task.assigned_node].tasks_completed += 1
                
            return result
            
        except Exception as e:
            task.error = str(e)
            task.completed_time = time.time()
            
            if task.assigned_node in self.nodes:
                self.nodes[task.assigned_node].tasks_failed += 1
                
            raise
            
    def _execute_mtj_simulation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute MTJ device simulation."""
        # Simulate MTJ device behavior
        resistance_high = data.get('resistance_high', 10000)
        resistance_low = data.get('resistance_low', 5000)
        switching_voltage = data.get('switching_voltage', 0.3)
        
        # Simulate computation time
        time.sleep(data.get('simulation_time', 0.01))
        
        # Calculate results
        resistance_ratio = resistance_high / resistance_low
        switching_energy = (switching_voltage ** 2) / resistance_low * 1e12  # pJ
        
        return {
            'resistance_ratio': resistance_ratio,
            'switching_energy_pj': switching_energy,
            'stability_factor': resistance_ratio * 26,  # kT units
            'simulation_status': 'completed'
        }
        
    def _execute_crossbar_computation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute crossbar array computation."""
        rows = data.get('rows', 128)
        cols = data.get('cols', 128)
        weights = data.get('weights', [[1.0] * cols for _ in range(rows)])
        inputs = data.get('inputs', [1.0] * cols)
        
        # Simulate matrix-vector multiplication
        time.sleep(len(weights) * len(inputs) / 1000000)  # Simulate computation
        
        outputs = []
        for row in weights:
            output = sum(w * i for w, i in zip(row, inputs))
            outputs.append(output)
            
        return {
            'outputs': outputs,
            'computation_time_ms': len(weights) * len(inputs) / 10000,
            'energy_consumption_nj': len(weights) * len(inputs) * 10e-3,
            'status': 'completed'
        }
        
    def _execute_verilog_generation(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute Verilog hardware generation."""
        module_name = data.get('module_name', 'spintron_module')
        crossbar_size = data.get('crossbar_size', (64, 64))
        target_frequency = data.get('target_frequency', 50e6)
        
        # Simulate Verilog generation
        time.sleep(0.05)  # Simulate generation time
        
        verilog_code = f"""
module {module_name} (
    input clk,
    input rst,
    input [{crossbar_size[1]-1}:0] inputs,
    output [{crossbar_size[0]-1}:0] outputs
);
    // Spintronic crossbar array implementation
    // Generated for {crossbar_size[0]}x{crossbar_size[1]} array
    // Target frequency: {target_frequency/1e6:.1f} MHz
endmodule
"""
        
        return {
            'verilog_code': verilog_code,
            'module_name': module_name,
            'estimated_area_mm2': (crossbar_size[0] * crossbar_size[1]) / 10000,
            'estimated_power_mw': target_frequency / 1e6 * 10,
            'status': 'completed'
        }
        
    def _execute_neural_inference(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute neural network inference."""
        input_data = data.get('input_data', [1.0] * 784)
        layer_sizes = data.get('layer_sizes', [784, 128, 64, 10])
        
        # Simulate neural network forward pass
        time.sleep(len(layer_sizes) * 0.001)
        
        current_output = input_data
        for i in range(len(layer_sizes) - 1):
            # Simulate layer computation
            next_size = layer_sizes[i + 1]
            current_output = [sum(current_output) / len(current_output)] * next_size
            
        return {
            'output': current_output,
            'inference_time_ms': len(layer_sizes) * 2,
            'energy_consumption_nj': sum(layer_sizes) * 10e-3,
            'accuracy_estimate': 0.95,
            'status': 'completed'
        }
        
    def _execute_weight_mapping(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute weight mapping to crossbar arrays."""
        weights = data.get('weights', [[1.0, -0.5], [0.3, 0.8]])
        crossbar_size = data.get('crossbar_size', (64, 64))
        quantization_bits = data.get('quantization_bits', 4)
        
        # Simulate weight mapping
        time.sleep(0.02)
        
        num_levels = 2 ** quantization_bits
        mapped_weights = []
        
        for row in weights:
            mapped_row = []
            for weight in row:
                # Quantize weight
                quantized = round((weight + 1) / 2 * (num_levels - 1))
                quantized = max(0, min(quantized, num_levels - 1))
                mapped_row.append(quantized)
            mapped_weights.append(mapped_row)
            
        return {
            'mapped_weights': mapped_weights,
            'quantization_levels': num_levels,
            'mapping_efficiency': 0.92,
            'crossbar_utilization': len(weights) * len(weights[0]) / (crossbar_size[0] * crossbar_size[1]),
            'status': 'completed'
        }
        
    def _execute_performance_analysis(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Execute performance analysis."""
        metrics = data.get('metrics', {})
        
        # Simulate analysis
        time.sleep(0.01)
        
        analysis_result = {
            'throughput_analysis': {
                'peak_throughput_ops_per_sec': 10000,
                'average_throughput_ops_per_sec': 7500,
                'bottleneck_analysis': 'Memory bandwidth limited'
            },
            'energy_analysis': {
                'total_energy_consumption_nj': 1250.5,
                'energy_per_operation_pj': 12.5,
                'energy_efficiency_tops_per_watt': 80.5
            },
            'latency_analysis': {
                'average_latency_ms': 15.2,
                'p95_latency_ms': 23.8,
                'p99_latency_ms': 31.4
            },
            'recommendations': [
                'Increase cache size to improve memory bandwidth',
                'Consider voltage scaling for energy optimization',
                'Implement pipelining for latency reduction'
            ],
            'status': 'completed'
        }
        
        return analysis_result
        
    def start_processing(self) -> None:
        """Start distributed task processing."""
        if self.running:
            return
            
        self.running = True
        
        # Start worker threads
        for i in range(self.local_node.processing_units):
            worker_thread = threading.Thread(
                target=self._worker_loop,
                name=f"Worker-{i}",
                daemon=True
            )
            worker_thread.start()
            self.worker_threads.append(worker_thread)
            
        # Start monitoring thread
        monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="Monitor",
            daemon=True
        )
        monitor_thread.start()
        self.worker_threads.append(monitor_thread)
        
        print(f"Started processing with {len(self.worker_threads)} threads")
        
    def stop_processing(self) -> None:
        """Stop distributed task processing."""
        self.running = False
        print("Stopping distributed processing...")
        
    def _worker_loop(self) -> None:
        """Main worker loop for processing tasks."""
        while self.running:
            try:
                # Get task from queue with timeout
                priority_task = self.task_queue.get(timeout=1.0)
                _, task = priority_task
                
                # Check if task is assigned to this node
                if task.assigned_node != self.node_id:
                    # Forward to appropriate node (simulated)
                    continue
                    
                # Execute task
                self.execute_task(task)
                
                # Move to completed tasks
                if task.task_id in self.active_tasks:
                    del self.active_tasks[task.task_id]
                self.completed_tasks[task.task_id] = task
                self.task_history.append(task)
                
                # Update performance metrics
                self._update_performance_metrics(task)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Worker error: {e}")
                
    def _monitoring_loop(self) -> None:
        """Monitoring loop for system health and load balancing."""
        while self.running:
            try:
                # Update node health
                self._update_node_health()
                
                # Perform load balancing
                self.load_balancer.rebalance_if_needed(self.nodes, self.active_tasks)
                
                # Clean up old completed tasks
                self._cleanup_old_tasks()
                
                time.sleep(5.0)  # Monitor every 5 seconds
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                
    def _update_node_health(self) -> None:
        """Update health status of all nodes."""
        current_time = time.time()
        
        for node in self.nodes.values():
            # Update local node
            if node.node_id == self.node_id:
                node.last_heartbeat = current_time
                node.current_load = len([t for t in self.active_tasks.values() 
                                       if t.assigned_node == node.node_id]) / node.max_capacity * 100
                
            # Check for dead nodes
            elif current_time - node.last_heartbeat > 30.0:  # 30 second timeout
                node.status = "dead"
                self._reassign_node_tasks(node.node_id)
                
    def _cleanup_old_tasks(self) -> None:
        """Clean up old completed tasks to free memory."""
        current_time = time.time()
        cutoff_time = current_time - 3600  # Keep tasks for 1 hour
        
        tasks_to_remove = []
        for task_id, task in self.completed_tasks.items():
            if task.completed_time and task.completed_time < cutoff_time:
                tasks_to_remove.append(task_id)
                
        for task_id in tasks_to_remove:
            del self.completed_tasks[task_id]
            
    def _update_performance_metrics(self, task: DistributedTask) -> None:
        """Update performance metrics based on completed task."""
        if task.started_time and task.completed_time:
            execution_time = task.completed_time - task.started_time
            
            metrics = {
                'timestamp': task.completed_time,
                'task_type': task.task_type,
                'execution_time_ms': execution_time * 1000,
                'memory_used_mb': task.required_memory_mb,
                'node_id': task.assigned_node,
                'success': task.error is None
            }
            
            self.performance_metrics.append(metrics)
            
            # Keep only recent metrics
            if len(self.performance_metrics) > 1000:
                self.performance_metrics = self.performance_metrics[-1000:]
                
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status."""
        current_time = time.time()
        uptime = current_time - self.start_time
        
        # Calculate task statistics
        total_tasks = len(self.completed_tasks) + len(self.active_tasks)
        completed_tasks = len(self.completed_tasks)
        success_rate = 0.0
        
        if self.completed_tasks:
            successful_tasks = sum(1 for task in self.completed_tasks.values() if task.error is None)
            success_rate = successful_tasks / len(self.completed_tasks)
            
        # Calculate throughput
        recent_tasks = [task for task in self.completed_tasks.values() 
                       if task.completed_time and task.completed_time > current_time - 60]
        throughput_per_minute = len(recent_tasks)
        
        # Node statistics
        active_nodes = sum(1 for node in self.nodes.values() if node.status == "active")
        total_capacity = sum(node.max_capacity for node in self.nodes.values())
        used_capacity = sum(node.current_load for node in self.nodes.values())
        
        return {
            'uptime_seconds': uptime,
            'node_id': self.node_id,
            'role': self.role.value,
            'nodes': {
                'total': len(self.nodes),
                'active': active_nodes,
                'total_capacity': total_capacity,
                'used_capacity': used_capacity,
                'utilization': used_capacity / total_capacity if total_capacity > 0 else 0
            },
            'tasks': {
                'total': total_tasks,
                'active': len(self.active_tasks),
                'completed': completed_tasks,
                'success_rate': success_rate,
                'throughput_per_minute': throughput_per_minute
            },
            'performance': {
                'metrics_collected': len(self.performance_metrics),
                'average_execution_time_ms': sum(m['execution_time_ms'] for m in self.performance_metrics[-100:]) / min(100, len(self.performance_metrics)) if self.performance_metrics else 0
            },
            'status': 'running' if self.running else 'stopped'
        }


class IntelligentLoadBalancer:
    """Intelligent load balancer for distributed spintronic computing."""
    
    def __init__(self):
        self.balancing_algorithm = "adaptive_weighted"
        self.rebalancing_threshold = 0.2  # 20% load difference
        self.prediction_window = 10  # seconds
        
    def select_node(self, task: DistributedTask, nodes: Dict[str, ComputeNode]) -> Optional[ComputeNode]:
        """Select the best node for a given task."""
        available_nodes = [node for node in nodes.values() 
                          if node.status == "active" and self._can_handle_task(node, task)]
        
        if not available_nodes:
            return None
            
        # Score nodes based on multiple factors
        node_scores = {}
        for node in available_nodes:
            score = self._calculate_node_score(node, task)
            node_scores[node.node_id] = (score, node)
            
        # Return node with highest score
        best_node_id = max(node_scores.keys(), key=lambda k: node_scores[k][0])
        return node_scores[best_node_id][1]
        
    def _can_handle_task(self, node: ComputeNode, task: DistributedTask) -> bool:
        """Check if node can handle the given task."""
        # Check memory requirements
        if task.required_memory_mb > node.available_memory_mb:
            return False
            
        # Check capabilities
        task_capability_map = {
            "mtj_simulation": "mtj_simulation",
            "crossbar_computation": "crossbar_computation",
            "verilog_generation": "verilog_generation",
            "neural_inference": "neural_inference"
        }
        
        required_capability = task_capability_map.get(task.task_type)
        if required_capability and not node.capabilities.get(required_capability, False):
            return False
            
        return True
        
    def _calculate_node_score(self, node: ComputeNode, task: DistributedTask) -> float:
        """Calculate suitability score for node-task pairing."""
        # Base score components
        load_score = max(0, 1.0 - (node.current_load / node.max_capacity))
        memory_score = min(1.0, node.available_memory_mb / task.required_memory_mb)
        capability_score = 1.0 if node.capabilities.get(task.task_type.replace('_', '_'), False) else 0.5
        
        # Network proximity score (simplified)
        network_score = 1.0  # In real implementation, this would consider network topology
        
        # Reliability score based on past performance
        total_tasks = node.tasks_completed + node.tasks_failed
        reliability_score = node.tasks_completed / total_tasks if total_tasks > 0 else 1.0
        
        # Combined score with weights
        score = (load_score * 0.3 + 
                memory_score * 0.2 + 
                capability_score * 0.2 + 
                network_score * 0.1 + 
                reliability_score * 0.2)
        
        return score
        
    def rebalance_if_needed(self, nodes: Dict[str, ComputeNode], active_tasks: Dict[str, DistributedTask]) -> None:
        """Perform load rebalancing if needed."""
        active_nodes = [node for node in nodes.values() if node.status == "active"]
        
        if len(active_nodes) < 2:
            return
            
        # Calculate load imbalance
        loads = [node.current_load for node in active_nodes]
        max_load = max(loads)
        min_load = min(loads)
        
        if max_load - min_load > self.rebalancing_threshold * 100:
            print(f"Load imbalance detected: {max_load:.1f}% vs {min_load:.1f}%")
            # In a real implementation, this would trigger task migration


class ResourceMonitor:
    """Monitor system resources for distributed computing."""
    
    def __init__(self):
        self.metrics_history: List[LoadBalancingMetrics] = []
        
    def collect_metrics(self, nodes: Dict[str, ComputeNode], active_tasks: Dict[str, DistributedTask]) -> LoadBalancingMetrics:
        """Collect current system metrics."""
        # Calculate aggregate metrics
        total_nodes = len([n for n in nodes.values() if n.status == "active"])
        avg_cpu = sum(n.current_load for n in nodes.values()) / total_nodes if total_nodes > 0 else 0
        avg_memory = sum(n.available_memory_mb for n in nodes.values()) / total_nodes if total_nodes > 0 else 0
        
        metrics = LoadBalancingMetrics(
            cpu_utilization=avg_cpu,
            memory_utilization=100 - avg_memory,  # Simplified
            network_utilization=20.0,  # Mock value
            task_queue_length=len(active_tasks),
            average_response_time_ms=50.0,  # Mock value
            throughput_tasks_per_sec=10.0  # Mock value
        )
        
        self.metrics_history.append(metrics)
        
        # Keep only recent metrics
        if len(self.metrics_history) > 100:
            self.metrics_history = self.metrics_history[-100:]
            
        return metrics