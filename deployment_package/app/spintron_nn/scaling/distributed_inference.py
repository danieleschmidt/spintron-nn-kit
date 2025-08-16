"""
Advanced Distributed Inference System for SpinTron-NN-Kit.

This module provides high-performance distributed inference capabilities:
- Dynamic inference pipeline partitioning
- Intelligent task scheduling and load balancing
- Fault-tolerant worker management
- Real-time performance optimization
- Energy-aware workload distribution
"""

import asyncio
import time
import json
import threading
from typing import Dict, List, Optional, Any, Callable, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque
import uuid
import statistics
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import multiprocessing as mp

from ..utils.monitoring import get_system_monitor
from ..utils.error_handling import SpintronError, robust_operation
from .auto_scaler import ScalingConfig, InstanceMetrics


class TaskPriority(Enum):
    """Task priority levels."""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class WorkerStatus(Enum):
    """Worker status states."""
    INITIALIZING = "initializing"
    IDLE = "idle"
    BUSY = "busy"
    ERROR = "error"
    OFFLINE = "offline"


@dataclass
class InferenceTask:
    """Represents a single inference task."""
    task_id: str
    model_path: str
    input_data: Any
    priority: TaskPriority = TaskPriority.NORMAL
    created_at: float = None
    deadline: Optional[float] = None
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskResult:
    """Result of an inference task."""
    task_id: str
    result: Any
    worker_id: str
    execution_time: float
    energy_consumed: float
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class WorkerConfig:
    """Configuration for inference workers."""
    worker_id: str
    max_concurrent_tasks: int = 1
    supported_models: List[str] = None
    gpu_memory_gb: float = 8.0
    energy_efficiency_pj_per_mac: float = 15.0
    heartbeat_interval: float = 10.0
    
    def __post_init__(self):
        if self.supported_models is None:
            self.supported_models = []


class DistributedTaskScheduler:
    """Intelligent task scheduler for distributed inference."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.monitor = get_system_monitor()
        
        # Task queues by priority
        self.task_queues = {
            TaskPriority.CRITICAL: queue.PriorityQueue(),
            TaskPriority.HIGH: queue.PriorityQueue(),
            TaskPriority.NORMAL: queue.PriorityQueue(),
            TaskPriority.LOW: queue.PriorityQueue()
        }
        
        # Worker registry
        self.workers = {}
        self.worker_metrics = {}
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = deque(maxlen=1000)
        
        # Scheduling statistics
        self.scheduling_stats = {
            'tasks_scheduled': 0,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'average_queue_time': 0.0,
            'average_execution_time': 0.0
        }
        
        # Scheduling thread
        self.running = False
        self.scheduler_thread = None
        
    def start(self):
        """Start the task scheduler."""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduling_loop, daemon=True)
        self.scheduler_thread.start()
    
    def stop(self):
        """Stop the task scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5.0)
    
    def register_worker(self, config: WorkerConfig):
        """Register a new worker."""
        self.workers[config.worker_id] = {
            'config': config,
            'status': WorkerStatus.IDLE,
            'current_tasks': set(),
            'last_heartbeat': time.time(),
            'total_tasks_completed': 0,
            'total_execution_time': 0.0,
            'total_energy_consumed': 0.0
        }
        
        print(f"Worker {config.worker_id} registered")
    
    def unregister_worker(self, worker_id: str):
        """Unregister a worker."""
        if worker_id in self.workers:
            # Reassign active tasks
            worker_info = self.workers[worker_id]
            for task_id in worker_info['current_tasks']:
                if task_id in self.active_tasks:
                    task = self.active_tasks[task_id]
                    self.submit_task(task)  # Resubmit the task
            
            del self.workers[worker_id]
            print(f"Worker {worker_id} unregistered")
    
    def submit_task(self, task: InferenceTask) -> str:
        """Submit a task for execution."""
        if task.task_id is None:
            task.task_id = str(uuid.uuid4())
        
        # Add to appropriate priority queue
        priority_value = task.priority.value
        queue_item = (-priority_value, task.created_at, task)  # Negative for max-heap behavior
        
        self.task_queues[task.priority].put(queue_item)
        self.scheduling_stats['tasks_scheduled'] += 1
        
        print(f"Task {task.task_id} submitted with priority {task.priority.name}")
        return task.task_id
    
    def _scheduling_loop(self):
        """Main scheduling loop."""
        while self.running:
            try:
                # Process tasks by priority
                for priority in [TaskPriority.CRITICAL, TaskPriority.HIGH, 
                               TaskPriority.NORMAL, TaskPriority.LOW]:
                    
                    if not self.task_queues[priority].empty():
                        try:
                            # Get task from queue (non-blocking)
                            _, created_at, task = self.task_queues[priority].get_nowait()
                            
                            # Find optimal worker
                            worker_id = self._select_optimal_worker(task)
                            
                            if worker_id:
                                self._assign_task_to_worker(task, worker_id)
                            else:
                                # No available worker, put task back
                                queue_item = (-task.priority.value, created_at, task)
                                self.task_queues[priority].put(queue_item)
                                
                        except queue.Empty:
                            continue
                
                # Update worker heartbeats and status
                self._update_worker_status()
                
                # Sleep before next iteration
                time.sleep(0.1)  # 100ms scheduling cycle
                
            except Exception as e:
                print(f"Error in scheduling loop: {e}")
                time.sleep(1.0)
    
    def _select_optimal_worker(self, task: InferenceTask) -> Optional[str]:
        """Select optimal worker for task using intelligent scheduling."""
        available_workers = []
        
        for worker_id, worker_info in self.workers.items():
            config = worker_info['config']
            
            # Check if worker is available
            if worker_info['status'] != WorkerStatus.IDLE:
                continue
            
            # Check if worker can handle more tasks
            if len(worker_info['current_tasks']) >= config.max_concurrent_tasks:
                continue
            
            # Check model compatibility
            if (hasattr(task, 'model_path') and task.model_path and 
                config.supported_models and 
                task.model_path not in config.supported_models):
                continue
            
            # Calculate worker score based on multiple factors
            score = self._calculate_worker_score(worker_id, task)
            available_workers.append((worker_id, score))
        
        if not available_workers:
            return None
        
        # Select worker with highest score
        available_workers.sort(key=lambda x: x[1], reverse=True)
        return available_workers[0][0]
    
    def _calculate_worker_score(self, worker_id: str, task: InferenceTask) -> float:
        """Calculate suitability score for worker-task assignment."""
        worker_info = self.workers[worker_id]
        config = worker_info['config']
        
        score = 100.0  # Base score
        
        # Energy efficiency factor (higher is better)
        if config.energy_efficiency_pj_per_mac > 0:
            efficiency_score = 20.0 / config.energy_efficiency_pj_per_mac  # Inverse relationship
            score += efficiency_score * 10
        
        # Current load factor (lower load is better)
        current_load = len(worker_info['current_tasks']) / config.max_concurrent_tasks
        score -= current_load * 20
        
        # Performance history factor
        if worker_info['total_tasks_completed'] > 0:
            avg_execution_time = worker_info['total_execution_time'] / worker_info['total_tasks_completed']
            # Lower execution time is better
            score += max(0, (10.0 - avg_execution_time)) * 5
        
        # Deadline urgency factor
        if task.deadline:
            time_remaining = task.deadline - time.time()
            if time_remaining < 30:  # Less than 30 seconds
                score += 50  # Urgent priority
            elif time_remaining < 60:  # Less than 60 seconds
                score += 25  # High priority
        
        # Recent heartbeat factor (fresher is better)
        time_since_heartbeat = time.time() - worker_info['last_heartbeat']
        if time_since_heartbeat < 10:
            score += 10
        elif time_since_heartbeat > 30:
            score -= 20
        
        return score
    
    def _assign_task_to_worker(self, task: InferenceTask, worker_id: str):
        """Assign task to specific worker."""
        self.active_tasks[task.task_id] = task
        self.workers[worker_id]['current_tasks'].add(task.task_id)
        self.workers[worker_id]['status'] = WorkerStatus.BUSY
        
        # Calculate queue time
        queue_time = time.time() - task.created_at
        self.scheduling_stats['average_queue_time'] = (
            (self.scheduling_stats['average_queue_time'] * self.scheduling_stats['tasks_completed'] + queue_time) /
            (self.scheduling_stats['tasks_completed'] + 1)
        )
        
        print(f"Task {task.task_id} assigned to worker {worker_id} (queue time: {queue_time:.2f}s)")
        
        # Record assignment in monitoring
        self.monitor.record_operation(
            "task_assignment",
            queue_time,
            success=True,
            tags={
                "worker_id": worker_id,
                "task_priority": task.priority.name,
                "queue_time_ms": str(int(queue_time * 1000))
            }
        )
    
    def complete_task(self, result: TaskResult):
        """Mark task as completed."""
        task_id = result.task_id
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            del self.active_tasks[task_id]
            
            # Update worker status
            if result.worker_id in self.workers:
                worker_info = self.workers[result.worker_id]
                worker_info['current_tasks'].discard(task_id)
                
                if result.success:
                    worker_info['total_tasks_completed'] += 1
                    worker_info['total_execution_time'] += result.execution_time
                    worker_info['total_energy_consumed'] += result.energy_consumed
                    self.scheduling_stats['tasks_completed'] += 1
                else:
                    self.scheduling_stats['tasks_failed'] += 1
                
                # Update worker status
                if len(worker_info['current_tasks']) == 0:
                    worker_info['status'] = WorkerStatus.IDLE
            
            # Update execution time statistics
            self.scheduling_stats['average_execution_time'] = (
                (self.scheduling_stats['average_execution_time'] * (self.scheduling_stats['tasks_completed'] - 1) + 
                 result.execution_time) / self.scheduling_stats['tasks_completed']
            ) if self.scheduling_stats['tasks_completed'] > 0 else result.execution_time
            
            # Store completed task
            self.completed_tasks.append(result)
            
            # Call task callback if provided
            if task.callback and result.success:
                try:
                    task.callback(result)
                except Exception as e:
                    print(f"Error in task callback: {e}")
            
            print(f"Task {task_id} completed by worker {result.worker_id} "
                  f"(execution time: {result.execution_time:.2f}s, success: {result.success})")
            
            # Record completion in monitoring
            self.monitor.record_operation(
                "task_completion",
                result.execution_time,
                success=result.success,
                tags={
                    "worker_id": result.worker_id,
                    "energy_consumed_pj": str(int(result.energy_consumed * 1000))
                }
            )
    
    def _update_worker_status(self):
        """Update worker status based on heartbeats."""
        current_time = time.time()
        
        for worker_id, worker_info in self.workers.items():
            time_since_heartbeat = current_time - worker_info['last_heartbeat']
            
            if time_since_heartbeat > 60:  # No heartbeat for 60 seconds
                if worker_info['status'] != WorkerStatus.OFFLINE:
                    worker_info['status'] = WorkerStatus.OFFLINE
                    print(f"Worker {worker_id} marked as offline")
            elif time_since_heartbeat > 30:  # No heartbeat for 30 seconds
                if worker_info['status'] == WorkerStatus.IDLE:
                    worker_info['status'] = WorkerStatus.ERROR
                    print(f"Worker {worker_id} may have issues")
    
    def heartbeat(self, worker_id: str, metrics: Optional[InstanceMetrics] = None):
        """Receive heartbeat from worker."""
        if worker_id in self.workers:
            self.workers[worker_id]['last_heartbeat'] = time.time()
            
            # Update status if worker was offline
            if self.workers[worker_id]['status'] == WorkerStatus.OFFLINE:
                self.workers[worker_id]['status'] = WorkerStatus.IDLE
                print(f"Worker {worker_id} back online")
            
            # Store metrics if provided
            if metrics:
                self.worker_metrics[worker_id] = metrics
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status and statistics."""
        # Queue lengths
        queue_lengths = {}
        for priority, task_queue in self.task_queues.items():
            queue_lengths[priority.name] = task_queue.qsize()
        
        # Worker status summary
        worker_status_counts = {}
        for status in WorkerStatus:
            worker_status_counts[status.name] = sum(
                1 for w in self.workers.values() if w['status'] == status
            )
        
        # Recent performance metrics
        recent_tasks = list(self.completed_tasks)[-100:]  # Last 100 tasks
        recent_execution_times = [t.execution_time for t in recent_tasks if t.success]
        recent_energy_consumption = [t.energy_consumed for t in recent_tasks if t.success]
        
        return {
            'scheduler_running': self.running,
            'total_workers': len(self.workers),
            'worker_status_counts': worker_status_counts,
            'queue_lengths': queue_lengths,
            'active_tasks': len(self.active_tasks),
            'scheduling_statistics': self.scheduling_stats,
            'recent_performance': {
                'avg_execution_time': statistics.mean(recent_execution_times) if recent_execution_times else 0,
                'avg_energy_consumption': statistics.mean(recent_energy_consumption) if recent_energy_consumption else 0,
                'task_success_rate': len(recent_execution_times) / max(1, len(recent_tasks)),
                'total_recent_tasks': len(recent_tasks)
            }
        }
    
    def get_worker_status(self, worker_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed status for specific worker."""
        if worker_id not in self.workers:
            return None
        
        worker_info = self.workers[worker_id]
        metrics = self.worker_metrics.get(worker_id)
        
        return {
            'worker_id': worker_id,
            'config': asdict(worker_info['config']),
            'status': worker_info['status'].value,
            'current_tasks': len(worker_info['current_tasks']),
            'total_tasks_completed': worker_info['total_tasks_completed'],
            'average_execution_time': (
                worker_info['total_execution_time'] / worker_info['total_tasks_completed']
                if worker_info['total_tasks_completed'] > 0 else 0
            ),
            'total_energy_consumed': worker_info['total_energy_consumed'],
            'last_heartbeat': worker_info['last_heartbeat'],
            'time_since_heartbeat': time.time() - worker_info['last_heartbeat'],
            'metrics': asdict(metrics) if metrics else None
        }


class DistributedInferenceCluster:
    """High-level distributed inference cluster manager."""
    
    def __init__(self, config: ScalingConfig):
        self.config = config
        self.scheduler = DistributedTaskScheduler(config)
        self.monitor = get_system_monitor()
        
        # Cluster state
        self.cluster_id = str(uuid.uuid4())
        self.start_time = time.time()
        
        # Performance tracking
        self.cluster_metrics = {
            'total_inferences': 0,
            'total_energy_consumed': 0.0,
            'uptime_seconds': 0.0,
            'peak_throughput': 0.0,
            'average_latency': 0.0
        }
    
    def start(self):
        """Start the distributed inference cluster."""
        self.scheduler.start()
        print(f"Distributed inference cluster {self.cluster_id} started")
    
    def stop(self):
        """Stop the distributed inference cluster."""
        self.scheduler.stop()
        print(f"Distributed inference cluster {self.cluster_id} stopped")
    
    def add_worker(self, worker_config: WorkerConfig):
        """Add a new worker to the cluster."""
        self.scheduler.register_worker(worker_config)
    
    def remove_worker(self, worker_id: str):
        """Remove a worker from the cluster."""
        self.scheduler.unregister_worker(worker_id)
    
    async def inference(self, model_path: str, input_data: Any, 
                       priority: TaskPriority = TaskPriority.NORMAL,
                       timeout: float = 30.0) -> TaskResult:
        """Perform distributed inference."""
        task = InferenceTask(
            task_id=str(uuid.uuid4()),
            model_path=model_path,
            input_data=input_data,
            priority=priority,
            deadline=time.time() + timeout
        )
        
        # Submit task
        task_id = self.scheduler.submit_task(task)
        
        # Wait for completion
        start_time = time.time()
        while time.time() - start_time < timeout:
            # Check if task is completed
            for result in self.scheduler.completed_tasks:
                if result.task_id == task_id:
                    return result
            
            await asyncio.sleep(0.1)
        
        # Timeout - return error result
        return TaskResult(
            task_id=task_id,
            result=None,
            worker_id="",
            execution_time=timeout,
            energy_consumed=0.0,
            success=False,
            error="Task timed out"
        )
    
    def batch_inference(self, tasks: List[Tuple[str, Any]], 
                       priority: TaskPriority = TaskPriority.NORMAL) -> List[str]:
        """Submit multiple inference tasks."""
        task_ids = []
        
        for model_path, input_data in tasks:
            task = InferenceTask(
                task_id=str(uuid.uuid4()),
                model_path=model_path,
                input_data=input_data,
                priority=priority
            )
            task_id = self.scheduler.submit_task(task)
            task_ids.append(task_id)
        
        return task_ids
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        scheduler_status = self.scheduler.get_status()
        
        # Update cluster metrics
        self.cluster_metrics['uptime_seconds'] = time.time() - self.start_time
        
        return {
            'cluster_id': self.cluster_id,
            'uptime_seconds': self.cluster_metrics['uptime_seconds'],
            'cluster_metrics': self.cluster_metrics,
            'scheduler_status': scheduler_status,
            'configuration': asdict(self.config)
        }
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        status = self.get_cluster_status()
        
        # Calculate derived metrics
        uptime_hours = self.cluster_metrics['uptime_seconds'] / 3600
        inferences_per_hour = (
            self.cluster_metrics['total_inferences'] / uptime_hours
            if uptime_hours > 0 else 0
        )
        
        energy_per_inference = (
            self.cluster_metrics['total_energy_consumed'] / self.cluster_metrics['total_inferences']
            if self.cluster_metrics['total_inferences'] > 0 else 0
        )
        
        return {
            'cluster_overview': {
                'cluster_id': self.cluster_id,
                'uptime_hours': uptime_hours,
                'total_workers': status['scheduler_status']['total_workers'],
                'active_workers': status['scheduler_status']['worker_status_counts'].get('IDLE', 0) + 
                               status['scheduler_status']['worker_status_counts'].get('BUSY', 0)
            },
            'performance_metrics': {
                'total_inferences': self.cluster_metrics['total_inferences'],
                'inferences_per_hour': inferences_per_hour,
                'average_latency_ms': self.cluster_metrics['average_latency'] * 1000,
                'peak_throughput_qps': self.cluster_metrics['peak_throughput']
            },
            'energy_metrics': {
                'total_energy_consumed_j': self.cluster_metrics['total_energy_consumed'],
                'energy_per_inference_pj': energy_per_inference * 1e12,
                'estimated_power_w': self.cluster_metrics['total_energy_consumed'] / max(1, self.cluster_metrics['uptime_seconds'])
            },
            'scheduler_details': status['scheduler_status']
        }


# Convenience functions for easy distributed inference setup

def create_distributed_cluster(min_workers: int = 1, max_workers: int = 10,
                             target_latency_ms: float = 500) -> DistributedInferenceCluster:
    """Create a distributed inference cluster with default configuration."""
    config = ScalingConfig(
        min_instances=min_workers,
        max_instances=max_workers,
        max_response_time_ms=target_latency_ms,
        target_cpu_utilization=0.7,
        scale_up_threshold=0.8,
        scale_down_threshold=0.3,
        cooldown_period_s=30.0
    )
    
    cluster = DistributedInferenceCluster(config)
    return cluster


def create_worker_config(worker_id: str, gpu_memory: float = 8.0, 
                        concurrent_tasks: int = 1,
                        supported_models: List[str] = None) -> WorkerConfig:
    """Create worker configuration with sensible defaults."""
    return WorkerConfig(
        worker_id=worker_id,
        max_concurrent_tasks=concurrent_tasks,
        supported_models=supported_models or [],
        gpu_memory_gb=gpu_memory,
        energy_efficiency_pj_per_mac=15.0,
        heartbeat_interval=10.0
    )