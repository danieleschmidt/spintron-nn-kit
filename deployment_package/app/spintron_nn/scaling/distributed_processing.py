"""
Distributed processing system for SpinTron-NN-Kit.

This module provides:
- Distributed inference across multiple workers
- Task scheduling and load balancing
- Fault-tolerant processing pipelines
- Parallel batch processing optimization
"""

import time
import queue
import threading
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Callable, Any, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
import hashlib


class TaskStatus(Enum):
    """Task execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkerType(Enum):
    """Types of workers for different tasks."""
    THREAD = "thread"
    PROCESS = "process"
    DISTRIBUTED = "distributed"


@dataclass
class Task:
    """Task specification for distributed processing."""
    task_id: str
    task_type: str
    payload: Dict[str, Any]
    priority: int = 0
    timeout: float = 60.0
    retry_count: int = 0
    max_retries: int = 3
    created_at: float = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()


@dataclass
class TaskResult:
    """Result of task execution."""
    task_id: str
    status: TaskStatus
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    worker_id: str = None
    completed_at: float = None
    
    def __post_init__(self):
        if self.completed_at is None:
            self.completed_at = time.time()


@dataclass
class WorkerStats:
    """Statistics for a worker."""
    worker_id: str
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_execution_time: float = 0.0
    average_execution_time: float = 0.0
    last_task_completed_at: float = 0.0
    is_active: bool = True


class TaskScheduler:
    """Intelligent task scheduler with priority and load balancing."""
    
    def __init__(self, max_queue_size: int = 1000):
        self.task_queue = queue.PriorityQueue(maxsize=max_queue_size)
        self.pending_tasks = {}
        self.completed_tasks = {}
        self.failed_tasks = {}
        
        self.scheduler_running = False
        self.scheduler_thread = None
        
    def submit_task(self, task: Task) -> bool:
        """Submit a task for execution.
        
        Args:
            task: Task to submit
            
        Returns:
            True if task was queued successfully
        """
        try:
            # Priority queue uses negative priority for max-heap behavior
            self.task_queue.put((-task.priority, task.created_at, task), timeout=1.0)
            self.pending_tasks[task.task_id] = task
            return True
        except queue.Full:
            return False
    
    def get_next_task(self, timeout: float = 1.0) -> Optional[Task]:
        """Get next task from queue.
        
        Args:
            timeout: Timeout for queue operation
            
        Returns:
            Next task or None if timeout
        """
        try:
            priority, created_at, task = self.task_queue.get(timeout=timeout)
            return task
        except queue.Empty:
            return None
    
    def mark_task_completed(self, task_result: TaskResult):
        """Mark task as completed and store result.
        
        Args:
            task_result: Result of task execution
        """
        task_id = task_result.task_id
        
        if task_id in self.pending_tasks:
            del self.pending_tasks[task_id]
        
        if task_result.status == TaskStatus.COMPLETED:
            self.completed_tasks[task_id] = task_result
        else:
            self.failed_tasks[task_id] = task_result
            
            # Check if task should be retried
            if task_id in self.pending_tasks:
                original_task = self.pending_tasks[task_id]
                if original_task.retry_count < original_task.max_retries:
                    original_task.retry_count += 1
                    self.submit_task(original_task)
    
    def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Get status of a specific task.
        
        Args:
            task_id: Task identifier
            
        Returns:
            Task status or None if not found
        """
        if task_id in self.pending_tasks:
            return TaskStatus.PENDING
        elif task_id in self.completed_tasks:
            return TaskStatus.COMPLETED
        elif task_id in self.failed_tasks:
            return TaskStatus.FAILED
        else:
            return None
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics.
        
        Returns:
            Dictionary with queue statistics
        """
        return {
            "pending_tasks": len(self.pending_tasks),
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
            "queue_size": self.task_queue.qsize(),
            "queue_empty": self.task_queue.empty(),
            "queue_full": self.task_queue.full()
        }


class WorkerPool:
    """Pool of workers for executing tasks."""
    
    def __init__(self, 
                 worker_count: int = None,
                 worker_type: WorkerType = WorkerType.THREAD,
                 task_handler: Callable[[Task], Any] = None):
        """Initialize worker pool.
        
        Args:
            worker_count: Number of workers (defaults to CPU count)
            worker_type: Type of workers to create
            task_handler: Function to handle tasks
        """
        self.worker_count = worker_count or mp.cpu_count()
        self.worker_type = worker_type
        self.task_handler = task_handler or self._default_task_handler
        
        self.workers = {}
        self.worker_stats = {}
        self.executor = None
        
        self.running = False
    
    def start(self):
        """Start the worker pool."""
        self.running = True
        
        if self.worker_type == WorkerType.THREAD:
            self.executor = ThreadPoolExecutor(max_workers=self.worker_count)
        elif self.worker_type == WorkerType.PROCESS:
            self.executor = ProcessPoolExecutor(max_workers=self.worker_count)
        
        # Initialize worker stats
        for i in range(self.worker_count):
            worker_id = f"worker_{i}"
            self.worker_stats[worker_id] = WorkerStats(worker_id=worker_id)
    
    def stop(self):
        """Stop the worker pool."""
        self.running = False
        if self.executor:
            self.executor.shutdown(wait=True)
    
    def submit_task(self, task: Task) -> Optional[threading.Thread]:
        """Submit task to worker pool.
        
        Args:
            task: Task to execute
            
        Returns:
            Future object or None if pool not running
        """
        if not self.running or not self.executor:
            return None
        
        future = self.executor.submit(self._execute_task_with_stats, task)
        return future
    
    def _execute_task_with_stats(self, task: Task) -> TaskResult:
        """Execute task and update worker statistics.
        
        Args:
            task: Task to execute
            
        Returns:
            Task result
        """
        worker_id = f"worker_{threading.current_thread().ident}"
        start_time = time.time()
        
        try:
            result = self.task_handler(task)
            execution_time = time.time() - start_time
            
            # Update worker stats
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                stats.tasks_completed += 1
                stats.total_execution_time += execution_time
                stats.average_execution_time = stats.total_execution_time / stats.tasks_completed
                stats.last_task_completed_at = time.time()
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                worker_id=worker_id
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # Update worker stats for failure
            if worker_id in self.worker_stats:
                stats = self.worker_stats[worker_id]
                stats.tasks_failed += 1
            
            return TaskResult(
                task_id=task.task_id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                worker_id=worker_id
            )
    
    def _default_task_handler(self, task: Task) -> Any:
        """Default task handler for demonstration.
        
        Args:
            task: Task to handle
            
        Returns:
            Task result
        """
        # Simulate some work
        time.sleep(0.1)
        
        if task.task_type == "inference":
            return {
                "prediction": [0.8, 0.2],
                "confidence": 0.95,
                "processing_time": 0.1
            }
        elif task.task_type == "training":
            return {
                "epoch": task.payload.get("epoch", 1),
                "loss": 0.05,
                "accuracy": 0.92
            }
        else:
            return {"status": "completed", "task_type": task.task_type}
    
    def get_worker_stats(self) -> Dict[str, WorkerStats]:
        """Get statistics for all workers.
        
        Returns:
            Dictionary mapping worker IDs to their statistics
        """
        return dict(self.worker_stats)
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get overall pool statistics.
        
        Returns:
            Dictionary with pool statistics
        """
        total_completed = sum(stats.tasks_completed for stats in self.worker_stats.values())
        total_failed = sum(stats.tasks_failed for stats in self.worker_stats.values())
        
        avg_execution_times = [stats.average_execution_time for stats in self.worker_stats.values() 
                              if stats.average_execution_time > 0]
        overall_avg_time = sum(avg_execution_times) / len(avg_execution_times) if avg_execution_times else 0
        
        return {
            "worker_count": self.worker_count,
            "worker_type": self.worker_type.value,
            "total_tasks_completed": total_completed,
            "total_tasks_failed": total_failed,
            "success_rate": total_completed / max(1, total_completed + total_failed),
            "overall_average_execution_time": overall_avg_time,
            "pool_running": self.running
        }


class DistributedProcessor:
    """Main distributed processing coordinator."""
    
    def __init__(self, 
                 worker_count: int = None,
                 max_queue_size: int = 1000,
                 worker_type: WorkerType = WorkerType.THREAD):
        """Initialize distributed processor.
        
        Args:
            worker_count: Number of workers
            max_queue_size: Maximum task queue size
            worker_type: Type of workers
        """
        self.scheduler = TaskScheduler(max_queue_size)
        self.worker_pool = WorkerPool(worker_count, worker_type)
        
        self.processing_thread = None
        self.running = False
        
        # Performance monitoring
        self.throughput_history = []
        self.latency_history = []
    
    def start(self):
        """Start the distributed processor."""
        self.running = True
        self.worker_pool.start()
        
        # Start processing loop
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
    
    def stop(self):
        """Stop the distributed processor."""
        self.running = False
        self.worker_pool.stop()
        
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)
    
    def _processing_loop(self):
        """Main processing loop that coordinates tasks and workers."""
        while self.running:
            try:
                # Get next task from scheduler
                task = self.scheduler.get_next_task(timeout=0.5)
                if task is None:
                    continue
                
                # Submit to worker pool
                future = self.worker_pool.submit_task(task)
                if future is None:
                    # Worker pool not available, requeue task
                    self.scheduler.submit_task(task)
                    time.sleep(0.1)
                    continue
                
                # Handle task completion asynchronously
                def handle_completion(future_obj):
                    try:
                        result = future_obj.result(timeout=task.timeout)
                        self.scheduler.mark_task_completed(result)
                        self._update_performance_metrics(result)
                    except Exception as e:
                        error_result = TaskResult(
                            task_id=task.task_id,
                            status=TaskStatus.FAILED,
                            error=str(e)
                        )
                        self.scheduler.mark_task_completed(error_result)
                
                # Register callback for completion
                future.add_done_callback(handle_completion)
                
            except Exception as e:
                print(f"Error in processing loop: {e}")
                time.sleep(1.0)
    
    def submit_task(self, task: Task) -> bool:
        """Submit task for distributed processing.
        
        Args:
            task: Task to process
            
        Returns:
            True if task was submitted successfully
        """
        return self.scheduler.submit_task(task)
    
    def submit_batch_tasks(self, tasks: List[Task]) -> List[bool]:
        """Submit multiple tasks as a batch.
        
        Args:
            tasks: List of tasks to submit
            
        Returns:
            List of success status for each task
        """
        results = []
        for task in tasks:
            success = self.submit_task(task)
            results.append(success)
        return results
    
    def wait_for_completion(self, task_ids: List[str], timeout: float = 60.0) -> Dict[str, TaskResult]:
        """Wait for specific tasks to complete.
        
        Args:
            task_ids: List of task IDs to wait for
            timeout: Maximum time to wait
            
        Returns:
            Dictionary mapping task IDs to results
        """
        start_time = time.time()
        results = {}
        
        while len(results) < len(task_ids) and time.time() - start_time < timeout:
            for task_id in task_ids:
                if task_id not in results:
                    status = self.scheduler.get_task_status(task_id)
                    if status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                        if task_id in self.scheduler.completed_tasks:
                            results[task_id] = self.scheduler.completed_tasks[task_id]
                        elif task_id in self.scheduler.failed_tasks:
                            results[task_id] = self.scheduler.failed_tasks[task_id]
            
            if len(results) < len(task_ids):
                time.sleep(0.1)
        
        return results
    
    def _update_performance_metrics(self, result: TaskResult):
        """Update performance metrics based on task result.
        
        Args:
            result: Task result to analyze
        """
        if result.status == TaskStatus.COMPLETED:
            self.latency_history.append(result.execution_time)
            
            # Keep only recent history
            if len(self.latency_history) > 1000:
                self.latency_history = self.latency_history[-1000:]
        
        # Calculate throughput (tasks per second)
        current_time = time.time()
        self.throughput_history.append((current_time, 1))
        
        # Remove old throughput data (older than 60 seconds)
        cutoff_time = current_time - 60
        self.throughput_history = [(t, count) for t, count in self.throughput_history if t > cutoff_time]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Calculate current throughput (tasks per second)
        if self.throughput_history:
            recent_tasks = len(self.throughput_history)
            time_window = 60  # Last 60 seconds
            throughput = recent_tasks / time_window
        else:
            throughput = 0.0
        
        # Calculate latency statistics
        latency_stats = {}
        if self.latency_history:
            import statistics
            latency_stats = {
                "mean": statistics.mean(self.latency_history),
                "median": statistics.median(self.latency_history),
                "p95": sorted(self.latency_history)[int(0.95 * len(self.latency_history))],
                "min": min(self.latency_history),
                "max": max(self.latency_history)
            }
        
        return {
            "throughput_tasks_per_second": throughput,
            "latency_stats": latency_stats,
            "queue_stats": self.scheduler.get_queue_stats(),
            "worker_pool_stats": self.worker_pool.get_pool_stats(),
            "total_latency_samples": len(self.latency_history),
            "total_throughput_samples": len(self.throughput_history)
        }


class ParallelInference:
    """Specialized parallel inference processor for SpinTron-NN-Kit."""
    
    def __init__(self, model_instances: List[Any], batch_size: int = 32):
        """Initialize parallel inference system.
        
        Args:
            model_instances: List of model instances for parallel processing
            batch_size: Batch size for processing
        """
        self.model_instances = model_instances
        self.batch_size = batch_size
        
        # Create distributed processor
        self.processor = DistributedProcessor(
            worker_count=len(model_instances),
            worker_type=WorkerType.THREAD
        )
        
        # Custom task handler for inference
        self.processor.worker_pool.task_handler = self._inference_task_handler
    
    def start(self):
        """Start parallel inference system."""
        self.processor.start()
    
    def stop(self):
        """Stop parallel inference system."""
        self.processor.stop()
    
    def _inference_task_handler(self, task: Task) -> Any:
        """Handle inference tasks with actual models.
        
        Args:
            task: Inference task
            
        Returns:
            Inference result
        """
        if task.task_type != "inference":
            raise ValueError(f"Unsupported task type: {task.task_type}")
        
        # Get model instance (simple round-robin)
        worker_idx = hash(task.task_id) % len(self.model_instances)
        model = self.model_instances[worker_idx]
        
        # Extract input data
        input_data = task.payload.get("input_data")
        if input_data is None:
            raise ValueError("No input data provided for inference")
        
        # Simulate inference (replace with actual model call)
        time.sleep(0.01)  # Simulate processing time
        
        return {
            "prediction": [0.8, 0.1, 0.1],  # Simulated output
            "confidence": 0.95,
            "model_id": worker_idx,
            "processing_time": 0.01
        }
    
    def infer_batch(self, batch_inputs: List[Any], timeout: float = 30.0) -> List[Any]:
        """Perform parallel inference on a batch of inputs.
        
        Args:
            batch_inputs: List of input data
            timeout: Timeout for batch processing
            
        Returns:
            List of inference results
        """
        # Create tasks for each input
        tasks = []
        for i, input_data in enumerate(batch_inputs):
            task = Task(
                task_id=f"inference_{int(time.time())}_{i}",
                task_type="inference",
                payload={"input_data": input_data},
                priority=1
            )
            tasks.append(task)
        
        # Submit all tasks
        submit_results = self.processor.submit_batch_tasks(tasks)
        if not all(submit_results):
            raise RuntimeError("Failed to submit all tasks for batch inference")
        
        # Wait for completion
        task_ids = [task.task_id for task in tasks]
        results = self.processor.wait_for_completion(task_ids, timeout)
        
        # Extract results in order
        ordered_results = []
        for task in tasks:
            if task.task_id in results:
                result = results[task.task_id]
                if result.status == TaskStatus.COMPLETED:
                    ordered_results.append(result.result)
                else:
                    ordered_results.append({"error": result.error})
            else:
                ordered_results.append({"error": "timeout"})
        
        return ordered_results
    
    def get_inference_stats(self) -> Dict[str, Any]:
        """Get inference-specific statistics.
        
        Returns:
            Dictionary with inference statistics
        """
        base_stats = self.processor.get_performance_stats()
        
        # Add inference-specific metrics
        base_stats.update({
            "model_instances": len(self.model_instances),
            "batch_size": self.batch_size,
            "parallel_efficiency": min(1.0, base_stats["throughput_tasks_per_second"] / len(self.model_instances))
        })
        
        return base_stats