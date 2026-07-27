"""Task scheduler with priority queuing, async execution, and retry logic."""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Dict, List, Optional

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class TaskPriority(int, Enum):
    """Priority levels – lower value = higher priority."""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class TaskStatus(str, Enum):
    """Lifecycle states for a scheduled task."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass(order=True)
class Task:
    """Represents a schedulable unit of work."""
    sort_key: tuple = field(compare=True)
    task_id: str = field(compare=False)
    name: str = field(compare=False)
    func: Callable = field(compare=False, repr=False)
    args: tuple = field(compare=False, default=())
    kwargs: Dict[str, Any] = field(compare=False, default_factory=dict)
    priority: TaskPriority = field(compare=False, default=TaskPriority.NORMAL)
    status: TaskStatus = field(compare=False, default=TaskStatus.PENDING)
    max_retries: int = field(compare=False, default=3)
    retry_count: int = field(compare=False, default=0)
    result: Any = field(compare=False, default=None)
    error: Optional[str] = field(compare=False, default=None)
    created_at: float = field(compare=False, default_factory=time.time)
    completed_at: Optional[float] = field(compare=False, default=None)

    def __post_init__(self) -> None:
        self.sort_key = (self.priority, self.created_at)


class TaskScheduler:
    """Schedules and executes AI tasks with priority, retries, and status tracking.

    Supports both synchronous and asynchronous execution with a configurable
    worker pool.

    Args:
        config: ThundersConfig instance.
        max_workers: Maximum number of concurrent workers.
        default_retry: Default retry count for submitted tasks.

    Example::

        scheduler = TaskScheduler(config)
        tid = scheduler.submit("train", train_fn, priority=TaskPriority.HIGH)
        scheduler.run_all()
        status = scheduler.get_status(tid)
    """

    def __init__(
        self,
        config: ThundersConfig,
        max_workers: int = 4,
        default_retry: int = 3,
    ) -> None:
        self._config = config
        self._max_workers = max_workers
        self._default_retry = default_retry
        self._queue: List[Task] = []
        self._tasks: Dict[str, Task] = {}
        self._results: Dict[str, Any] = {}
        logger.info("TaskScheduler ready – max_workers=%d", max_workers)

    # ------------------------------------------------------------------
    # Submission
    # ------------------------------------------------------------------

    def submit(
        self,
        name: str,
        func: Callable,
        args: tuple = (),
        kwargs: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: Optional[int] = None,
    ) -> str:
        """Submit a new task to the scheduler.

        Args:
            name: Human-readable task name.
            func: Callable to execute.
            args: Positional arguments for *func*.
            kwargs: Keyword arguments for *func*.
            priority: Task priority level.
            max_retries: Override default retry count.

        Returns:
            Unique task ID.
        """
        task_id = uuid.uuid4().hex[:12]
        task = Task(
            sort_key=(priority, time.time()),
            task_id=task_id,
            name=name,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            max_retries=max_retries if max_retries is not None else self._default_retry,
        )
        self._queue.append(task)
        self._queue.sort()
        self._tasks[task_id] = task
        logger.info("Task submitted: %s (id=%s, priority=%s)", name, task_id, priority.name)
        return task_id

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def run_all(self) -> None:
        """Execute all pending tasks synchronously, respecting concurrency."""
        import heapq

        running: List[Task] = []
        queue = list(self._queue)
        heapq.heapify(queue)

        while queue or running:
            # Fill up to max_workers
            while queue and len(running) < self._max_workers:
                task = heapq.heappop(queue)
                self._execute_task(task)
                running.append(task)

            running = [t for t in running if t.status == TaskStatus.RUNNING]

    async def run_all_async(self) -> None:
        """Execute all pending tasks asynchronously."""
        sem = asyncio.Semaphore(self._max_workers)

        async def _run(task: Task) -> None:
            async with sem:
                await self._execute_task_async(task)

        await asyncio.gather(*[_run(t) for t in self._queue if t.status == TaskStatus.PENDING])

    def _execute_task(self, task: Task) -> None:
        """Run a single task with retry logic."""
        task.status = TaskStatus.RUNNING
        try:
            result = task.func(*task.args, **task.kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self._results[task.task_id] = result
            logger.info("Task completed: %s (id=%s)", task.name, task.task_id)
        except Exception as exc:
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.RETRYING
                logger.warning(
                    "Task %s failed (attempt %d/%d): %s – retrying",
                    task.task_id, task.retry_count, task.max_retries, exc,
                )
                self._execute_task(task)
            else:
                task.status = TaskStatus.FAILED
                task.error = str(exc)
                logger.error("Task %s failed permanently: %s", task.task_id, exc)

    async def _execute_task_async(self, task: Task) -> None:
        """Run a single task asynchronously with retry logic."""
        task.status = TaskStatus.RUNNING
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = await asyncio.to_thread(task.func, *task.args, **task.kwargs)
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = time.time()
            self._results[task.task_id] = result
        except Exception as exc:
            task.retry_count += 1
            if task.retry_count <= task.max_retries:
                task.status = TaskStatus.RETRYING
                await self._execute_task_async(task)
            else:
                task.status = TaskStatus.FAILED
                task.error = str(exc)

    # ------------------------------------------------------------------
    # Status / results
    # ------------------------------------------------------------------

    def get_status(self, task_id: str) -> Optional[TaskStatus]:
        """Return the current status of a task."""
        task = self._tasks.get(task_id)
        return task.status if task else None

    def get_result(self, task_id: str) -> Any:
        """Return the result of a completed task."""
        return self._results.get(task_id)

    def cancel(self, task_id: str) -> bool:
        """Cancel a pending task.

        Returns:
            *True* if the task was successfully cancelled.
        """
        task = self._tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            self._queue = [t for t in self._queue if t.task_id != task_id]
            logger.info("Task cancelled: %s", task_id)
            return True
        return False

    @property
    def pending_count(self) -> int:
        """Number of pending tasks in the queue."""
        return sum(1 for t in self._queue if t.status == TaskStatus.PENDING)
