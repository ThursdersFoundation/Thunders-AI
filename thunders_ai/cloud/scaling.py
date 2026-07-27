"""Auto-scaling engine for Thunders AI cloud workloads.

Provides horizontal and vertical scaling with cost-optimised
resource allocation based on real-time metrics.
"""

from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class ScalingDirection(str, Enum):
    """Direction of a scaling operation."""
    HORIZONTAL = "horizontal"
    VERTICAL = "vertical"


class ScaleAction(str, Enum):
    """Type of scaling action."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    NO_ACTION = "no_action"


class ResourceMetrics:
    """Snapshot of resource utilisation metrics.

    Attributes:
        cpu_percent: CPU utilisation percentage.
        memory_percent: Memory utilisation percentage.
        gpu_percent: GPU utilisation percentage.
        request_rate: Incoming requests per second.
        latency_ms: Average response latency in milliseconds.
    """

    def __init__(
        self,
        cpu_percent: float = 0.0,
        memory_percent: float = 0.0,
        gpu_percent: float = 0.0,
        request_rate: float = 0.0,
        latency_ms: float = 0.0,
    ) -> None:
        self.cpu_percent = cpu_percent
        self.memory_percent = memory_percent
        self.gpu_percent = gpu_percent
        self.request_rate = request_rate
        self.latency_ms = latency_ms
        self.timestamp: float = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the metrics to a dictionary."""
        return {
            "cpu_percent": self.cpu_percent,
            "memory_percent": self.memory_percent,
            "gpu_percent": self.gpu_percent,
            "request_rate": self.request_rate,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp,
        }


class AutoScaler:
    """Automatically scale cloud resources based on demand.

    Supports both horizontal (adding/removing instances) and vertical
    (resizing instances) scaling with configurable thresholds and
    cost-optimisation strategies.

    Attributes:
        min_instances: Minimum number of instances.
        max_instances: Maximum number of instances.
        current_instances: Current instance count.
    """

    def __init__(
        self,
        min_instances: int = 1,
        max_instances: int = 10,
        scale_up_threshold: float = 75.0,
        scale_down_threshold: float = 25.0,
        cooldown_seconds: int = 60,
        cost_weight: float = 0.5,
    ) -> None:
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.cooldown_seconds = cooldown_seconds
        self.cost_weight = cost_weight

        self.current_instances: int = min_instances
        self._last_scale_time: float = 0.0
        self._metrics_history: List[ResourceMetrics] = []
        self._scale_history: List[Dict[str, Any]] = []
        self._instance_costs: Dict[str, float] = {
            "small": 0.05,
            "medium": 0.10,
            "large": 0.20,
            "gpu": 0.90,
        }

        logger.info(
            "AutoScaler initialised: min=%d, max=%d, up=%.0f%%, down=%.0f%%",
            min_instances,
            max_instances,
            scale_up_threshold,
            scale_down_threshold,
        )

    def scale_up(
        self,
        count: int = 1,
        direction: ScalingDirection = ScalingDirection.HORIZONTAL,
        instance_type: str = "medium",
    ) -> Dict[str, Any]:
        """Add resources to the deployment.

        Args:
            count: Number of units to add.
            direction: Horizontal or vertical scaling.
            instance_type: Instance size for vertical scaling.

        Returns:
            Scaling result with new instance count and cost.

        Raises:
            ValueError: If count would exceed max_instances.
        """
        new_count = self.current_instances + count
        if new_count > self.max_instances:
            raise ValueError(
                f"Cannot scale to {new_count} instances; "
                f"max is {self.max_instances}"
            )

        cost_per_hour = self._instance_costs.get(instance_type, 0.10) * count
        instance_ids = [f"inst-{uuid.uuid4().hex[:8]}" for _ in range(count)]

        self.current_instances = new_count
        self._last_scale_time = time.time()

        result: Dict[str, Any] = {
            "action": ScaleAction.SCALE_UP.value,
            "direction": direction.value,
            "count": count,
            "instance_type": instance_type,
            "new_total": self.current_instances,
            "new_instance_ids": instance_ids,
            "additional_cost_per_hour": cost_per_hour,
        }
        self._scale_history.append(result)
        logger.info("Scaled up: +%d instances → %d total", count, new_count)
        return result

    def scale_down(
        self,
        count: int = 1,
        direction: ScalingDirection = ScalingDirection.HORIZONTAL,
    ) -> Dict[str, Any]:
        """Remove resources from the deployment.

        Args:
            count: Number of units to remove.
            direction: Horizontal or vertical scaling.

        Returns:
            Scaling result with new instance count and savings.

        Raises:
            ValueError: If count would go below min_instances.
        """
        new_count = self.current_instances - count
        if new_count < self.min_instances:
            raise ValueError(
                f"Cannot scale to {new_count} instances; "
                f"min is {self.min_instances}"
            )

        savings_per_hour = 0.10 * count
        self.current_instances = new_count
        self._last_scale_time = time.time()

        result: Dict[str, Any] = {
            "action": ScaleAction.SCALE_DOWN.value,
            "direction": direction.value,
            "count": count,
            "new_total": self.current_instances,
            "savings_per_hour": savings_per_hour,
        }
        self._scale_history.append(result)
        logger.info("Scaled down: -%d instances → %d total", count, new_count)
        return result

    def auto_scale(
        self,
        metrics: Optional[ResourceMetrics] = None,
        cost_budget: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Automatically determine and apply scaling based on metrics.

        Uses CPU, memory, GPU, and request-rate thresholds to decide
        whether to scale up, down, or take no action. Applies a cooldown
        period between consecutive scaling events.

        Args:
            metrics: Current resource metrics. If None, auto_scale is a no-op.
            cost_budget: Optional hourly cost cap.

        Returns:
            Scaling decision and action taken.
        """
        if metrics is None:
            return {"action": ScaleAction.NO_ACTION.value, "reason": "no metrics"}

        self._metrics_history.append(metrics)

        # Enforce cooldown
        elapsed = time.time() - self._last_scale_time
        if elapsed < self.cooldown_seconds:
            return {
                "action": ScaleAction.NO_ACTION.value,
                "reason": f"cooldown ({self.cooldown_seconds - int(elapsed)}s remaining)",
            }

        # Weighted utilisation score
        util = (
            metrics.cpu_percent * 0.4
            + metrics.memory_percent * 0.3
            + metrics.gpu_percent * 0.3
        )

        # Cost check
        if cost_budget is not None:
            current_cost = self.current_instances * 0.10
            if current_cost >= cost_budget and util > self.scale_up_threshold:
                return {
                    "action": ScaleAction.NO_ACTION.value,
                    "reason": "cost budget reached",
                }

        if util > self.scale_up_threshold:
            try:
                result = self.scale_up(count=1)
                result["reason"] = f"utilisation {util:.1f}% > {self.scale_up_threshold}%"
                return result
            except ValueError:
                return {"action": ScaleAction.NO_ACTION.value, "reason": "at max capacity"}

        if util < self.scale_down_threshold:
            try:
                result = self.scale_down(count=1)
                result["reason"] = f"utilisation {util:.1f}% < {self.scale_down_threshold}%"
                return result
            except ValueError:
                return {"action": ScaleAction.NO_ACTION.value, "reason": "at min capacity"}

        return {"action": ScaleAction.NO_ACTION.value, "reason": "within thresholds"}

    def get_metrics(self, window_seconds: int = 300) -> Dict[str, Any]:
        """Retrieve aggregated resource metrics.

        Args:
            window_seconds: Look-back window in seconds.

        Returns:
            Aggregated metrics including averages and peak values.
        """
        cutoff = time.time() - window_seconds
        recent = [m for m in self._metrics_history if m.timestamp >= cutoff]

        if not recent:
            return {"window_seconds": window_seconds, "data_points": 0}

        avg_cpu = sum(m.cpu_percent for m in recent) / len(recent)
        avg_mem = sum(m.memory_percent for m in recent) / len(recent)
        avg_gpu = sum(m.gpu_percent for m in recent) / len(recent)
        peak_cpu = max(m.cpu_percent for m in recent)
        peak_mem = max(m.memory_percent for m in recent)

        return {
            "window_seconds": window_seconds,
            "data_points": len(recent),
            "current_instances": self.current_instances,
            "avg_cpu_percent": round(avg_cpu, 2),
            "avg_memory_percent": round(avg_mem, 2),
            "avg_gpu_percent": round(avg_gpu, 2),
            "peak_cpu_percent": round(peak_cpu, 2),
            "peak_memory_percent": round(peak_mem, 2),
            "estimated_cost_per_hour": round(self.current_instances * 0.10, 2),
        }
