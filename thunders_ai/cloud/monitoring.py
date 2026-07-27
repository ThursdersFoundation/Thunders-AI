"""Monitoring and observability for Thunders AI cloud deployments.

Provides real-time system monitoring, metric logging, alerting,
and Prometheus-compatible metric export.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import defaultdict
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class MetricType(str, Enum):
    """Metric types for Prometheus compatibility."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


class Alert:
    """Represents a monitoring alert.

    Attributes:
        alert_id: Unique identifier.
        severity: Alert severity.
        message: Human-readable description.
        metric_name: The metric that triggered the alert.
        threshold: The threshold that was breached.
        value: The current metric value.
    """

    def __init__(
        self,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        threshold: float,
        value: float,
    ) -> None:
        self.alert_id = f"alert-{uuid.uuid4().hex[:8]}"
        self.severity = severity
        self.message = message
        self.metric_name = metric_name
        self.threshold = threshold
        self.value = value
        self.timestamp: float = time.time()
        self.acknowledged: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the alert."""
        return {
            "alert_id": self.alert_id,
            "severity": self.severity.value,
            "message": self.message,
            "metric_name": self.metric_name,
            "threshold": self.threshold,
            "value": self.value,
            "timestamp": self.timestamp,
            "acknowledged": self.acknowledged,
        }


class MonitoringSystem:
    """Centralised monitoring and observability for Thunders AI.

    Collects, stores, and analyses system metrics with alerting
    and Prometheus-compatible export.

    Attributes:
        service_name: Name of the monitored service.
        metrics: Stored metric samples keyed by name.
    """

    def __init__(
        self,
        service_name: str = "thunders-ai",
        retention_seconds: int = 86400,
        alert_callbacks: Optional[List[Callable[[Alert], None]]] = None,
    ) -> None:
        self.service_name = service_name
        self.retention_seconds = retention_seconds
        self._alert_callbacks = alert_callbacks or []
        self.metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._metric_types: Dict[str, MetricType] = {}
        self._alert_rules: List[Dict[str, Any]] = []
        self._active_alerts: Dict[str, Alert] = {}
        self._performance_baseline: Dict[str, float] = {}

        logger.info(
            "MonitoringSystem initialised: service=%s, retention=%ds",
            service_name,
            retention_seconds,
        )

    def monitor(
        self,
        targets: Optional[List[str]] = None,
        interval_seconds: int = 10,
    ) -> Dict[str, Any]:
        """Start monitoring specified targets.

        Args:
            targets: List of deployment IDs or service endpoints to monitor.
            interval_seconds: Scraping interval.

        Returns:
            Monitoring session details.
        """
        targets = targets or ["default"]
        session_id = f"mon-{uuid.uuid4().hex[:8]}"

        session: Dict[str, Any] = {
            "session_id": session_id,
            "targets": targets,
            "interval_seconds": interval_seconds,
            "started_at": time.time(),
            "status": "active",
        }

        logger.info(
            "Monitoring session %s started: %d targets, interval=%ds",
            session_id,
            len(targets),
            interval_seconds,
        )
        return session

    def log_metrics(
        self,
        name: str,
        value: float,
        metric_type: MetricType = MetricType.GAUGE,
        labels: Optional[Dict[str, str]] = None,
        timestamp: Optional[float] = None,
    ) -> None:
        """Log a metric sample.

        Args:
            name: Metric name (e.g. 'cpu_utilization_percent').
            value: Current metric value.
            metric_type: Prometheus metric type.
            labels: Optional key-value labels for dimensional metrics.
            timestamp: Override timestamp (defaults to now).
        """
        labels = labels or {}
        timestamp = timestamp or time.time()
        self._metric_types[name] = metric_type

        sample: Dict[str, Any] = {
            "value": value,
            "labels": labels,
            "timestamp": timestamp,
        }
        self.metrics[name].append(sample)
        self._check_alert_rules(name, value)
        self._cleanup_old_metrics()

        logger.debug("Metric logged: %s=%.4f", name, value)

    def alert(
        self,
        severity: AlertSeverity,
        message: str,
        metric_name: str,
        threshold: float,
        value: float,
    ) -> Alert:
        """Create and dispatch a monitoring alert.

        Args:
            severity: Alert severity level.
            message: Human-readable description.
            metric_name: The metric that triggered the alert.
            threshold: The configured threshold.
            value: The current value.

        Returns:
            The created Alert object.
        """
        alert = Alert(
            severity=severity,
            message=message,
            metric_name=metric_name,
            threshold=threshold,
            value=value,
        )
        self._active_alerts[alert.alert_id] = alert

        for callback in self._alert_callbacks:
            try:
                callback(alert)
            except Exception as exc:
                logger.error("Alert callback failed: %s", exc)

        logger.warning(
            "Alert [%s]: %s (metric=%s, value=%.2f, threshold=%.2f)",
            severity.value,
            message,
            metric_name,
            value,
            threshold,
        )
        return alert

    def get_dashboard(self, time_range_seconds: int = 3600) -> Dict[str, Any]:
        """Generate dashboard data for visualisation.

        Args:
            time_range_seconds: Look-back window.

        Returns:
            Dashboard payload with metric summaries and active alerts.
        """
        cutoff = time.time() - time_range_seconds
        panels: Dict[str, Any] = {}

        for name, samples in self.metrics.items():
            recent = [s for s in samples if s["timestamp"] >= cutoff]
            if not recent:
                continue
            values = [s["value"] for s in recent]
            panels[name] = {
                "type": self._metric_types.get(name, MetricType.GAUGE).value,
                "current": values[-1],
                "min": min(values),
                "max": max(values),
                "avg": round(sum(values) / len(values), 4),
                "samples": len(recent),
            }

        return {
            "service": self.service_name,
            "time_range_seconds": time_range_seconds,
            "generated_at": time.time(),
            "panels": panels,
            "active_alerts": {
                aid: a.to_dict() for aid, a in self._active_alerts.items()
                if not a.acknowledged
            },
        }

    def track_performance(
        self,
        operation: str,
        duration_seconds: float,
        success: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Track the performance of an operation.

        Args:
            operation: Name of the measured operation.
            duration_seconds: Elapsed time in seconds.
            success: Whether the operation succeeded.
            metadata: Additional context.

        Returns:
            Performance tracking record.
        """
        metric_name = f"perf_{operation}"
        self.log_metrics(
            name=f"{metric_name}_duration_seconds",
            value=duration_seconds,
            metric_type=MetricType.HISTOGRAM,
            labels={"operation": operation, "success": str(success)},
        )
        self.log_metrics(
            name=f"{metric_name}_total",
            value=1.0,
            metric_type=MetricType.COUNTER,
            labels={"operation": operation, "success": str(success)},
        )

        record: Dict[str, Any] = {
            "operation": operation,
            "duration_seconds": round(duration_seconds, 6),
            "success": success,
            "metadata": metadata or {},
            "timestamp": time.time(),
        }

        # Update baseline for anomaly detection
        history = self.metrics.get(f"{metric_name}_duration_seconds", [])
        if len(history) >= 10:
            recent_durations = [s["value"] for s in history[-10:]]
            self._performance_baseline[operation] = sum(recent_durations) / len(
                recent_durations
            )

        return record

    def export_prometheus(self) -> str:
        """Export all metrics in Prometheus text exposition format.

        Returns:
            Prometheus-formatted metric string.
        """
        lines: List[str] = []
        for name, samples in self.metrics.items():
            mtype = self._metric_types.get(name, MetricType.GAUGE)
            lines.append(f"# TYPE {name} {mtype.value}")
            for sample in samples[-1:]:  # latest sample only
                labels_str = ""
                if sample["labels"]:
                    parts = [f'{k}="{v}"' for k, v in sample["labels"].items()]
                    labels_str = "{" + ",".join(parts) + "}"
                lines.append(f"{name}{labels_str} {sample['value']}")
        return "\n".join(lines) + "\n"

    # -- Internal helpers ---------------------------------------------------

    def add_alert_rule(
        self,
        metric_name: str,
        threshold: float,
        comparison: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        message_template: str = "",
    ) -> None:
        """Add a rule that triggers an alert when a metric breaches a threshold."""
        self._alert_rules.append({
            "metric_name": metric_name,
            "threshold": threshold,
            "comparison": comparison,
            "severity": severity,
            "message_template": message_template or f"{metric_name} {{comparison}} {threshold}",
        })

    def _check_alert_rules(self, metric_name: str, value: float) -> None:
        """Evaluate alert rules against the latest metric value."""
        for rule in self._alert_rules:
            if rule["metric_name"] != metric_name:
                continue
            triggered = False
            comp = rule["comparison"]
            thr = rule["threshold"]
            if comp == "gt" and value > thr:
                triggered = True
            elif comp == "lt" and value < thr:
                triggered = True
            elif comp == "gte" and value >= thr:
                triggered = True
            elif comp == "lte" and value <= thr:
                triggered = True

            if triggered:
                self.alert(
                    severity=rule["severity"],
                    message=rule["message_template"].format(
                        comparison=comp, threshold=thr
                    ),
                    metric_name=metric_name,
                    threshold=thr,
                    value=value,
                )

    def _cleanup_old_metrics(self) -> None:
        """Remove metric samples older than the retention period."""
        cutoff = time.time() - self.retention_seconds
        for name in list(self.metrics.keys()):
            self.metrics[name] = [
                s for s in self.metrics[name] if s["timestamp"] >= cutoff
            ]
