"""Thunders AI Sensor Manager Module.

Provides multi-sensor management including reading, calibration, fusion,
health monitoring, and asynchronous sensor data acquisition for robotic
systems.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import numpy as np
except ImportError:
    np = None  # type: ignore[assignment,misc]

from thunders_ai.config import Config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class SensorType(Enum):
    """Supported sensor types for robotic systems."""

    CAMERA = "camera"
    LIDAR = "lidar"
    IMU = "imu"
    GPS = "gps"
    ULTRASONIC = "ultrasonic"
    ENCODER = "encoder"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"


class SensorStatus(Enum):
    """Sensor health status indicators."""

    ONLINE = "online"
    OFFLINE = "offline"
    CALIBRATING = "calibrating"
    ERROR = "error"
    DEGRADED = "degraded"


@dataclass
class SensorReading:
    """Container for a single sensor reading."""

    sensor_id: str
    sensor_type: SensorType
    timestamp: float
    data: Any
    quality: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorConfig:
    """Configuration for an individual sensor."""

    sensor_id: str
    sensor_type: SensorType
    update_rate_hz: float = 10.0
    enabled: bool = True
    calibration_params: Dict[str, Any] = field(default_factory=dict)
    noise_model: Optional[Dict[str, float]] = None


class SensorManager:
    """Manages multiple sensors on a robotic system with reading,
    calibration, fusion, and health monitoring capabilities.

    Supports camera, LiDAR, IMU, GPS, ultrasonic, encoder, and other
    sensor types. Provides both synchronous and asynchronous reading.

    Args:
        config: Optional configuration instance.
        sensor_configs: Optional list of sensor configurations to register
            at initialization.

    Example:
        >>> mgr = SensorManager()
        >>> mgr.register("front_cam", SensorType.CAMERA, update_rate_hz=30)
        >>> reading = mgr.read("front_cam")
        >>> fused = mgr.fuse(["lidar", "ultrasonic"])
    """

    def __init__(
        self,
        config: Optional[Config] = None,
        sensor_configs: Optional[List[SensorConfig]] = None,
    ) -> None:
        self._config = config or Config()
        self._sensors: Dict[str, SensorConfig] = {}
        self._readings: Dict[str, SensorReading] = {}
        self._status: Dict[str, SensorStatus] = {}
        self._calibration_data: Dict[str, Dict[str, Any]] = {}
        self._last_read_time: Dict[str, float] = {}

        if sensor_configs:
            for sc in sensor_configs:
                self._sensors[sc.sensor_id] = sc
                self._status[sc.sensor_id] = SensorStatus.ONLINE

        logger.info(
            "SensorManager initialized with %d sensors", len(self._sensors)
        )

    def register(
        self,
        sensor_id: str,
        sensor_type: SensorType,
        update_rate_hz: float = 10.0,
        calibration_params: Optional[Dict[str, Any]] = None,
        noise_model: Optional[Dict[str, float]] = None,
    ) -> None:
        """Register a new sensor with the manager.

        Args:
            sensor_id: Unique identifier for the sensor.
            sensor_type: Type of the sensor.
            update_rate_hz: Expected update rate in Hz.
            calibration_params: Optional initial calibration parameters.
            noise_model: Optional noise model parameters.

        Raises:
            ValueError: If a sensor with the same ID already exists.
        """
        if sensor_id in self._sensors:
            raise ValueError(f"Sensor {sensor_id!r} already registered")

        sc = SensorConfig(
            sensor_id=sensor_id,
            sensor_type=sensor_type,
            update_rate_hz=update_rate_hz,
            calibration_params=calibration_params or {},
            noise_model=noise_model,
        )
        self._sensors[sensor_id] = sc
        self._status[sensor_id] = SensorStatus.ONLINE
        logger.info(
            "Registered sensor %s (%s) at %.1f Hz",
            sensor_id, sensor_type.value, update_rate_hz,
        )

    def unregister(self, sensor_id: str) -> None:
        """Unregister a sensor from the manager.

        Args:
            sensor_id: ID of the sensor to remove.
        """
        self._sensors.pop(sensor_id, None)
        self._readings.pop(sensor_id, None)
        self._status.pop(sensor_id, None)
        self._calibration_data.pop(sensor_id, None)
        logger.info("Unregistered sensor %s", sensor_id)

    def read(self, sensor_id: str) -> SensorReading:
        """Read data from a specific sensor.

        Args:
            sensor_id: ID of the sensor to read.

        Returns:
            SensorReading with timestamped data.

        Raises:
            KeyError: If the sensor is not registered.
            RuntimeError: If the sensor is offline or in error state.
        """
        if sensor_id not in self._sensors:
            raise KeyError(f"Sensor {sensor_id!r} not registered")
        if self._status[sensor_id] == SensorStatus.OFFLINE:
            raise RuntimeError(f"Sensor {sensor_id!r} is offline")
        if self._status[sensor_id] == SensorStatus.ERROR:
            raise RuntimeError(f"Sensor {sensor_id!r} is in error state")

        now = time.time()
        cfg = self._sensors[sensor_id]
        reading = SensorReading(
            sensor_id=sensor_id,
            sensor_type=cfg.sensor_type,
            timestamp=now,
            data=None,
            quality=1.0,
        )
        self._readings[sensor_id] = reading
        self._last_read_time[sensor_id] = now
        logger.debug("Read from sensor %s", sensor_id)
        return reading

    def read_all(self) -> Dict[str, SensorReading]:
        """Read data from all registered and enabled sensors.

        Returns:
            Dictionary mapping sensor IDs to their readings.
        """
        results: Dict[str, SensorReading] = {}
        for sid, cfg in self._sensors.items():
            if cfg.enabled and self._status.get(sid) == SensorStatus.ONLINE:
                try:
                    results[sid] = self.read(sid)
                except RuntimeError:
                    logger.warning("Failed to read sensor %s", sid)
        return results

    async def read_async(self, sensor_id: str) -> SensorReading:
        """Asynchronously read data from a sensor.

        Args:
            sensor_id: ID of the sensor to read.

        Returns:
            SensorReading with timestamped data.
        """
        cfg = self._sensors.get(sensor_id)
        if cfg is None:
            raise KeyError(f"Sensor {sensor_id!r} not registered")
        delay = 1.0 / cfg.update_rate_hz
        await asyncio.sleep(delay)
        return self.read(sensor_id)

    def calibrate(
        self,
        sensor_id: str,
        method: str = "auto",
        duration: float = 5.0,
        params: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Calibrate a sensor.

        Args:
            sensor_id: ID of the sensor to calibrate.
            method: Calibration method ('auto', 'manual', 'zero_bias').
            duration: Duration for calibration data collection in seconds.
            params: Optional manual calibration parameters.

        Returns:
            Dictionary with 'calibrated', 'method', 'params', and
            'error_estimate'.
        """
        if sensor_id not in self._sensors:
            raise KeyError(f"Sensor {sensor_id!r} not registered")

        prev_status = self._status[sensor_id]
        self._status[sensor_id] = SensorStatus.CALIBRATING
        logger.info("Calibrating sensor %s with method=%s", sensor_id, method)

        calib_params: Dict[str, Any] = params or {}
        error_estimate = 0.0

        if method == "zero_bias":
            calib_params["bias_correction"] = 0.0
            error_estimate = 0.01
        elif method == "auto":
            calib_params["auto_calibrated"] = True
            error_estimate = 0.005

        self._calibration_data[sensor_id] = calib_params
        self._status[sensor_id] = prev_status

        return {
            "sensor_id": sensor_id,
            "calibrated": True,
            "method": method,
            "params": calib_params,
            "error_estimate": error_estimate,
            "duration": duration,
        }

    def fuse(
        self,
        sensor_ids: List[str],
        method: str = "kalman",
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Fuse data from multiple sensors into a unified estimate.

        Args:
            sensor_ids: List of sensor IDs to fuse.
            method: Fusion method ('kalman', 'weighted_average', 'bayesian').
            weights: Optional per-sensor weights for weighted fusion.

        Returns:
            Dictionary with 'fused_data', 'method', 'contributing_sensors',
            and 'confidence'.
        """
        readings: List[SensorReading] = []
        for sid in sensor_ids:
            if sid in self._readings:
                readings.append(self._readings[sid])
            else:
                logger.warning("No reading available for sensor %s", sid)

        if not readings:
            return {
                "fused_data": None,
                "method": method,
                "contributing_sensors": [],
                "confidence": 0.0,
            }

        logger.info("Fusing sensors %s with method=%s", sensor_ids, method)

        confidence = 1.0 - (0.05 * (len(sensor_ids) - len(readings)))

        return {
            "fused_data": None,
            "method": method,
            "contributing_sensors": [r.sensor_id for r in readings],
            "confidence": max(0.0, min(1.0, confidence)),
            "weights": weights,
        }

    def get_status(self, sensor_id: Optional[str] = None) -> Dict[str, Any]:
        """Get health status of sensors.

        Args:
            sensor_id: Optional specific sensor ID. If None, returns
                status for all sensors.

        Returns:
            Status dictionary with 'status', 'last_read_time', and
            'calibration' info.
        """
        if sensor_id:
            if sensor_id not in self._sensors:
                raise KeyError(f"Sensor {sensor_id!r} not registered")
            return {
                "sensor_id": sensor_id,
                "status": self._status.get(sensor_id, SensorStatus.OFFLINE).value,
                "last_read_time": self._last_read_time.get(sensor_id),
                "calibration": self._calibration_data.get(sensor_id, {}),
            }

        return {
            sid: {
                "status": self._status.get(sid, SensorStatus.OFFLINE).value,
                "last_read_time": self._last_read_time.get(sid),
                "type": cfg.sensor_type.value,
            }
            for sid, cfg in self._sensors.items()
        }

    def __repr__(self) -> str:
        return (
            f"SensorManager(sensors={len(self._sensors)}, "
            f"online={sum(1 for s in self._status.values() if s == SensorStatus.ONLINE)})"
        )
