"""Validation utilities for Thunders AI.

Provides static validation methods for model names, API keys,
device strings, images, audio, and configuration dictionaries.
Each method returns a (bool, str) tuple indicating validity
and an error message.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class Validators:
    """Static validation methods for common Thunders AI inputs.

    Every public method returns a ``(bool, str)`` tuple where the
    first element indicates validity and the second contains an
    error message (empty string when valid).
    """

    # Pre-compiled patterns for performance
    _MODEL_NAME_PATTERN = re.compile(
        r"^[a-zA-Z0-9][a-zA-Z0-9_\-./]*[a-zA-Z0-9]$|^[a-zA-Z0-9]$"
    )
    _API_KEY_PATTERN = re.compile(r"^[a-zA-Z0-9\-_]{20,}$")
    _DEVICE_PATTERN = re.compile(
        r"^(cpu|cuda(:\d+)?|mps|xpu|tpu|auto|cpu:?\d*|gpu(:\d+)?)$"
    )
    _IMAGE_EXTENSIONS = {
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff", ".tif", ".svg",
    }
    _AUDIO_EXTENSIONS = {
        ".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a", ".wma", ".opus",
    }

    @staticmethod
    def validate_model_name(name: str) -> Tuple[bool, str]:
        """Validate a model name.

        Model names must be 1-128 characters, start and end with
        alphanumeric characters, and may contain hyphens, underscores,
        dots, and forward slashes.

        Args:
            name: Model name to validate.

        Returns:
            (is_valid, error_message) tuple.
        """
        if not name:
            return False, "model name must not be empty"

        if len(name) > 128:
            return False, f"model name too long ({len(name)} chars; max 128)"

        if len(name) < 1:
            return False, "model name must be at least 1 character"

        if not Validators._MODEL_NAME_PATTERN.match(name):
            return (
                False,
                "model name must start and end with alphanumeric characters "
                "and contain only letters, digits, hyphens, underscores, dots, or slashes",
            )

        # Reject consecutive separators
        if re.search(r"[./\-_]{2,}", name):
            return False, "model name must not contain consecutive separators"

        return True, ""

    @staticmethod
    def validate_api_key(key: str) -> Tuple[bool, str]:
        """Validate an API key format.

        API keys must be at least 20 characters of alphanumeric
        characters, hyphens, or underscores.

        Args:
            key: API key string to validate.

        Returns:
            (is_valid, error_message) tuple.
        """
        if not key:
            return False, "API key must not be empty"

        if len(key) < 20:
            return False, f"API key too short ({len(key)} chars; min 20)"

        if len(key) > 512:
            return False, f"API key too long ({len(key)} chars; max 512)"

        if not Validators._API_KEY_PATTERN.match(key):
            return (
                False,
                "API key must contain only alphanumeric characters, hyphens, or underscores",
            )

        return True, ""

    @staticmethod
    def validate_device(device: str) -> Tuple[bool, str]:
        """Validate a device string.

        Supported devices: cpu, cuda, cuda:N, mps, xpu, tpu, auto, gpu, gpu:N.

        Args:
            device: Device string to validate.

        Returns:
            (is_valid, error_message) tuple.
        """
        if not device:
            return False, "device string must not be empty"

        device_lower = device.lower().strip()
        if not Validators._DEVICE_PATTERN.match(device_lower):
            return (
                False,
                f"unsupported device '{device}'; expected cpu, cuda[:N], mps, xpu, tpu, auto, or gpu[:N]",
            )

        # Validate device index is within range for cuda/gpu
        match = re.match(r"^(cuda|gpu):(\d+)$", device_lower)
        if match:
            index = int(match.group(2))
            if index > 255:
                return False, f"device index {index} is out of range (0-255)"

        return True, ""

    @staticmethod
    def validate_image(
        image: Union[str, Path, bytes, Dict[str, Any]],
        max_size_mb: float = 50.0,
    ) -> Tuple[bool, str]:
        """Validate an image input.

        Accepts file paths, raw bytes, or image metadata dicts.
        For file paths, checks extension and optionally file size.

        Args:
            image: Image path, bytes, or metadata dict.
            max_size_mb: Maximum file size in megabytes (for path validation).

        Returns:
            (is_valid, error_message) tuple.
        """
        if image is None:
            return False, "image must not be None"

        if isinstance(image, (str, Path)):
            path = Path(image)
            ext = path.suffix.lower()
            if ext not in Validators._IMAGE_EXTENSIONS:
                return (
                    False,
                    f"unsupported image extension '{ext}'; "
                    f"expected one of {sorted(Validators._IMAGE_EXTENSIONS)}",
                )
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    return (
                        False,
                        f"image file too large ({size_mb:.1f} MB; max {max_size_mb} MB)",
                    )
            return True, ""

        if isinstance(image, bytes):
            if len(image) == 0:
                return False, "image bytes must not be empty"
            size_mb = len(image) / (1024 * 1024)
            if size_mb > max_size_mb:
                return (
                    False,
                    f"image bytes too large ({size_mb:.1f} MB; max {max_size_mb} MB)",
                )
            return True, ""

        if isinstance(image, dict):
            required_keys = {"data", "format"}
            if not required_keys.intersection(image.keys()):
                return False, "image dict must contain 'data' and/or 'format' keys"
            return True, ""

        return False, f"unsupported image type: {type(image).__name__}"

    @staticmethod
    def validate_audio(
        audio: Union[str, Path, bytes, Dict[str, Any]],
        max_size_mb: float = 200.0,
        max_duration_seconds: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Validate an audio input.

        Accepts file paths, raw bytes, or audio metadata dicts.

        Args:
            audio: Audio path, bytes, or metadata dict.
            max_size_mb: Maximum file size in megabytes.
            max_duration_seconds: Optional maximum duration.

        Returns:
            (is_valid, error_message) tuple.
        """
        if audio is None:
            return False, "audio must not be None"

        if isinstance(audio, (str, Path)):
            path = Path(audio)
            ext = path.suffix.lower()
            if ext not in Validators._AUDIO_EXTENSIONS:
                return (
                    False,
                    f"unsupported audio extension '{ext}'; "
                    f"expected one of {sorted(Validators._AUDIO_EXTENSIONS)}",
                )
            if path.exists():
                size_mb = path.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    return (
                        False,
                        f"audio file too large ({size_mb:.1f} MB; max {max_size_mb} MB)",
                    )
            return True, ""

        if isinstance(audio, bytes):
            if len(audio) == 0:
                return False, "audio bytes must not be empty"
            size_mb = len(audio) / (1024 * 1024)
            if size_mb > max_size_mb:
                return (
                    False,
                    f"audio bytes too large ({size_mb:.1f} MB; max {max_size_mb} MB)",
                )
            return True, ""

        if isinstance(audio, dict):
            if "duration" in audio and max_duration_seconds is not None:
                duration = audio["duration"]
                if isinstance(duration, (int, float)) and duration > max_duration_seconds:
                    return (
                        False,
                        f"audio duration {duration:.1f}s exceeds max {max_duration_seconds}s",
                    )
            return True, ""

        return False, f"unsupported audio type: {type(audio).__name__}"

    @staticmethod
    def validate_config(
        config: Dict[str, Any],
        schema: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, str]:
        """Validate a configuration dictionary.

        If a schema is provided, checks that all required keys are
        present and that values match the expected types.

        Args:
            config: Configuration dictionary to validate.
            schema: Optional schema with 'required' keys and 'types' mapping.

        Returns:
            (is_valid, error_message) tuple.
        """
        if not isinstance(config, dict):
            return False, f"config must be a dict, got {type(config).__name__}"

        if config is None:
            return False, "config must not be None"

        if schema:
            # Check required keys
            required = schema.get("required", [])
            for key in required:
                if key not in config:
                    return False, f"missing required config key: '{key}'"

            # Check types
            type_map = schema.get("types", {})
            for key, expected_type in type_map.items():
                if key in config:
                    actual = config[key]
                    if isinstance(expected_type, tuple):
                        if not isinstance(actual, expected_type):
                            type_names = " or ".join(t.__name__ for t in expected_type)
                            return (
                                False,
                                f"config key '{key}' must be {type_names}, "
                                f"got {type(actual).__name__}",
                            )
                    else:
                        if not isinstance(actual, expected_type):
                            return (
                                False,
                                f"config key '{key}' must be {expected_type.__name__}, "
                                f"got {type(actual).__name__}",
                            )

            # Check value constraints
            constraints = schema.get("constraints", {})
            for key, constraint_fn in constraints.items():
                if key in config:
                    try:
                        is_valid, msg = constraint_fn(config[key])
                        if not is_valid:
                            return False, f"config key '{key}': {msg}"
                    except Exception as exc:
                        return False, f"config key '{key}' constraint error: {exc}"

        return True, ""
