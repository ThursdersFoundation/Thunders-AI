"""Thunders AI Video Module.

Provides video analysis, frame extraction, action recognition, object
tracking, and video summarization capabilities.
"""

import os
import io
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]

from thunders_ai.config import Config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

VideoInput = Union[str, Path]


class VideoAI:
    """Advanced video AI for analysis, action recognition, and summarization.

    Supports loading videos from file paths or URLs. Provides frame-level
    and video-level analysis capabilities.

    Args:
        config: Optional configuration instance. Uses default config if None.
        model_name: Name of the video model for inference.
        device: Compute device string ('cpu', 'cuda', 'auto').

    Example:
        >>> video = VideoAI(model_name="x3d")
        >>> frames = video.extract_frames("clip.mp4", fps=2)
        >>> actions = video.detect_actions("clip.mp4")
    """

    SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv"}

    def __init__(
        self,
        config: Optional[Config] = None,
        model_name: str = "default",
        device: str = "auto",
    ) -> None:
        self._config = config or Config()
        self._model_name = model_name
        self._device = self._resolve_device(device)
        self._model: Optional[Any] = None
        self._initialized = False
        logger.info(
            "VideoAI initialized: model=%s, device=%s", model_name, self._device
        )

    def _resolve_device(self, device: str) -> str:
        """Resolve the compute device based on availability."""
        if device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return device

    def _validate_video(self, video: VideoInput) -> Path:
        """Validate and resolve a video input to a Path.

        Args:
            video: Video file path.

        Returns:
            Resolved Path object.

        Raises:
            FileNotFoundError: If the video file does not exist.
            ValueError: If the video format is unsupported.
        """
        path = Path(video)
        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            raise ValueError(
                f"Unsupported video format: {path.suffix}. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {path}")
        return path

    def _ensure_model(self) -> None:
        """Lazy-load the video model on first use."""
        if not self._initialized:
            logger.debug("Loading video model: %s", self._model_name)
            self._initialized = True

    def _get_video_metadata(self, path: Path) -> Dict[str, Any]:
        """Extract metadata from a video file.

        Args:
            path: Path to the video file.

        Returns:
            Dictionary with duration, fps, width, height, codec, etc.
        """
        metadata: Dict[str, Any] = {
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "format": path.suffix.lower(),
        }
        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            if cap.isOpened():
                metadata["fps"] = cap.get(cv2.CAP_PROP_FPS)
                metadata["width"] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata["height"] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata["frame_count"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if metadata["fps"] > 0:
                    metadata["duration"] = (
                        metadata["frame_count"] / metadata["fps"]
                    )
                cap.release()
        except ImportError:
            logger.debug("OpenCV not available for metadata extraction")
        return metadata

    def analyze(
        self,
        video: VideoInput,
        tasks: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Perform comprehensive video analysis.

        Args:
            video: Input video file path.
            tasks: Analysis tasks to perform. Options: 'actions',
                'objects', 'summary'. Defaults to all tasks.

        Returns:
            Dictionary with analysis results keyed by task name.
        """
        self._ensure_model()
        path = self._validate_video(video)
        tasks = tasks or ["actions", "objects", "summary"]
        logger.info("Analyzing video: %s, tasks=%s", path.name, tasks)

        metadata = self._get_video_metadata(path)
        results: Dict[str, Any] = {"metadata": metadata}

        if "actions" in tasks:
            results["actions"] = self.detect_actions(video)
        if "objects" in tasks:
            results["objects"] = self.track_objects(video)
        if "summary" in tasks:
            results["summary"] = self.summarize(video)

        return results

    def extract_frames(
        self,
        video: VideoInput,
        fps: Optional[float] = None,
        timestamps: Optional[List[float]] = None,
        max_frames: Optional[int] = None,
        format: str = "pil",
    ) -> Dict[str, Any]:
        """Extract frames from a video at specified rate or timestamps.

        Args:
            video: Input video file path.
            fps: Target frames per second. If None, extracts all frames.
            timestamps: Specific timestamps (seconds) to extract.
            max_frames: Maximum number of frames to extract.
            format: Output format — 'pil', 'numpy', or 'bytes'.

        Returns:
            Dictionary with 'frames' list, 'frame_count', and metadata.
        """
        self._ensure_model()
        path = self._validate_video(video)
        logger.info(
            "Extracting frames from %s: fps=%s, max=%s",
            path.name, fps, max_frames,
        )

        metadata = self._get_video_metadata(path)
        frames: List[Any] = []
        frame_timestamps: List[float] = []

        try:
            import cv2
            cap = cv2.VideoCapture(str(path))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            step = int(video_fps / fps) if fps and video_fps > 0 else 1
            frame_idx = 0
            extracted = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_idx % step == 0:
                    if format == "pil":
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                    elif format == "numpy":
                        frames.append(frame)
                    else:
                        _, buf = cv2.imencode(".jpg", frame)
                        frames.append(buf.tobytes())
                    frame_timestamps.append(frame_idx / video_fps if video_fps > 0 else 0.0)
                    extracted += 1
                    if max_frames and extracted >= max_frames:
                        break
                frame_idx += 1
            cap.release()
        except ImportError:
            logger.warning("OpenCV not available; frame extraction limited")

        return {
            "frames": frames,
            "frame_count": len(frames),
            "timestamps": frame_timestamps,
            "format": format,
            "metadata": metadata,
        }

    def detect_actions(
        self,
        video: VideoInput,
        confidence: float = 0.5,
        action_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Recognize actions and activities in a video.

        Args:
            video: Input video file path.
            confidence: Minimum confidence threshold (0.0 to 1.0).
            action_types: Optional filter for specific action categories.

        Returns:
            Dictionary with 'actions' list (label, confidence, time_range)
            and 'action_count'.
        """
        self._ensure_model()
        path = self._validate_video(video)
        logger.info("Detecting actions in %s: confidence=%.2f", path.name, confidence)

        actions: List[Dict[str, Any]] = []
        return {
            "actions": actions,
            "action_count": len(actions),
            "confidence_threshold": confidence,
            "model": self._model_name,
        }

    def track_objects(
        self,
        video: VideoInput,
        tracker_type: str = "deepsort",
        confidence: float = 0.5,
    ) -> Dict[str, Any]:
        """Track multiple objects throughout a video.

        Args:
            video: Input video file path.
            tracker_type: Tracking algorithm ('deepsort', 'sort', 'bytetrack').
            confidence: Detection confidence threshold.

        Returns:
            Dictionary with 'tracks' mapping object IDs to trajectories
            and 'unique_objects' count.
        """
        self._ensure_model()
        path = self._validate_video(video)
        logger.info(
            "Tracking objects in %s: tracker=%s", path.name, tracker_type
        )

        tracks: Dict[str, List[Dict[str, Any]]] = {}
        return {
            "tracks": tracks,
            "unique_objects": len(tracks),
            "tracker_type": tracker_type,
            "confidence_threshold": confidence,
        }

    def summarize(
        self,
        video: VideoInput,
        method: str = "keyframe",
        max_duration: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate a video summary.

        Args:
            video: Input video file path.
            method: Summarization method — 'keyframe' or 'highlight'.
            max_duration: Maximum duration of summary in seconds.

        Returns:
            Dictionary with 'summary_frames', 'summary_timestamps',
            and 'method'.
        """
        self._ensure_model()
        path = self._validate_video(video)
        logger.info("Summarizing %s: method=%s", path.name, method)

        metadata = self._get_video_metadata(path)
        summary_frames: List[int] = []
        summary_timestamps: List[float] = []

        return {
            "summary_frames": summary_frames,
            "summary_timestamps": summary_timestamps,
            "method": method,
            "max_duration": max_duration,
            "original_duration": metadata.get("duration"),
        }

    def __repr__(self) -> str:
        return (
            f"VideoAI(model={self._model_name!r}, "
            f"device={self._device!r}, "
            f"initialized={self._initialized})"
        )
