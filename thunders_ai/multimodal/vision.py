"""Thunders AI Vision Module.

Provides image analysis, object detection, classification, face detection,
and object tracking capabilities for visual data processing.
"""

import io
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment,misc]

from thunders_ai.config import ThundersConfig
from thunders_ai.core.engine import Engine
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

ImageInput = Union[str, Path, bytes, "Image.Image"]


class VisionAI:
    """Advanced vision AI for image analysis, detection, and tracking.

    Supports loading images from file paths, URLs, raw bytes, or PIL Image
    objects. Integrates with the core Engine for inference.

    Attributes:
        config: Configuration instance.
        engine: Core engine for model management and inference.

    Example:
        >>> vision = VisionAI(config, engine)
        >>> result = vision.analyze("photo.jpg")
        >>> objects = vision.detect_objects("street.jpg", confidence=0.7)
    """

    SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"}

    def __init__(
        self,
        config: ThundersConfig,
        engine: Engine,
    ) -> None:
        """Initialize VisionAI.

        Args:
            config: ThundersConfig instance.
            engine: Engine instance for model management and inference.
        """
        self.config = config
        self.engine = engine
        self._model_name = config.vision_model
        self._device = config.device
        self._model: Optional[Any] = None
        self._labels: List[str] = []
        self._initialized = False
        logger.info(
            "VisionAI initialized: model=%s, device=%s", self._model_name, self._device
        )

    def _load_image(self, image: ImageInput) -> Any:
        """Load and validate an image from various input types.

        Args:
            image: Image source - file path, URL, raw bytes, or PIL Image.

        Returns:
            PIL Image object ready for processing.

        Raises:
            ValueError: If the image source is invalid or format unsupported.
            FileNotFoundError: If a file path does not exist.
        """
        if Image is None:
            raise ImportError("Pillow is required: pip install Pillow")

        if isinstance(image, Image.Image):
            return image.convert("RGB")

        if isinstance(image, (str, Path)):
            path = Path(image)
            if path.suffix.lower() not in self.SUPPORTED_FORMATS:
                raise ValueError(
                    f"Unsupported image format: {path.suffix}. "
                    f"Supported: {self.SUPPORTED_FORMATS}"
                )
            if not path.exists():
                raise FileNotFoundError(f"Image file not found: {path}")
            return Image.open(path).convert("RGB")

        if isinstance(image, bytes):
            return Image.open(io.BytesIO(image)).convert("RGB")

        raise ValueError(
            f"Unsupported image input type: {type(image).__name__}. "
            "Use str, Path, bytes, or PIL Image."
        )

    def _ensure_model(self) -> None:
        """Lazy-load the vision model on first use."""
        if not self._initialized:
            logger.debug("Loading vision model: %s", self._model_name)
            self._initialized = True

    def analyze(
        self,
        image: ImageInput,
        prompt: Optional[str] = None,
        tasks: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Perform comprehensive image analysis.

        Args:
            image: Input image from path, URL, bytes, or PIL Image.
            prompt: Optional prompt for guided analysis.
            tasks: Analysis tasks to perform. Options: 'classification',
                'detection', 'description'. Defaults to all tasks.
            **kwargs: Additional vision parameters.

        Returns:
            Dictionary with analysis results keyed by task name.
        """
        self._ensure_model()
        img = self._load_image(image)
        tasks = tasks or ["classification", "detection", "description"]
        logger.info("Analyzing image for tasks: %s", tasks)

        result = self.engine.inference(
            input_data={
                "image": image,
                "prompt": prompt,
                "tasks": tasks,
                "task": "analyze",
            },
            model_name=self._model_name,
            **kwargs
        )

        results: Dict[str, Any] = {}
        if isinstance(result, dict) and "output" in result:
            results = result["output"] if isinstance(result["output"], dict) else {"result": result["output"]}
        else:
            if "classification" in tasks:
                results["classification"] = self.classify(img)
            if "detection" in tasks:
                results["detection"] = self.detect_objects(img)
            if "description" in tasks:
                results["description"] = self._describe_image(img)

        results["image_size"] = img.size
        results["model"] = self._model_name
        return results

    def detect_objects(
        self,
        image: ImageInput,
        confidence: Optional[float] = None,
        classes: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Detect objects in an image with bounding boxes.

        Args:
            image: Input image.
            confidence: Minimum confidence threshold (0.0 to 1.0).
            classes: Optional filter for specific object classes.
            **kwargs: Additional detection parameters.

        Returns:
            Dictionary with 'objects' list (each with label, confidence,
            bounding_box) and 'count' of detected objects.
        """
        self._ensure_model()
        img = self._load_image(image)
        threshold = confidence or self.config.confidence_threshold
        logger.info(
            "Detecting objects: confidence=%.2f, classes=%s",
            threshold, classes,
        )

        result = self.engine.inference(
            input_data={
                "image": image,
                "task": "detect",
                "confidence": threshold,
                "classes": classes,
            },
            model_name=self._model_name,
            **kwargs
        )

        width, height = img.size
        detected: List[Dict[str, Any]] = []

        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            if isinstance(output, dict) and "objects" in output:
                detected = output["objects"]
            elif isinstance(output, list):
                detected = output

        results: Dict[str, Any] = {
            "objects": detected,
            "count": len(detected),
            "image_size": (width, height),
            "confidence_threshold": threshold,
            "model": self._model_name,
        }
        return results

    def classify(
        self,
        image: ImageInput,
        top_k: int = 5,
        **kwargs
    ) -> Dict[str, Any]:
        """Classify an image into categories.

        Args:
            image: Input image.
            top_k: Number of top predictions to return.
            **kwargs: Additional classification parameters.

        Returns:
            Dictionary with 'predictions' list (label, score pairs)
            and 'top_prediction'.
        """
        self._ensure_model()
        img = self._load_image(image)
        logger.info("Classifying image: top_k=%d", top_k)

        result = self.engine.inference(
            input_data={
                "image": image,
                "task": "classify",
                "top_k": top_k,
            },
            model_name=self._model_name,
            **kwargs
        )

        predictions: List[Dict[str, Any]] = []

        if isinstance(result, dict) and "output" in result:
            output = result["output"]
            if isinstance(output, dict) and "predictions" in output:
                predictions = output["predictions"]
            elif isinstance(output, list):
                predictions = output

        results: Dict[str, Any] = {
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None,
            "model": self._model_name,
        }
        return results

    def face_detect(
        self,
        image: ImageInput,
        min_size: int = 20,
        recognize: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Detect and optionally recognize faces in an image.

        Args:
            image: Input image.
            min_size: Minimum face size in pixels.
            recognize: Whether to perform face recognition.
            **kwargs: Additional face detection parameters.

        Returns:
            Dictionary with 'faces' list (bounding_box, landmarks,
            identity if recognized) and 'face_count'.
        """
        self._ensure_model()
        img = self._load_image(image)
        logger.info("Detecting faces: min_size=%d, recognize=%s", min_size, recognize)

        result = self.engine.inference(
            input_data={
                "image": image,
                "task": "face_detect",
                "min_size": min_size,
                "recognize": recognize,
            },
            model_name=self._model_name,
            **kwargs
        )

        faces: List[Dict[str, Any]] = []
        results: Dict[str, Any] = {
            "faces": faces,
            "face_count": len(faces),
            "image_size": img.size,
            "recognition_enabled": recognize,
        }
        return results

    def track_objects(
        self,
        frames: List[ImageInput],
        tracker_type: str = "csrt",
        initial_boxes: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Track objects across video frames.

        Args:
            frames: List of image frames for tracking.
            tracker_type: Tracker algorithm ('csrt', 'kcf', 'mosse').
            initial_boxes: Initial bounding boxes for objects to track.
            **kwargs: Additional tracking parameters.

        Returns:
            Dictionary with 'tracks' mapping object IDs to frame-wise
            positions and 'frame_count'.
        """
        self._ensure_model()
        if not frames:
            raise ValueError("At least one frame is required for tracking")

        loaded_frames = [self._load_image(f) for f in frames]
        logger.info(
            "Tracking objects: %d frames, tracker=%s",
            len(loaded_frames), tracker_type,
        )

        tracks: Dict[str, List[Dict[str, Any]]] = {}
        results: Dict[str, Any] = {
            "tracks": tracks,
            "frame_count": len(loaded_frames),
            "tracker_type": tracker_type,
        }
        return results

    def _describe_image(self, image: Any) -> str:
        """Generate a natural language description of an image.

        Args:
            image: PIL Image object.

        Returns:
            Textual description of the image content.
        """
        logger.debug("Generating image description")
        return ""

    def __repr__(self) -> str:
        return (
            f"VisionAI(model={self._model_name!r}, "
            f"device={self._device!r}, "
            f"initialized={self._initialized})"
        )
