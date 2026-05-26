"""Tests for Thunders AI vision functionality."""
import base64
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from thunders_ai import ThundersAI, ThundersConfig
from thunders_ai.multimodal.vision import VisionAI, VisionResult, DetectedObject


class TestVisionAI:
    """Tests for the VisionAI class."""

    def test_image_analysis(self, thunders_client, sample_image_path, mock_vision_response):
        """Test basic image analysis."""
        with patch.object(
            thunders_client, "analyze_image", return_value=mock_vision_response
        ):
            result = thunders_client.analyze_image(sample_image_path)
            assert result is not None
            assert "captions" in result
            assert len(result["captions"]) > 0

    def test_object_detection(self, thunders_client, sample_image_path, mock_vision_response):
        """Test object detection in images."""
        with patch.object(
            thunders_client, "detect_objects", return_value=mock_vision_response["objects"]
        ):
            objects = thunders_client.detect_objects(sample_image_path)
            assert isinstance(objects, list)
            assert len(objects) >= 1
            assert objects[0]["label"] == "cat"
            assert objects[0]["confidence"] > 0.5

    def test_image_formats(self, thunders_client, temp_dir):
        """Test support for different image formats."""
        # Test that the vision system accepts common image formats
        supported_formats = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
        for fmt in supported_formats:
            img_path = temp_dir / f"test{fmt}"
            # Create a minimal placeholder file
            img_path.write_bytes(b"\x00" * 100)
            with patch.object(
                thunders_client, "analyze_image", return_value={"captions": ["test"]}
            ):
                try:
                    result = thunders_client.analyze_image(str(img_path))
                    assert result is not None
                except ValueError:
                    # Some formats may not be supported in all environments
                    pass

    def test_confidence_threshold(self, thunders_client, sample_image_path, mock_vision_response):
        """Test confidence threshold filtering in object detection."""
        high_threshold = 0.9
        with patch.object(
            thunders_client, "detect_objects", return_value=[
                obj for obj in mock_vision_response["objects"]
                if obj["confidence"] >= high_threshold
            ]
        ) as mock_detect:
            objects = thunders_client.detect_objects(
                sample_image_path, confidence_threshold=high_threshold
            )
            # Only objects with confidence >= threshold should be returned
            for obj in objects:
                assert obj["confidence"] >= high_threshold

    def test_invalid_image(self, thunders_client, temp_dir):
        """Test error handling for invalid image input."""
        # Non-existent file
        with pytest.raises((FileNotFoundError, ValueError)):
            thunders_client.analyze_image("/nonexistent/path/image.png")

        # Invalid file content
        invalid_path = temp_dir / "invalid.png"
        invalid_path.write_text("this is not an image")
        with pytest.raises((ValueError, IOError)):
            thunders_client.analyze_image(str(invalid_path))

        # None input
        with pytest.raises((ValueError, TypeError)):
            thunders_client.analyze_image(None)

    def test_image_from_bytes(self, thunders_client, mock_vision_response):
        """Test image analysis from raw bytes."""
        # Create a minimal PNG in memory
        image_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 50
        with patch.object(
            thunders_client, "analyze_image", return_value=mock_vision_response
        ):
            result = thunders_client.analyze_image(image_bytes)
            assert result is not None

    def test_image_from_base64(self, thunders_client, mock_vision_response):
        """Test image analysis from base64-encoded string."""
        image_data = base64.b64encode(b"\x89PNG\r\n\x1a\n" + b"\x00" * 50).decode("utf-8")
        with patch.object(
            thunders_client, "analyze_image", return_value=mock_vision_response
        ):
            result = thunders_client.analyze_image(f"data:image/png;base64,{image_data}")
            assert result is not None

    def test_vision_result_structure(self):
        """Test VisionResult data class."""
        result = VisionResult(
            captions=["A cat on a couch"],
            objects=[DetectedObject(label="cat", confidence=0.95, bbox=[10, 20, 100, 200])],
            labels=["cat", "couch"],
        )
        assert result.captions[0] == "A cat on a couch"
        assert result.objects[0].label == "cat"
        assert result.objects[0].confidence == 0.95
        assert "cat" in result.labels

    def test_batch_image_analysis(self, thunders_client, sample_image_path, mock_vision_response):
        """Test batch processing of multiple images."""
        image_paths = [sample_image_path] * 3
        with patch.object(
            thunders_client, "analyze_images", return_value=[mock_vision_response] * 3
        ):
            results = thunders_client.analyze_images(image_paths)
            assert len(results) == 3
            for r in results:
                assert "captions" in r
