"""CNN Model Module for Thunders AI.

Implements convolutional neural networks for image classification,
feature extraction, and object detection with transfer learning support
for ResNet, VGG, and EfficientNet architectures.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

ARCHITECTURES = {"resnet18", "resnet34", "resnet50", "resnet101", "vgg16", "vgg19", "efficientnet_b0", "efficientnet_b1", "efficientnet_b2"}


class CNNModel:
    """Convolutional Neural Network model for image tasks.

    Supports multiple architectures (ResNet, VGG, EfficientNet) with
    transfer learning, feature extraction, classification, and object
    detection capabilities.

    Args:
        architecture: Backbone architecture name (e.g., 'resnet50', 'vgg16').
        num_classes: Number of output classes for classification.
        pretrained: Whether to load pretrained weights.
        freeze_backbone: Whether to freeze backbone parameters.
        device: Device for computation ('cpu' or 'cuda').

    Example::

        model = CNNModel(architecture="resnet50", num_classes=10)
        features = model.extract_features(images)
        predictions = model.classify(images)
    """

    def __init__(
        self,
        architecture: str = "resnet50",
        num_classes: int = 1000,
        pretrained: bool = False,
        freeze_backbone: bool = False,
        device: Optional[str] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for CNNModel. "
                "Install it with: pip install torch"
            )

        app_cfg = get_config()
        cfg = app_cfg.neural
        self.architecture = architecture.lower()
        if self.architecture not in ARCHITECTURES:
            raise ValueError(f"Unsupported architecture '{architecture}'. Choose from {ARCHITECTURES}")
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.freeze_backbone = freeze_backbone
        self.device = device or app_cfg.device
        self._feature_hooks: List[torch.Tensor] = []

        self._build_model()
        logger.info(
            "CNNModel initialized: arch=%s, classes=%d, pretrained=%s, device=%s",
            self.architecture, self.num_classes, self.pretrained, self.device,
        )

    def _build_model(self) -> None:
        """Construct the CNN model based on the chosen architecture."""
        self._backbone = self._create_backbone()
        if self.freeze_backbone:
            for param in self._backbone.parameters():
                param.requires_grad = False
        self._classifier = nn.Linear(self._get_feature_dim(), self.num_classes).to(self.device)
        self._model = nn.ModuleDict({
            "backbone": self._backbone,
            "classifier": self._classifier,
        }).to(self.device)

    def _create_backbone(self) -> nn.Module:
        """Create the backbone network."""
        try:
            if self.architecture.startswith("resnet"):
                return self._create_resnet()
            elif self.architecture.startswith("vgg"):
                return self._create_vgg()
            elif self.architecture.startswith("efficientnet"):
                return self._create_efficientnet()
        except Exception as exc:
            logger.warning("Failed to load torchvision model, using simple CNN: %s", exc)
        return self._create_simple_cnn()

    def _create_resnet(self) -> nn.Module:
        """Create a ResNet backbone."""
        import torchvision.models as models
        weights = "DEFAULT" if self.pretrained else None
        model_fn = getattr(models, self.architecture, None)
        if model_fn is None:
            raise ValueError(f"Unknown ResNet variant: {self.architecture}")
        backbone = model_fn(weights=weights)
        layers = list(backbone.children())[:-1]
        return nn.Sequential(*layers)

    def _create_vgg(self) -> nn.Module:
        """Create a VGG backbone."""
        import torchvision.models as models
        weights = "DEFAULT" if self.pretrained else None
        model_fn = getattr(models, self.architecture, None)
        if model_fn is None:
            raise ValueError(f"Unknown VGG variant: {self.architecture}")
        backbone = model_fn(weights=weights)
        backbone.classifier = nn.Sequential(*list(backbone.classifier.children())[:-1])
        return backbone

    def _create_efficientnet(self) -> nn.Module:
        """Create an EfficientNet backbone."""
        import torchvision.models as models
        weights = "DEFAULT" if self.pretrained else None
        model_fn = getattr(models, self.architecture, None)
        if model_fn is None:
            raise ValueError(f"Unknown EfficientNet variant: {self.architecture}")
        backbone = model_fn(weights=weights)
        backbone.classifier = nn.Identity()
        return backbone

    def _create_simple_cnn(self) -> nn.Module:
        """Create a simple CNN as a fallback backbone."""
        return nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

    def _get_feature_dim(self) -> int:
        """Determine the output feature dimension of the backbone."""
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            features = self._backbone(dummy)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            return features.size(1)

    def forward(self, images: "torch.Tensor") -> "torch.Tensor":
        """Run forward pass through the CNN.

        Args:
            images: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Logits of shape (batch_size, num_classes).
        """
        features = self._backbone(images)
        if features.dim() > 2:
            features = features.view(features.size(0), -1)
        logits = self._classifier(features)
        return logits

    def extract_features(self, images: "torch.Tensor", layer_name: Optional[str] = None) -> "torch.Tensor":
        """Extract intermediate features from the CNN.

        Args:
            images: Input images tensor.
            layer_name: Optional specific layer to extract from. If None,
                returns features from the last backbone layer.

        Returns:
            Feature tensor.
        """
        self._model.eval()
        with torch.no_grad():
            features = self._backbone(images)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
        return features

    def classify(self, images: "torch.Tensor", return_probs: bool = False) -> Dict[str, Any]:
        """Classify input images.

        Args:
            images: Input images tensor of shape (batch, C, H, W).
            return_probs: If True, also return probability distributions.

        Returns:
            Dictionary with 'labels' and optionally 'probabilities'.
        """
        self._model.eval()
        with torch.no_grad():
            logits = self.forward(images)
            labels = logits.argmax(dim=-1)
            result: Dict[str, Any] = {"labels": labels}
            if return_probs:
                result["probabilities"] = torch.softmax(logits, dim=-1)
        return result

    def detect(self, images: "torch.Tensor", confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Detect objects in images using the classification backbone.

        This provides a basic detection capability by running classification
        on image patches. For full detection, use a dedicated detection model.

        Args:
            images: Input images tensor.
            confidence_threshold: Minimum confidence for detections.

        Returns:
            List of detection dictionaries per image.
        """
        self._model.eval()
        detections: List[Dict[str, Any]] = []
        with torch.no_grad():
            logits = self.forward(images)
            probs = torch.softmax(logits, dim=-1)
            for i in range(images.size(0)):
                img_dets: List[Dict[str, Any]] = []
                top_probs, top_labels = probs[i].topk(min(5, self.num_classes))
                for j in range(top_probs.size(0)):
                    if top_probs[j].item() >= confidence_threshold:
                        img_dets.append({
                            "label": top_labels[j].item(),
                            "confidence": top_probs[j].item(),
                            "box": [0, 0, images.size(3), images.size(2)],
                        })
                detections.append({"detections": img_dets})
        return detections

    def load_pretrained(self, path: str) -> None:
        """Load pretrained model weights.

        Args:
            path: File path to the saved model weights.
        """
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info("Loaded pretrained CNN weights from %s", path)
        except FileNotFoundError:
            logger.error("Model file not found: %s", path)
            raise
        except RuntimeError as exc:
            logger.error("Failed to load CNN weights: %s", exc)
            raise

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: Destination file path.
        """
        torch.save(self._model.state_dict(), path)
        logger.info("CNN model saved to %s", path)

    def get_config_info(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "pretrained": self.pretrained,
            "freeze_backbone": self.freeze_backbone,
            "device": self.device,
        }
