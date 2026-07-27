"""RNN Model Module for Thunders AI.

Implements RNN/LSTM/GRU models for sequential data processing with
support for bidirectional configurations and optional attention mechanisms.
"""

from __future__ import annotations

import math
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

RNN_TYPES = {"rnn", "lstm", "gru"}

if HAS_TORCH:

    class _AttentionLayer(nn.Module):
        """Bahdanau-style attention mechanism for RNN sequences."""

        def __init__(self, hidden_size: int) -> None:
            super().__init__()
            self.attn = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1, bias=False)

        def forward(
            self, hidden: "torch.Tensor", encoder_outputs: "torch.Tensor"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            seq_len = encoder_outputs.size(1)
            hidden_expanded = hidden.unsqueeze(1).repeat(1, seq_len, 1)
            energy = torch.tanh(self.attn(torch.cat((hidden_expanded, encoder_outputs), dim=2)))
            attention_scores = self.v(energy).squeeze(2)
            attention_weights = F.softmax(attention_scores, dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
            return context, attention_weights


class RNNModel:
    """Recurrent Neural Network model for sequential data.

    Supports RNN, LSTM, and GRU cells with optional bidirectional
    processing and attention mechanisms.

    Args:
        input_size: Dimension of input features per timestep.
        hidden_size: Number of hidden units in each RNN layer.
        num_layers: Number of stacked RNN layers.
        rnn_type: Type of recurrent cell ('rnn', 'lstm', or 'gru').
        bidirectional: Whether to use a bidirectional RNN.
        use_attention: Whether to apply attention over outputs.
        dropout: Dropout probability between RNN layers.
        num_classes: If > 0, adds a classification head.
        device: Device for computation.

    Example::

        model = RNNModel(input_size=128, hidden_size=256, rnn_type="lstm")
        output = model.forward(sequences)
        encoded = model.encode_sequence(sequences)
    """

    def __init__(
        self,
        input_size: int = 128,
        hidden_size: Optional[int] = None,
        num_layers: Optional[int] = None,
        rnn_type: str = "lstm",
        bidirectional: bool = False,
        use_attention: bool = False,
        dropout: float = 0.1,
        num_classes: int = 0,
        device: Optional[str] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for RNNModel. "
                "Install it with: pip install torch"
            )

        app_cfg = get_config()
        cfg = app_cfg.neural
        self.input_size = input_size
        self.hidden_size = hidden_size or cfg.rnn_hidden_size
        self.num_layers = num_layers or cfg.rnn_num_layers
        self.rnn_type = rnn_type.lower()
        if self.rnn_type not in RNN_TYPES:
            raise ValueError(f"Unsupported RNN type '{rnn_type}'. Choose from {RNN_TYPES}")
        self.bidirectional = bidirectional or cfg.rnn_bidirectional
        self.use_attention = use_attention
        self.dropout = dropout
        self.num_classes = num_classes
        self.device = device or app_cfg.device
        self.num_directions = 2 if self.bidirectional else 1

        self._build_model()
        logger.info(
            "RNNModel initialized: type=%s, hidden=%d, layers=%d, bidir=%s, attn=%s",
            self.rnn_type, self.hidden_size, self.num_layers,
            self.bidirectional, self.use_attention,
        )

    def _build_model(self) -> None:
        """Construct the RNN model layers."""
        rnn_cls = {"rnn": nn.RNN, "lstm": nn.LSTM, "gru": nn.GRU}[self.rnn_type]
        self._rnn = rnn_cls(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=self.bidirectional,
            dropout=self.dropout if self.num_layers > 1 else 0.0,
        )

        effective_hidden = self.hidden_size * self.num_directions

        if self.use_attention:
            self._attention = _AttentionLayer(effective_hidden)
        else:
            self._attention = None

        if self.num_classes > 0:
            self._classifier = nn.Linear(effective_hidden, self.num_classes)
        else:
            self._classifier = None

        self._model = nn.ModuleDict({
            "rnn": self._rnn,
            **({"attention": self._attention} if self._attention else {}),
            **({"classifier": self._classifier} if self._classifier else {}),
        }).to(self.device)

    def forward(
        self, sequences: "torch.Tensor", lengths: Optional["torch.Tensor"] = None
    ) -> Dict[str, "torch.Tensor"]:
        """Run forward pass through the RNN.

        Args:
            sequences: Input tensor of shape (batch_size, seq_len, input_size).
            lengths: Optional tensor of actual sequence lengths for packing.

        Returns:
            Dictionary with 'outputs', 'hidden', and optionally 'attention_weights'.
        """
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            rnn_out, hidden = self._rnn(packed)
            outputs, _ = nn.utils.rnn.pad_packed_sequence(rnn_out, batch_first=True)
        else:
            outputs, hidden = self._rnn(sequences)

        result: Dict[str, torch.Tensor] = {"outputs": outputs, "hidden": hidden}

        if self._attention is not None:
            if self.rnn_type == "lstm":
                query = hidden[0][-1] if not self.bidirectional else torch.cat([hidden[0][-2], hidden[0][-1]], dim=1)
            else:
                query = hidden[-1] if not self.bidirectional else torch.cat([hidden[-2], hidden[-1]], dim=1)
            context, attn_weights = self._attention(query, outputs)
            result["context"] = context
            result["attention_weights"] = attn_weights

        if self._classifier is not None:
            if self._attention is not None:
                class_input = result.get("context", outputs[:, -1, :])
            else:
                class_input = outputs[:, -1, :]
            result["logits"] = self._classifier(class_input)

        return result

    def predict(
        self, sequences: "torch.Tensor", horizon: int = 1
    ) -> "torch.Tensor":
        """Predict future sequence values autoregressively.

        Args:
            sequences: Input sequence tensor of shape (batch, seq_len, input_size).
            horizon: Number of future timesteps to predict.

        Returns:
            Predicted tensor of shape (batch, horizon, input_size).
        """
        self._model.eval()
        batch_size = sequences.size(0)
        predictions_list: List[torch.Tensor] = []

        with torch.no_grad():
            result = self.forward(sequences)
            outputs = result["outputs"]
            last_output = outputs[:, -1:, :]
            predictions_list.append(last_output)

            current_input = last_output
            for _ in range(horizon - 1):
                step_result = self.forward(current_input)
                step_output = step_result["outputs"][:, -1:, :]
                predictions_list.append(step_output)
                current_input = step_output

        return torch.cat(predictions_list, dim=1)

    def encode_sequence(self, sequences: "torch.Tensor") -> "torch.Tensor":
        """Encode a sequence into a fixed-size representation.

        Args:
            sequences: Input sequence tensor of shape (batch, seq_len, input_size).

        Returns:
            Encoded tensor of shape (batch, hidden_size * num_directions).
        """
        self._model.eval()
        with torch.no_grad():
            result = self.forward(sequences)
            if self._attention is not None and "context" in result:
                return result["context"]
            outputs = result["outputs"]
            return outputs[:, -1, :]

    def load_pretrained(self, path: str) -> None:
        """Load pretrained model weights from disk.

        Args:
            path: File path to the saved model state dict.
        """
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info("Loaded pretrained RNN weights from %s", path)
        except FileNotFoundError:
            logger.error("Model file not found: %s", path)
            raise
        except RuntimeError as exc:
            logger.error("Failed to load RNN weights: %s", exc)
            raise

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: Destination file path.
        """
        torch.save(self._model.state_dict(), path)
        logger.info("RNN model saved to %s", path)

    def get_config_info(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "rnn_type": self.rnn_type,
            "bidirectional": self.bidirectional,
            "use_attention": self.use_attention,
            "dropout": self.dropout,
            "num_classes": self.num_classes,
            "device": self.device,
        }
