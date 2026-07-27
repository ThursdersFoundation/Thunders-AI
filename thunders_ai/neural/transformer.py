"""Transformer Model Module for Thunders AI.

Wraps PyTorch Transformer architecture with configurable layers, heads,
and hidden size. Supports text generation, encoding, decoding, and
attention visualization.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

try:
    import torch
    import torch.nn as nn
    from torch.nn import TransformerEncoder, TransformerEncoderLayer
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class TransformerModel:
    """Transformer-based model for sequence processing and generation.

    Wraps PyTorch's Transformer architecture with a configurable number of
    layers, attention heads, and hidden dimensions. Supports forward pass,
    text generation, encoding, decoding, and attention visualization.

    Args:
        vocab_size: Size of the vocabulary.
        num_layers: Number of transformer encoder/decoder layers.
        num_heads: Number of attention heads.
        hidden_size: Dimension of feed-forward hidden layer.
        dropout: Dropout probability.
        max_seq_len: Maximum sequence length for positional encoding.
        device: Device to run the model on ('cpu' or 'cuda').

    Example::

        model = TransformerModel(vocab_size=30000)
        output = model.forward(input_ids)
        generated = model.generate(input_ids, max_new_tokens=50)
    """

    def __init__(
        self,
        vocab_size: int = 30000,
        num_layers: Optional[int] = None,
        num_heads: Optional[int] = None,
        hidden_size: Optional[int] = None,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        device: Optional[str] = None,
    ) -> None:
        if not HAS_TORCH:
            raise ImportError(
                "PyTorch is required for TransformerModel. "
                "Install it with: pip install torch"
            )

        app_cfg = get_config()
        cfg = app_cfg.neural
        self.vocab_size = vocab_size
        self.num_layers = num_layers or cfg.transformer_layers
        self.num_heads = num_heads or cfg.transformer_heads
        self.hidden_size = hidden_size or cfg.transformer_hidden
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.device = device or app_cfg.device

        self._attention_weights: Optional[List[torch.Tensor]] = None
        self._build_model()
        logger.info(
            "TransformerModel initialized: layers=%d, heads=%d, hidden=%d, device=%s",
            self.num_layers, self.num_heads, self.hidden_size, self.device,
        )

    def _build_model(self) -> None:
        """Construct the transformer model layers."""
        d_model = self.hidden_size
        self._embedding = nn.Embedding(self.vocab_size, d_model)
        self._pos_encoder = _PositionalEncoding(d_model, self.dropout, self.max_seq_len)

        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=self.num_heads,
            dim_feedforward=d_model * 4,
            dropout=self.dropout,
            batch_first=True,
        )
        self._encoder = TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self._decoder = nn.Linear(d_model, self.vocab_size)

        self._model = nn.ModuleDict({
            "embedding": self._embedding,
            "pos_encoder": self._pos_encoder,
            "encoder": self._encoder,
            "decoder": self._decoder,
        }).to(self.device)

    def forward(self, input_ids: "torch.Tensor", mask: Optional["torch.Tensor"] = None) -> "torch.Tensor":
        """Run forward pass through the transformer.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).
            mask: Optional attention mask.

        Returns:
            Logits tensor of shape (batch_size, seq_len, vocab_size).
        """
        self._model.train() if self._model.training else self._model.eval()
        x = self._embedding(input_ids) * math.sqrt(self.hidden_size)
        x = self._pos_encoder(x)

        if mask is not None:
            output = self._encoder(x, src_key_padding_mask=mask)
        else:
            output = self._encoder(x)

        logits = self._decoder(output)
        return logits

    def generate(
        self,
        input_ids: "torch.Tensor",
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ) -> "torch.Tensor":
        """Generate tokens autoregressively.

        Args:
            input_ids: Prompt token IDs of shape (batch_size, seq_len).
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature; lower = more deterministic.
            top_k: If set, only sample from top-k logits.
            top_p: If set, use nucleus sampling with this threshold.

        Returns:
            Generated token IDs including the prompt.
        """
        self._model.eval()
        generated = input_ids.clone()

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if generated.size(1) >= self.max_seq_len:
                    break
                logits = self.forward(generated)
                next_logits = logits[:, -1, :] / max(temperature, 1e-8)

                if top_k is not None:
                    top_k = min(top_k, next_logits.size(-1))
                    indices_to_remove = next_logits < torch.topk(next_logits, top_k)[0][..., -1, None]
                    next_logits[indices_to_remove] = float("-inf")

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_logits[indices_to_remove] = float("-inf")

                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)

        return generated

    def encode(self, input_ids: "torch.Tensor") -> "torch.Tensor":
        """Encode a sequence into hidden representations.

        Args:
            input_ids: Token IDs of shape (batch_size, seq_len).

        Returns:
            Encoded representations of shape (batch_size, seq_len, hidden_size).
        """
        self._model.eval()
        with torch.no_grad():
            x = self._embedding(input_ids) * math.sqrt(self.hidden_size)
            x = self._pos_encoder(x)
            encoded = self._encoder(x)
        return encoded

    def decode(self, encoded: "torch.Tensor") -> "torch.Tensor":
        """Decode hidden representations back to vocabulary logits.

        Args:
            encoded: Encoded tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        return self._decoder(encoded)

    def get_attention_weights(self) -> Optional[List["torch.Tensor"]]:
        """Retrieve stored attention weights for visualization.

        Returns:
            List of attention weight tensors, or None if not available.
        """
        return self._attention_weights

    def load_pretrained(self, path: str) -> None:
        """Load pretrained model weights from disk.

        Args:
            path: File path to the saved model state dict.

        Raises:
            FileNotFoundError: If the model file does not exist.
            RuntimeError: If the state dict cannot be loaded.
        """
        try:
            state_dict = torch.load(path, map_location=self.device, weights_only=True)
            self._model.load_state_dict(state_dict)
            logger.info("Loaded pretrained weights from %s", path)
        except FileNotFoundError:
            logger.error("Pretrained model not found at %s", path)
            raise
        except RuntimeError as exc:
            logger.error("Failed to load model weights: %s", exc)
            raise

    def save(self, path: str) -> None:
        """Save model weights to disk.

        Args:
            path: Destination file path for the state dict.
        """
        torch.save(self._model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def get_config_info(self) -> Dict[str, Any]:
        """Return model configuration as a dictionary."""
        return {
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "hidden_size": self.hidden_size,
            "dropout": self.dropout,
            "max_seq_len": self.max_seq_len,
            "device": self.device,
        }


if HAS_TORCH:

    class _PositionalEncoding(nn.Module):
        """Sinusoidal positional encoding as described in 'Attention Is All You Need'."""

        def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512) -> None:
            super().__init__()
            self.dropout = nn.Dropout(p=dropout)
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)

        def forward(self, x: "torch.Tensor") -> "torch.Tensor":
            x = x + self.pe[:, : x.size(1)]
            return self.dropout(x)
