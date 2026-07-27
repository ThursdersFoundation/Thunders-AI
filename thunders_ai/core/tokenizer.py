"""Tokenizer wrapper supporting BPE, WordPiece, and SentencePiece algorithms."""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from thunders_ai.config import ThundersConfig
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class TokenizerBackend(str, Enum):
    """Supported tokenizer backends."""
    BPE = "bpe"
    WORDPIECE = "wordpiece"
    SENTENCEPIECE = "sentencepiece"


# Default special token sets per backend
_SPECIAL_TOKENS: Dict[TokenizerBackend, Dict[str, str]] = {
    TokenizerBackend.BPE: {
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
    },
    TokenizerBackend.WORDPIECE: {
        "pad_token": "[PAD]",
        "bos_token": "[CLS]",
        "eos_token": "[SEP]",
        "unk_token": "[UNK]",
        "mask_token": "[MASK]",
    },
    TokenizerBackend.SENTENCEPIECE: {
        "pad_token": "<pad>",
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "mask_token": "<mask>",
    },
}


class Tokenizer:
    """Unified tokenizer interface wrapping HuggingFace tokenizers.

    Provides encode/decode, special-token management, and custom vocabulary
    support across BPE, WordPiece, and SentencePiece backends.

    Args:
        config: ThundersConfig instance.
        model_name: Pretrained tokenizer name or local path.

    Example::

        tok = Tokenizer(config, "gpt2")
        ids = tok.encode("Hello world")
        text = tok.decode(ids)
    """

    def __init__(
        self,
        config: ThundersConfig,
        model_name: Optional[str] = None,
    ) -> None:
        self._config = config
        self._model_name = model_name or getattr(config, "model_name", None)
        self._backend = TokenizerBackend(
            getattr(config, "tokenizer_backend", "bpe")
        )
        self._tokenizer: Optional[Any] = None
        self._custom_vocab: Dict[str, int] = {}

        if self._model_name:
            self._load_pretrained(self._model_name)
        logger.info("Tokenizer initialized – backend=%s", self._backend.value)

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_pretrained(self, model_name: str) -> None:
        """Load a pretrained tokenizer from HuggingFace Hub or local path."""
        try:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            logger.info("Loaded pretrained tokenizer: %s", model_name)
        except Exception as exc:
            logger.warning("Could not load tokenizer '%s': %s", model_name, exc)
            self._tokenizer = None

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        truncation: bool = True,
    ) -> List[int]:
        """Encode text into a list of token IDs.

        Args:
            text: Input string.
            add_special_tokens: Whether to prepend/append special tokens.
            max_length: Truncate to this maximum length.
            truncation: Whether to truncate if text exceeds *max_length*.

        Returns:
            List of integer token IDs.
        """
        if self._tokenizer is not None:
            encoded = self._tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                max_length=max_length,
                truncation=truncation,
            )
            return encoded["input_ids"]

        # Fallback: simple whitespace tokenizer with custom vocab
        tokens = text.split()
        ids = [self._custom_vocab.get(t, self._custom_vocab.get("<unk>", 0)) for t in tokens]
        if add_special_tokens:
            bos = self._custom_vocab.get("<s>", 1)
            eos = self._custom_vocab.get("</s>", 2)
            ids = [bos] + ids + [eos]
        if max_length and truncation and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def decode(
        self,
        token_ids: List[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode a list of token IDs back to text.

        Args:
            token_ids: Token IDs to decode.
            skip_special_tokens: Remove special tokens from output.

        Returns:
            Decoded string.
        """
        if self._tokenizer is not None:
            return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

        # Fallback: reverse custom vocab
        id_to_token = {v: k for k, v in self._custom_vocab.items()}
        tokens = [id_to_token.get(i, "<unk>") for i in token_ids]
        if skip_special_tokens:
            special = set(_SPECIAL_TOKENS[self._backend].values())
            tokens = [t for t in tokens if t not in special]
        return " ".join(tokens)

    # ------------------------------------------------------------------
    # Custom vocabulary
    # ------------------------------------------------------------------

    def add_tokens(self, tokens: List[str]) -> int:
        """Add custom tokens to the tokenizer vocabulary.

        Args:
            tokens: List of new token strings.

        Returns:
            Number of tokens actually added.
        """
        added = 0
        for token in tokens:
            if token not in self._custom_vocab:
                self._custom_vocab[token] = len(self._custom_vocab)
                added += 1
        if self._tokenizer is not None:
            added = self._tokenizer.add_tokens(tokens)
        logger.debug("Added %d custom tokens.", added)
        return added

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        if self._tokenizer is not None:
            return self._tokenizer.vocab_size
        return len(self._custom_vocab)

    @property
    def special_tokens(self) -> Dict[str, str]:
        """Return the special-token mapping for the current backend."""
        return _SPECIAL_TOKENS.get(self._backend, {})

    def token_to_id(self, token: str) -> int:
        """Map a single token string to its ID."""
        if self._tokenizer is not None:
            return self._tokenizer.convert_tokens_to_ids(token)
        return self._custom_vocab.get(token, self._custom_vocab.get("<unk>", 0))

    def id_to_token(self, token_id: int) -> str:
        """Map a token ID to its string representation."""
        if self._tokenizer is not None:
            return self._tokenizer.convert_ids_to_tokens(token_id)
        id_to_token = {v: k for k, v in self._custom_vocab.items()}
        return id_to_token.get(token_id, "<unk>")
