"""Encryption Module for Thunders AI.

Provides AES-256-GCM encryption, SHA-256 hashing, digital signatures,
key generation, and key rotation using the cryptography library with
pure-Python fallback implementations.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import struct
import time
from typing import Any, Dict, List, Optional, Tuple

try:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec, utils as asym_utils
    from cryptography.hazmat.primitives.asymmetric.ec import (
        EllipticCurvePrivateNumbers,
        EllipticCurvePublicNumbers,
        SECP256R1,
    )
    from cryptography.exceptions import InvalidSignature as CryptoInvalidSignature
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)


class EncryptionSystem:
    """Comprehensive encryption and hashing system.

    Provides AES-256-GCM encryption/decryption, SHA-256 hashing,
    ECDSA digital signatures, key generation, and key rotation.

    Args:
        key: Optional 32-byte encryption key. Auto-generated if not provided.
        signature_key: Optional ECDSA private key for signing.
        key_rotation_days: Number of days before automatic key rotation warning.

    Example::

        enc = EncryptionSystem()
        ciphertext = enc.encrypt(b"secret data")
        plaintext = enc.decrypt(ciphertext)
        digest = enc.hash(b"important data")
    """

    KEY_SIZE = 32  # 256 bits for AES-256
    NONCE_SIZE = 12  # 96-bit nonce for GCM
    TAG_SIZE = 16  # 128-bit authentication tag

    def __init__(
        self,
        key: Optional[bytes] = None,
        key_rotation_days: Optional[int] = None,
    ) -> None:
        cfg = get_config().security
        self._key = key or self.generate_key()
        self._key_rotation_days = key_rotation_days or cfg.key_rotation_days
        self._key_created_at = time.time()
        self._key_history: List[Dict[str, Any]] = []
        self._signing_key: Optional[Any] = None
        self._verification_key: Optional[Any] = None

        if HAS_CRYPTOGRAPHY:
            self._signing_key = ec.generate_private_key(SECP256R1())
            self._verification_key = self._signing_key.public_key()
            self._aesgcm = AESGCM(self._key)

        logger.info("EncryptionSystem initialized with %s backend", "cryptography" if HAS_CRYPTOGRAPHY else "fallback")

    def generate_key(self, size: int = KEY_SIZE) -> bytes:
        """Generate a cryptographically secure random key.

        Args:
            size: Key size in bytes (default: 32 for AES-256).

        Returns:
            Random bytes of the specified length.
        """
        key = secrets.token_bytes(size)
        logger.debug("Generated new encryption key (%d bytes)", size)
        return key

    def encrypt(self, plaintext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt.
            associated_data: Optional authenticated but non-encrypted data.

        Returns:
            Ciphertext with prepended nonce (nonce + ciphertext + tag).

        Raises:
            ValueError: If plaintext is empty.
        """
        if not plaintext:
            raise ValueError("Plaintext cannot be empty")

        nonce = os.urandom(self.NONCE_SIZE)

        if HAS_CRYPTOGRAPHY:
            ciphertext = self._aesgcm.encrypt(nonce, plaintext, associated_data)
        else:
            ciphertext = self._xor_encrypt(plaintext, nonce)

        result = nonce + ciphertext
        logger.debug("Encrypted %d bytes -> %d bytes", len(plaintext), len(result))
        return result

    def decrypt(self, ciphertext: bytes, associated_data: Optional[bytes] = None) -> bytes:
        """Decrypt data using AES-256-GCM.

        Args:
            ciphertext: Encrypted data with prepended nonce.
            associated_data: Optional authenticated data (must match encryption).

        Returns:
            Decrypted plaintext bytes.

        Raises:
            ValueError: If ciphertext is too short or decryption fails.
        """
        if len(ciphertext) < self.NONCE_SIZE + self.TAG_SIZE:
            raise ValueError("Ciphertext is too short to be valid")

        nonce = ciphertext[: self.NONCE_SIZE]
        encrypted_data = ciphertext[self.NONCE_SIZE:]

        try:
            if HAS_CRYPTOGRAPHY:
                plaintext = self._aesgcm.decrypt(nonce, encrypted_data, associated_data)
            else:
                plaintext = self._xor_encrypt(encrypted_data, nonce)
        except Exception as exc:
            logger.error("Decryption failed: %s", exc)
            raise ValueError(f"Decryption failed: {exc}") from exc

        logger.debug("Decrypted %d bytes -> %d bytes", len(ciphertext), len(plaintext))
        return plaintext

    def _xor_encrypt(self, data: bytes, nonce: bytes) -> bytes:
        """Fallback XOR-based encryption (NOT secure, for testing only).

        Args:
            data: Data to encrypt/decrypt.
            nonce: Nonce bytes for pseudo-random stream.

        Returns:
            XOR-encrypted data.
        """
        stream = hashlib.sha256(self._key + nonce).digest()
        result = bytearray(len(data))
        for i, byte in enumerate(data):
            key_byte = stream[i % len(stream)]
            result[i] = byte ^ key_byte
        return bytes(result)

    def hash(self, data: bytes, algorithm: str = "sha256", salt: Optional[bytes] = None) -> bytes:
        """Compute a secure hash of the given data.

        Args:
            data: Data to hash.
            algorithm: Hash algorithm ('sha256', 'sha384', 'sha512').
            salt: Optional salt to prepend before hashing.

        Returns:
            Hash digest bytes.
        """
        algo_map = {
            "sha256": hashlib.sha256,
            "sha384": hashlib.sha384,
            "sha512": hashlib.sha512,
        }
        algo_lower = algorithm.lower()
        if algo_lower not in algo_map:
            raise ValueError(f"Unsupported hash algorithm '{algorithm}'. Choose from {list(algo_map.keys())}")

        hasher = algo_map[algo_lower]()
        if salt:
            hasher.update(salt)
        hasher.update(data)
        digest = hasher.digest()
        logger.debug("Hashed %d bytes using %s", len(data), algo_lower)
        return digest

    def sign(self, data: bytes) -> bytes:
        """Create a digital signature for the given data.

        Uses ECDSA with SHA-256 when cryptography library is available,
        falls back to HMAC-SHA256.

        Args:
            data: Data to sign.

        Returns:
            Signature bytes.
        """
        if HAS_CRYPTOGRAPHY and self._signing_key is not None:
            signature = self._signing_key.sign(
                data,
                ec.ECDSA(hashes.SHA256()),
            )
        else:
            signature = hmac.new(self._key, data, hashlib.sha256).digest()

        logger.debug("Created signature for %d bytes", len(data))
        return signature

    def verify(self, data: bytes, signature: bytes) -> bool:
        """Verify a digital signature.

        Args:
            data: Original data that was signed.
            signature: Signature to verify.

        Returns:
            True if the signature is valid, False otherwise.
        """
        try:
            if HAS_CRYPTOGRAPHY and self._verification_key is not None:
                self._verification_key.verify(signature, data, ec.ECDSA(hashes.SHA256()))
                return True
            else:
                expected = hmac.new(self._key, data, hashlib.sha256).digest()
                return hmac.compare_digest(expected, signature)
        except CryptoInvalidSignature:
            logger.warning("Invalid signature detected")
            return False
        except Exception as exc:
            logger.warning("Signature verification failed: %s", exc)
            return False

    def rotate_key(self, new_key: Optional[bytes] = None) -> bytes:
        """Rotate the encryption key, archiving the old one.

        Args:
            new_key: Optional new key. Auto-generated if not provided.

        Returns:
            The new encryption key.
        """
        old_key_info = {
            "key_hash": self.hash(self._key).hex()[:16],
            "created_at": self._key_created_at,
            "retired_at": time.time(),
        }
        self._key_history.append(old_key_info)

        self._key = new_key or self.generate_key()
        self._key_created_at = time.time()

        if HAS_CRYPTOGRAPHY:
            self._aesgcm = AESGCM(self._key)

        logger.info("Encryption key rotated (archived %d previous keys)", len(self._key_history))
        return self._key

    def is_key_rotation_due(self) -> bool:
        """Check if the current key is due for rotation.

        Returns:
            True if the key has exceeded the rotation period.
        """
        age_days = (time.time() - self._key_created_at) / 86400
        return age_days >= self._key_rotation_days

    def get_key_info(self) -> Dict[str, Any]:
        """Return information about the current key (non-sensitive)."""
        return {
            "key_hash": self.hash(self._key).hex()[:16],
            "created_at": self._key_created_at,
            "rotation_days": self._key_rotation_days,
            "rotation_due": self.is_key_rotation_due(),
            "previous_keys": len(self._key_history),
            "backend": "cryptography" if HAS_CRYPTOGRAPHY else "fallback",
        }
