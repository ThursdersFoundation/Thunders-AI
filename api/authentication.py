"""Thunders AI API Authentication - Key and JWT authentication with role-based access.

Provides API key and JWT token authentication for securing API endpoints,
along with utilities for key generation, verification, and role-based
permission management.
"""

import hashlib
import hmac
import logging
import os
import secrets
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Default secret for HMAC operations (override via environment variable)
_SECRET_KEY: str = os.environ.get("THUNDERS_SECRET_KEY", "change-me-in-production-32ch!")


class Role(str, Enum):
    """User roles for role-based access control.

    Roles are hierarchical: ADMIN includes all permissions of USER,
    and USER includes all permissions of GUEST.
    """
    GUEST = "guest"
    USER = "user"
    ADMIN = "admin"


# Permission sets for each role
ROLE_PERMISSIONS: Dict[Role, Set[str]] = {
    Role.GUEST: {"chat:read", "models:list", "health:check"},
    Role.USER: {
        "chat:read", "chat:write",
        "vision:analyze",
        "speech:tts", "speech:stt",
        "models:list", "health:check",
    },
    Role.ADMIN: {
        "chat:read", "chat:write",
        "vision:analyze",
        "speech:tts", "speech:stt",
        "robotics:navigate",
        "models:list", "models:manage",
        "health:check", "admin:all",
    },
}


@dataclass
class APIKeyRecord:
    """Represents a stored API key with metadata.

    Attributes:
        key_hash: SHA-256 hash of the API key for secure storage.
        key_prefix: First 8 characters of the key for identification.
        name: Human-readable name for the key.
        role: Role assigned to this key.
        created_at: Timestamp when the key was created.
        expires_at: Optional expiration timestamp.
        is_active: Whether the key is currently active.
        metadata: Additional metadata.
    """
    key_hash: str
    key_prefix: str
    name: str
    role: Role
    created_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class APIKeyAuth:
    """API key authentication provider.

    Manages creation, verification, and revocation of API keys.
    Keys are stored as hashes to prevent leakage.
    """

    def __init__(self, secret_key: Optional[str] = None) -> None:
        """Initialize the API key auth provider.

        Args:
            secret_key: Secret key for HMAC operations. Falls back to
                the THUNDERS_SECRET_KEY environment variable.
        """
        self._secret = secret_key or _SECRET_KEY
        self._keys: Dict[str, APIKeyRecord] = {}

    def create_key(
        self,
        name: str,
        role: Role = Role.USER,
        expires_in_seconds: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generate a new API key.

        Args:
            name: Human-readable name for the key.
            role: Role to assign to this key.
            expires_in_seconds: Optional TTL in seconds.
            metadata: Optional additional metadata.

        Returns:
            The raw API key string (shown only once).
        """
        raw_key = f"thunders_{secrets.token_hex(24)}"
        key_hash = self._hash_key(raw_key)
        key_prefix = raw_key[:12]
        expires_at = time.time() + expires_in_seconds if expires_in_seconds else None

        self._keys[key_hash] = APIKeyRecord(
            key_hash=key_hash,
            key_prefix=key_prefix,
            name=name,
            role=role,
            expires_at=expires_at,
            metadata=metadata or {},
        )
        logger.info("API key created: prefix=%s, name=%s, role=%s", key_prefix, name, role.value)
        return raw_key

    def verify_key(self, raw_key: str) -> Optional[APIKeyRecord]:
        """Verify an API key and return its record.

        Args:
            raw_key: The raw API key string to verify.

        Returns:
            The APIKeyRecord if valid, None otherwise.
        """
        key_hash = self._hash_key(raw_key)
        record = self._keys.get(key_hash)

        if record is None:
            logger.warning("API key not found: %s...", raw_key[:12])
            return None

        if not record.is_active:
            logger.warning("API key is inactive: %s", record.key_prefix)
            return None

        if record.expires_at and time.time() > record.expires_at:
            logger.warning("API key expired: %s", record.key_prefix)
            return None

        return record

    def revoke_key(self, key_prefix: str) -> bool:
        """Revoke an API key by its prefix.

        Args:
            key_prefix: The first 12 characters of the key.

        Returns:
            True if a key was revoked, False if not found.
        """
        for record in self._keys.values():
            if record.key_prefix == key_prefix:
                record.is_active = False
                logger.info("API key revoked: %s", key_prefix)
                return True
        return False

    def has_permission(self, raw_key: str, permission: str) -> bool:
        """Check if an API key has a specific permission.

        Args:
            raw_key: The raw API key string.
            permission: Permission string to check (e.g., 'chat:write').

        Returns:
            True if the key is valid and has the permission.
        """
        record = self.verify_key(raw_key)
        if record is None:
            return False
        perms = ROLE_PERMISSIONS.get(record.role, set())
        return permission in perms

    def _hash_key(self, raw_key: str) -> str:
        """Compute a SHA-256 hash of the API key.

        Args:
            raw_key: The raw API key string.

        Returns:
            Hex-encoded SHA-256 hash.
        """
        return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


class JWTAuth:
    """JWT token authentication provider.

    Provides lightweight JWT-like token creation and verification
    using HMAC-SHA256. Not a full JWT library — for production,
    consider using PyJWT or python-jose.
    """

    def __init__(self, secret_key: Optional[str] = None) -> None:
        """Initialize the JWT auth provider.

        Args:
            secret_key: Secret key for signing tokens.
        """
        self._secret = secret_key or _SECRET_KEY

    def create_token(
        self,
        subject: str,
        role: Role = Role.USER,
        expires_in_seconds: int = 3600,
        claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a signed JWT-like token.

        Args:
            subject: Subject identifier (e.g., user ID).
            role: Role to embed in the token.
            expires_in_seconds: Token TTL in seconds.
            claims: Optional additional claims.

        Returns:
            Encoded token string.
        """
        import base64
        import json

        payload = {
            "sub": subject,
            "role": role.value,
            "iat": int(time.time()),
            "exp": int(time.time()) + expires_in_seconds,
            "jti": secrets.token_hex(8),
            **(claims or {}),
        }
        header = {"alg": "HS256", "typ": "JWT"}
        header_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
        payload_b64 = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(
            self._secret.encode(), signing_input.encode(), hashlib.sha256
        ).hexdigest()
        return f"{signing_input}.{signature}"

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify and decode a JWT-like token.

        Args:
            token: The token string to verify.

        Returns:
            Decoded payload dict if valid, None otherwise.
        """
        import base64
        import json

        parts = token.split(".")
        if len(parts) != 3:
            logger.warning("Invalid token format: expected 3 parts")
            return None

        header_b64, payload_b64, signature = parts
        signing_input = f"{header_b64}.{payload_b64}"
        expected_sig = hmac.new(
            self._secret.encode(), signing_input.encode(), hashlib.sha256
        ).hexdigest()

        if not hmac.compare_digest(signature, expected_sig):
            logger.warning("Token signature verification failed")
            return None

        # Restore base64 padding
        padding = 4 - len(payload_b64) % 4
        padded = payload_b64 + "=" * padding if padding != 4 else payload_b64
        try:
            payload = json.loads(base64.urlsafe_b64decode(padded))
        except Exception as exc:
            logger.warning("Failed to decode token payload: %s", exc)
            return None

        if payload.get("exp", 0) < time.time():
            logger.warning("Token expired for subject: %s", payload.get("sub"))
            return None

        return payload

    def has_permission(self, token: str, permission: str) -> bool:
        """Check if a JWT token grants a specific permission.

        Args:
            token: The token string.
            permission: Permission string to check.

        Returns:
            True if the token is valid and grants the permission.
        """
        payload = self.verify_token(token)
        if payload is None:
            return False
        role_str = payload.get("role", "guest")
        try:
            role = Role(role_str)
        except ValueError:
            return False
        return permission in ROLE_PERMISSIONS.get(role, set())


# ---------------------------------------------------------------------------
# Module-level convenience functions
# ---------------------------------------------------------------------------

_default_key_auth = APIKeyAuth()
_default_jwt_auth = JWTAuth()


def create_api_key(
    name: str,
    role: Role = Role.USER,
    expires_in_seconds: Optional[int] = None,
) -> str:
    """Create a new API key using the default auth provider.

    Args:
        name: Human-readable name for the key.
        role: Role to assign.
        expires_in_seconds: Optional TTL in seconds.

    Returns:
        The raw API key string.
    """
    return _default_key_auth.create_key(name, role, expires_in_seconds)


def verify_api_key(raw_key: str) -> Optional[APIKeyRecord]:
    """Verify an API key using the default auth provider.

    Args:
        raw_key: The raw API key string.

    Returns:
        The APIKeyRecord if valid, None otherwise.
    """
    return _default_key_auth.verify_key(raw_key)
