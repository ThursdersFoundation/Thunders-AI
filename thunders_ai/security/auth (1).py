"""Authentication and Authorization Module for Thunders AI.

Provides user authentication, JWT token management, role-based access
control (RBAC), and API key management.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import time
import base64
from typing import Any, Dict, List, Optional, Set, Tuple

from thunders_ai.config import get_config
from thunders_ai.logger import get_logger

logger = get_logger(__name__)

# Role hierarchy: higher index = more permissions
DEFAULT_ROLES: Dict[str, Set[str]] = {
    "viewer": {"read"},
    "editor": {"read", "write"},
    "admin": {"read", "write", "delete", "manage_users"},
    "superadmin": {"read", "write", "delete", "manage_users", "manage_system"},
}


class _JWTCodec:
    """Minimal JWT encode/decode using HMAC-SHA256."""

    @staticmethod
    def _b64url_encode(data: bytes) -> str:
        return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")

    @staticmethod
    def _b64url_decode(data: str) -> bytes:
        padding = 4 - len(data) % 4
        if padding != 4:
            data += "=" * padding
        return base64.urlsafe_b64decode(data)

    @classmethod
    def encode(cls, payload: Dict[str, Any], secret: str, algorithm: str = "HS256") -> str:
        """Encode a JWT token.

        Args:
            payload: Token claims dictionary.
            secret: Signing secret key.
            algorithm: Signing algorithm (only HS256 supported).

        Returns:
            Encoded JWT string.
        """
        header = {"alg": algorithm, "typ": "JWT"}
        header_b64 = cls._b64url_encode(json.dumps(header, separators=(",", ":")).encode())
        payload_b64 = cls._b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
        signing_input = f"{header_b64}.{payload_b64}"
        signature = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
        signature_b64 = cls._b64url_encode(signature)
        return f"{signing_input}.{signature_b64}"

    @classmethod
    def decode(cls, token: str, secret: str) -> Optional[Dict[str, Any]]:
        """Decode and verify a JWT token.

        Args:
            token: Encoded JWT string.
            secret: Signing secret key.

        Returns:
            Decoded payload if valid, None otherwise.
        """
        try:
            parts = token.split(".")
            if len(parts) != 3:
                return None
            header_b64, payload_b64, signature_b64 = parts
            signing_input = f"{header_b64}.{payload_b64}"
            expected_sig = hmac.new(secret.encode(), signing_input.encode(), hashlib.sha256).digest()
            actual_sig = cls._b64url_decode(signature_b64)
            if not hmac.compare_digest(expected_sig, actual_sig):
                return None
            payload = json.loads(cls._b64url_decode(payload_b64))
            return payload
        except Exception as exc:
            logger.warning("JWT decode failed: %s", exc)
            return None


class AuthManager:
    """Authentication and authorization manager.

    Provides user authentication, JWT token lifecycle management,
    role-based access control (RBAC), and API key management.

    Args:
        secret_key: Secret key for JWT signing. Auto-generated if not provided.
        token_expiry_minutes: JWT token expiration time in minutes.
        roles: Custom role-to-permissions mapping.

    Example::

        auth = AuthManager()
        token = auth.create_token(user_id="alice", role="admin")
        is_valid = auth.verify_token(token)
        has_perm = auth.authorize("alice", "delete")
    """

    def __init__(
        self,
        secret_key: Optional[str] = None,
        token_expiry_minutes: Optional[int] = None,
        roles: Optional[Dict[str, Set[str]]] = None,
    ) -> None:
        cfg = get_config().security
        self._secret_key = secret_key or cfg.jwt_secret_key or secrets.token_hex(32)
        self._token_expiry_minutes = token_expiry_minutes or cfg.jwt_expiry_minutes
        self._jwt_algorithm = cfg.jwt_algorithm
        self._roles = roles or dict(DEFAULT_ROLES)

        self._users: Dict[str, Dict[str, Any]] = {}
        self._revoked_tokens: Set[str] = set()
        self._api_keys: Dict[str, Dict[str, Any]] = {}

        logger.info("AuthManager initialized (token expiry: %d min)", self._token_expiry_minutes)

    def register_user(
        self, user_id: str, password: str, role: str = "viewer", metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Register a new user.

        Args:
            user_id: Unique user identifier.
            password: Plain-text password (will be hashed).
            role: Assigned role name.
            metadata: Optional user metadata.

        Returns:
            True if registration succeeded, False if user already exists.
        """
        if user_id in self._users:
            logger.warning("User '%s' already exists", user_id)
            return False
        if role not in self._roles:
            logger.warning("Unknown role '%s' for user '%s'", role, user_id)
            return False

        password_hash = hashlib.sha256((password + self._secret_key).encode()).hexdigest()
        self._users[user_id] = {
            "password_hash": password_hash,
            "role": role,
            "created_at": time.time(),
            "metadata": metadata or {},
        }
        logger.info("User '%s' registered with role '%s'", user_id, role)
        return True

    def authenticate(self, user_id: str, password: str) -> Optional[str]:
        """Authenticate a user with credentials.

        Args:
            user_id: User identifier.
            password: Plain-text password.

        Returns:
            JWT token string if authentication succeeds, None otherwise.
        """
        user = self._users.get(user_id)
        if user is None:
            logger.warning("Authentication failed: unknown user '%s'", user_id)
            return None

        password_hash = hashlib.sha256((password + self._secret_key).encode()).hexdigest()
        if not hmac.compare_digest(password_hash, user["password_hash"]):
            logger.warning("Authentication failed: invalid password for '%s'", user_id)
            return None

        token = self.create_token(user_id=user_id, role=user["role"])
        logger.info("User '%s' authenticated successfully", user_id)
        return token

    def authorize(self, user_id: str, permission: str) -> bool:
        """Check if a user has a specific permission.

        Args:
            user_id: User identifier.
            permission: Permission string to check (e.g., 'read', 'write').

        Returns:
            True if the user has the permission, False otherwise.
        """
        user = self._users.get(user_id)
        if user is None:
            logger.warning("Authorization check for unknown user '%s'", user_id)
            return False

        role = user["role"]
        permissions = self._roles.get(role, set())
        has_permission = permission in permissions
        if not has_permission:
            logger.debug("User '%s' (role: %s) lacks permission '%s'", user_id, role, permission)
        return has_permission

    def create_token(
        self,
        user_id: str,
        role: str = "viewer",
        extra_claims: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a new JWT token.

        Args:
            user_id: Subject user identifier.
            role: User role to embed in the token.
            extra_claims: Additional claims to include.

        Returns:
            Encoded JWT token string.
        """
        now = time.time()
        payload: Dict[str, Any] = {
            "sub": user_id,
            "role": role,
            "iat": now,
            "exp": now + self._token_expiry_minutes * 60,
            "jti": secrets.token_hex(16),
        }
        if extra_claims:
            payload.update(extra_claims)

        token = _JWTCodec.encode(payload, self._secret_key, self._jwt_algorithm)
        logger.debug("Created token for user '%s' (role: %s)", user_id, role)
        return token

    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify a JWT token and return its payload.

        Args:
            token: JWT token string to verify.

        Returns:
            Token payload if valid, None otherwise.
        """
        if token in self._revoked_tokens:
            logger.warning("Revoked token presented")
            return None

        payload = _JWTCodec.decode(token, self._secret_key)
        if payload is None:
            logger.warning("Invalid JWT token signature")
            return None

        if payload.get("exp", 0) < time.time():
            logger.warning("Expired JWT token for user '%s'", payload.get("sub"))
            return None

        return payload

    def revoke_token(self, token: str) -> bool:
        """Revoke a JWT token, preventing further use.

        Args:
            token: JWT token string to revoke.

        Returns:
            True if the token was revoked, False if already revoked.
        """
        if token in self._revoked_tokens:
            return False
        self._revoked_tokens.add(token)
        logger.info("Token revoked (jti: %s)", token[-16:])
        return True

    def create_api_key(self, user_id: str, permissions: Optional[Set[str]] = None, name: Optional[str] = None) -> str:
        """Create a new API key for programmatic access.

        Args:
            user_id: Associated user identifier.
            permissions: Set of permissions for this key. Defaults to user's role permissions.
            name: Optional human-readable name for the key.

        Returns:
            Generated API key string.
        """
        api_key = f"tk-{secrets.token_hex(24)}"
        user = self._users.get(user_id)
        if permissions is None and user:
            permissions = self._roles.get(user.get("role", "viewer"), set())

        self._api_keys[api_key] = {
            "user_id": user_id,
            "permissions": permissions or set(),
            "name": name or f"key-{len(self._api_keys) + 1}",
            "created_at": time.time(),
            "last_used": None,
        }
        logger.info("API key created for user '%s' (name: %s)", user_id, name)
        return api_key

    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key and return its metadata.

        Args:
            api_key: API key string to verify.

        Returns:
            API key metadata if valid, None otherwise.
        """
        key_data = self._api_keys.get(api_key)
        if key_data is None:
            logger.warning("Invalid API key presented")
            return None
        key_data["last_used"] = time.time()
        return key_data

    def revoke_api_key(self, api_key: str) -> bool:
        """Revoke an API key.

        Args:
            api_key: API key to revoke.

        Returns:
            True if revoked, False if not found.
        """
        if api_key in self._api_keys:
            del self._api_keys[api_key]
            logger.info("API key revoked")
            return True
        return False

    def get_user_permissions(self, user_id: str) -> Set[str]:
        """Get all permissions for a user based on their role.

        Args:
            user_id: User identifier.

        Returns:
            Set of permission strings.
        """
        user = self._users.get(user_id)
        if user is None:
            return set()
        return self._roles.get(user["role"], set())

    def add_role(self, role_name: str, permissions: Set[str]) -> None:
        """Add or update a role definition.

        Args:
            role_name: Name of the role.
            permissions: Set of permissions for the role.
        """
        self._roles[role_name] = permissions
        logger.info("Role '%s' defined with permissions: %s", role_name, permissions)

    def get_info(self) -> Dict[str, Any]:
        """Return non-sensitive manager configuration info."""
        return {
            "token_expiry_minutes": self._token_expiry_minutes,
            "jwt_algorithm": self._jwt_algorithm,
            "registered_users": len(self._users),
            "defined_roles": list(self._roles.keys()),
            "active_api_keys": len(self._api_keys),
            "revoked_tokens": len(self._revoked_tokens),
        }
