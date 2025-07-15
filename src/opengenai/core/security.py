"""
Security and Authentication for OpenGenAI
Comprehensive security features including authentication, authorization, and rate limiting.
"""

import base64
import hashlib
import hmac
import json
import re
import secrets
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
from cryptography.fernet import Fernet
from passlib.context import CryptContext

from opengenai.core.exceptions import (
    AuthenticationError,
    SecurityError,
    ValidationError,
)
from opengenai.core.logging import audit_logger, get_logger

logger = get_logger("opengenai.security")
audit_logger = audit_logger


class PasswordManager:
    """Secure password management with hashing and validation."""

    def __init__(self):
        """Initialize password manager."""
        self.pwd_context = CryptContext(
            schemes=["bcrypt", "argon2"],
            default="bcrypt",
            bcrypt__rounds=12,
            argon2__time_cost=2,
            argon2__memory_cost=102400,
            argon2__parallelism=8,
        )

    def hash_password(self, password: str) -> str:
        """Hash a password securely."""
        if not password:
            raise ValidationError("Password cannot be empty")

        try:
            return self.pwd_context.hash(password)
        except Exception as e:
            logger.error("Failed to hash password", error=str(e))
            raise SecurityError("Password hashing failed") from e

    def verify_password(self, password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        if not password or not hashed_password:
            return False

        try:
            return self.pwd_context.verify(password, hashed_password)
        except Exception as e:
            logger.error("Failed to verify password", error=str(e))
            return False

    def needs_update(self, hashed_password: str) -> bool:
        """Check if password hash needs updating."""
        try:
            return self.pwd_context.needs_update(hashed_password)
        except Exception:
            return True

    def generate_password(self, length: int = 16) -> str:
        """Generate a secure random password."""
        if length < 8:
            raise ValidationError("Password length must be at least 8 characters")

        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*"
        password = ''.join(secrets.choice(alphabet) for _ in range(length))

        # Ensure password meets complexity requirements
        if not self.validate_password_strength(password):
            return self.generate_password(length)

        return password

    def validate_password_strength(self, password: str) -> bool:
        """Validate password strength."""
        if len(password) < 8:
            return False

        # Check for at least one lowercase, uppercase, digit, and special character
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)

        return has_lower and has_upper and has_digit and has_special


class TokenManager:
    """JWT token management for authentication."""

    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        """Initialize token manager."""
        if not secret_key:
            raise ValidationError("Secret key cannot be empty")

        self.secret_key = secret_key
        self.algorithm = algorithm
        self.default_expiry = timedelta(hours=24)

    def create_token(
        self,
        data: dict[str, Any],
        expires_in: timedelta | None = None,
    ) -> str:
        """Create a JWT token."""
        try:
            payload = data.copy()
            now = datetime.now(UTC)

            payload.update(
                {
                    "iat": now,
                    "exp": now + (expires_in or self.default_expiry),
                    "jti": secrets.token_hex(16),  # JWT ID for token revocation
                }
            )

            token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)

            logger.info(
                "Token created",
                user_id=data.get("user_id"),
                expires_in=str(expires_in or self.default_expiry),
            )

            return token
        except Exception as e:
            logger.error("Failed to create token", error=str(e))
            raise SecurityError("Token creation failed") from e

    def verify_token(self, token: str) -> dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(
                token, self.secret_key, algorithms=[self.algorithm], options={"verify_exp": True}
            )

            logger.debug(
                "Token verified",
                user_id=payload.get("user_id"),
                jti=payload.get("jti"),
            )

            return payload
        except jwt.JWTError as e:
            logger.warning("Token verification failed", error=str(e))
            raise AuthenticationError("Invalid token") from e

    def refresh_token(self, token: str) -> str:
        """Refresh a JWT token."""
        try:
            payload = self.verify_token(token)

            # Remove old timestamp fields
            payload.pop("iat", None)
            payload.pop("exp", None)
            payload.pop("jti", None)

            # Create new token
            return self.create_token(payload)
        except Exception as e:
            logger.error("Failed to refresh token", error=str(e))
            raise SecurityError("Token refresh failed") from e

    def extract_user_id(self, token: str) -> str | None:
        """Extract user ID from token."""
        try:
            payload = self.verify_token(token)
            return payload.get("user_id")
        except Exception:
            return None


class APIKeyManager:
    """API key management for service authentication."""

    def __init__(self, secret_key: str):
        """Initialize API key manager."""
        self.secret_key = secret_key

    def generate_api_key(self, user_id: str, scope: list[str]) -> str:
        """Generate a new API key."""
        try:
            key_data = {
                "user_id": user_id,
                "scope": scope,
                "created_at": datetime.now(UTC).isoformat(),
                "random": secrets.token_hex(16),
            }

            key_json = json.dumps(key_data, sort_keys=True)
            key_bytes = key_json.encode('utf-8')

            # Create HMAC signature
            signature = hmac.new(
                self.secret_key.encode('utf-8'), key_bytes, hashlib.sha256
            ).hexdigest()

            # Combine key data and signature
            api_key = base64.b64encode(key_bytes).decode('utf-8') + '.' + signature

            logger.info(
                "API key generated",
                user_id=user_id,
                scope=scope,
            )

            return api_key
        except Exception as e:
            logger.error("Failed to generate API key", error=str(e))
            raise SecurityError("API key generation failed") from e

    def verify_api_key(self, api_key: str) -> dict[str, Any]:
        """Verify an API key."""
        try:
            # Split key and signature
            parts = api_key.split('.')
            if len(parts) != 2:
                raise ValueError("Invalid API key format")

            key_data_b64, signature = parts
            key_bytes = base64.b64decode(key_data_b64)

            # Verify signature
            expected_signature = hmac.new(
                self.secret_key.encode('utf-8'), key_bytes, hashlib.sha256
            ).hexdigest()

            if not hmac.compare_digest(signature, expected_signature):
                raise ValueError("Invalid API key signature")

            # Parse key data
            key_data = json.loads(key_bytes.decode('utf-8'))

            logger.debug(
                "API key verified",
                user_id=key_data.get("user_id"),
            )

            return key_data
        except Exception as e:
            logger.warning("API key verification failed", error=str(e))
            raise AuthenticationError("Invalid API key") from e


class EncryptionManager:
    """Data encryption and decryption utilities."""

    def __init__(self, encryption_key: str | None = None):
        """Initialize encryption manager."""
        if encryption_key:
            self.key = encryption_key.encode('utf-8')
        else:
            self.key = Fernet.generate_key()

        self.cipher = Fernet(self.key)

    def encrypt(self, data: str | bytes) -> str:
        """Encrypt data."""
        try:
            if isinstance(data, str):
                data = data.encode('utf-8')

            encrypted = self.cipher.encrypt(data)
            return base64.b64encode(encrypted).decode('utf-8')
        except Exception as e:
            logger.error("Failed to encrypt data", error=str(e))
            raise SecurityError("Data encryption failed") from e

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data."""
        try:
            encrypted_bytes = base64.b64decode(encrypted_data)
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode('utf-8')
        except Exception as e:
            logger.error("Failed to decrypt data", error=str(e))
            raise SecurityError("Data decryption failed") from e

    def encrypt_dict(self, data: dict[str, Any]) -> str:
        """Encrypt a dictionary."""
        try:
            json_data = json.dumps(data, sort_keys=True)
            return self.encrypt(json_data)
        except Exception as e:
            logger.error("Failed to encrypt dictionary", error=str(e))
            raise SecurityError("Dictionary encryption failed") from e

    def decrypt_dict(self, encrypted_data: str) -> dict[str, Any]:
        """Decrypt a dictionary."""
        try:
            json_data = self.decrypt(encrypted_data)
            return json.loads(json_data)
        except Exception as e:
            logger.error("Failed to decrypt dictionary", error=str(e))
            raise SecurityError("Dictionary decryption failed") from e

    @staticmethod
    def generate_key() -> str:
        """Generate a new encryption key."""
        return Fernet.generate_key().decode('utf-8')


class RateLimiter:
    """Rate limiting for API endpoints."""

    def __init__(self, max_requests: int, window_seconds: int):
        """Initialize rate limiter."""
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: dict[str, list[float]] = {}

    def is_allowed(self, identifier: str) -> bool:
        """Check if request is allowed."""
        now = time.time()
        window_start = now - self.window_seconds

        # Clean old requests
        if identifier in self.requests:
            self.requests[identifier] = [
                req_time for req_time in self.requests[identifier] if req_time > window_start
            ]
        else:
            self.requests[identifier] = []

        # Check if under limit
        if len(self.requests[identifier]) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded",
                identifier=identifier,
                current_requests=len(self.requests[identifier]),
                max_requests=self.max_requests,
            )
            return False

        # Add current request
        self.requests[identifier].append(now)
        return True

    def get_remaining_requests(self, identifier: str) -> int:
        """Get remaining requests for identifier."""
        if identifier not in self.requests:
            return self.max_requests

        return max(0, self.max_requests - len(self.requests[identifier]))

    def get_reset_time(self, identifier: str) -> float:
        """Get time when rate limit resets."""
        if identifier not in self.requests or not self.requests[identifier]:
            return time.time()

        oldest_request = min(self.requests[identifier])
        return oldest_request + self.window_seconds


class InputValidator:
    """Input validation and sanitization utilities."""

    # Common regex patterns
    EMAIL_PATTERN = re.compile(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$')
    USERNAME_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,50}$')
    UUID_PATTERN = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$')

    @staticmethod
    def validate_email(email: str) -> bool:
        """Validate email address."""
        return bool(InputValidator.EMAIL_PATTERN.match(email))

    @staticmethod
    def validate_username(username: str) -> bool:
        """Validate username."""
        return bool(InputValidator.USERNAME_PATTERN.match(username))

    @staticmethod
    def validate_uuid(uuid_str: str) -> bool:
        """Validate UUID format."""
        return bool(InputValidator.UUID_PATTERN.match(uuid_str))

    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not value:
            return ""

        # Remove null bytes and control characters
        sanitized = ''.join(char for char in value if ord(char) >= 32 or char in '\t\n\r')

        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]

        return sanitized.strip()

    @staticmethod
    def validate_json(json_str: str) -> bool:
        """Validate JSON string."""
        try:
            json.loads(json_str)
            return True
        except (json.JSONDecodeError, TypeError):
            return False

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename for safe storage."""
        # Remove or replace dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
        sanitized = re.sub(r'\.\.+', '.', sanitized)  # Remove multiple dots
        sanitized = sanitized.strip('. ')  # Remove leading/trailing dots and spaces

        # Ensure filename is not empty
        if not sanitized:
            sanitized = "unnamed_file"

        return sanitized[:255]  # Limit length


class SecurityHeaders:
    """Security headers for web responses."""

    @staticmethod
    def get_security_headers() -> dict[str, str]:
        """Get recommended security headers."""
        return {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }

    @staticmethod
    def get_cors_headers(
        allowed_origins: list[str],
        allowed_methods: list[str],
        allowed_headers: list[str],
    ) -> dict[str, str]:
        """Get CORS headers."""
        return {
            "Access-Control-Allow-Origin": ",".join(allowed_origins),
            "Access-Control-Allow-Methods": ",".join(allowed_methods),
            "Access-Control-Allow-Headers": ",".join(allowed_headers),
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Max-Age": "86400",
        }


class SecureStorage:
    """Secure storage utilities for sensitive data."""

    def __init__(self, storage_dir: Path, encryption_key: str):
        """Initialize secure storage."""
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.encryption = EncryptionManager(encryption_key)

    def store(self, key: str, data: Any) -> None:
        """Store data securely."""
        try:
            # Sanitize key for filename
            safe_key = InputValidator.sanitize_filename(key)
            file_path = self.storage_dir / f"{safe_key}.enc"

            # Encrypt data
            if isinstance(data, dict):
                encrypted_data = self.encryption.encrypt_dict(data)
            else:
                encrypted_data = self.encryption.encrypt(str(data))

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(encrypted_data)

            logger.info("Data stored securely", key=key)
        except Exception as e:
            logger.error("Failed to store data", key=key, error=str(e))
            raise SecurityError("Secure storage failed") from e

    def retrieve(self, key: str) -> Any:
        """Retrieve data securely."""
        try:
            # Sanitize key for filename
            safe_key = InputValidator.sanitize_filename(key)
            file_path = self.storage_dir / f"{safe_key}.enc"

            if not file_path.exists():
                raise FileNotFoundError(f"Key not found: {key}")

            # Read encrypted data
            with open(file_path, encoding='utf-8') as f:
                encrypted_data = f.read()

            # Try to decrypt as dict first, then as string
            try:
                return self.encryption.decrypt_dict(encrypted_data)
            except:
                return self.encryption.decrypt(encrypted_data)
        except Exception as e:
            logger.error("Failed to retrieve data", key=key, error=str(e))
            raise SecurityError("Secure retrieval failed") from e

    def delete(self, key: str) -> None:
        """Delete stored data."""
        try:
            safe_key = InputValidator.sanitize_filename(key)
            file_path = self.storage_dir / f"{safe_key}.enc"

            if file_path.exists():
                file_path.unlink()
                logger.info("Data deleted securely", key=key)
        except Exception as e:
            logger.error("Failed to delete data", key=key, error=str(e))
            raise SecurityError("Secure deletion failed") from e

    def exists(self, key: str) -> bool:
        """Check if key exists."""
        safe_key = InputValidator.sanitize_filename(key)
        file_path = self.storage_dir / f"{safe_key}.enc"
        return file_path.exists()


class AuditTrail:
    """Audit trail for security events."""

    def __init__(self, storage: SecureStorage):
        """Initialize audit trail."""
        self.storage = storage

    def log_authentication(
        self,
        user_id: str,
        action: str,
        success: bool,
        ip_address: str | None = None,
        user_agent: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log authentication event."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "authentication",
            "user_id": user_id,
            "action": action,
            "success": success,
            "ip_address": ip_address,
            "user_agent": user_agent,
            "details": details or {},
        }

        self._store_event(event)
        audit_logger.log_authentication(user_id, action, success, details)

    def log_authorization(
        self,
        user_id: str,
        resource: str,
        action: str,
        granted: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log authorization event."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "authorization",
            "user_id": user_id,
            "resource": resource,
            "action": action,
            "granted": granted,
            "details": details or {},
        }

        self._store_event(event)
        audit_logger.log_authorization(user_id, resource, action, granted, details)

    def log_data_access(
        self,
        user_id: str,
        resource: str,
        operation: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log data access event."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "data_access",
            "user_id": user_id,
            "resource": resource,
            "operation": operation,
            "details": details or {},
        }

        self._store_event(event)
        audit_logger.log_data_access(user_id, resource, operation, details)

    def log_security_event(
        self,
        event_type: str,
        description: str,
        severity: str = "INFO",
        details: dict[str, Any] | None = None,
    ) -> None:
        """Log security event."""
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event_type": "security",
            "security_event": event_type,
            "description": description,
            "severity": severity,
            "details": details or {},
        }

        self._store_event(event)
        audit_logger.log_system_event(event_type, description, details)

    def _store_event(self, event: dict[str, Any]) -> None:
        """Store audit event."""
        try:
            event_id = f"audit_{int(time.time() * 1000000)}"
            self.storage.store(event_id, event)
        except Exception as e:
            logger.error("Failed to store audit event", error=str(e))


# Global security instances
password_manager = PasswordManager()
encryption_manager = EncryptionManager()
input_validator = InputValidator()


def generate_secure_token(length: int = 32) -> str:
    """Generate a secure random token."""
    return secrets.token_urlsafe(length)


def generate_csrf_token() -> str:
    """Generate a CSRF token."""
    return secrets.token_urlsafe(32)


def verify_csrf_token(token: str, expected_token: str) -> bool:
    """Verify CSRF token."""
    if not token or not expected_token:
        return False
    return hmac.compare_digest(token, expected_token)


def hash_sensitive_data(data: str, salt: str | None = None) -> tuple[str, str]:
    """Hash sensitive data with salt."""
    if salt is None:
        salt = secrets.token_hex(32)

    # Create hash
    hash_obj = hashlib.pbkdf2_hmac('sha256', data.encode('utf-8'), salt.encode('utf-8'), 100000)
    hashed = base64.b64encode(hash_obj).decode('utf-8')

    return hashed, salt


def verify_sensitive_data(data: str, hashed: str, salt: str) -> bool:
    """Verify sensitive data against hash."""
    try:
        expected_hash, _ = hash_sensitive_data(data, salt)
        return hmac.compare_digest(hashed, expected_hash)
    except Exception:
        return False


def create_security_manager(
    secret_key: str,
    encryption_key: str | None = None,
    storage_dir: Path | None = None,
) -> dict[str, Any]:
    """Create a complete security manager."""
    managers = {
        "password": PasswordManager(),
        "token": TokenManager(secret_key),
        "api_key": APIKeyManager(secret_key),
        "encryption": EncryptionManager(encryption_key),
        "validator": InputValidator(),
    }

    if storage_dir:
        secure_storage = SecureStorage(storage_dir, encryption_key or secret_key)
        managers["storage"] = secure_storage
        managers["audit"] = AuditTrail(secure_storage)

    return managers
