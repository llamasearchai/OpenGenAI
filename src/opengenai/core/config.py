"""
OpenGenAI Configuration Management
Enhanced configuration with validation, environment support, and security.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field, PositiveInt, SecretStr
from pydantic_settings import BaseSettings


# --------------------------------------------------------------------------- #
#  LEAF CONFIGS
# --------------------------------------------------------------------------- #
class OpenAIConfig(BaseSettings):
    api_key: SecretStr = Field(..., alias="OPENAI_API_KEY")
    organization_id: str | None = Field(default=None, alias="OPENAI_ORG_ID")
    project_id: str | None = Field(default=None, alias="OPENAI_PROJECT_ID")
    model: str = "gpt-4o-mini"
    temperature: float = 0.0


class DatabaseConfig(BaseSettings):
    url: str = "sqlite:///./opengenai.db"
    pool_size: PositiveInt = 5
    max_overflow: int = 10
    pool_timeout: int = 30
    pool_recycle: int = 1800  # seconds


class RedisConfig(BaseSettings):
    url: str = "redis://localhost:6379/0"
    host: str = "localhost"
    port: int = 6379
    password: str | None = None
    db: int = 0


class APIConfig(BaseSettings):
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    reload: bool = False
    cors_origins: list[str] = ["*"]


class SecurityConfig(BaseSettings):
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    bcrypt_rounds: int = 12
    api_key_header: str = "X-API-KEY"


class MonitoringConfig(BaseSettings):
    metrics_enabled: bool = True
    tracing_enabled: bool = False
    logging_enabled: bool = True
    logging_level: str = "INFO"
    prometheus_enabled: bool = False


class AgentConfig(BaseSettings):
    max_agents: int = 32
    agent_timeout: int = 60  # seconds
    max_iterations: int = 100
    memory_window: int = 20
    enable_reflection: bool = False


class CacheConfig(BaseSettings):
    enabled: bool = True
    default_ttl: int = 3600  # seconds
    max_size: int = 10_000
    strategy: str = "LRU"
    compression_enabled: bool = False


# --------------------------------------------------------------------------- #
#  ROOT SETTINGS OBJECT
# --------------------------------------------------------------------------- #


class Settings(BaseSettings):
    openai: OpenAIConfig
    database: DatabaseConfig = DatabaseConfig()
    redis: RedisConfig = RedisConfig()
    api: APIConfig = APIConfig()
    security: SecurityConfig = SecurityConfig()

    def __init__(self, **values):
        # Ensure OpenAIConfig is initialized with required env or values
        if "openai" not in values:
            # Attempt to initialize OpenAIConfig with environment variable or raise error if missing
            try:
                api_key = os.getenv("OPENAI_API_KEY")
                if api_key is None:
                    raise ValueError("OPENAI_API_KEY environment variable must be set or provided in configuration.")
                values["openai"] = OpenAIConfig(OPENAI_API_KEY=SecretStr(api_key))
            except Exception as e:
                raise ValueError("OPENAI_API_KEY environment variable must be set or provided in configuration.") from e
        super().__init__(**values)
        self.monitoring = MonitoringConfig()
        self.agent = AgentConfig()
        self.cache = CacheConfig()

    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        extra = "ignore"


settings = Settings()  # ready for import
