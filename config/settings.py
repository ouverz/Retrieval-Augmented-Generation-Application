import logging
import os
from datetime import timedelta
from functools import lru_cache
from typing import Optional
from dataclasses import dataclass

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv(dotenv_path="./.env")


def setup_logging():
    """Configure basic logging for the application."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )


class LLMSettings(BaseModel):
    """Base settings for Language Model configurations."""

    temperature: float = 0.0
    max_tokens: Optional[int] = None
    max_retries: int = 3


class OpenAISettings(LLMSettings):
    """OpenAI-specific settings extending LLMSettings."""

    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    default_model: str = Field(default="gpt-4o")
    embedding_model: str = Field(default="text-embedding-3-small")
    timeout: float = 60.0


class DatabaseSettings(BaseModel):
    """Database connection settings."""

    service_url: str = Field(default_factory=lambda: os.getenv("TIMESCALE_SERVICE_URL"))


class VectorStoreSettings(BaseModel):
    """Settings for the VectorStore."""

    table_name: str = "embeddings"
    embedding_dimensions: int = 1536
    time_partition_interval: timedelta = timedelta(days=7)


class RedisSettings(BaseModel):
    """Redis connection and caching settings."""
    
    url: str = Field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379/0"))
    embedding_cache_ttl: int = 86400  # 24 hours for embeddings
    query_cache_ttl: int = 3600      # 1 hour for query results
    session_ttl: int = 1800          # 30 minutes for sessions
    max_connections: int = 20


class Settings(BaseModel):
    """Main settings class combining all sub-settings."""

    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)


@dataclass
class HybridSearchConfig:
    """Configuration for hybrid search using Reciprocal Rank Fusion (RRF).
    
    RRF combines rankings from different search methods without requiring
    weight tuning. The rrf_k parameter controls the fusion behavior:
    - Higher k values (e.g., 100) make the fusion more conservative
    - Lower k values (e.g., 20) give more weight to top-ranked results
    - Standard value of 60 provides balanced fusion
    """

    rrf_k: int = 60  # RRF k parameter for rank fusion
    bm25_top_k: int = 10  # Number of top BM25 results to retrieve
    vector_top_k: int = 10  # Number of top vector results to retrieve
    max_results: int = 20  # Maximum number of results to return

    def __post_init__(self):
        """Validate RRF parameters"""
        if self.rrf_k < 1:
            raise ValueError(f"rrf_k must be >= 1, got {self.rrf_k}")


@lru_cache()
def get_settings() -> Settings:
    """Create and return a cached instance of the Settings."""
    settings = Settings()
    setup_logging()
    
    # Validate required environment variables
    missing_vars = []
    if not settings.openai.api_key:
        missing_vars.append("OPENAI_API_KEY")
    if not settings.database.service_url:
        missing_vars.append("TIMESCALE_SERVICE_URL")
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return settings
