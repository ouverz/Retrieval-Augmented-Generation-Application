# app/deps.py
import os
from functools import lru_cache
from fastapi import Depends
from config.settings import get_settings
from backend.container import AppContainer
from backend.state_store import StateStore
from core.services.cache_service import CacheService, get_cache_service
from dotenv import load_dotenv

# NEW THREAD-SAFE DEPENDENCY INJECTION SYSTEM
from backend.container import ImmutableAppContainer, create_immutable_container
from typing import Annotated
import logging

# DEPRECATED: Global container instance - will be removed in future versions
_global_container = None
logger = logging.getLogger(__name__)

load_dotenv()

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


@lru_cache()
def settings():
    return get_settings()


def pg_dsn() -> str:
    # prefer an explicit env, else derive from your settings
    settings = get_settings()
    return os.getenv(
        "PG_DSN",
        settings.database.service_url
        or f"postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/postgres",
    )


def state_store(dsn: str = Depends(pg_dsn)) -> StateStore:
    return StateStore(dsn)


# DEPRECATED: This function uses global state and will be removed
def app_container(dsn: str = Depends(pg_dsn)) -> AppContainer:
    """DEPRECATED: Uses global state. Use get_immutable_container() instead."""
    global _global_container
    if _global_container is None:
        data_dir = os.getenv(
            "DATA_DIR", "./data/documents"
        )  # Use the correct data directory
        _global_container = AppContainer(data_dir=data_dir, pg_dsn=dsn)
    return _global_container


# DEPRECATED: Alias for backward compatibility - will be removed
def get_app_container() -> AppContainer:
    """DEPRECATED: Get app container instance for non-dependency injection usage."""
    return app_container()


def cache_service() -> CacheService:
    """Get cache service dependency for injection."""
    return get_cache_service()


# Thread-safe container factory (no global state)
def get_immutable_container(dsn: str = Depends(pg_dsn)) -> ImmutableAppContainer:
    """
    Create new immutable container instance for dependency injection.
    Thread-safe, no global state, fully testable.
    """
    data_dir = os.getenv("DATA_DIR", "./data/documents")
    return create_immutable_container(data_dir=data_dir, pg_dsn=dsn)


# Type annotations for new dependency injection system
ImmutableContainerDep = Annotated[
    ImmutableAppContainer, Depends(get_immutable_container)
]
SettingsDep = Annotated[object, Depends(settings)]
StateStoreDep = Annotated[StateStore, Depends(state_store)]
CacheServiceDep = Annotated[CacheService, Depends(cache_service)]
