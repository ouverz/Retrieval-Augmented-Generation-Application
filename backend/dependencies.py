# backend/deps.py
import os
from functools import lru_cache
from fastapi import Depends
from config.settings import get_settings
from backend.container import AppContainer
from backend.state_store import StateStore
from dotenv import load_dotenv

# Global container instance to maintain state across requests
_global_container = None

load_dotenv()

POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")


@lru_cache()
def settings():
    return get_settings()


@lru_cache()
def pg_dsn() -> str:
    # prefer an explicit env, else derive from your settings
    settings = get_settings()
    return os.getenv(
        "PG_DSN",
        settings.database.service_url
        or f"postgresql://postgres:{POSTGRES_PASSWORD}@localhost:5432/postgres",
    )


@lru_cache()
def state_store(dsn: str = Depends(pg_dsn)) -> StateStore:
    return StateStore(dsn)


def app_container(dsn: str = Depends(pg_dsn)) -> AppContainer:
    global _global_container
    if _global_container is None:
        data_dir = os.getenv(
            "DATA_DIR", "./data/documents"
        )  # Use the correct data directory
        _global_container = AppContainer(data_dir=data_dir, pg_dsn=dsn)
    return _global_container