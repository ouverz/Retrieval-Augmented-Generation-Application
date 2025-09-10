# app/routers/cache.py
import logging
from fastapi import APIRouter, HTTPException, Depends, Request
from typing import Dict, Any
from enum import Enum
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.dependencies import app_container
from backend.container import AppContainer
from backend.security import require_api_key

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


class CacheType(str, Enum):
    """Cache types for clearing operations."""
    all = "all"
    embeddings = "embeddings"
    queries = "queries"
    sessions = "sessions"


@router.get("/stats")
@limiter.limit("30/minute")
def get_cache_stats(
    request: Request,
    container: AppContainer = Depends(app_container),
    api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Get detailed cache statistics and health information."""
    try:
        if not container.cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        stats = container.cache_service.get_cache_stats()
        logger.info("[Backend] Cache stats retrieved")
        return stats
        
    except Exception as e:
        logger.error(f"[Backend] Failed to get cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get cache stats: {str(e)}")


@router.post("/clear")
@limiter.limit("10/hour")
def clear_cache(
    request: Request,
    cache_type: CacheType = CacheType.all,
    container: AppContainer = Depends(app_container),
    api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Clear cache by type (all, embeddings, queries, sessions)."""
    try:
        if not container.cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        success = container.cache_service.clear_cache(cache_type.value)
        
        if success:
            logger.info(f"[Backend] Cleared {cache_type.value} cache")
            return {
                "status": "success",
                "message": f"Cleared {cache_type.value} cache",
                "cache_type": cache_type.value
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to clear {cache_type.value} cache")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Backend] Failed to clear cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")


@router.get("/health")
@limiter.limit("60/minute")
def get_cache_health(
    request: Request,
    container: AppContainer = Depends(app_container),
    api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Get cache service health status."""
    try:
        if not container.cache_service:
            return {
                "available": False,
                "error": "Cache service not initialized"
            }
        
        is_available = container.cache_service.is_available()
        stats = container.cache_service.get_cache_stats() if is_available else {}
        
        return {
            "available": is_available,
            "stats": stats,
            "degraded_mode": not is_available
        }
        
    except Exception as e:
        logger.error(f"[Backend] Failed to check cache health: {e}")
        return {
            "available": False,
            "error": str(e),
            "degraded_mode": True
        }




@router.post("/session")
@limiter.limit("20/hour")
def create_session(
    request: Request,
    user_id: str = None,
    container: AppContainer = Depends(app_container),
    api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Create a new user session."""
    try:
        if not container.cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        session_id = container.cache_service.create_session(user_id)
        logger.info(f"[Backend] Created session: {session_id}")
        
        return {
            "session_id": session_id,
            "user_id": user_id or "anonymous"
        }
        
    except Exception as e:
        logger.error(f"[Backend] Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@router.get("/session/{session_id}")
@limiter.limit("60/minute")
def get_session(
    session_id: str,
    request: Request,
    container: AppContainer = Depends(app_container),
    api_key: str = Depends(require_api_key)
) -> Dict[str, Any]:
    """Get session data by session ID."""
    try:
        if not container.cache_service:
            raise HTTPException(status_code=503, detail="Cache service not available")
        
        session_data = container.cache_service.get_session(session_id)
        
        if session_data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return session_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[Backend] Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")