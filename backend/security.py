# backend/security.py
import os
import logging
from typing import Optional
from fastapi import HTTPException, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

logger = logging.getLogger(__name__)

# API Key Configuration
API_KEY_HEADER = "X-API-Key"
VALID_API_KEYS = set()

def load_api_keys():
    """Load valid API keys from environment variables."""
    global VALID_API_KEYS
    
    # Load API keys from environment (comma-separated)
    api_keys_env = os.getenv("RAG_API_KEYS", "")
    if api_keys_env:
        VALID_API_KEYS = set(key.strip() for key in api_keys_env.split(",") if key.strip())
        logger.info(f"Loaded {len(VALID_API_KEYS)} API keys from environment")
    else:
        logger.warning("No API keys configured - all endpoints will be open")

# Load API keys on module import
load_api_keys()

class APIKeyAuth:
    """API Key authentication dependency."""
    
    def __init__(self, required: bool = True):
        self.required = required
    
    async def __call__(self, request: Request) -> Optional[str]:
        """Validate API key from request headers.
        
        Args:
            request: FastAPI request object
            
        Returns:
            API key if valid, None if not required
            
        Raises:
            HTTPException: If authentication fails
        """
        # Skip authentication if no API keys are configured
        if not VALID_API_KEYS:
            if self.required:
                logger.warning("API key required but none configured")
            return None
        
        # Get API key from header
        api_key = request.headers.get(API_KEY_HEADER)
        
        if not api_key:
            if self.required:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="API key required",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
            return None
        
        # Validate API key
        if api_key not in VALID_API_KEYS:
            logger.warning(f"Invalid API key attempt from {request.client.host if request.client else 'unknown'}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid API key",
                headers={"WWW-Authenticate": "ApiKey"},
            )
        
        logger.debug(f"Valid API key used from {request.client.host if request.client else 'unknown'}")
        return api_key

# Authentication dependencies
require_api_key = APIKeyAuth(required=True)
optional_api_key = APIKeyAuth(required=False)