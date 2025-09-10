# app/main.py
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from backend.routers import query, ingest, init, cache
from backend.dependencies import app_container
from backend.security import load_api_keys
from fastapi import Depends

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Rate Limiting Configuration
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="RAG Service", version="1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8501",  # Streamlit default
        "http://localhost:3000",  # React default
        "http://127.0.0.1:8501",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    
    # Essential security headers
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    response.headers["Content-Security-Policy"] = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'"
    
    return response

# Request Size Limitation Middleware
@app.middleware("http")
async def limit_request_size(request: Request, call_next):
    # Limit request body size to 110MB (slightly more than file upload limit)
    max_size = 110 * 1024 * 1024
    
    if hasattr(request, "headers") and "content-length" in request.headers:
        content_length = int(request.headers["content-length"])
        if content_length > max_size:
            return JSONResponse(
                status_code=413,
                content={"detail": "Request too large"}
            )
    
    response = await call_next(request)
    return response


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Generate a correlation ID for error tracking
    import uuid
    error_id = str(uuid.uuid4())[:8]
    
    # Log detailed error information (server-side only)
    logger.error(
        f"Global exception [{error_id}] on {request.url}: {str(exc)}", 
        exc_info=True,
        extra={"error_id": error_id, "url": str(request.url)}
    )
    
    # Return generic error to client (no sensitive information)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error occurred",
            "error_id": error_id,
            "message": "Please contact support with this error ID if the problem persists"
        }
    )


# HTTP exception handler
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    logger.warning(f"HTTP {exc.status_code} on {request.url}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )


@app.get("/healthz")
def healthz():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/health")
def health(container = Depends(app_container)):
    """Detailed health check including cache status."""
    try:
        cache_health = container.get_cache_health()
        
        return {
            "status": "ok",
            "service_ready": container.is_ready(),
            "cache": cache_health
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "message": "Service health check failed",
            "cache": {"status": "unknown"}
        }




@app.on_event("startup")
async def startup_event():
    logger.info("RAG Service starting up...")
    # Load API keys during startup
    load_api_keys()


@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("RAG Service shutting down...")
    try:
        # Import here to avoid circular imports
        from backend.dependencies import get_app_container
        container = get_app_container()
        if hasattr(container, 'vector_engine') and container.vector_engine and hasattr(container.vector_engine, 'vector_store'):
            container.vector_engine.vector_store.close()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.warning(f"Error during shutdown cleanup: {e}")


app.include_router(init.router, prefix="/init", tags=["init"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(cache.router, prefix="/cache", tags=["cache"])
