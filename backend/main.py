# app/main.py
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from backend.routers import query, ingest, init, cache, clean_processing
from backend.dependencies import get_app_container, ImmutableContainerDep

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG Service", version="1.0")


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Global exception on {request.url}: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
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
def health():
    """Detailed health check including cache status."""
    try:
        container = get_app_container()
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
            "error": str(e),
            "cache": {"error": "Unable to check cache status"}
        }


@app.get("/health/v2")
def health_v2(container: ImmutableContainerDep):
    """Thread-safe health check using immutable container."""
    try:
        cache_health = container.get_cache_health()
        
        return {
            "status": "ok",
            "service_ready": container.is_ready(),
            "container_type": "immutable",
            "thread_safe": True,
            "cache": cache_health
        }
    except Exception as e:
        logger.error(f"v2 Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "container_type": "immutable",
            "thread_safe": True,
            "cache": {"error": "Unable to check cache status"}
        }


@app.on_event("startup")
async def startup_event():
    logger.info("RAG Service starting up...")


@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("RAG Service shutting down...")
    try:
        container = get_app_container()
        if container.vector_engine and hasattr(container.vector_engine, 'vector_store'):
            container.vector_engine.vector_store.close()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.warning(f"Error during shutdown cleanup: {e}")


app.include_router(init.router, prefix="/init", tags=["init"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
app.include_router(cache.router, prefix="/cache", tags=["cache"])
app.include_router(clean_processing.router)
