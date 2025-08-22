# app/main.py
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from app.routers import query, ingest, init

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
    return {"status": "ok"}


@app.on_event("startup")
async def startup_event():
    logger.info("RAG Service starting up...")


@app.on_event("shutdown") 
async def shutdown_event():
    logger.info("RAG Service shutting down...")


app.include_router(init.router, prefix="/init", tags=["init"])
app.include_router(query.router, prefix="/query", tags=["query"])
app.include_router(ingest.router, prefix="/ingest", tags=["ingest"])
