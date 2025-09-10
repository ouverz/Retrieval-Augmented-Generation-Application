# app/routers/query.py
import time
import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional, Dict, Any
from slowapi import Limiter
from slowapi.util import get_remote_address
from backend.schemas.query import QueryRequest, QueryResponse
from backend.dependencies import app_container
from backend.container import AppContainer
from backend.security import require_api_key
from core.services.synthesis_service import synthesize_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.post("", response_model=QueryResponse)
@limiter.limit("10/minute")
async def query_endpoint(
    request: Request,
    req: QueryRequest, 
    container: AppContainer = Depends(app_container),
    api_key: str = Depends(require_api_key)
):
    try:
        logger.info(f"Received query: {req.query[:100]}...")
        
        if not container.is_ready():
            logger.error("Container not ready")
            raise HTTPException(412, "Service not initialized. POST /init first.")

        t0 = time.time()
        cache_hit = False
        cache_key = None
        
        # Try to get cached result if cache service is available
        cached_result = None
        if (container.cache_service and container.cache_service.is_available() 
            and container.hybrid_engine and hasattr(container.hybrid_engine, 'config')):
            try:
                cached_result = container.cache_service.get_cached_query_result(
                    req.query, req.top_k, container.hybrid_engine.config
                )
                if cached_result:
                    cache_hit = True
                    cache_key = container.cache_service._query_cache_key(
                        req.query, req.top_k, 
                        container.cache_service._config_hash(container.hybrid_engine.config)
                    )
                    logger.info(f"[API] Cache hit for query: {req.query[:50]}...")
            except Exception as e:
                logger.warning(f"[API] Cache lookup failed, proceeding without cache: {e}")
                
        if cached_result:
            # Use cached result
            ctx_df, response = cached_result
            latency_ms = int((time.time() - t0) * 1000)
        else:
            # Cache miss - perform normal query processing
            logger.info("[API] Cache miss - performing search")
            
            # Check if hybrid engine exists and is properly initialized
            if not container.hybrid_engine:
                logger.error("Hybrid engine is None")
                raise HTTPException(500, "Hybrid search engine not initialized")
            
            # Check if BM25 engine has built index
            if not container.bm25_engine or not container.bm25_engine.retriever:
                logger.error("BM25 engine not properly initialized")
                raise HTTPException(500, "BM25 search index not built. System may still be initializing.")
            
            # Check if LLM factory is available
            if not container.llm_factory:
                logger.error("LLM factory not initialized")
                raise HTTPException(500, "LLM factory not initialized")
            
            logger.info("Performing hybrid search...")
            
            # The HybridSearchEngine.search() returns (ctx_df, response) 
            # We'll use the response directly instead of calling synthesizer twice
            ctx_df, response = await container.hybrid_engine.search(
                req.query, 
                top_k=req.top_k,
                rrf_k=req.rrf_k
            )
            
            if response is None:
                logger.warning("HybridSearchEngine returned None response, falling back to direct synthesis")
                # Fallback to direct synthesis if hybrid engine didn't return response
                response = await synthesize_answer(
                    query=req.query,
                    context=ctx_df,
                    factory=container.llm_factory,
                    max_attempts=2,
                )
            
            latency_ms = int((time.time() - t0) * 1000)
            logger.info(f"Query completed in {latency_ms}ms")
            
            # Cache the result if cache service is available
            if (container.cache_service and container.cache_service.is_available() 
                and container.hybrid_engine and hasattr(container.hybrid_engine, 'config')):
                try:
                    success = container.cache_service.cache_query_result(
                        req.query, req.top_k, container.hybrid_engine.config,
                        ctx_df, response
                    )
                    if success:
                        logger.info("[API] Query result cached successfully")
                        cache_key = container.cache_service._query_cache_key(
                            req.query, req.top_k, 
                            container.cache_service._config_hash(container.hybrid_engine.config)
                        )
                except Exception as e:
                    logger.warning(f"[API] Failed to cache query result: {e}")
        
        # Update session history if session_id provided and cache service available
        if req.session_id and container.cache_service and container.cache_service.is_available():
            try:
                # Create a brief summary of the response
                response_summary = response.answer[:100] if hasattr(response, 'answer') else str(response)[:100]
                container.cache_service.update_session(req.session_id, req.query, response_summary)
                logger.info(f"[API] Updated session {req.session_id} with query history")
            except Exception as e:
                logger.warning(f"[API] Failed to update session history: {e}")
        
        # Create results table from ctx_df - now with true hybrid scores
        results_table = []
        for i, (_, row) in enumerate(ctx_df.iterrows()):
            metadata = row.get("metadata", {})
            
            # Extract true hybrid score components
            hybrid_score = round(metadata.get("rrf_score", metadata.get("hybrid_score", metadata.get("score", 0.0))), 4)
            bm25_score = round(metadata.get("bm25_score", 0.0), 4)
            vector_score = round(metadata.get("vector_score", 0.0), 4)
            vector_similarity = round(metadata.get("vector_similarity", 0.0), 4) if metadata.get("vector_similarity") else None
            quality_penalty = round(metadata.get("content_quality_penalty", 1.0), 2)
            
            # Engine information
            found_by = metadata.get("found_by_engines", [])
            engine_display = "+".join(found_by) if found_by else metadata.get("source_engine", "unknown")
            
            results_table.append({
                "rank": i + 1,
                "source_id": row.get("id", f"result_{i+1}"),
                "content_preview": (row.get("content", "")[:100] + "..." if len(row.get("content", "")) > 100 else row.get("content", "")),
                "hybrid_score": hybrid_score,
                "bm25_score": bm25_score,
                "vector_score": vector_score,
                "vector_similarity": vector_similarity,
                "quality_penalty": quality_penalty,
                "engines": engine_display,
                "bm25_rank": metadata.get("bm25_rank"),
                "vector_rank": metadata.get("vector_rank"),
                "distance": round(metadata.get("vector_distance", 0.0), 4) if metadata.get("vector_distance") else None,
            })
        
        # Compute evaluation metrics if requested
        # REMOVED: Evaluation metrics computation - evaluation system removed in lean branch
        evaluation_metrics = None
        
        # Create response using the synthesized response
        if hasattr(response, 'model_dump'):
            # Normal case: proper SynthesizedResponse object
            response_data = response.model_dump()
            response_data.update({
                "latency_ms": latency_ms,
                "results_table": results_table,
                "cache_hit": cache_hit,
                "cache_key": cache_key,
                "session_id": req.session_id,
                "evaluation_metrics": evaluation_metrics
            })
            return QueryResponse(**response_data)
        else:
            # Fallback case: synthesizer returned raw string or other type
            logger.error(f"Synthesizer returned unexpected type: {type(response)}")
            logger.error(f"Response content: {str(response)[:200]}...")
            
            # Create a fallback response
            fallback_response = {
                "thought_process": ["Error: Synthesizer returned unexpected format"],
                "answer": str(response) if response else "I encountered an error processing your request.",
                "enough_context": False,
                "confidence": 0.0,
                "citations": [],
                "precision": 0.0,
                "evidence_precision": "low",
                "latency_ms": latency_ms,
                "results_table": results_table,
                "cache_hit": cache_hit,
                "cache_key": cache_key,
                "session_id": req.session_id,
                "evaluation_metrics": None
            }
            return QueryResponse(**fallback_response)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Query processing failed: {str(e)}")


