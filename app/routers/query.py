# app/routers/query.py
import time
import logging
from fastapi import APIRouter, Depends, HTTPException
from app.schemas.query import QueryRequest, QueryResponse
from app.deps import app_container
from app.services.app_container import AppContainer
from src.services.synthesizer import synthesize_answer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("", response_model=QueryResponse)
async def query_endpoint(req: QueryRequest, container: AppContainer = Depends(app_container)):
    try:
        logger.info(f"Received query: {req.query[:100]}...")
        
        if not container.is_ready():
            logger.error("Container not ready")
            raise HTTPException(412, "Service not initialized. POST /init first.")

        t0 = time.time()
        
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
        ctx_df, response = await container.hybrid_engine.search(req.query, top_k=req.top_k)
        
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
        
        # Create results table from ctx_df - now with true hybrid scores
        results_table = []
        for i, (_, row) in enumerate(ctx_df.iterrows()):
            metadata = row.get("metadata", {})
            
            # Extract true hybrid score components
            hybrid_score = round(metadata.get("hybrid_score", metadata.get("score", 0.0)), 4)
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
        
        # Create response using the synthesized response
        if hasattr(response, 'model_dump'):
            # Normal case: proper SynthesizedResponse object
            response_data = response.model_dump()
            response_data.update({
                "latency_ms": latency_ms,
                "results_table": results_table
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
                "results_table": results_table
            }
            return QueryResponse(**fallback_response)
    
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        raise
    except Exception as e:
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(500, f"Query processing failed: {str(e)}")