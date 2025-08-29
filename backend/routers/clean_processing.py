"""
Clean processing router demonstrating new architecture patterns.
Shows separation of concerns and proper dependency injection.
"""

import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel

from backend.dependencies import ImmutableContainerDep
from core.interfaces.document_processing import ProcessingResult

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/processing/v2", tags=["Clean Processing"])


class ProcessingRequest(BaseModel):
    """Request model for document processing."""
    directory: str


class ProcessingResponse(BaseModel):
    """Response model for document processing results."""
    success: bool
    total_chunks: int
    processing_time_seconds: float
    errors: list[str]
    message: str


class SearchRequest(BaseModel):
    """Request model for clean search operations."""
    query: str
    top_k: int = 10
    search_type: str = "hybrid"  # vector, keyword, hybrid
    vector_weight: float = 0.6


class SearchResponse(BaseModel):
    """Response model for search results."""
    query: str
    total_results: int
    processing_time_ms: int
    search_type: str
    documents: list[Dict[str, Any]]
    metadata: Dict[str, Any]


@router.post("/process", response_model=ProcessingResponse)
async def process_documents_clean(
    request: ProcessingRequest,
    container: ImmutableContainerDep
) -> ProcessingResponse:
    """
    Process documents using clean architecture with proper separation of concerns.
    Demonstrates the new ProcessingOrchestrator pattern.
    """
    logger.info(f"Starting clean document processing for directory: {request.directory}")
    
    try:
        # Create clean processing orchestrator
        orchestrator = container.create_processing_orchestrator()
        
        # Execute complete processing pipeline
        result = orchestrator.process_and_index(request.directory)
        
        if result.success:
            message = f"Successfully processed {result.total_chunks} document chunks"
            logger.info(message)
        else:
            message = f"Processing completed with {len(result.errors)} errors"
            logger.warning(message)
        
        return ProcessingResponse(
            success=result.success,
            total_chunks=result.total_chunks,
            processing_time_seconds=result.processing_time_seconds,
            errors=result.errors,
            message=message
        )
        
    except Exception as e:
        logger.error(f"Clean processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.post("/search", response_model=SearchResponse)
async def search_clean(
    request: SearchRequest,
    container: ImmutableContainerDep
) -> SearchResponse:
    """
    Search using clean search engines with proper interface separation.
    Demonstrates the new SearchEngineFactory pattern.
    """
    logger.info(f"Clean search request: {request.query} (type: {request.search_type})")
    
    try:
        # Create search engine factory
        factory = container.create_search_engine_factory()
        
        # Create appropriate search engine
        if request.search_type == "vector":
            engine = factory.create_vector_engine()
            result = engine.search(request.query, request.top_k)
            
        elif request.search_type == "keyword":
            engine = factory.create_keyword_engine()
            result = engine.search(request.query, request.top_k)
            
        elif request.search_type == "hybrid":
            vector_engine = factory.create_vector_engine()
            keyword_engine = factory.create_keyword_engine()
            engine = factory.create_hybrid_engine(vector_engine, keyword_engine)
            result = engine.search(request.query, request.top_k, request.vector_weight)
            
        else:
            raise ValueError(f"Invalid search type: {request.search_type}")
        
        # Check if engines are ready
        if not engine.is_ready():
            raise RuntimeError(f"{request.search_type} search engine not ready")
        
        # Convert Document objects to dictionaries
        documents = []
        for doc in result.documents:
            doc_dict = {
                "content": doc.page_content,
                "metadata": doc.metadata
            }
            documents.append(doc_dict)
        
        logger.info(f"Clean search completed: {result.total_results} results in {result.processing_time_ms}ms")
        
        return SearchResponse(
            query=result.query,
            total_results=result.total_results,
            processing_time_ms=result.processing_time_ms,
            search_type=request.search_type,
            documents=documents,
            metadata=result.metadata
        )
        
    except Exception as e:
        logger.error(f"Clean search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/engines/status")
async def get_engines_status(container: ImmutableContainerDep) -> Dict[str, Any]:
    """
    Get status of all search engines using clean architecture.
    Demonstrates engine readiness checking and component inspection.
    """
    try:
        factory = container.create_search_engine_factory()
        
        # Create all engine types
        vector_engine = factory.create_vector_engine()
        keyword_engine = factory.create_keyword_engine() 
        hybrid_engine = factory.create_hybrid_engine(vector_engine, keyword_engine)
        
        # Check readiness
        status = {
            "vector_engine": {
                "ready": vector_engine.is_ready(),
                "type": "CleanVectorSearchEngine"
            },
            "keyword_engine": {
                "ready": keyword_engine.is_ready(),
                "type": "CleanKeywordSearchEngine"
            },
            "hybrid_engine": {
                "ready": hybrid_engine.is_ready(),
                "type": "CleanHybridSearchEngine",
                "components": list(hybrid_engine.get_component_engines().keys())
            }
        }
        
        # Overall system readiness
        status["system_ready"] = all(
            engine_status["ready"] 
            for engine_status in status.values()
            if isinstance(engine_status, dict) and "ready" in engine_status
        )
        
        logger.info(f"Engine status check: system_ready={status['system_ready']}")
        return status
        
    except Exception as e:
        logger.error(f"Engine status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")


@router.get("/architecture/info")
async def get_architecture_info() -> Dict[str, Any]:
    """
    Get information about the clean architecture implementation.
    Educational endpoint showing the new patterns.
    """
    return {
        "architecture_version": "v2_clean",
        "patterns": [
            "Interface Segregation Principle",
            "Dependency Injection",
            "Factory Pattern", 
            "Repository Pattern",
            "Orchestrator Pattern"
        ],
        "interfaces": {
            "DocumentProcessor": "Pure document processing without side effects",
            "DocumentRepository": "Document persistence abstraction",
            "SearchIndexBuilder": "Index building operations",
            "ProcessingOrchestrator": "Complete processing pipeline coordination",
            "SearchEngine": "Pure search operations with standardized results",
            "SearchEngineFactory": "Search engine creation and configuration"
        },
        "improvements": [
            "Clear separation of concerns",
            "Testable components with dependency injection",
            "Standardized error handling and result formats",
            "Thread-safe operations",
            "Document-level fusion with proper content alignment",
            "Immutable container architecture"
        ],
        "migration_path": [
            "Phase 1: Debug code cleanup ✓",
            "Phase 2: Global state management reform ✓", 
            "Phase 3: Document processing architecture reform (in progress)",
            "Phase 4: Complete migration to clean interfaces"
        ]
    }