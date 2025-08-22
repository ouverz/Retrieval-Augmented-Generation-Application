# app/routers/init.py
import hashlib
import os
import glob
import threading
import time
from typing import Dict, Any
from fastapi import APIRouter, Depends, BackgroundTasks
from app.schemas.ingest import InitRequest, InitResponse, InitStatusResponse
from app.deps import app_container, state_store
from app.services.app_container import AppContainer
from app.services.state_store import StateStore

from src.Processing_Documents import DocumentProcessor, RAGApplication
from src.VectorSearchEngine import VectorSearchEngine
from src.BM25SearchEngine import BM25SearchEngine
from src.HybridSearchEngine import HybridSearchEngine
from src.services.llm_factory import LLMFactory
from src.config.settings import HybridSearchConfig

router = APIRouter()

# Global initialization status
_init_status = {
    "status": "idle",
    "progress": 0.0,
    "message": "",
    "documents_processed": 0,
    "total_documents": 0,
    "error": None
}
_init_lock = threading.Lock()


def _hash_pdfs(folder: str) -> str:
    sha = hashlib.sha256()
    for path in sorted(glob.glob(os.path.join(folder, "*.pdf"))):
        sha.update(path.encode())
        stat = os.stat(path)
        sha.update(str(stat.st_mtime_ns).encode())
        sha.update(str(stat.st_size).encode())
    return sha.hexdigest()


def _update_status(status: str, progress: float = 0.0, message: str = "", 
                  documents_processed: int = 0, total_documents: int = 0, error: str = None):
    """Update initialization status"""
    global _init_status
    with _init_lock:
        _init_status.update({
            "status": status,
            "progress": progress,
            "message": message,
            "documents_processed": documents_processed,
            "total_documents": total_documents,
            "error": error
        })


@router.get("/status", response_model=InitStatusResponse)
def get_init_status():
    """Get current initialization status"""
    with _init_lock:
        return InitStatusResponse(**_init_status)


@router.get("/debug")
def debug_system_state(container: AppContainer = Depends(app_container)):
    """Debug endpoint to check system component states"""
    return {
        "container_ready": container.is_ready(),
        "has_hybrid_engine": container.hybrid_engine is not None,
        "has_bm25_engine": container.bm25_engine is not None,
        "bm25_has_retriever": container.bm25_engine.retriever if container.bm25_engine else None is not None,
        "bm25_document_count": len(container.bm25_engine.documents) if container.bm25_engine else 0,
        "has_vector_engine": container.vector_engine is not None,
        "has_llm_factory": container.llm_factory is not None,
        "has_doc_processor": container.doc_processor is not None,
        "has_rag_app": container.rag_app is not None,
        "data_dir": container.data_dir,
    }


def _run_initialization_background(container: AppContainer, dataset_hash: str, store: StateStore):
    """Background initialization task with progress tracking"""
    try:
        _update_status("running", 0.0, "Starting initialization...")
        
        # Count PDFs for progress tracking
        pdf_files = glob.glob(os.path.join(container.data_dir, "*.pdf"))
        total_pdfs = len(pdf_files)
        
        _update_status("running", 0.1, f"Found {total_pdfs} PDF files to process", 0, total_pdfs)
        time.sleep(0.5)  # Brief pause for UI
        
        # 1) Build engines first
        _update_status("running", 0.2, "Initializing search engines...", 0, total_pdfs)
        config = HybridSearchConfig()
        vector_engine = VectorSearchEngine()
        bm25_engine = BM25SearchEngine()
        
        # 2) Process PDFs with progress tracking
        _update_status("running", 0.3, "Creating document processor...", 0, total_pdfs)
        processor = DocumentProcessor(
            directory=container.data_dir,
            bm25_engine=bm25_engine,
            vector_engine=vector_engine
        )
        
        _update_status("running", 0.4, "Processing documents and creating embeddings...", 0, total_pdfs)
        processor.run_application()  # This processes PDFs and adds chunks to BM25 engine
        
        _update_status("running", 0.6, "Building BM25 search index...", total_pdfs, total_pdfs)
        try:
            # Load processed documents with proper UUIDs
            import pandas as pd
            docs_df = pd.read_csv("processed_rag_documents.csv")
            bm25_engine.build_index(docs_df)  # Pass DataFrame with UUIDs
        except ValueError as e:
            if "No documents to index" in str(e):
                raise ValueError(f"No documents found in {container.data_dir}. Please add PDF files and try again.")
            raise
        except FileNotFoundError:
            # Fallback to old method if CSV doesn't exist
            bm25_engine.build_index()  # Build from accumulated documents
        
        _update_status("running", 0.7, "Documents processed, building hybrid search engine...", total_pdfs, total_pdfs)
        
        # 3) Build RAG application and hybrid engine
        rag = RAGApplication(config=config)
        hybrid = HybridSearchEngine(
            bm25_engine=bm25_engine, 
            vector_engine=vector_engine, 
            config=config
        )
        
        _update_status("running", 0.8, "Initializing LLM factory...", total_pdfs, total_pdfs)
        
        # 4) LLM factory
        llm = LLMFactory()
        
        _update_status("running", 0.9, "Saving configuration...", total_pdfs, total_pdfs)
        
        # 5) Save into app container for reuse
        container.doc_processor = processor
        container.rag_app = rag
        container.vector_engine = vector_engine
        container.bm25_engine = bm25_engine
        container.hybrid_engine = hybrid
        container.llm_factory = llm

        # 6) Validate that engines are properly initialized
        if not bm25_engine.retriever:
            raise RuntimeError("BM25 engine failed to initialize properly")
        if not container.llm_factory:
            raise RuntimeError("LLM factory failed to initialize properly")
        
        # 7) Persist new dataset hash
        store.set("dataset_hash", dataset_hash)
        
        _update_status("completed", 1.0, f"Initialization completed! Processed {total_pdfs} documents.", 
                      total_pdfs, total_pdfs)
        
    except Exception as e:
        error_msg = str(e)
        print(f"Background initialization error: {error_msg}")
        import traceback
        traceback.print_exc()
        _update_status("error", 0.0, f"Error: {error_msg}", 0, 0, error_msg)


@router.post("", response_model=InitResponse)
def initialize(
    req: InitRequest,
    background_tasks: BackgroundTasks,
    store: StateStore = Depends(state_store),
    container: AppContainer = Depends(app_container),
):
    try:
        # Check if already running
        with _init_lock:
            if _init_status["status"] == "running":
                return InitResponse(initialized=False, reason="Initialization already in progress")
        
        dataset_hash = _hash_pdfs(container.data_dir)
        saved = store.get("dataset_hash")

        if (saved == dataset_hash) and container.is_ready() and not req.force:
            _update_status("completed", 1.0, "System already initialized")
            return InitResponse(initialized=True, reason="already-initialized (no changes)")

        # Start background initialization
        background_tasks.add_task(_run_initialization_background, container, dataset_hash, store)
        return InitResponse(initialized=False, reason="initialization-started")
    
    except Exception as e:
        print(f"Initialization startup error: {e}")
        import traceback
        traceback.print_exc()
        _update_status("error", 0.0, str(e), error=str(e))
        return InitResponse(initialized=False, reason=f"Error: {str(e)}")