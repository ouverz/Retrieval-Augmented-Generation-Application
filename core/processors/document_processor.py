from __future__ import annotations
import os
import uuid
import time
import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List, Dict, Any

from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

from core.utils.extract_keywords import extract_metadata_with_pypdf
from core.search.vector_search import VectorSearchEngine
from core.search.bm25_search import BM25SearchEngine
from core.search.hybrid_search import HybridSearchEngine
from core.interfaces.document_processing import (
    ProcessedDocument, ProcessingResult, DocumentProcessor as IDocumentProcessor,
    DocumentRepository, SearchIndexBuilder, ProcessingOrchestrator
)
from core.database.vector_store import VectorStore
from config.settings import HybridSearchConfig

logger = logging.getLogger(__name__)


class CleanDocumentProcessor(IDocumentProcessor):
    """Pure document processor - only handles file-to-structured-data conversion."""
    
    def __init__(self, chunker: Optional[HybridChunker] = None):
        self.converter = DocumentConverter()
        self.chunker = chunker or HybridChunker(tokenizer="BAAI/bge-m3")
        
    def process_file(self, file_path: str) -> ProcessingResult:
        """Process a single document file into structured format."""
        start_time = time.time()
        errors = []
        documents = []
        
        try:
            # Extract metadata
            try:
                metadata = extract_metadata_with_pypdf(file_path)
                metadata['source_file'] = Path(file_path).name
                metadata['file_path'] = file_path
            except Exception as e:
                errors.append(f"Metadata extraction failed: {str(e)}")
                metadata = {'source_file': Path(file_path).name, 'file_path': file_path}
            
            # Convert document using Docling
            try:
                result = self.converter.convert(source=file_path)
                doc = result.document
            except Exception as e:
                errors.append(f"Document conversion failed: {str(e)}")
                return ProcessingResult([], 0, time.time() - start_time, errors)
            
            # Chunk the document
            try:
                chunks = list(self.chunker.chunk(doc))
                if not chunks:
                    errors.append("No chunks generated from document")
                    return ProcessingResult([], 0, time.time() - start_time, errors)
            except Exception as e:
                errors.append(f"Document chunking failed: {str(e)}")
                return ProcessingResult([], 0, time.time() - start_time, errors)
            
            # Convert chunks to ProcessedDocument objects
            for i, chunk in enumerate(chunks):
                try:
                    # Generate unique ID
                    chunk_uuid = str(uuid.uuid4())
                    
                    # Extract content
                    content_raw = chunk.text
                    content_enriched = content_raw  # TODO: Add enrichment logic if needed
                    
                    # Extract keywords (basic implementation)
                    keywords = self._extract_keywords(content_raw)
                    
                    # Create processed document (without embeddings yet)
                    processed_doc = ProcessedDocument(
                        uuid_chunk=chunk_uuid,
                        content_raw=content_raw,
                        content_enriched=content_enriched,
                        embeddings=[],  # Will be populated by repository
                        metadata=metadata.copy(),
                        keywords=keywords,
                        file_path=file_path,
                        chunk_index=i
                    )
                    
                    documents.append(processed_doc)
                    
                except Exception as e:
                    errors.append(f"Chunk {i} processing failed: {str(e)}")
                    continue
            
            processing_time = time.time() - start_time
            logger.info(f"Processed {file_path}: {len(documents)} chunks in {processing_time:.2f}s")
            
            return ProcessingResult(
                documents=documents,
                total_chunks=len(documents),
                processing_time_seconds=processing_time,
                errors=errors
            )
            
        except Exception as e:
            errors.append(f"Unexpected error processing {file_path}: {str(e)}")
            return ProcessingResult([], 0, time.time() - start_time, errors)
    
    def process_directory(self, directory: str) -> ProcessingResult:
        """Process all PDF files in a directory."""
        start_time = time.time()
        all_documents = []
        all_errors = []
        
        if not os.path.exists(directory):
            return ProcessingResult(
                [], 0, 0.0, [f"Directory does not exist: {directory}"]
            )
        
        # Find all PDF files
        pdf_files = []
        for root, _, files in os.walk(directory):
            for file in files:
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(root, file))
        
        logger.info(f"Found {len(pdf_files)} PDF files in {directory}")
        
        # Process each file
        for file_path in pdf_files:
            result = self.process_file(file_path)
            all_documents.extend(result.documents)
            all_errors.extend(result.errors)
        
        total_time = time.time() - start_time
        logger.info(f"Directory processing complete: {len(all_documents)} total chunks")
        
        return ProcessingResult(
            documents=all_documents,
            total_chunks=len(all_documents),
            processing_time_seconds=total_time,
            errors=all_errors
        )
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content (basic implementation)."""
        # TODO: Implement proper keyword extraction
        # For now, return empty list
        return []


class DocumentProcessor:
    """Legacy PDF → Docling → HybridChunker → vectors upsert + BM25 accumulation."""

    def __init__(
        self,
        directory: str,
        bm25_engine: BM25SearchEngine,
        vector_engine: VectorSearchEngine,
        chunker: Optional[HybridChunker] = None,
    ) -> None:
        self.directory = directory
        self.converter = DocumentConverter()
        self.chunker = chunker or HybridChunker(tokenizer="BAAI/bge-m3")
        self.bm25 = bm25_engine
        self.vector = vector_engine

    def process_directory(self) -> pd.DataFrame:
        all_docs = pd.DataFrame(
            columns=[
                "uuid_chunk",
                "metadata",
                "keywords",
                "chunk_text",
                "chunk_enriched",
                "embeddings",
            ]
        )
        pdfs = [f for f in os.listdir(self.directory) if f.lower().endswith(".pdf")]
        for fname in pdfs:
            df = self.process_single_pdf(os.path.join(self.directory, fname))
            if isinstance(df, pd.DataFrame) and not df.empty:
                all_docs = pd.concat([all_docs, df], ignore_index=True)
        return all_docs

    def process_single_pdf(self, file_path: str) -> pd.DataFrame:
        import logging
        logger = logging.getLogger(__name__)
        
        try:
            meta = extract_metadata_with_pypdf(file_path)
        except Exception as e:
            logger.error(f"Failed to extract metadata from {file_path}: {e}")
            return pd.DataFrame()

        try:
            result = self.converter.convert(source=file_path)
            doc = result.document
        except Exception as e:
            logger.error(f"Failed to convert document {file_path}: {e}")
            return pd.DataFrame()

        try:
            chunks = list(self.chunker.chunk(doc))
            if not chunks:
                logger.warning(f"No chunks generated for {file_path}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to chunk document {file_path}: {e}")
            return pd.DataFrame()

        try:
            vec_engine = VectorSearchEngine(
                chunks, meta, os.path.basename(file_path), file_path, self.chunker
            )
            df = vec_engine.create_embeddings()
            if isinstance(df, pd.DataFrame) and not df.empty:
                ready = VectorSearchEngine.apply_prepare_record(df)
                vec_engine.upsert_records(ready)
                logger.info(f"Successfully processed {file_path}: {len(df)} chunks")
            else:
                logger.warning(f"Empty DataFrame returned for {file_path}")
                df = pd.DataFrame()
        except Exception as e:
            logger.error(f"Failed to create embeddings for {file_path}: {e}")
            df = pd.DataFrame()

        try:
            if self.bm25 is not None:
                self.bm25.add_chunks(file_path, chunks)
        except Exception as e:
            logger.error(f"Failed to add chunks to BM25 for {file_path}: {e}")

        return df

    def run_application(self) -> pd.DataFrame:
        return self.process_directory()


class VectorDocumentRepository(DocumentRepository):
    """Repository implementation using VectorStore for persistence."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        
    def save_documents(self, documents: List[ProcessedDocument]) -> bool:
        """Save processed documents to vector store."""
        if not documents:
            return True
            
        try:
            # Generate embeddings for documents that don't have them
            for doc in documents:
                if not doc.embeddings:
                    doc.embeddings = self.vector_store.get_embeddings(doc.content_enriched)
            
            # Convert to DataFrame format expected by vector store
            df = self._to_vector_store_format(documents)
            
            # Upsert to vector store
            self.vector_store.upsert(df)
            
            logger.info(f"Saved {len(documents)} documents to vector store")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save documents: {str(e)}")
            return False
    
    def get_all_documents(self) -> pd.DataFrame:
        """Retrieve all documents - not directly supported by VectorStore."""
        # VectorStore doesn't have a direct "get all" method
        # This would need to be implemented if required
        raise NotImplementedError("VectorStore doesn't support retrieving all documents")
    
    def clear_documents(self) -> bool:
        """Clear all documents from storage."""
        try:
            self.vector_store.delete(delete_all=True)
            logger.info("Cleared all documents from vector store")
            return True
        except Exception as e:
            logger.error(f"Failed to clear documents: {str(e)}")
            return False
    
    def _to_vector_store_format(self, documents: List[ProcessedDocument]) -> pd.DataFrame:
        """Convert ProcessedDocument list to VectorStore DataFrame format."""
        rows = []
        for doc in documents:
            # Prepare metadata for vector store
            metadata = doc.metadata.copy()
            metadata['keywords'] = doc.keywords
            metadata['chunk_index'] = doc.chunk_index
            metadata['file_name'] = Path(doc.file_path).name
            
            rows.append({
                'id': doc.uuid_chunk,
                'metadata': metadata,
                'contents': doc.content_enriched,  # Use enriched content
                'embedding': doc.embeddings
            })
        
        return pd.DataFrame(rows)


class BM25IndexBuilder(SearchIndexBuilder):
    """Index builder specifically for BM25 search indices."""
    
    def __init__(self, bm25_engine):
        self.bm25_engine = bm25_engine
        
    def build_vector_index(self, documents_df: pd.DataFrame) -> bool:
        """Not applicable for BM25 builder."""
        return True
        
    def build_bm25_index(self, documents_df: pd.DataFrame) -> bool:
        """Build BM25 index from documents DataFrame."""
        try:
            # Group documents by file for BM25 engine
            if 'file_path' in documents_df.columns:
                file_groups = documents_df.groupby('file_path')
                
                for file_path, group in file_groups:
                    # Convert DataFrame rows to chunk-like objects
                    chunks = []
                    for _, row in group.iterrows():
                        # Create a simple chunk object with text attribute
                        chunk_obj = type('Chunk', (), {'text': row['chunk_enriched']})() 
                        chunks.append(chunk_obj)
                    
                    # Add chunks to BM25 engine
                    self.bm25_engine.add_chunks(file_path, chunks)
            
            # Build the index
            self.bm25_engine.build_index()
            logger.info("BM25 index built successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to build BM25 index: {str(e)}")
            return False
            
    def build_hybrid_index(self, documents_df: pd.DataFrame) -> bool:
        """Build hybrid index - just build BM25 since vector is handled separately."""
        return self.build_bm25_index(documents_df)


class CleanProcessingOrchestrator(ProcessingOrchestrator):
    """Orchestrates the complete document processing pipeline."""
    
    def __init__(
        self,
        processor: IDocumentProcessor,
        repository: DocumentRepository,
        index_builder: SearchIndexBuilder
    ):
        self.processor = processor
        self.repository = repository
        self.index_builder = index_builder
        
    def process_and_index(self, directory: str) -> ProcessingResult:
        """Complete pipeline: process documents and build all search indices."""
        logger.info(f"Starting complete processing pipeline for {directory}")
        
        # Step 1: Process documents
        processing_result = self.processor.process_directory(directory)
        
        if not processing_result.success:
            logger.error(f"Document processing failed with {len(processing_result.errors)} errors")
            return processing_result
        
        if not processing_result.documents:
            logger.warning("No documents were processed")
            return processing_result
        
        # Step 2: Save to repository (generates embeddings)
        if not self.repository.save_documents(processing_result.documents):
            processing_result.errors.append("Failed to save documents to repository")
            return processing_result
        
        # Step 3: Build search indices
        documents_df = processing_result.to_dataframe()
        
        if not self.index_builder.build_bm25_index(documents_df):
            processing_result.errors.append("Failed to build BM25 index")
            
        if not self.index_builder.build_vector_index(documents_df):
            processing_result.errors.append("Failed to build vector index")
        
        logger.info("Complete processing pipeline finished successfully")
        return processing_result
    
    def is_ready(self) -> bool:
        """Check if all components are ready for search operations."""
        # Basic readiness check - could be enhanced
        return True


class RAGApplication:
    """Orchestrates shared engines; finalize builds BM25 and Hybrid."""

    def __init__(self, config: HybridSearchConfig) -> None:
        self.config = config
        self.bm25_engine = BM25SearchEngine()
        self.vector_engine = VectorSearchEngine()
        self.hybrid_engine: Optional[HybridSearchEngine] = None

    def finalize(self) -> None:
        if self.bm25_engine.retriever is None:
            self.bm25_engine.build_index()
        self.hybrid_engine = HybridSearchEngine(
            self.bm25_engine, self.vector_engine, self.config
        )

    def search(self, query: str):
        if not self.hybrid_engine:
            raise RuntimeError("Call finalize() before search().")
        return self.hybrid_engine.search(query)
