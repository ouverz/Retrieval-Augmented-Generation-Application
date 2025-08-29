"""
Integration tests for the new clean architecture implementation.
Tests the complete separation of concerns and proper interface usage.
"""

import pytest
import tempfile
import os
from unittest.mock import Mock, patch
import pandas as pd

from core.services.clean_document_processor import (
    CleanDocumentProcessor, VectorDocumentRepository, 
    BM25IndexBuilder, CleanProcessingOrchestrator
)
from core.services.clean_search_engines import (
    CleanVectorSearchEngine, CleanKeywordSearchEngine, 
    CleanHybridSearchEngine, CleanSearchEngineFactory
)
from core.interfaces.document_processing import ProcessedDocument
from core.interfaces.search_engines import SearchResult


class TestCleanDocumentProcessor:
    """Test the clean document processor implementation."""
    
    def test_processor_creates_structured_documents(self):
        """Test that processor creates properly structured ProcessedDocument objects."""
        processor = CleanDocumentProcessor()
        
        # Use simpler patch approach
        with patch('core.services.clean_document_processor.extract_metadata_with_pypdf') as mock_meta, \
             patch.object(processor, 'converter') as mock_converter, \
             patch.object(processor, 'chunker') as mock_chunker:
            
            # Setup mocks
            mock_doc = Mock()
            mock_converter.convert.return_value = Mock(document=mock_doc)
            
            mock_chunks = [Mock(text="Test chunk 1"), Mock(text="Test chunk 2")]
            mock_chunker.chunk.return_value = mock_chunks
            
            mock_meta.return_value = {"title": "Test Document"}
            
            # Test processing
            result = processor.process_file("test.pdf")
            
            # Verify structure
            assert result.success
            assert len(result.documents) == 2
            assert all(isinstance(doc, ProcessedDocument) for doc in result.documents)
            
            # Check first document structure
            doc = result.documents[0]
            assert doc.content_raw == "Test chunk 1"
            assert doc.content_enriched == "Test chunk 1" 
            assert doc.metadata["title"] == "Test Document"
            assert doc.chunk_index == 0
            assert doc.uuid_chunk  # Should have UUID
    
    def test_processor_handles_errors_gracefully(self):
        """Test that processor handles various error conditions."""
        processor = CleanDocumentProcessor()
        
        # Test file that doesn't exist
        result = processor.process_file("nonexistent.pdf")
        assert not result.success
        assert len(result.errors) > 0
    
    def test_directory_processing(self):
        """Test directory processing finds and processes PDF files."""
        processor = CleanDocumentProcessor()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            pdf_file = os.path.join(temp_dir, "test.pdf")
            with open(pdf_file, 'w') as f:
                f.write("dummy pdf content")
            
            # Create non-PDF file (should be ignored)
            txt_file = os.path.join(temp_dir, "test.txt")
            with open(txt_file, 'w') as f:
                f.write("dummy text content")
            
            # Mock successful processing for PDF
            with patch.object(processor, 'process_file') as mock_process:
                mock_process.return_value = Mock(
                    documents=[Mock()], 
                    errors=[], 
                    processing_time_seconds=0.1
                )
                
                result = processor.process_directory(temp_dir)
                
                # Should only process PDF file
                mock_process.assert_called_once_with(pdf_file)


class TestVectorDocumentRepository:
    """Test the vector document repository implementation."""
    
    def test_save_generates_embeddings(self):
        """Test that repository generates embeddings for documents."""
        mock_vector_store = Mock()
        mock_vector_store.get_embeddings.return_value = [0.1, 0.2, 0.3]
        
        repository = VectorDocumentRepository(mock_vector_store)
        
        # Create test document without embeddings
        doc = ProcessedDocument(
            uuid_chunk="test-uuid",
            content_raw="test content",
            content_enriched="enriched test content",
            embeddings=[],  # No embeddings initially
            metadata={"title": "test"},
            keywords=["test"],
            file_path="test.pdf",
            chunk_index=0
        )
        
        success = repository.save_documents([doc])
        
        # Should generate embeddings and save
        assert success
        mock_vector_store.get_embeddings.assert_called_with("enriched test content")
        mock_vector_store.upsert.assert_called_once()
        
        # Check document was modified with embeddings
        assert doc.embeddings == [0.1, 0.2, 0.3]
    
    def test_to_vector_store_format_conversion(self):
        """Test conversion of ProcessedDocument to vector store format."""
        mock_vector_store = Mock()
        repository = VectorDocumentRepository(mock_vector_store)
        
        doc = ProcessedDocument(
            uuid_chunk="test-uuid",
            content_raw="test content", 
            content_enriched="enriched test content",
            embeddings=[0.1, 0.2, 0.3],
            metadata={"title": "test"},
            keywords=["keyword1", "keyword2"],
            file_path="/path/to/test.pdf",
            chunk_index=5
        )
        
        df = repository._to_vector_store_format([doc])
        
        # Check DataFrame structure
        assert len(df) == 1
        row = df.iloc[0]
        assert row['id'] == "test-uuid"
        assert row['contents'] == "enriched test content"  # Uses enriched content
        assert row['embedding'] == [0.1, 0.2, 0.3]
        assert row['metadata']['keywords'] == ["keyword1", "keyword2"]
        assert row['metadata']['chunk_index'] == 5
        assert row['metadata']['file_name'] == "test.pdf"


class TestCleanSearchEngines:
    """Test the clean search engine implementations."""
    
    def test_vector_search_engine(self):
        """Test vector search engine implementation."""
        mock_vector_store = Mock()
        
        # Setup mock search results
        mock_df = pd.DataFrame({
            'id': ['uuid1', 'uuid2'],
            'content': ['content 1', 'content 2'],
            'distance': [0.1, 0.2],
            'title': ['title1', 'title2']
        })
        mock_vector_store.search.return_value = mock_df
        
        engine = CleanVectorSearchEngine(mock_vector_store)
        result = engine.search("test query", top_k=5)
        
        # Check result structure
        assert isinstance(result, SearchResult)
        assert result.query == "test query"
        assert len(result.documents) == 2
        assert result.total_results == 2
        assert result.metadata['search_type'] == 'vector'
        
        # Check document conversion
        doc1 = result.documents[0]
        assert doc1.page_content == 'content 1'
        assert doc1.metadata['id'] == 'uuid1'
        assert doc1.metadata['distance'] == 0.1
        assert doc1.metadata['title'] == 'title1'
    
    def test_keyword_search_engine(self):
        """Test BM25 keyword search engine implementation."""
        mock_bm25_engine = Mock()
        mock_bm25_engine.retriever = Mock()  # Make it ready
        
        # Setup mock search results
        mock_df = pd.DataFrame({
            'uuid_chunk': ['uuid1', 'uuid2'],
            'chunk_enriched': ['content 1', 'content 2'],
            'bm25_score': [0.8, 0.6],
            'file_name': ['file1.pdf', 'file2.pdf'],
            'chunk_index': [0, 1]
        })
        mock_bm25_engine.search.return_value = mock_df
        
        engine = CleanKeywordSearchEngine(mock_bm25_engine)
        result = engine.search("test query", top_k=5)
        
        # Check result structure
        assert isinstance(result, SearchResult)
        assert result.query == "test query"
        assert len(result.documents) == 2
        assert result.metadata['search_type'] == 'bm25'
        
        # Check document conversion
        doc1 = result.documents[0]
        assert doc1.page_content == 'content 1'  # Uses enriched content
        assert doc1.metadata['id'] == 'uuid1'
        assert doc1.metadata['score'] == 0.8
    
    def test_hybrid_search_document_fusion(self):
        """Test hybrid search document-level fusion."""
        mock_vector_engine = Mock()
        mock_keyword_engine = Mock()
        
        # Setup mock engines as ready
        mock_vector_engine.is_ready.return_value = True
        mock_keyword_engine.is_ready.return_value = True
        
        # Create mock search results with overlapping content
        from langchain.schema import Document
        
        vector_docs = [
            Document(page_content="same content", metadata={'distance': 0.1}),
            Document(page_content="vector only", metadata={'distance': 0.2})
        ]
        keyword_docs = [
            Document(page_content="same content", metadata={'score': 0.8}),
            Document(page_content="keyword only", metadata={'score': 0.6})
        ]
        
        mock_vector_engine.search.return_value = Mock(
            documents=vector_docs, 
            query="test",
            total_results=2,
            processing_time_ms=100,
            metadata={}
        )
        mock_keyword_engine.search.return_value = Mock(
            documents=keyword_docs,
            query="test", 
            total_results=2,
            processing_time_ms=100,
            metadata={}
        )
        
        engine = CleanHybridSearchEngine(mock_vector_engine, mock_keyword_engine)
        result = engine.search("test query", top_k=10, vector_weight=0.7)
        
        # Should have 3 unique documents (1 overlapping + 2 unique)
        assert len(result.documents) == 3
        assert result.metadata['search_type'] == 'hybrid'
        assert result.metadata['vector_weight'] == 0.7
        assert abs(result.metadata['keyword_weight'] - 0.3) < 0.0001
        
        # Check fusion metadata
        for doc in result.documents:
            assert 'hybrid_score' in doc.metadata
            assert 'vector_score' in doc.metadata  
            assert 'keyword_score' in doc.metadata


class TestProcessingOrchestrator:
    """Test the complete processing orchestrator."""
    
    def test_complete_pipeline_execution(self):
        """Test the complete end-to-end processing pipeline."""
        # Create mocks for all components
        mock_processor = Mock()
        mock_repository = Mock()
        mock_index_builder = Mock()
        
        # Setup successful processing
        mock_docs = [Mock()]
        mock_result = Mock(
            success=True,
            documents=mock_docs,
            errors=[],
            to_dataframe=Mock(return_value=pd.DataFrame())
        )
        mock_processor.process_directory.return_value = mock_result
        mock_repository.save_documents.return_value = True
        mock_index_builder.build_bm25_index.return_value = True
        mock_index_builder.build_vector_index.return_value = True
        
        orchestrator = CleanProcessingOrchestrator(
            mock_processor, mock_repository, mock_index_builder
        )
        
        result = orchestrator.process_and_index("test_directory")
        
        # Verify complete pipeline execution
        mock_processor.process_directory.assert_called_once_with("test_directory")
        mock_repository.save_documents.assert_called_once_with(mock_docs)
        mock_index_builder.build_bm25_index.assert_called_once()
        mock_index_builder.build_vector_index.assert_called_once()
        
        assert result.success
    
    def test_pipeline_handles_component_failures(self):
        """Test pipeline handles failures in individual components."""
        mock_processor = Mock()
        mock_repository = Mock()
        mock_index_builder = Mock()
        
        # Setup repository failure
        mock_result = Mock(success=True, documents=[Mock()], errors=[])
        mock_processor.process_directory.return_value = mock_result
        mock_repository.save_documents.return_value = False  # Repository fails
        
        orchestrator = CleanProcessingOrchestrator(
            mock_processor, mock_repository, mock_index_builder
        )
        
        result = orchestrator.process_and_index("test_directory")
        
        # Should capture repository failure - the result should be the original processing result
        # but with added errors from orchestrator
        assert "Failed to save documents to repository" in result.errors


class TestSearchEngineFactory:
    """Test the search engine factory implementation."""
    
    def test_factory_creates_engines(self):
        """Test that factory creates proper engine instances."""
        mock_vector_store = Mock()
        mock_bm25_engine = Mock()
        
        factory = CleanSearchEngineFactory(mock_vector_store, mock_bm25_engine)
        
        # Test engine creation
        vector_engine = factory.create_vector_engine()
        assert isinstance(vector_engine, CleanVectorSearchEngine)
        assert vector_engine.vector_store == mock_vector_store
        
        keyword_engine = factory.create_keyword_engine() 
        assert isinstance(keyword_engine, CleanKeywordSearchEngine)
        assert keyword_engine.bm25_engine == mock_bm25_engine
        
        hybrid_engine = factory.create_hybrid_engine(vector_engine, keyword_engine)
        assert isinstance(hybrid_engine, CleanHybridSearchEngine)
        assert hybrid_engine.vector_engine == vector_engine
        assert hybrid_engine.keyword_engine == keyword_engine


# Integration test combining multiple components
class TestIntegratedCleanArchitecture:
    """Integration tests for the complete clean architecture."""
    
    @pytest.mark.integration
    def test_end_to_end_architecture_flow(self):
        """Test the complete flow from processing to search using clean architecture."""
        # This would be a more complex integration test
        # that sets up real components and tests the full flow
        # Placeholder for now - would require actual database and file setup
        pass
    
    def test_interface_compliance(self):
        """Test that all implementations properly implement their interfaces."""
        # Test that concrete classes implement all required methods
        from core.interfaces.document_processing import DocumentProcessor, DocumentRepository
        from core.interfaces.search_engines import SearchEngine, SearchEngineFactory
        
        # Check that our implementations have all required methods
        processor = CleanDocumentProcessor()
        assert hasattr(processor, 'process_file')
        assert hasattr(processor, 'process_directory')
        assert callable(processor.process_file)
        assert callable(processor.process_directory)
        
        # Similar checks for other components...
        # This ensures interface compliance without runtime errors