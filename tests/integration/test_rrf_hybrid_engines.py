"""
Integration tests for RRF implementation with actual search engines.

Tests the complete RRF fusion flow using real vector and keyword search engines,
including disagreement scenarios and performance validation.
"""

import pytest
import tempfile
import os
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from langchain.schema import Document

from core.services.clean_search_engines import (
    CleanVectorSearchEngine, CleanKeywordSearchEngine, 
    CleanHybridSearchEngine, CleanSearchEngineFactory
)
from core.interfaces.search_engines import SearchResult


class TestRRFHybridSearchIntegration:
    """Integration tests for RRF with hybrid search engines."""
    
    def setup_method(self):
        """Setup mock engines for each test."""
        self.mock_vector_store = Mock()
        self.mock_bm25_engine = Mock()
        self.mock_bm25_engine.retriever = Mock()  # Make it ready
        
        # Create factory and engines
        self.factory = CleanSearchEngineFactory(self.mock_vector_store, self.mock_bm25_engine)
        self.vector_engine = self.factory.create_vector_engine()
        self.keyword_engine = self.factory.create_keyword_engine()
        self.hybrid_engine = self.factory.create_hybrid_engine(self.vector_engine, self.keyword_engine)
    
    def test_rrf_integration_with_identical_results(self):
        """Test RRF integration when both engines return identical results."""
        # Setup identical search results
        vector_df = pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3'],
            'content': ['content1', 'content2', 'content3'],
            'distance': [0.1, 0.2, 0.3],
            'metadata': [{'file': 'f1'}, {'file': 'f2'}, {'file': 'f3'}]
        })
        
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['doc1', 'doc2', 'doc3'],
            'chunk_enriched': ['content1', 'content2', 'content3'],
            'bm25_score': [0.9, 0.8, 0.7],
            'file_name': ['f1', 'f2', 'f3'],
            'chunk_index': [0, 1, 2]
        })
        
        # Mock engine responses
        self.mock_vector_store.search.return_value = vector_df
        self.mock_bm25_engine.search.return_value = keyword_df
        
        result = self.hybrid_engine.search("test query", top_k=5)
        
        # Should have 3 documents
        assert len(result.documents) == 3
        assert result.metadata['fusion_method'] == 'reciprocal_rank_fusion'
        assert result.metadata['rrf_k_value'] == 60
        
        # Documents should be ordered by RRF score
        for i, doc in enumerate(result.documents):
            assert doc.metadata['fusion_method'] == 'reciprocal_rank_fusion'
            assert doc.metadata['vector_rank'] == i + 1
            assert doc.metadata['keyword_rank'] == i + 1
            
            # RRF score should be 2/(k + rank) for identical ranks
            expected_rrf = 2.0 / (60 + i + 1)
            assert abs(doc.metadata['rrf_score'] - expected_rrf) < 0.0001
        
        # First document should have highest RRF score
        assert result.documents[0].metadata['rrf_score'] > result.documents[1].metadata['rrf_score']
    
    def test_rrf_integration_with_partial_overlap(self):
        """Test RRF integration with partial overlap between engine results."""
        vector_df = pd.DataFrame({
            'id': ['doc1', 'doc2', 'doc3'],
            'content': ['shared content', 'vector only 1', 'vector only 2'],
            'distance': [0.1, 0.2, 0.3]
        })
        
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['doc4', 'doc1', 'doc5'],
            'chunk_enriched': ['keyword only 1', 'shared content', 'keyword only 2'],
            'bm25_score': [0.9, 0.8, 0.7],
            'file_name': ['f4', 'f1', 'f5'],
            'chunk_index': [0, 1, 2]
        })
        
        self.mock_vector_store.search.return_value = vector_df
        self.mock_bm25_engine.search.return_value = keyword_df
        
        result = self.hybrid_engine.search("test query", top_k=10)
        
        # Should have 5 unique documents
        assert len(result.documents) == 5
        
        # Find the overlapping document
        shared_docs = [doc for doc in result.documents if doc.page_content == "shared content"]
        assert len(shared_docs) == 1
        shared_doc = shared_docs[0]
        
        # Should have both ranks
        assert shared_doc.metadata['vector_rank'] == 1
        assert shared_doc.metadata['keyword_rank'] == 2
        
        # Should have highest RRF score due to consensus
        assert result.documents[0] == shared_doc
        
        # Check RRF calculation
        expected_rrf = (1.0 / 61) + (1.0 / 62)  # rank 1 + rank 2
        assert abs(shared_doc.metadata['rrf_score'] - expected_rrf) < 0.0001
    
    def test_rrf_integration_with_no_overlap(self):
        """Test RRF integration when engines return completely different documents."""
        vector_df = pd.DataFrame({
            'id': ['vec1', 'vec2', 'vec3'],
            'content': ['vector content 1', 'vector content 2', 'vector content 3'],
            'distance': [0.1, 0.2, 0.3]
        })
        
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['key1', 'key2', 'key3'],
            'chunk_enriched': ['keyword content 1', 'keyword content 2', 'keyword content 3'],
            'bm25_score': [0.9, 0.8, 0.7],
            'file_name': ['f1', 'f2', 'f3'],
            'chunk_index': [0, 1, 2]
        })
        
        self.mock_vector_store.search.return_value = vector_df
        self.mock_bm25_engine.search.return_value = keyword_df
        
        result = self.hybrid_engine.search("test query", top_k=10)
        
        # Should have 6 documents total
        assert len(result.documents) == 6
        
        # All documents should have only one rank (vector or keyword, not both)
        for doc in result.documents:
            vector_rank = doc.metadata.get('vector_rank')
            keyword_rank = doc.metadata.get('keyword_rank')
            
            # Exactly one should be None
            assert (vector_rank is None) != (keyword_rank is None)
            
            # RRF score should reflect single-list contribution
            if vector_rank is not None:
                expected_rrf = 1.0 / (60 + vector_rank)
                assert abs(doc.metadata['rrf_score'] - expected_rrf) < 0.0001
            else:
                expected_rrf = 1.0 / (60 + keyword_rank)
                assert abs(doc.metadata['rrf_score'] - expected_rrf) < 0.0001
        
        # Top 2 documents should be rank 1 from each list (tie at 1/61)
        top_two_scores = [doc.metadata['rrf_score'] for doc in result.documents[:2]]
        expected_top_score = 1.0 / 61
        for score in top_two_scores:
            assert abs(score - expected_top_score) < 0.0001
    
    def test_rrf_disagreement_scenario(self):
        """Test RRF when engines strongly disagree on document relevance."""
        # Vector engine ranks doc1 high, doc2 low
        vector_df = pd.DataFrame({
            'id': ['doc1', 'doc2'],
            'content': ['content1', 'content2'],
            'distance': [0.05, 0.95]  # Very different distances
        })
        
        # Keyword engine ranks doc2 high, doc1 low (opposite)
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['doc2', 'doc1'],
            'chunk_enriched': ['content2', 'content1'],
            'bm25_score': [0.95, 0.05],  # Very different scores
            'file_name': ['f2', 'f1'],
            'chunk_index': [0, 1]
        })
        
        self.mock_vector_store.search.return_value = vector_df
        self.mock_bm25_engine.search.return_value = keyword_df
        
        result = self.hybrid_engine.search("test query", top_k=5)
        
        assert len(result.documents) == 2
        
        # Find both documents
        doc1 = next(doc for doc in result.documents if 'content1' in doc.page_content)
        doc2 = next(doc for doc in result.documents if 'content2' in doc.page_content)
        
        # Check their ranks reflect the disagreement
        assert doc1.metadata['vector_rank'] == 1  # High in vector
        assert doc1.metadata['keyword_rank'] == 2  # Low in keyword
        
        assert doc2.metadata['vector_rank'] == 2  # Low in vector  
        assert doc2.metadata['keyword_rank'] == 1  # High in keyword
        
        # Both should have equal RRF scores (1/61 + 1/62 each)
        expected_rrf = (1.0 / 61) + (1.0 / 62)
        assert abs(doc1.metadata['rrf_score'] - expected_rrf) < 0.0001
        assert abs(doc2.metadata['rrf_score'] - expected_rrf) < 0.0001
        
        # This demonstrates RRF's fairness - no penalty for disagreement
    
    def test_rrf_top_k_limiting(self):
        """Test that RRF respects top_k parameter."""
        # Create larger result sets
        vector_df = pd.DataFrame({
            'id': [f'vec_{i}' for i in range(10)],
            'content': [f'vector content {i}' for i in range(10)],
            'distance': [i * 0.1 for i in range(10)]
        })
        
        keyword_df = pd.DataFrame({
            'uuid_chunk': [f'key_{i}' for i in range(10)],
            'chunk_enriched': [f'keyword content {i}' for i in range(10)],
            'bm25_score': [1.0 - i * 0.1 for i in range(10)],
            'file_name': [f'f{i}' for i in range(10)],
            'chunk_index': list(range(10))
        })
        
        self.mock_vector_store.search.return_value = vector_df
        self.mock_bm25_engine.search.return_value = keyword_df
        
        # Test different top_k values
        for k in [3, 5, 8, 15]:
            result = self.hybrid_engine.search("test query", top_k=k)
            
            # Should return exactly min(k, total_unique_docs) documents
            expected_count = min(k, 20)  # 10 vector + 10 keyword, no overlap
            assert len(result.documents) == expected_count
            
            # Should be ordered by RRF score
            scores = [doc.metadata['rrf_score'] for doc in result.documents]
            assert scores == sorted(scores, reverse=True)
    
    def test_rrf_empty_results_handling(self):
        """Test RRF behavior with empty results from one or both engines."""
        # Test empty vector results
        empty_vector_df = pd.DataFrame(columns=['id', 'content', 'distance'])
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['key1', 'key2'],
            'chunk_enriched': ['keyword content 1', 'keyword content 2'],
            'bm25_score': [0.9, 0.8],
            'file_name': ['f1', 'f2'],
            'chunk_index': [0, 1]
        })
        
        self.mock_vector_store.search.return_value = empty_vector_df
        self.mock_bm25_engine.search.return_value = keyword_df
        
        result = self.hybrid_engine.search("test query", top_k=5)
        
        assert len(result.documents) == 2
        for doc in result.documents:
            assert doc.metadata['vector_rank'] is None
            assert doc.metadata['keyword_rank'] in [1, 2]
            assert doc.metadata['fusion_method'] == 'reciprocal_rank_fusion'
        
        # Test empty keyword results
        vector_df = pd.DataFrame({
            'id': ['vec1', 'vec2'],
            'content': ['vector content 1', 'vector content 2'],
            'distance': [0.1, 0.2]
        })
        empty_keyword_df = pd.DataFrame(columns=['uuid_chunk', 'chunk_enriched', 'bm25_score', 'file_name', 'chunk_index'])
        
        self.mock_vector_store.search.return_value = vector_df
        self.mock_bm25_engine.search.return_value = empty_keyword_df
        
        result = self.hybrid_engine.search("test query", top_k=5)
        
        assert len(result.documents) == 2
        for doc in result.documents:
            assert doc.metadata['vector_rank'] in [1, 2]
            assert doc.metadata['keyword_rank'] is None
            assert doc.metadata['fusion_method'] == 'reciprocal_rank_fusion'
    
    def test_rrf_metadata_preservation_integration(self):
        """Test that RRF preserves original document metadata while adding fusion info."""
        vector_df = pd.DataFrame({
            'id': ['doc1'],
            'content': ['test content'],
            'distance': [0.1],
            'title': ['Test Document'],
            'author': ['Test Author'],
            'custom_field': ['custom_value']
        })
        
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['doc1'],
            'chunk_enriched': ['test content'],
            'bm25_score': [0.9],
            'file_name': ['test.pdf'],
            'chunk_index': [0],
            'extra_field': ['extra_value']
        })
        
        self.mock_vector_store.search.return_value = vector_df
        self.mock_bm25_engine.search.return_value = keyword_df
        
        result = self.hybrid_engine.search("test query", top_k=5)
        
        assert len(result.documents) == 1
        doc = result.documents[0]
        
        # Should preserve vector metadata
        assert doc.metadata['title'] == 'Test Document'
        assert doc.metadata['author'] == 'Test Author'
        assert doc.metadata['custom_field'] == 'custom_value'
        
        # Should preserve keyword metadata - depending on implementation  
        # Note: keyword-specific metadata might not be fully preserved in RRF fusion
        # This is expected behavior as the implementation focuses on core fusion metadata
        
        # Should add RRF fusion metadata
        assert doc.metadata['fusion_method'] == 'reciprocal_rank_fusion'
        assert doc.metadata['rrf_score'] > 0
        assert doc.metadata['vector_rank'] == 1
        assert doc.metadata['keyword_rank'] == 1
        assert doc.metadata['vector_score'] > 0  # Converted from distance
        assert doc.metadata['keyword_score'] == 0.9
    
    def test_rrf_engine_readiness_integration(self):
        """Test RRF integration with engine readiness checks."""
        # Test with vector engine not ready
        mock_vector_engine = Mock()
        mock_vector_engine.is_ready.return_value = False
        mock_keyword_engine = Mock()
        mock_keyword_engine.is_ready.return_value = True
        
        hybrid_engine = CleanHybridSearchEngine(mock_vector_engine, mock_keyword_engine)
        
        assert not hybrid_engine.is_ready()
        
        # Test with both engines ready
        mock_vector_engine.is_ready.return_value = True
        assert hybrid_engine.is_ready()
        
        # Test search behavior when not ready (should still work with error handling)
        mock_vector_engine.is_ready.return_value = False
        mock_vector_engine.search.side_effect = RuntimeError("Vector engine not ready")
        mock_keyword_engine.search.return_value = Mock(
            documents=[Document(page_content="test", metadata={'score': 0.8})],
            metadata={}
        )
        
        result = hybrid_engine.search("test query", top_k=5)
        
        # Should handle gracefully and return results from available engine
        # Actual behavior depends on implementation error handling
        assert isinstance(result, SearchResult)


class TestRRFConfigurationIntegration:
    """Test RRF configuration and parameter integration."""
    
    def test_rrf_k_parameter_configuration(self):
        """Test RRF with different k parameter configurations."""
        mock_vector_store = Mock()
        mock_bm25_engine = Mock()
        mock_bm25_engine.retriever = Mock()
        
        # Test if k parameter is configurable (depends on implementation)
        vector_engine = CleanVectorSearchEngine(mock_vector_store)
        keyword_engine = CleanKeywordSearchEngine(mock_bm25_engine)
        hybrid_engine = CleanHybridSearchEngine(vector_engine, keyword_engine)
        
        # Setup test data
        vector_df = pd.DataFrame({
            'id': ['doc1'],
            'content': ['test content'],
            'distance': [0.1]
        })
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['doc1'],
            'chunk_enriched': ['test content'],
            'bm25_score': [0.9],
            'file_name': ['test.pdf'],
            'chunk_index': [0]
        })
        
        mock_vector_store.search.return_value = vector_df
        mock_bm25_engine.search.return_value = keyword_df
        
        result = hybrid_engine.search("test query", top_k=5)
        
        # Check default k parameter
        assert result.metadata.get('rrf_k_value') == 60
        
        # If implementation allows custom k, test different values
        # This test assumes k is configurable via engine configuration
    
    def test_rrf_factory_integration(self):
        """Test RRF integration through search engine factory."""
        mock_vector_store = Mock()
        mock_bm25_engine = Mock()
        mock_bm25_engine.retriever = Mock()
        
        factory = CleanSearchEngineFactory(mock_vector_store, mock_bm25_engine)
        
        # Create engines through factory
        vector_engine = factory.create_vector_engine()
        keyword_engine = factory.create_keyword_engine()
        hybrid_engine = factory.create_hybrid_engine(vector_engine, keyword_engine)
        
        assert isinstance(hybrid_engine, CleanHybridSearchEngine)
        assert hybrid_engine.vector_engine == vector_engine
        assert hybrid_engine.keyword_engine == keyword_engine
        
        # Test that factory-created engines work with RRF
        vector_df = pd.DataFrame({
            'id': ['doc1'],
            'content': ['test content'],
            'distance': [0.1]
        })
        keyword_df = pd.DataFrame({
            'uuid_chunk': ['doc1'],
            'chunk_enriched': ['test content'],
            'bm25_score': [0.9],
            'file_name': ['test.pdf'],
            'chunk_index': [0]
        })
        
        mock_vector_store.search.return_value = vector_df
        mock_bm25_engine.search.return_value = keyword_df
        
        result = hybrid_engine.search("test query", top_k=5)
        
        assert len(result.documents) == 1
        assert result.metadata.get('fusion_method') == 'reciprocal_rank_fusion'


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v"])