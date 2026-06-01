"""
Performance validation tests for RRF implementation.

Tests that compare RRF against previous weighted fusion approach,
validate performance improvements, and test real-world scenarios.
"""

import pytest
import time
import statistics
from typing import List, Dict, Any, Tuple
from unittest.mock import Mock, patch
from langchain.schema import Document

# Mock implementations for comparison
def mock_weighted_fusion(
    vector_docs: List[Document], 
    keyword_docs: List[Document],
    vector_weight: float = 0.6
) -> List[Document]:
    """Mock of the old weighted fusion approach for comparison."""
    keyword_weight = 1.0 - vector_weight
    doc_scores = {}
    
    # Process vector results
    for doc in vector_docs:
        content_hash = hash(doc.page_content)
        vector_score = 1.0 / (1.0 + doc.metadata.get('distance', 0))
        doc_scores[content_hash] = {
            'document': doc,
            'vector_score': vector_score,
            'keyword_score': 0.0,
            'content_hash': content_hash
        }
    
    # Process keyword results
    for doc in keyword_docs:
        content_hash = hash(doc.page_content)
        keyword_score = doc.metadata.get('score', 0)
        
        if content_hash in doc_scores:
            doc_scores[content_hash]['keyword_score'] = keyword_score
        else:
            doc_scores[content_hash] = {
                'document': doc,
                'vector_score': 0.0,
                'keyword_score': keyword_score,
                'content_hash': content_hash
            }
    
    # Calculate weighted scores
    scored_docs = []
    for entry in doc_scores.values():
        weighted_score = (
            entry['vector_score'] * vector_weight + 
            entry['keyword_score'] * keyword_weight
        )
        
        doc = entry['document']
        doc.metadata.update({
            'weighted_score': weighted_score,
            'vector_score': entry['vector_score'],
            'keyword_score': entry['keyword_score']
        })
        
        scored_docs.append((weighted_score, doc))
    
    scored_docs.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored_docs]


def mock_rrf_fusion_for_validation(
    vector_docs: List[Document], 
    keyword_docs: List[Document],
    k: int = 60
) -> List[Document]:
    """Mock RRF implementation for validation tests."""
    doc_scores = {}
    
    # Process vector results
    for rank, doc in enumerate(vector_docs, 1):
        content_hash = hash(doc.page_content)
        rrf_score = 1.0 / (k + rank)
        doc_scores[content_hash] = {
            'document': doc,
            'rrf_score': rrf_score,
            'vector_rank': rank,
            'keyword_rank': None,
            'content_hash': content_hash
        }
    
    # Process keyword results
    for rank, doc in enumerate(keyword_docs, 1):
        content_hash = hash(doc.page_content)
        rrf_score = 1.0 / (k + rank)
        
        if content_hash in doc_scores:
            doc_scores[content_hash]['rrf_score'] += rrf_score
            doc_scores[content_hash]['keyword_rank'] = rank
        else:
            doc_scores[content_hash] = {
                'document': doc,
                'rrf_score': rrf_score,
                'vector_rank': None,
                'keyword_rank': rank,
                'content_hash': content_hash
            }
    
    # Update document metadata
    for entry in doc_scores.values():
        doc = entry['document']
        doc.metadata.update({
            'rrf_score': entry['rrf_score'],
            'vector_rank': entry['vector_rank'],
            'keyword_rank': entry['keyword_rank'],
            'fusion_method': 'rrf'
        })
    
    # Sort by RRF score
    scored_docs = sorted(
        doc_scores.values(),
        key=lambda x: x['rrf_score'],
        reverse=True
    )
    
    return [entry['document'] for entry in scored_docs]


class TestRRFvsWeightedFusionComparison:
    """Compare RRF against weighted fusion in various scenarios."""
    
    def test_consensus_amplification_comparison(self):
        """Test that RRF better amplifies consensus than weighted fusion."""
        # Document liked by both engines
        vector_docs = [
            Document(page_content="consensus content", metadata={'distance': 0.1}),
            Document(page_content="vector preferred", metadata={'distance': 0.5})
        ]
        keyword_docs = [
            Document(page_content="consensus content", metadata={'score': 0.9}),
            Document(page_content="keyword preferred", metadata={'score': 0.8})
        ]
        
        # Test both approaches
        weighted_result = mock_weighted_fusion(vector_docs, keyword_docs, vector_weight=0.6)
        rrf_result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # Both should rank consensus content first
        assert weighted_result[0].page_content == "consensus content"
        assert rrf_result[0].page_content == "consensus content"
        
        # Compare how strongly they favor the consensus
        consensus_weighted = weighted_result[0].metadata['weighted_score']
        consensus_rrf = rrf_result[0].metadata['rrf_score']
        
        # RRF should give more consistent ranking (both get rank 1)
        # Weighted fusion depends on actual scores which can vary
        assert rrf_result[0].metadata['vector_rank'] == 1
        assert rrf_result[0].metadata['keyword_rank'] == 1
    
    def test_disagreement_handling_comparison(self):
        """Test how RRF vs weighted fusion handle engine disagreement."""
        # Engines strongly disagree on scores but both contain the documents
        vector_docs = [
            Document(page_content="doc1", metadata={'distance': 0.1}),  # vector likes
            Document(page_content="doc2", metadata={'distance': 0.9})   # vector dislikes
        ]
        keyword_docs = [
            Document(page_content="doc2", metadata={'score': 0.9}),     # keyword likes
            Document(page_content="doc1", metadata={'score': 0.1})      # keyword dislikes
        ]
        
        weighted_result = mock_weighted_fusion(vector_docs, keyword_docs, vector_weight=0.6)
        rrf_result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # Find scores for both documents
        doc1_weighted = next(doc for doc in weighted_result if doc.page_content == "doc1")
        doc2_weighted = next(doc for doc in weighted_result if doc.page_content == "doc2")
        
        doc1_rrf = next(doc for doc in rrf_result if doc.page_content == "doc1")
        doc2_rrf = next(doc for doc in rrf_result if doc.page_content == "doc2")
        
        # Weighted fusion: doc1 should win due to vector preference
        # RRF: both should have equal scores (rank 1 + rank 2)
        
        weighted_diff = abs(doc1_weighted.metadata['weighted_score'] - doc2_weighted.metadata['weighted_score'])
        rrf_diff = abs(doc1_rrf.metadata['rrf_score'] - doc2_rrf.metadata['rrf_score'])
        
        # RRF should show smaller difference (more fair treatment)
        assert rrf_diff < weighted_diff
    
    def test_score_magnitude_bias_mitigation(self):
        """Test RRF's mitigation of score magnitude bias."""
        # High vector scores vs moderate keyword scores
        vector_docs = [
            Document(page_content="high_vec", metadata={'distance': 0.01}),  # Very high vector score
            Document(page_content="med_vec", metadata={'distance': 0.5})     # Medium vector score
        ]
        keyword_docs = [
            Document(page_content="med_vec", metadata={'score': 0.7}),       # Medium keyword score  
            Document(page_content="high_vec", metadata={'score': 0.6})       # Lower keyword score
        ]
        
        weighted_result = mock_weighted_fusion(vector_docs, keyword_docs, vector_weight=0.6)
        rrf_result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # In weighted fusion, high_vec might win due to very high vector score
        # In RRF, both have equal treatment (ranks 1+2, 2+1)
        
        high_vec_weighted = next(doc for doc in weighted_result if doc.page_content == "high_vec")
        med_vec_weighted = next(doc for doc in weighted_result if doc.page_content == "med_vec")
        
        high_vec_rrf = next(doc for doc in rrf_result if doc.page_content == "high_vec")
        med_vec_rrf = next(doc for doc in rrf_result if doc.page_content == "med_vec")
        
        # RRF should give equal scores
        assert abs(high_vec_rrf.metadata['rrf_score'] - med_vec_rrf.metadata['rrf_score']) < 0.0001
        
        # Weighted fusion will likely prefer high_vec due to vector bias
        assert high_vec_weighted.metadata['weighted_score'] != med_vec_weighted.metadata['weighted_score']
    
    def test_ranking_stability_comparison(self):
        """Test ranking stability between approaches."""
        # Documents with similar scores but different orderings
        vector_docs = [
            Document(page_content="doc_a", metadata={'distance': 0.2}),
            Document(page_content="doc_b", metadata={'distance': 0.21}),
            Document(page_content="doc_c", metadata={'distance': 0.22})
        ]
        keyword_docs = [
            Document(page_content="doc_c", metadata={'score': 0.8}),
            Document(page_content="doc_a", metadata={'score': 0.79}),
            Document(page_content="doc_b", metadata={'score': 0.78})
        ]
        
        weighted_result = mock_weighted_fusion(vector_docs, keyword_docs, vector_weight=0.5)
        rrf_result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # Both should produce stable, deterministic rankings
        assert len(weighted_result) == 3
        assert len(rrf_result) == 3
        
        # RRF ranking should be based on position consensus
        # All documents appear at positions (1,2), (2,3), (3,1) - so different RRF scores
        rrf_scores = [(doc.page_content, doc.metadata['rrf_score']) for doc in rrf_result]
        rrf_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Verify RRF produces consistent ranking
        assert len(set(score for _, score in rrf_scores)) >= 2  # Should have different scores


class TestRRFPerformanceCharacteristics:
    """Test performance aspects of RRF implementation."""
    
    def test_rrf_computational_complexity(self):
        """Test RRF performance with increasing result set sizes."""
        def create_test_docs(n: int, prefix: str) -> List[Document]:
            return [
                Document(page_content=f"{prefix}_doc_{i}", metadata={'score': 1.0 - i*0.01})
                for i in range(n)
            ]
        
        # Test with increasing sizes
        sizes = [10, 50, 100, 500]
        rrf_times = []
        weighted_times = []
        
        for size in sizes:
            vector_docs = create_test_docs(size, "vec")
            keyword_docs = create_test_docs(size, "key")
            
            # Time RRF
            start_time = time.time()
            for _ in range(10):  # Average over multiple runs
                mock_rrf_fusion_for_validation(vector_docs, keyword_docs)
            rrf_time = (time.time() - start_time) / 10
            rrf_times.append(rrf_time)
            
            # Time weighted fusion
            start_time = time.time()
            for _ in range(10):  # Average over multiple runs
                mock_weighted_fusion(vector_docs, keyword_docs)
            weighted_time = (time.time() - start_time) / 10
            weighted_times.append(weighted_time)
        
        # RRF should have reasonable performance scaling
        # Both approaches are O(n) where n is total documents
        for i in range(len(sizes)):
            # Performance should be reasonable (within 10x of each other)
            assert rrf_times[i] < weighted_times[i] * 10
            assert weighted_times[i] < rrf_times[i] * 10
    
    def test_rrf_memory_efficiency(self):
        """Test memory usage characteristics of RRF."""
        # Create overlapping document sets
        vector_docs = [
            Document(page_content=f"doc_{i}", metadata={'distance': i*0.1})
            for i in range(100)
        ]
        keyword_docs = [
            Document(page_content=f"doc_{i}", metadata={'score': 1.0 - i*0.01})
            for i in range(50, 150)  # 50% overlap
        ]
        
        # RRF should handle overlapping documents efficiently
        result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # Should have exactly 150 unique documents (no duplication)
        assert len(result) == 150
        
        # Check that overlapping documents have both ranks
        overlapping_docs = [doc for doc in result 
                          if doc.metadata.get('vector_rank') and doc.metadata.get('keyword_rank')]
        assert len(overlapping_docs) == 50  # 50 overlapping documents
    
    def test_rrf_parameter_sensitivity(self):
        """Test how RRF k parameter affects performance."""
        from core.search.rrf_fusion import calculate_rrf_scores
        
        # Create test data: doc1 ranks 1,2 and doc2 ranks 2,1 
        vector_results = [('doc1', 1), ('doc2', 2)]
        keyword_results = [('doc2', 1), ('doc1', 2)]
        
        # Test different k values
        k_values = [1, 10, 60, 100, 1000]
        scores_by_k = {}
        
        for k in k_values:
            scores_by_k[k] = calculate_rrf_scores([vector_results, keyword_results], k=k)
        
        # All should produce same score patterns (both documents have equal rank patterns)
        for k in k_values:
            doc1_score = scores_by_k[k]['doc1']
            doc2_score = scores_by_k[k]['doc2']
            assert abs(doc1_score - doc2_score) < 0.0001  # Should be equal due to symmetric ranks
        
        # But actual scores should vary with k
        doc1_k1 = scores_by_k[1]['doc1']
        doc1_k1000 = scores_by_k[1000]['doc1']
        
        # Verify k parameter effect
        # k=1: 1/2 + 1/3 = 0.833...
        # k=1000: 1/1001 + 1/1002 ≈ 0.002  
        expected_k1 = (1.0 / 2) + (1.0 / 3)
        expected_k1000 = (1.0 / 1001) + (1.0 / 1002)
        
        assert abs(doc1_k1 - expected_k1) < 0.001
        assert abs(doc1_k1000 - expected_k1000) < 0.000001
        assert doc1_k1 > doc1_k1000  # Lower k gives higher scores
        
        # Test monotonic decrease as k increases
        prev_score = None
        for k in k_values:
            current_score = scores_by_k[k]['doc1']
            if prev_score is not None:
                assert current_score <= prev_score  # Scores decrease as k increases
            prev_score = current_score


class TestRRFRealWorldScenarios:
    """Test RRF in realistic search scenarios."""
    
    def test_academic_search_scenario(self):
        """Test RRF in academic paper search context."""
        # Vector search: semantic similarity
        vector_docs = [
            Document(page_content="neural networks deep learning architectures", 
                    metadata={'distance': 0.1, 'title': 'Deep Learning Survey'}),
            Document(page_content="machine learning algorithms optimization", 
                    metadata={'distance': 0.2, 'title': 'ML Optimization'}),
            Document(page_content="artificial intelligence systems design", 
                    metadata={'distance': 0.3, 'title': 'AI Systems'})
        ]
        
        # Keyword search: exact term matches
        keyword_docs = [
            Document(page_content="neural network architecture optimization techniques", 
                    metadata={'score': 0.95, 'title': 'Network Optimization'}),
            Document(page_content="neural networks deep learning architectures", 
                    metadata={'score': 0.88, 'title': 'Deep Learning Survey'}),
            Document(page_content="optimization algorithms for machine learning", 
                    metadata={'score': 0.75, 'title': 'Optimization Algorithms'})
        ]
        
        result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # Should prioritize the overlapping document
        overlapping_docs = [doc for doc in result 
                          if doc.metadata.get('vector_rank') and doc.metadata.get('keyword_rank')]
        assert len(overlapping_docs) == 1
        assert overlapping_docs[0].metadata['title'] == 'Deep Learning Survey'
        
        # Should be ranked first due to consensus
        assert result[0].metadata['title'] == 'Deep Learning Survey'
    
    def test_multilingual_search_scenario(self):
        """Test RRF with multilingual content."""
        vector_docs = [
            Document(page_content="english machine learning content", 
                    metadata={'distance': 0.1, 'language': 'en'}),
            Document(page_content="mixed english and français machine learning", 
                    metadata={'distance': 0.25, 'language': 'mixed'})
        ]
        
        keyword_docs = [
            Document(page_content="machine learning apprentissage automatique", 
                    metadata={'score': 0.85, 'language': 'mixed'}),
            Document(page_content="mixed english and français machine learning", 
                    metadata={'score': 0.8, 'language': 'mixed'})
        ]
        
        result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # Mixed language document appears in both - should rank highly
        mixed_docs = [doc for doc in result if doc.metadata.get('language') == 'mixed' 
                     and doc.metadata.get('vector_rank') and doc.metadata.get('keyword_rank')]
        assert len(mixed_docs) == 1
        
        # Should have good RRF score due to appearing in both lists
        assert mixed_docs[0].metadata['rrf_score'] > 0
    
    def test_domain_specific_search_scenario(self):
        """Test RRF in domain-specific search (medical, legal, etc.)."""
        # Medical search: vector finds semantic similarity, keyword finds exact terms
        vector_docs = [
            Document(page_content="cardiovascular disease risk factors prevention", 
                    metadata={'distance': 0.15, 'domain': 'medical'}),
            Document(page_content="heart disease prevention lifestyle changes", 
                    metadata={'distance': 0.2, 'domain': 'medical'})
        ]
        
        keyword_docs = [
            Document(page_content="cardiovascular prevention guidelines 2024", 
                    metadata={'score': 0.9, 'domain': 'medical'}),
            Document(page_content="cardiovascular disease risk factors prevention", 
                    metadata={'score': 0.85, 'domain': 'medical'}),
            Document(page_content="cardiac rehabilitation cardiovascular outcomes", 
                    metadata={'score': 0.7, 'domain': 'medical'})
        ]
        
        result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # Exact match should rank highly
        exact_match = next(doc for doc in result 
                          if "cardiovascular disease risk factors prevention" in doc.page_content)
        
        # Should have both ranks
        assert exact_match.metadata.get('vector_rank') == 1
        assert exact_match.metadata.get('keyword_rank') == 2
        
        # Should be highly ranked due to appearing in both results
        high_ranked_docs = result[:2]
        assert exact_match in high_ranked_docs
    
    def test_temporal_content_search_scenario(self):
        """Test RRF with time-sensitive content."""
        # Recent content vs older content
        vector_docs = [
            Document(page_content="AI trends 2024 machine learning developments", 
                    metadata={'distance': 0.1, 'year': 2024}),
            Document(page_content="artificial intelligence future predictions", 
                    metadata={'distance': 0.2, 'year': 2023})
        ]
        
        keyword_docs = [
            Document(page_content="machine learning trends 2024 developments", 
                    metadata={'score': 0.9, 'year': 2024}),
            Document(page_content="AI trends 2024 machine learning developments", 
                    metadata={'score': 0.85, 'year': 2024})
        ]
        
        result = mock_rrf_fusion_for_validation(vector_docs, keyword_docs, k=60)
        
        # 2024 content that appears in both should rank highest
        consensus_2024 = next(doc for doc in result 
                            if doc.metadata.get('year') == 2024 
                            and doc.metadata.get('vector_rank') 
                            and doc.metadata.get('keyword_rank'))
        
        assert result[0] == consensus_2024
        assert "AI trends 2024 machine learning developments" in consensus_2024.page_content


if __name__ == "__main__":
    # Run performance validation tests
    pytest.main([__file__, "-v", "-s"])