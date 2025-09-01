"""Comprehensive unit tests for RRF fusion algorithm."""

import pytest
from core.search.rrf_fusion import calculate_rrf_scores, fuse_search_results, RRFFusion


class TestCalculateRRFScores:
    """Test suite for calculate_rrf_scores function."""
    
    def test_single_engine_basic_scoring(self):
        """Test RRF scoring with a single search engine."""
        ranked_results = [
            [("doc1", 1), ("doc2", 2), ("doc3", 3)]
        ]
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        # Expected scores: doc1=1/61, doc2=1/62, doc3=1/63
        expected = {
            "doc1": 1.0 / 61,
            "doc2": 1.0 / 62,
            "doc3": 1.0 / 63
        }
        
        assert len(scores) == 3
        for doc_id, expected_score in expected.items():
            assert abs(scores[doc_id] - expected_score) < 1e-10
    
    def test_multiple_engines_no_overlap(self):
        """Test RRF scoring with multiple engines and no document overlap."""
        ranked_results = [
            [("doc1", 1), ("doc2", 2)],  # Engine 1
            [("doc3", 1), ("doc4", 2)]   # Engine 2
        ]
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        expected = {
            "doc1": 1.0 / 61,
            "doc2": 1.0 / 62,
            "doc3": 1.0 / 61,
            "doc4": 1.0 / 62
        }
        
        assert len(scores) == 4
        for doc_id, expected_score in expected.items():
            assert abs(scores[doc_id] - expected_score) < 1e-10
    
    def test_multiple_engines_with_overlap(self):
        """Test RRF scoring with overlapping documents from multiple engines."""
        ranked_results = [
            [("doc1", 1), ("doc2", 2), ("doc3", 3)],  # Engine 1
            [("doc2", 1), ("doc3", 2), ("doc4", 3)]   # Engine 2
        ]
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        expected = {
            "doc1": 1.0 / 61,                    # Only in engine 1, rank 1
            "doc2": 1.0 / 62 + 1.0 / 61,        # Engine 1 rank 2 + engine 2 rank 1
            "doc3": 1.0 / 63 + 1.0 / 62,        # Engine 1 rank 3 + engine 2 rank 2  
            "doc4": 1.0 / 63                     # Only in engine 2, rank 3
        }
        
        assert len(scores) == 4
        for doc_id, expected_score in expected.items():
            assert abs(scores[doc_id] - expected_score) < 1e-10
    
    def test_different_k_values(self):
        """Test RRF scoring with different k parameter values."""
        ranked_results = [[("doc1", 1), ("doc2", 2)]]
        
        # Test k=0
        scores_k0 = calculate_rrf_scores(ranked_results, k=0)
        assert abs(scores_k0["doc1"] - 1.0) < 1e-10
        assert abs(scores_k0["doc2"] - 0.5) < 1e-10
        
        # Test k=10
        scores_k10 = calculate_rrf_scores(ranked_results, k=10)
        assert abs(scores_k10["doc1"] - 1.0/11) < 1e-10
        assert abs(scores_k10["doc2"] - 1.0/12) < 1e-10
        
        # Test k=100
        scores_k100 = calculate_rrf_scores(ranked_results, k=100)
        assert abs(scores_k100["doc1"] - 1.0/101) < 1e-10
        assert abs(scores_k100["doc2"] - 1.0/102) < 1e-10
    
    def test_empty_results_list(self):
        """Test error handling for empty ranked_results list."""
        with pytest.raises(ValueError, match="ranked_results cannot be empty"):
            calculate_rrf_scores([], k=60)
    
    def test_negative_k_parameter(self):
        """Test error handling for negative k parameter."""
        ranked_results = [[("doc1", 1)]]
        with pytest.raises(ValueError, match="k parameter must be non-negative"):
            calculate_rrf_scores(ranked_results, k=-1)
    
    def test_empty_inner_lists(self):
        """Test handling of empty result lists from engines."""
        ranked_results = [
            [("doc1", 1), ("doc2", 2)],  # Engine 1 with results
            [],                           # Engine 2 with no results
            [("doc3", 1)]                # Engine 3 with results
        ]
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        expected = {
            "doc1": 1.0 / 61,
            "doc2": 1.0 / 62,
            "doc3": 1.0 / 61
        }
        
        assert len(scores) == 3
        for doc_id, expected_score in expected.items():
            assert abs(scores[doc_id] - expected_score) < 1e-10
    
    def test_all_empty_inner_lists(self):
        """Test handling when all engine result lists are empty."""
        ranked_results = [[], [], []]
        scores = calculate_rrf_scores(ranked_results, k=60)
        assert scores == {}
    
    def test_invalid_ranks(self):
        """Test handling of invalid rank values (zero or negative)."""
        ranked_results = [
            [("doc1", 1), ("doc2", 0), ("doc3", -1), ("doc4", 2)]
        ]
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        # Should only include doc1 and doc4 (valid ranks)
        expected = {
            "doc1": 1.0 / 61,
            "doc4": 1.0 / 62
        }
        
        assert len(scores) == 2
        for doc_id, expected_score in expected.items():
            assert abs(scores[doc_id] - expected_score) < 1e-10


class TestFuseSearchResults:
    """Test suite for fuse_search_results function."""
    
    def test_basic_fusion_and_sorting(self):
        """Test basic fusion and sorting of results."""
        ranked_results = [
            [("doc1", 1), ("doc2", 3)],  # Engine 1
            [("doc2", 1), ("doc3", 2)]   # Engine 2
        ]
        
        fused = fuse_search_results(ranked_results, k=60)
        
        # doc2 should have highest score (appears in both engines with good ranks)
        # Expected scores: doc2 = 1/63 + 1/61, doc1 = 1/61, doc3 = 1/62
        assert len(fused) == 3
        assert fused[0][0] == "doc2"  # Highest score
        assert fused[1][0] == "doc1"  # Second highest
        assert fused[2][0] == "doc3"  # Lowest score
        
        # Verify scores are in descending order
        for i in range(len(fused) - 1):
            assert fused[i][1] >= fused[i + 1][1]
    
    def test_top_k_limiting(self):
        """Test top_k parameter limits results correctly."""
        ranked_results = [
            [("doc1", 1), ("doc2", 2), ("doc3", 3), ("doc4", 4)]
        ]
        
        # Test top_k=2
        fused = fuse_search_results(ranked_results, k=60, top_k=2)
        assert len(fused) == 2
        assert fused[0][0] == "doc1"
        assert fused[1][0] == "doc2"
        
        # Test top_k=None (all results)
        fused_all = fuse_search_results(ranked_results, k=60, top_k=None)
        assert len(fused_all) == 4
        
        # Test top_k=0 (should return empty list)
        fused_zero = fuse_search_results(ranked_results, k=60, top_k=0)
        assert len(fused_zero) == 0
    
    def test_tie_handling(self):
        """Test consistent handling of tied scores."""
        # Create scenario where two documents have identical RRF scores
        ranked_results = [
            [("doc1", 1)],  # Score: 1/61
            [("doc2", 1)]   # Score: 1/61
        ]
        
        fused = fuse_search_results(ranked_results, k=60)
        assert len(fused) == 2
        assert abs(fused[0][1] - fused[1][1]) < 1e-10  # Same scores
        
        # Verify deterministic ordering (should be consistent)
        doc_ids = [doc_id for doc_id, _ in fused]
        assert sorted(doc_ids) == ["doc1", "doc2"]


class TestRRFFusion:
    """Test suite for RRFFusion class."""
    
    def test_initialization(self):
        """Test RRFFusion initialization."""
        # Default k=60
        rrf = RRFFusion()
        assert rrf.k == 60
        
        # Custom k
        rrf_custom = RRFFusion(k=100)
        assert rrf_custom.k == 100
    
    def test_initialization_negative_k(self):
        """Test error handling for negative k in initialization."""
        with pytest.raises(ValueError, match="k parameter must be non-negative"):
            RRFFusion(k=-5)
    
    def test_calculate_scores_method(self):
        """Test calculate_scores method uses configured k parameter."""
        rrf = RRFFusion(k=10)
        ranked_results = [[("doc1", 1), ("doc2", 2)]]
        
        scores = rrf.calculate_scores(ranked_results)
        
        expected = {
            "doc1": 1.0 / 11,
            "doc2": 1.0 / 12
        }
        
        assert len(scores) == 2
        for doc_id, expected_score in expected.items():
            assert abs(scores[doc_id] - expected_score) < 1e-10
    
    def test_fuse_results_method(self):
        """Test fuse_results method uses configured k parameter."""
        rrf = RRFFusion(k=10)
        ranked_results = [
            [("doc1", 1), ("doc2", 2)],
            [("doc2", 1), ("doc3", 2)]
        ]
        
        fused = rrf.fuse_results(ranked_results, top_k=2)
        
        assert len(fused) == 2
        # doc2 should be first (highest combined score)
        assert fused[0][0] == "doc2"
        
        # Verify scores are computed with k=10
        expected_doc2_score = 1.0/12 + 1.0/11  # rank 2 in engine 1 + rank 1 in engine 2
        assert abs(fused[0][1] - expected_doc2_score) < 1e-10


class TestRRFEdgeCases:
    """Test edge cases and disagreement scenarios."""
    
    def test_complete_disagreement(self):
        """Test scenario where engines completely disagree on rankings."""
        ranked_results = [
            [("doc1", 1), ("doc2", 2), ("doc3", 3)],  # Engine 1 ranking
            [("doc3", 1), ("doc2", 2), ("doc1", 3)]   # Engine 2 reverse ranking
        ]
        
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        # Calculate expected scores
        doc1_score = 1.0/61 + 1.0/63  # rank 1 + rank 3
        doc2_score = 1.0/62 + 1.0/62  # rank 2 + rank 2  
        doc3_score = 1.0/63 + 1.0/61  # rank 3 + rank 1
        
        assert abs(scores["doc1"] - doc1_score) < 1e-10
        assert abs(scores["doc2"] - doc2_score) < 1e-10
        assert abs(scores["doc3"] - doc3_score) < 1e-10
        
        # doc1 and doc3 should have equal scores (symmetric disagreement)
        assert abs(scores["doc1"] - scores["doc3"]) < 1e-10
        # doc2 should have slightly lower score (consistent but worse rank 2)
        assert scores["doc1"] > scores["doc2"]
        assert scores["doc3"] > scores["doc2"]
    
    def test_single_document_multiple_engines(self):
        """Test same document appearing in multiple engines at different ranks."""
        ranked_results = [
            [("doc1", 1)],    # Best rank in engine 1
            [("doc1", 5)],    # Worse rank in engine 2
            [("doc1", 2)]     # Medium rank in engine 3
        ]
        
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        expected_score = 1.0/61 + 1.0/65 + 1.0/62
        assert len(scores) == 1
        assert abs(scores["doc1"] - expected_score) < 1e-10
    
    def test_large_rank_values(self):
        """Test handling of large rank values."""
        ranked_results = [
            [("doc1", 1000), ("doc2", 2000)]
        ]
        
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        expected = {
            "doc1": 1.0 / 1060,
            "doc2": 1.0 / 2060
        }
        
        for doc_id, expected_score in expected.items():
            assert abs(scores[doc_id] - expected_score) < 1e-10
    
    def test_many_engines(self):
        """Test RRF with many search engines."""
        num_engines = 10
        ranked_results = []
        
        # Each engine returns doc1 at rank 1
        for i in range(num_engines):
            ranked_results.append([("doc1", 1)])
        
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        # doc1 should appear in all engines at rank 1
        expected_score = num_engines * (1.0 / 61)
        assert abs(scores["doc1"] - expected_score) < 1e-10
    
    def test_mixed_document_sets(self):
        """Test engines returning completely different document sets."""
        ranked_results = [
            [("eng1_doc1", 1), ("eng1_doc2", 2)],     # Engine 1 docs
            [("eng2_doc1", 1), ("eng2_doc2", 2)],     # Engine 2 docs  
            [("eng3_doc1", 1), ("eng3_doc2", 2)]      # Engine 3 docs
        ]
        
        scores = calculate_rrf_scores(ranked_results, k=60)
        
        # Should have 6 documents total, each appearing in exactly one engine
        assert len(scores) == 6
        
        # All rank-1 documents should have equal scores
        rank1_docs = ["eng1_doc1", "eng2_doc1", "eng3_doc1"]
        rank1_score = 1.0 / 61
        for doc in rank1_docs:
            assert abs(scores[doc] - rank1_score) < 1e-10
        
        # All rank-2 documents should have equal scores
        rank2_docs = ["eng1_doc2", "eng2_doc2", "eng3_doc2"]
        rank2_score = 1.0 / 62
        for doc in rank2_docs:
            assert abs(scores[doc] - rank2_score) < 1e-10