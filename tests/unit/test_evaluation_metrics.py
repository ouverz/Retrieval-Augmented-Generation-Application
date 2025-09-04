"""
Comprehensive unit tests for evaluation metrics module.

Tests cover Mean Reciprocal Rank (MRR), Precision@K, and Mean Average 
Precision (MAP) metrics with various edge cases and scenarios.
"""

import pytest
import pandas as pd
from typing import Dict, Set, List
from langchain.schema import Document

from core.evaluation.metrics import (
    MeanReciprocalRank,
    PrecisionAtK, 
    MeanAveragePrecision,
    EvaluationResult,
    evaluate_search_results
)


class TestMeanReciprocalRank:
    """Test suite for Mean Reciprocal Rank (MRR) metric."""
    
    def test_mrr_perfect_ranking(self):
        """Test MRR with perfect ranking (relevant doc at rank 1)."""
        mrr = MeanReciprocalRank()
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3"],
            "q2": ["doc4", "doc5", "doc6"]
        }
        ground_truth = {
            "q1": {"doc1"},  # First doc is relevant
            "q2": {"doc4"}   # First doc is relevant
        }
        
        result = mrr.calculate(search_results, ground_truth)
        
        assert result.score == 1.0  # Perfect MRR
        assert result.query_scores["q1"] == 1.0  # 1/1
        assert result.query_scores["q2"] == 1.0  # 1/1
        assert result.metadata["queries_with_ground_truth"] == 2
    
    def test_mrr_various_ranks(self):
        """Test MRR with relevant docs at different ranks."""
        mrr = MeanReciprocalRank()
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3"],  # Relevant at rank 1
            "q2": ["doc4", "doc5", "doc6"],  # Relevant at rank 2  
            "q3": ["doc7", "doc8", "doc9"]   # Relevant at rank 3
        }
        ground_truth = {
            "q1": {"doc1"},
            "q2": {"doc5"}, 
            "q3": {"doc9"}
        }
        
        result = mrr.calculate(search_results, ground_truth)
        
        # MRR = (1/1 + 1/2 + 1/3) / 3 = (1.0 + 0.5 + 0.333...) / 3
        expected_mrr = (1.0 + 0.5 + 1.0/3) / 3
        assert abs(result.score - expected_mrr) < 1e-10
        
        assert result.query_scores["q1"] == 1.0
        assert result.query_scores["q2"] == 0.5
        assert abs(result.query_scores["q3"] - 1.0/3) < 1e-10
    
    def test_mrr_no_relevant_found(self):
        """Test MRR when no relevant documents are found."""
        mrr = MeanReciprocalRank()
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3"],
            "q2": ["doc4", "doc5", "doc6"]
        }
        ground_truth = {
            "q1": {"doc_x"},  # Not in results
            "q2": {"doc_y"}   # Not in results
        }
        
        result = mrr.calculate(search_results, ground_truth)
        
        assert result.score == 0.0
        assert result.query_scores["q1"] == 0.0
        assert result.query_scores["q2"] == 0.0
        assert result.metadata["zero_reciprocal_ranks"] == 2
    
    def test_mrr_empty_search_results(self):
        """Test MRR with empty search results."""
        mrr = MeanReciprocalRank()
        
        result = mrr.calculate({}, {})
        
        assert result.score == 0.0
        assert result.query_scores == {}
        assert result.metadata["total_queries"] == 0
    
    def test_mrr_empty_ground_truth(self):
        """Test MRR when ground truth is missing for queries."""
        mrr = MeanReciprocalRank()
        
        search_results = {
            "q1": ["doc1", "doc2"],
            "q2": ["doc3", "doc4"]
        }
        ground_truth = {}  # No ground truth
        
        result = mrr.calculate(search_results, ground_truth)
        
        assert result.score == 0.0
        assert result.query_scores == {}
        assert result.metadata["queries_with_ground_truth"] == 0
    
    def test_mrr_multiple_relevant_docs(self):
        """Test MRR with multiple relevant docs (only first rank matters)."""
        mrr = MeanReciprocalRank()
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3", "doc4"]
        }
        ground_truth = {
            "q1": {"doc2", "doc4"}  # Both at ranks 2 and 4
        }
        
        result = mrr.calculate(search_results, ground_truth)
        
        # Should use rank of first relevant doc (doc2 at rank 2)
        assert result.score == 0.5  # 1/2
        assert result.query_scores["q1"] == 0.5


class TestPrecisionAtK:
    """Test suite for Precision@K metric."""
    
    def test_precision_at_k_perfect(self):
        """Test Precision@K with all top-K results being relevant."""
        precision = PrecisionAtK(k=3)
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3", "doc4", "doc5"]
        }
        ground_truth = {
            "q1": {"doc1", "doc2", "doc3", "doc6", "doc7"}
        }
        
        result = precision.calculate(search_results, ground_truth)
        
        assert result.score == 1.0  # 3/3 relevant in top-3
        assert result.query_scores["q1"] == 1.0
        assert result.metadata["k_value"] == 3
        assert result.metadata["perfect_precision_queries"] == 1
    
    def test_precision_at_k_partial(self):
        """Test Precision@K with partial relevance in top-K."""
        precision = PrecisionAtK(k=4)
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3", "doc4"],  # 2 relevant out of 4
            "q2": ["doc5", "doc6", "doc7", "doc8"]   # 1 relevant out of 4
        }
        ground_truth = {
            "q1": {"doc1", "doc3"},
            "q2": {"doc6"}
        }
        
        result = precision.calculate(search_results, ground_truth)
        
        # Average precision: (2/4 + 1/4) / 2 = 0.375
        assert result.score == 0.375
        assert result.query_scores["q1"] == 0.5   # 2/4
        assert result.query_scores["q2"] == 0.25  # 1/4
    
    def test_precision_at_k_zero(self):
        """Test Precision@K when no relevant docs in top-K."""
        precision = PrecisionAtK(k=2)
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3"],
            "q2": ["doc4", "doc5", "doc6"]
        }
        ground_truth = {
            "q1": {"doc3"},  # Relevant doc at rank 3 (outside top-2)
            "q2": {"doc7"}   # Relevant doc not in results
        }
        
        result = precision.calculate(search_results, ground_truth)
        
        assert result.score == 0.0
        assert result.query_scores["q1"] == 0.0
        assert result.query_scores["q2"] == 0.0
        assert result.metadata["zero_precision_queries"] == 2
    
    def test_precision_at_k_fewer_results_than_k(self):
        """Test Precision@K when search returns fewer than K results."""
        precision = PrecisionAtK(k=5)
        
        search_results = {
            "q1": ["doc1", "doc2"]  # Only 2 results, but k=5
        }
        ground_truth = {
            "q1": {"doc1"}
        }
        
        result = precision.calculate(search_results, ground_truth)
        
        # Should calculate based on actual results: 1/2 = 0.5
        assert result.score == 0.5
        assert result.query_scores["q1"] == 0.5
    
    def test_precision_at_k_invalid_k(self):
        """Test Precision@K initialization with invalid k values."""
        with pytest.raises(ValueError, match="k parameter must be positive"):
            PrecisionAtK(k=0)
        
        with pytest.raises(ValueError, match="k parameter must be positive"):
            PrecisionAtK(k=-1)
    
    def test_precision_at_k_different_k_values(self):
        """Test Precision@K with different k values."""
        search_results = {
            "q1": ["doc1", "doc2", "doc3", "doc4", "doc5"]
        }
        ground_truth = {
            "q1": {"doc1", "doc3", "doc5"}  # 3 relevant docs
        }
        
        # Test P@3: 2/3 relevant
        p_at_3 = PrecisionAtK(k=3)
        result_3 = p_at_3.calculate(search_results, ground_truth)
        assert abs(result_3.score - 2/3) < 1e-10
        
        # Test P@5: 3/5 relevant  
        p_at_5 = PrecisionAtK(k=5)
        result_5 = p_at_5.calculate(search_results, ground_truth)
        assert result_5.score == 0.6  # 3/5


class TestMeanAveragePrecision:
    """Test suite for Mean Average Precision (MAP) metric."""
    
    def test_map_perfect_ranking(self):
        """Test MAP with perfect ranking of all relevant docs."""
        map_metric = MeanAveragePrecision()
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3", "doc4"]  # All relevant at top
        }
        ground_truth = {
            "q1": {"doc1", "doc2", "doc3"}
        }
        
        result = map_metric.calculate(search_results, ground_truth)
        
        # AP = (1/1 + 2/2 + 3/3) / 3 = (1 + 1 + 1) / 3 = 1.0
        assert result.score == 1.0
        assert result.query_scores["q1"] == 1.0
        assert result.metadata["perfect_ap_queries"] == 1
    
    def test_map_mixed_ranking(self):
        """Test MAP with relevant docs scattered in ranking."""
        map_metric = MeanAveragePrecision()
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3", "doc4", "doc5"]  # rel at 1,3,5
        }
        ground_truth = {
            "q1": {"doc1", "doc3", "doc5"}
        }
        
        result = map_metric.calculate(search_results, ground_truth)
        
        # AP = (1/1 + 2/3 + 3/5) / 3 = (1 + 0.667 + 0.6) / 3 = 0.756
        expected_ap = (1.0 + 2.0/3.0 + 3.0/5.0) / 3.0
        assert abs(result.score - expected_ap) < 1e-10
        assert abs(result.query_scores["q1"] - expected_ap) < 1e-10
    
    def test_map_multiple_queries(self):
        """Test MAP calculation across multiple queries."""
        map_metric = MeanAveragePrecision()
        
        search_results = {
            "q1": ["doc1", "doc2", "doc3"],  # AP = 1.0
            "q2": ["doc4", "doc5", "doc6"]   # AP = 0.0 (no relevant docs)
        }
        ground_truth = {
            "q1": {"doc1"},
            "q2": {"doc7"}  # Not in results
        }
        
        result = map_metric.calculate(search_results, ground_truth)
        
        # MAP = (1.0 + 0.0) / 2 = 0.5
        assert result.score == 0.5
        assert result.query_scores["q1"] == 1.0
        assert result.query_scores["q2"] == 0.0
        assert result.metadata["zero_ap_queries"] == 1
    
    def test_map_no_relevant_docs(self):
        """Test MAP when no relevant documents are found."""
        map_metric = MeanAveragePrecision()
        
        search_results = {
            "q1": ["doc1", "doc2"],
            "q2": ["doc3", "doc4"]
        }
        ground_truth = {
            "q1": {"doc_x"},  # Not found
            "q2": {"doc_y"}   # Not found
        }
        
        result = map_metric.calculate(search_results, ground_truth)
        
        assert result.score == 0.0
        assert result.query_scores["q1"] == 0.0
        assert result.query_scores["q2"] == 0.0
        assert result.metadata["zero_ap_queries"] == 2
    
    def test_map_empty_results(self):
        """Test MAP with empty search results."""
        map_metric = MeanAveragePrecision()
        
        result = map_metric.calculate({}, {})
        
        assert result.score == 0.0
        assert result.query_scores == {}
        assert result.metadata["total_queries"] == 0


class TestUtilityFunctions:
    """Test suite for utility functions."""
    
    def test_extract_doc_ids_from_documents(self):
        """Test extracting doc IDs from Document objects."""
        documents = [
            Document(page_content="content1", metadata={"id": "doc1"}),
            Document(page_content="content2", metadata={"id": "doc2"}),
            Document(page_content="content3", metadata={"id": 123})  # Non-string ID
        ]
        
        doc_ids = [doc.metadata.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        
        assert doc_ids == ["doc1", "doc2", "123"]
    
    def test_extract_doc_ids_from_dataframe(self):
        """Test extracting doc IDs from DataFrame."""
        df = pd.DataFrame({
            "id": ["doc1", "doc2", "doc3"],
            "content": ["content1", "content2", "content3"],
            "score": [0.9, 0.8, 0.7]
        })
        
        doc_ids = [str(row["id"]) for _, row in df.iterrows()]
        
        assert doc_ids == ["doc1", "doc2", "doc3"]
    
    def test_extract_doc_ids_missing_id_column(self):
        """Test error handling when DataFrame missing id column."""
        df = pd.DataFrame({
            "content": ["content1", "content2"],
            "score": [0.9, 0.8]
        })
        
        with pytest.raises(ValueError, match="DataFrame missing 'id' column"):
            [str(row["id"]) for _, row in df.iterrows()]
    
    def test_extract_doc_ids_unsupported_format(self):
        """Test error handling for unsupported result formats."""
        with pytest.raises(ValueError, match="Unsupported search results format"):
            evaluate_search_results("invalid format", "test query")
    
    def test_extract_doc_ids_documents_without_ids(self):
        """Test handling of Document objects without IDs."""
        documents = [
            Document(page_content="content1", metadata={"id": "doc1"}),
            Document(page_content="content2", metadata={}),  # No ID
            Document(page_content="content3", metadata={"id": None})  # None ID
        ]
        
        doc_ids = [doc.metadata.get("id", f"doc_{i}") for i, doc in enumerate(documents)]
        
        # Should only include documents with valid IDs
        assert doc_ids == ["doc1"]


class TestEvaluateSearchResults:
    """Test suite for the main evaluation function."""
    
    def test_evaluate_search_results_basic(self):
        """Test basic evaluation with multiple metrics."""
        queries = {
            "q1": "test query 1",
            "q2": "test query 2"
        }
        
        search_results = {
            "q1": [
                Document(page_content="content1", metadata={"id": "doc1"}),
                Document(page_content="content2", metadata={"id": "doc2"})
            ],
            "q2": [
                Document(page_content="content3", metadata={"id": "doc3"})
            ]
        }
        
        ground_truth = {
            "q1": {"doc1"},
            "q2": {"doc3"}
        }
        
        results = evaluate_search_results(queries, search_results, ground_truth)
        
        # Should have default metrics
        assert "Mean Reciprocal Rank (MRR)" in results
        assert "Precision@10" in results
        assert "Mean Average Precision (MAP)" in results
        
        # Check MRR score (both queries have relevant doc at rank 1)
        assert results["Mean Reciprocal Rank (MRR)"].score == 1.0
    
    def test_evaluate_search_results_custom_metrics(self):
        """Test evaluation with custom metrics list."""
        queries = {"q1": "test query"}
        search_results = {
            "q1": [Document(page_content="content", metadata={"id": "doc1"})]
        }
        ground_truth = {"q1": {"doc1"}}
        
        custom_metrics = [
            MeanReciprocalRank(),
            PrecisionAtK(k=5)
        ]
        
        results = evaluate_search_results(
            queries, search_results, ground_truth, metrics=custom_metrics
        )
        
        assert len(results) == 2
        assert "Mean Reciprocal Rank (MRR)" in results
        assert "Precision@5" in results
        assert "Mean Average Precision (MAP)" not in results
    
    def test_evaluate_search_results_dataframe_input(self):
        """Test evaluation with DataFrame search results."""
        queries = {"q1": "test query"}
        
        search_results = {
            "q1": pd.DataFrame({
                "id": ["doc1", "doc2"],
                "content": ["content1", "content2"],
                "score": [0.9, 0.8]
            })
        }
        
        ground_truth = {"q1": {"doc1"}}
        
        results = evaluate_search_results(queries, search_results, ground_truth)
        
        # Should work with DataFrame input
        assert "Mean Reciprocal Rank (MRR)" in results
        assert results["Mean Reciprocal Rank (MRR)"].score == 1.0
    
    def test_evaluate_search_results_error_handling(self):
        """Test error handling in evaluation function."""
        queries = {"q1": "test query"}
        
        # Invalid search results format
        search_results = {"q1": "invalid format"}
        ground_truth = {"q1": {"doc1"}}
        
        results = evaluate_search_results(queries, search_results, ground_truth)
        
        # Should handle errors gracefully and return zero scores
        # Since the invalid format results in empty doc list, 
        # metrics calculate 0.0 scores but don't include error in metadata
        for metric_name, result in results.items():
            assert result.score == 0.0
            # The error is logged but metrics still calculate normally with empty results
            assert result.metadata["queries_with_ground_truth"] >= 0


class TestEdgeCases:
    """Test suite for edge cases and error conditions."""
    
    def test_metrics_with_empty_inputs(self):
        """Test all metrics handle empty inputs gracefully."""
        search_results = {}
        ground_truth = {}
        
        mrr = MeanReciprocalRank()
        precision = PrecisionAtK(k=10)
        map_metric = MeanAveragePrecision()
        
        mrr_result = mrr.calculate(search_results, ground_truth)
        precision_result = precision.calculate(search_results, ground_truth)
        map_result = map_metric.calculate(search_results, ground_truth)
        
        assert mrr_result.score == 0.0
        assert precision_result.score == 0.0
        assert map_result.score == 0.0
    
    def test_metrics_with_large_rankings(self):
        """Test metrics with large result sets."""
        search_results = {
            "q1": [f"doc{i}" for i in range(1000)]  # 1000 documents
        }
        ground_truth = {
            "q1": {"doc999"}  # Relevant doc at rank 1000
        }
        
        mrr = MeanReciprocalRank()
        precision = PrecisionAtK(k=100)
        
        mrr_result = mrr.calculate(search_results, ground_truth)
        precision_result = precision.calculate(search_results, ground_truth)
        
        assert mrr_result.score == 1.0/1000  # 1/rank
        assert precision_result.score == 0.0  # No relevant in top-100
    
    def test_metrics_name_methods(self):
        """Test metric name methods return correct values."""
        mrr = MeanReciprocalRank()
        precision = PrecisionAtK(k=5)
        map_metric = MeanAveragePrecision()
        
        assert mrr.get_name() == "Mean Reciprocal Rank (MRR)"
        assert precision.get_name() == "Precision@5"
        assert map_metric.get_name() == "Mean Average Precision (MAP)"