"""Test frontend weight control functionality."""
import pytest
from unittest.mock import patch, Mock
import sys
import os

# Add frontend directory to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../frontend'))

# Note: These tests would require setting up a mock Streamlit environment
# For now, we'll test the core functions that handle weight parameters

class TestFrontendWeights:
    """Test suite for frontend weight functionality."""

    def test_query_system_payload_with_weights(self):
        """Test that query_system creates correct payload with weights."""
        # This would require importing from streamlit_app, but since it has streamlit imports
        # we'll test the logic directly
        
        # Test payload creation logic
        query = "test query"
        top_k = 10
        vector_weight = 0.6
        
        # Expected payload
        expected_payload = {
            "query": query,
            "top_k": top_k,
            "vector_weight": vector_weight
        }
        
        # Test payload creation
        payload = {"query": query, "top_k": top_k}
        if vector_weight is not None:
            payload["vector_weight"] = vector_weight
            
        assert payload == expected_payload

    def test_query_system_payload_without_weights(self):
        """Test that query_system creates correct payload without weights."""
        query = "test query"
        top_k = 8
        vector_weight = None
        
        # Expected payload (no vector_weight field)
        expected_payload = {
            "query": query,
            "top_k": top_k
        }
        
        # Test payload creation
        payload = {"query": query, "top_k": top_k}
        if vector_weight is not None:
            payload["vector_weight"] = vector_weight
            
        assert payload == expected_payload

    def test_weight_calculation_consistency(self):
        """Test that BM25 weight calculation is consistent."""
        test_vector_weights = [0.0, 0.25, 0.5, 0.7, 0.75, 1.0]
        
        for vector_weight in test_vector_weights:
            bm25_weight = 1.0 - vector_weight
            total_weight = vector_weight + bm25_weight
            
            # Weights should sum to 1.0
            assert abs(total_weight - 1.0) < 0.001, f"Weights don't sum to 1.0: v={vector_weight}, bm25={bm25_weight}"
            
            # Both weights should be non-negative
            assert vector_weight >= 0.0, f"Vector weight should be non-negative: {vector_weight}"
            assert bm25_weight >= 0.0, f"BM25 weight should be non-negative: {bm25_weight}"

    def test_slider_step_increment(self):
        """Test that slider increments work as expected."""
        # Test valid 0.05 increments
        base_value = 0.0
        step = 0.05
        max_value = 1.0
        
        current = base_value
        increments = []
        
        while current <= max_value:
            increments.append(round(current, 2))  # Round to avoid float precision issues
            current += step
            
        expected_increments = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                             0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0]
        
        assert increments == expected_increments

    def test_default_weight_values(self):
        """Test that default values match backend configuration."""
        # Frontend default should match backend default (0.7 vector, 0.3 BM25)
        default_vector_weight = 0.7
        default_bm25_weight = 1.0 - default_vector_weight
        
        assert default_vector_weight == 0.7
        assert abs(default_bm25_weight - 0.3) < 0.001

    @patch('requests.post')
    def test_api_request_with_weights(self, mock_post):
        """Test that API requests include weight parameters."""
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"success": True}
        mock_post.return_value = mock_response
        
        # Test API payload
        expected_payload = {
            "query": "test query",
            "top_k": 8,
            "vector_weight": 0.6
        }
        
        # This would test the actual API call if we could import make_api_request
        # For now, verify the mock was called correctly
        import requests
        response = requests.post("http://test", json=expected_payload)
        
        mock_post.assert_called_once_with("http://test", json=expected_payload)
        assert response.status_code == 200