#!/usr/bin/env python3
"""
Demo script to showcase RRF-specific logging and monitoring capabilities.
This script demonstrates the enhanced logging features for Reciprocal Rank Fusion.
"""

import logging
import sys
import os
import json
from typing import Dict, Any

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

def setup_logging():
    """Setup structured logging to demonstrate RRF monitoring capabilities."""
    
    class StructuredFormatter(logging.Formatter):
        """Custom formatter to display structured logging data."""
        
        def format(self, record):
            # Base format
            base_msg = super().format(record)
            
            # Add structured data if present
            if hasattr(record, 'extra') and record.extra:
                structured_data = json.dumps(record.extra, indent=2)
                return f"{base_msg}\n  Structured Data: {structured_data}"
            
            return base_msg
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Add structured formatter to the handler
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = StructuredFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    
    # Remove default handler and add our custom one
    logger.handlers.clear()
    logger.addHandler(handler)
    
    return logger

def demo_rrf_logging_features():
    """Demonstrate the RRF logging and monitoring features."""
    
    logger = logging.getLogger("rrf_demo")
    
    print("🔍 RRF Logging & Monitoring Demo")
    print("=" * 50)
    
    # Simulate RRF search initiation
    logger.info("RRF hybrid search initiated", extra={
        "query_length": 45,
        "query_preview": "What are the benefits of bedtime routines for children?",
        "top_k": 10,
        "rrf_k_value": 60,
        "search_type": "hybrid_rrf",
        "fusion_method": "reciprocal_rank_fusion"
    })
    
    # Simulate component search performance
    logger.debug("Component engine performance", extra={
        "vector_search_time_ms": 145,
        "keyword_search_time_ms": 67,
        "vector_results": 12,
        "keyword_results": 8,
        "query_preview": "What are the benefits of bedtime routines..."
    })
    
    # Simulate RRF scoring details
    logger.debug("RRF scoring #1", extra={
        "raw_rrf_score": 0.033056,
        "final_rrf_score": 0.033056,
        "bm25_rank": 1,
        "vector_rank": 3,
        "quality_penalty": 1.0,
        "found_by_engines": ["bm25", "vector"],
        "rrf_k_value": 60,
        "content_preview": "Bedtime routines provide structure and predictability for children..."
    })
    
    # Simulate RRF fusion completion
    logger.debug("RRF fusion completed", extra={
        "rrf_processing_time_ms": 23,
        "total_unique_documents": 15,
        "both_engines_found": 5,
        "bm25_only_found": 3,
        "vector_only_found": 7,
        "agreement_percentage": 33.33,
        "avg_rrf_score": 0.018764,
        "top_rrf_score": 0.033056,
        "rrf_k_used": 60
    })
    
    # Simulate final search completion
    logger.info("RRF hybrid search completed", extra={
        "total_processing_time_ms": 267,
        "fusion_time_ms": 23,
        "vector_search_time_ms": 145,
        "keyword_search_time_ms": 67,
        "input_bm25_results": 8,
        "input_vector_results": 12,
        "unique_results_after_fusion": 15,
        "final_results_count": 10,
        "bm25_only_count": 3,
        "vector_only_count": 7,
        "both_engines_count": 5,
        "engine_agreement_rate": 50.0,
        "avg_rrf_score": 0.018764,
        "top_rrf_score": 0.033056,
        "rrf_k_value": 60,
        "fusion_method": "reciprocal_rank_fusion",
        "deduplication_applied": True
    })
    
    print("\n📊 Sample Document Metadata Structure:")
    print("-" * 40)
    
    sample_metadata = {
        "rrf_score": 0.033056,
        "bm25_rank": 1,
        "vector_rank": 3,
        "found_by_engines": ["bm25", "vector"],
        "rrf_k_value": 60,
        "fusion_method": "reciprocal_rank_fusion",
        "vector_distance": 0.234,
        "content_quality_penalty": 1.0
    }
    
    print(json.dumps(sample_metadata, indent=2))
    
    print("\n🎯 Key RRF Monitoring Metrics:")
    print("-" * 40)
    print("• RRF Score Distribution (high/medium/low buckets)")
    print("• Engine Agreement Rate (documents found by both engines)")
    print("• RRF K Parameter Impact on scoring")
    print("• Component timing breakdown (vector vs keyword search)")
    print("• Fusion processing time")
    print("• Rank position correlation between engines")
    
    print("\n✅ RRF Logging Enhancement Complete!")
    print("All structured logging is ready for production monitoring.")

if __name__ == "__main__":
    setup_logging()
    demo_rrf_logging_features()