#!/usr/bin/env python3
"""
Simple test to verify the evaluation fix by directly testing the metrics calculation.
"""
import pandas as pd
from core.evaluation.metrics import create_synthetic_relevance_judgments, evaluate_search_results
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_dataframe(query_theme: str, num_docs: int = 5):
    """Create test dataframe with different score patterns based on query theme."""
    
    # Different score patterns for different themes
    if "machine learning" in query_theme.lower():
        scores = [0.85, 0.72, 0.58, 0.34, 0.21]  # High-to-low pattern
    elif "solar energy" in query_theme.lower():
        scores = [0.91, 0.45, 0.38, 0.29, 0.15]  # High first, then lower
    elif "exercise" in query_theme.lower():
        scores = [0.67, 0.62, 0.51, 0.44, 0.39]  # More uniform scores
    elif "quantum" in query_theme.lower():
        scores = [0.95, 0.88, 0.23, 0.18, 0.12]  # Strong top 2, weak rest
    else:  # climate change
        scores = [0.79, 0.76, 0.71, 0.65, 0.31]  # Gradual decline
    
    data = []
    for i, score in enumerate(scores):
        metadata = {
            "id": f"{query_theme.replace(' ', '_').lower()}_doc_{i+1}",
            "hybrid_score": score,
            "vector_score": score * 0.6,
            "bm25_score": score * 0.4,
            "source_engine": "hybrid"
        }
        
        data.append({
            "id": metadata["id"],
            "content": f"Document {i+1} about {query_theme} with score {score}",
            "metadata": metadata
        })
    
    return pd.DataFrame(data)

def test_evaluation_fix():
    """Test that different queries produce different evaluation metrics."""
    
    test_cases = [
        ("What is machine learning?", "machine learning"),
        ("How does solar energy work?", "solar energy"),  
        ("What are the benefits of exercise?", "exercise"),
        ("Explain quantum computing", "quantum computing"),
        ("What is climate change?", "climate change")
    ]
    
    results = {}
    
    logger.info("🔍 Testing evaluation metrics with different score patterns...")
    
    for query, theme in test_cases:
        logger.info(f"\n📊 Testing query: {query}")
        
        # Create test data with theme-specific score patterns
        ctx_df = create_test_dataframe(theme)
        
        # Show the score pattern
        logger.info("   Score pattern:")
        for i, row in ctx_df.iterrows():
            metadata = row["metadata"]
            logger.info(f"     Doc {i+1}: {metadata['hybrid_score']}")
        
        # Create synthetic relevance judgments using the fixed algorithm
        relevant_docs = create_synthetic_relevance_judgments(ctx_df)
        logger.info(f"   Relevant docs: {len(relevant_docs)} out of {len(ctx_df)}")
        logger.info(f"   Relevant IDs: {list(relevant_docs)}")
        
        # Compute evaluation metrics
        eval_results = evaluate_search_results(
            ctx_df=ctx_df,
            query=query,
            relevance_judgments=relevant_docs
        )
        
        # Extract key metrics
        key_metrics = {
            "mrr": eval_results.get("mrr").value if eval_results.get("mrr") else None,
            "map": eval_results.get("map").value if eval_results.get("map") else None,
            "precision_at_3": eval_results.get("precision_at_3").value if eval_results.get("precision_at_3") else None,
            "precision_at_5": eval_results.get("precision_at_5").value if eval_results.get("precision_at_5") else None,
            "recall_at_3": eval_results.get("recall_at_3").value if eval_results.get("recall_at_3") else None,
            "ndcg_at_5": eval_results.get("ndcg_at_5").value if eval_results.get("ndcg_at_5") else None,
        }
        
        results[query] = key_metrics
        
        logger.info(f"   📈 Metrics:")
        for metric, value in key_metrics.items():
            logger.info(f"     {metric}: {value}")
    
    # Analyze results for diversity
    logger.info(f"\n{'='*60}")
    logger.info("🔍 ANALYSIS: Checking metric diversity")
    logger.info(f"{'='*60}")
    
    # Check if all metrics are identical
    first_result = list(results.values())[0]
    all_identical = True
    
    for query, metrics in list(results.items())[1:]:
        for metric_name in first_result.keys():
            if first_result[metric_name] != metrics[metric_name]:
                all_identical = False
                break
    
    logger.info(f"\n📊 Comparison Summary:")
    for i, (query, metrics) in enumerate(results.items(), 1):
        logger.info(f"  Query {i}: {query[:30]}...")
        logger.info(f"    MRR: {metrics['mrr']:.3f}, MAP: {metrics['map']:.3f}, P@3: {metrics['precision_at_3']:.3f}, P@5: {metrics['precision_at_5']:.3f}")
    
    logger.info(f"\n{'='*60}")
    logger.info("🏁 FINAL VERDICT")
    logger.info(f"{'='*60}")
    
    if all_identical:
        logger.error("❌ ISSUE: All metrics are still identical!")
        logger.error("   The fix may not have addressed the root cause.")
        return False
    else:
        logger.info("✅ SUCCESS: Metrics vary appropriately between queries!")
        logger.info("   The evaluation fix is working correctly.")
        
        # Show some examples of differences
        different_metrics = []
        for metric_name in first_result.keys():
            values = [metrics[metric_name] for metrics in results.values()]
            if len(set(values)) > 1:  # More than 1 unique value
                different_metrics.append(f"{metric_name}: {values}")
        
        logger.info("   📈 Metrics showing variation:")
        for diff in different_metrics[:3]:  # Show first 3 examples
            logger.info(f"     {diff}")
        
        return True

if __name__ == "__main__":
    success = test_evaluation_fix()
    if success:
        print("\n✅ Evaluation fix verification PASSED")
        exit(0)
    else:
        print("\n❌ Evaluation fix verification FAILED") 
        exit(1)