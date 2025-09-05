#!/usr/bin/env python3
"""
Realistic diagnostic script to test evaluation caching with actual RRF scores.
This simulates the real-world scenario where different queries might return
identical evaluation metrics due to caching or evaluation logic issues.
"""
import pandas as pd
import json
from typing import Dict, Any, Set
from core.services.evaluation_service import EvaluationService
from core.evaluation.metrics import evaluate_search_results, create_synthetic_relevance_judgments


def create_realistic_search_results():
    """Create realistic search results that might cause evaluation issues."""
    
    # Scenario 1: Two different queries that return the same top documents in the same order
    # This might happen if both queries are semantically similar or if the search system
    # has limited diversity in results
    
    query1_results = pd.DataFrame({
        'id': ['paper_001', 'paper_002', 'paper_003', 'paper_004', 'paper_005'],
        'content': [
            'Machine learning algorithms for natural language processing applications in modern AI systems...',
            'Deep neural networks and their applications in computer vision and image recognition tasks...',
            'Reinforcement learning techniques for autonomous agent development and decision making...',
            'Natural language processing using transformer models and attention mechanisms for text...',
            'Computer vision applications in medical imaging and diagnostic systems using deep learning...'
        ],
        'metadata': [
            {
                'id': 'paper_001',
                'hybrid_score': 0.92, 
                'rrf_score': 0.92,
                'bm25_score': 0.85, 
                'vector_score': 0.89,
                'bm25_rank': 1,
                'vector_rank': 2
            },
            {
                'id': 'paper_002', 
                'hybrid_score': 0.87,
                'rrf_score': 0.87, 
                'bm25_score': 0.82, 
                'vector_score': 0.81,
                'bm25_rank': 2,
                'vector_rank': 3
            },
            {
                'id': 'paper_003',
                'hybrid_score': 0.81,
                'rrf_score': 0.81,
                'bm25_score': 0.79, 
                'vector_score': 0.76,
                'bm25_rank': 3,
                'vector_rank': 4
            },
            {
                'id': 'paper_004',
                'hybrid_score': 0.76,
                'rrf_score': 0.76,
                'bm25_score': 0.74, 
                'vector_score': 0.72,
                'bm25_rank': 4,
                'vector_rank': 5
            },
            {
                'id': 'paper_005',
                'hybrid_score': 0.71,
                'rrf_score': 0.71,
                'bm25_score': 0.68, 
                'vector_score': 0.67,
                'bm25_rank': 5,
                'vector_rank': 6
            }
        ]
    })
    
    query2_results = pd.DataFrame({
        'id': ['paper_001', 'paper_002', 'paper_003', 'paper_004', 'paper_005'],
        'content': [
            'Machine learning algorithms for natural language processing applications in modern AI systems...',
            'Deep neural networks and their applications in computer vision and image recognition tasks...',
            'Reinforcement learning techniques for autonomous agent development and decision making...',
            'Natural language processing using transformer models and attention mechanisms for text...',
            'Computer vision applications in medical imaging and diagnostic systems using deep learning...'
        ],
        'metadata': [
            {
                'id': 'paper_001',
                'hybrid_score': 0.92,  # Same scores as query1!
                'rrf_score': 0.92,
                'bm25_score': 0.85, 
                'vector_score': 0.89,
                'bm25_rank': 1,
                'vector_rank': 2
            },
            {
                'id': 'paper_002', 
                'hybrid_score': 0.87,  # Same scores as query1!
                'rrf_score': 0.87,
                'bm25_score': 0.82, 
                'vector_score': 0.81,
                'bm25_rank': 2,
                'vector_rank': 3
            },
            {
                'id': 'paper_003',
                'hybrid_score': 0.81,  # Same scores as query1!
                'rrf_score': 0.81,
                'bm25_score': 0.79, 
                'vector_score': 0.76,
                'bm25_rank': 3,
                'vector_rank': 4
            },
            {
                'id': 'paper_004',
                'hybrid_score': 0.76,  # Same scores as query1!
                'rrf_score': 0.76,
                'bm25_score': 0.74, 
                'vector_score': 0.72,
                'bm25_rank': 4,
                'vector_rank': 5
            },
            {
                'id': 'paper_005',
                'hybrid_score': 0.71,  # Same scores as query1!
                'rrf_score': 0.71,
                'bm25_score': 0.68, 
                'vector_score': 0.67,
                'bm25_rank': 5,
                'vector_rank': 6
            }
        ]
    })
    
    # Scenario 2: Different queries with genuinely different results
    query3_results = pd.DataFrame({
        'id': ['paper_010', 'paper_011', 'paper_012', 'paper_013', 'paper_014'],
        'content': [
            'Database optimization techniques for large-scale distributed systems and performance tuning...',
            'Web development frameworks and their impact on modern application architectures...',
            'Cybersecurity protocols for protecting sensitive data in cloud computing environments...',
            'Software engineering best practices for agile development teams and project management...',
            'Mobile application development using cross-platform frameworks and native technologies...'
        ],
        'metadata': [
            {
                'id': 'paper_010',
                'hybrid_score': 0.95,
                'rrf_score': 0.95,
                'bm25_score': 0.90, 
                'vector_score': 0.92,
                'bm25_rank': 1,
                'vector_rank': 1
            },
            {
                'id': 'paper_011', 
                'hybrid_score': 0.88,
                'rrf_score': 0.88,
                'bm25_score': 0.85, 
                'vector_score': 0.84,
                'bm25_rank': 2,
                'vector_rank': 2
            },
            {
                'id': 'paper_012',
                'hybrid_score': 0.82,
                'rrf_score': 0.82,
                'bm25_score': 0.78, 
                'vector_score': 0.79,
                'bm25_rank': 3,
                'vector_rank': 3
            },
            {
                'id': 'paper_013',
                'hybrid_score': 0.77,
                'rrf_score': 0.77,
                'bm25_score': 0.73, 
                'vector_score': 0.74,
                'bm25_rank': 4,
                'vector_rank': 4
            },
            {
                'id': 'paper_014',
                'hybrid_score': 0.69,
                'rrf_score': 0.69,
                'bm25_score': 0.66, 
                'vector_score': 0.65,
                'bm25_rank': 5,
                'vector_rank': 5
            }
        ]
    })
    
    return {
        'query1': query1_results,
        'query2': query2_results,  # Same results as query1 - this should cause issues!
        'query3': query3_results   # Different results
    }


def test_realistic_evaluation_caching():
    """Test evaluation with realistic data that might cause cache issues."""
    print("🎯 REALISTIC EVALUATION CACHING TEST")
    print("=" * 60)
    
    # Create evaluation service
    eval_service = EvaluationService(cache_service=None)
    
    # Test queries
    queries = {
        'query1': "What are the latest advances in machine learning and AI?",
        'query2': "How does artificial intelligence work in modern applications?",
        'query3': "Best practices for database optimization and performance"
    }
    
    # Get realistic search results
    search_results = create_realistic_search_results()
    
    print("Testing evaluation with realistic search results:")
    print()
    
    evaluation_results = {}
    
    for query_key, query_text in queries.items():
        print(f"📝 Query {query_key}: {query_text}")
        ctx_df = search_results[query_key]
        
        # Show context info
        ctx_hash = eval_service._hash_context_df(ctx_df)
        cache_key = eval_service._get_cache_key(query_text, ctx_hash)
        
        print(f"   Context: {len(ctx_df)} documents")
        print(f"   Context hash: {ctx_hash}")
        print(f"   Cache key: {cache_key[:16]}...")
        
        # Show document IDs and scores
        doc_info = []
        for _, row in ctx_df.iterrows():
            metadata = row.get('metadata', {})
            doc_id = row.get('id') or metadata.get('id', 'unknown')
            score = metadata.get('hybrid_score', metadata.get('rrf_score', 0.0))
            doc_info.append(f"{doc_id}:{score}")
        
        print(f"   Doc signatures: {', '.join(doc_info)}")
        
        # Evaluate
        try:
            metrics = eval_service.evaluate_query_results(
                query=query_text,
                ctx_df=ctx_df,
                use_cache=False  # Disable cache to see actual evaluation behavior
            )
            
            if metrics:
                mrr_value = metrics.mrr.value if metrics.mrr else "N/A"
                map_value = metrics.map_score.value if metrics.map_score else "N/A"
                p5_value = metrics.precision_at_k[5].value if metrics.precision_at_k and 5 in metrics.precision_at_k else "N/A"
                
                print(f"   ✅ Evaluation results: MRR={mrr_value}, MAP={map_value}, P@5={p5_value}")
                
                evaluation_results[query_key] = {
                    'query': query_text,
                    'ctx_hash': ctx_hash,
                    'cache_key': cache_key,
                    'mrr': mrr_value,
                    'map': map_value,
                    'p5': p5_value
                }
            else:
                print(f"   ❌ No evaluation results returned")
                evaluation_results[query_key] = None
                
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            evaluation_results[query_key] = None
        
        print()
    
    return evaluation_results


def analyze_evaluation_results(results: Dict[str, Any]):
    """Analyze evaluation results for suspicious patterns."""
    print("🔍 EVALUATION RESULTS ANALYSIS")
    print("=" * 60)
    
    # Group by metrics values
    metric_groups = {}
    
    for query_key, result in results.items():
        if result is None:
            continue
            
        metrics_signature = f"MRR:{result['mrr']}_MAP:{result['map']}_P5:{result['p5']}"
        
        if metrics_signature not in metric_groups:
            metric_groups[metrics_signature] = []
        metric_groups[metrics_signature].append((query_key, result))
    
    # Find suspicious identical metrics
    identical_metrics = {k: v for k, v in metric_groups.items() if len(v) > 1}
    
    if identical_metrics:
        print("🚨 SUSPICIOUS: Found queries with identical evaluation metrics!")
        for metrics_sig, items in identical_metrics.items():
            print(f"\n❌ Identical metrics pattern: {metrics_sig}")
            for query_key, result in items:
                query_short = result['query'][:40] + "..." if len(result['query']) > 40 else result['query']
                print(f"   - {query_key}: '{query_short}'")
                print(f"     Context hash: {result['ctx_hash']}")
                
        print("\n💡 This suggests either:")
        print("   1. The search system returns identical results for different queries")
        print("   2. The evaluation logic is flawed")
        print("   3. There's a caching issue")
        
    else:
        print("✅ No suspicious identical evaluation metrics found")
    
    # Check context hash collisions
    ctx_hash_groups = {}
    for query_key, result in results.items():
        if result is None:
            continue
        ctx_hash = result['ctx_hash']
        if ctx_hash not in ctx_hash_groups:
            ctx_hash_groups[ctx_hash] = []
        ctx_hash_groups[ctx_hash].append((query_key, result))
    
    print(f"\n📊 Context Hash Analysis:")
    for ctx_hash, items in ctx_hash_groups.items():
        print(f"   Hash {ctx_hash}: {len(items)} queries")
        if len(items) > 1:
            print(f"      ⚠️  Multiple queries have identical context:")
            for query_key, result in items:
                query_short = result['query'][:35] + "..." if len(result['query']) > 35 else result['query']
                print(f"         - {query_key}: '{query_short}'")


def test_synthetic_relevance_judgments():
    """Test synthetic relevance judgment creation logic."""
    print("\n🔬 SYNTHETIC RELEVANCE JUDGMENTS TEST")
    print("=" * 60)
    
    search_results = create_realistic_search_results()
    
    for query_key, ctx_df in search_results.items():
        print(f"\n📋 Query {query_key}:")
        
        # Test synthetic relevance judgments with different thresholds
        for threshold in [0.3, 0.5, 0.7, 0.8]:
            relevant_docs = create_synthetic_relevance_judgments(ctx_df, threshold)
            print(f"   Threshold {threshold}: {len(relevant_docs)} relevant docs - {list(relevant_docs) if relevant_docs else 'None'}")
            
            # If using this threshold, what would the metrics be?
            if relevant_docs:
                doc_ids = [str(row.get('id') or row.get('metadata', {}).get('id', f'doc_{i}')) 
                          for i, (_, row) in enumerate(ctx_df.iterrows())]
                
                results = evaluate_search_results(
                    ctx_df=ctx_df,
                    query=f"test_query_{query_key}",
                    relevance_judgments=relevant_docs,
                    k_values=[1, 5]
                )
                
                if results:
                    mrr = results.get('mrr', {}).value if 'mrr' in results else 0.0
                    map_val = results.get('map', {}).value if 'map' in results else 0.0
                    print(f"      → MRR: {mrr:.3f}, MAP: {map_val:.3f}")


def main():
    """Run realistic evaluation diagnostics."""
    print("🔧 REALISTIC EVALUATION CACHE DIAGNOSTICS")
    print("=" * 70)
    print("Testing with realistic search results to identify evaluation issues.")
    print("=" * 70)
    
    # Test 1: Realistic evaluation with potential cache issues
    results = test_realistic_evaluation_caching()
    
    # Test 2: Analyze results for patterns
    analyze_evaluation_results(results)
    
    # Test 3: Test synthetic relevance judgment logic
    test_synthetic_relevance_judgments()
    
    print("\n💡 INVESTIGATION SUMMARY")
    print("=" * 60)
    print("If different queries show identical evaluation metrics:")
    print("1. Check if search system returns same documents/scores")
    print("2. Verify synthetic relevance judgment logic")  
    print("3. Look for context hash collisions")
    print("4. Check evaluation metric calculation logic")
    print("5. Clear evaluation cache and test fresh")


if __name__ == "__main__":
    main()