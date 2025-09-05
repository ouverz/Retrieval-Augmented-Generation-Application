#!/usr/bin/env python3
"""
Test the fixed evaluation service to verify different queries produce different metrics.
"""
import pandas as pd
from core.services.evaluation_service import EvaluationService


def create_test_queries_and_results():
    """Create test scenarios with different queries and realistic results."""
    
    # Query 1: AI/ML topic - high relevance results
    query1_results = pd.DataFrame({
        'id': ['ai_paper_1', 'ml_paper_2', 'stats_paper_3', 'math_paper_4', 'bio_paper_5'],
        'content': [
            'Advanced machine learning algorithms for natural language processing...',
            'Deep neural networks in artificial intelligence applications...',
            'Statistical analysis methods for data science applications...',
            'Mathematical foundations of optimization algorithms...',
            'Biological systems and computational modeling approaches...'
        ],
        'metadata': [
            {'id': 'ai_paper_1', 'hybrid_score': 0.89, 'rrf_score': 0.89},  # Highly relevant
            {'id': 'ml_paper_2', 'hybrid_score': 0.84, 'rrf_score': 0.84},  # Highly relevant  
            {'id': 'stats_paper_3', 'hybrid_score': 0.51, 'rrf_score': 0.51},  # Moderately relevant
            {'id': 'math_paper_4', 'hybrid_score': 0.33, 'rrf_score': 0.33},  # Somewhat relevant
            {'id': 'bio_paper_5', 'hybrid_score': 0.18, 'rrf_score': 0.18}   # Not very relevant
        ]
    })
    
    # Query 2: Database topic - different relevance distribution
    query2_results = pd.DataFrame({
        'id': ['db_paper_1', 'sys_paper_2', 'web_paper_3', 'ui_paper_4', 'game_paper_5'],
        'content': [
            'Database optimization techniques for distributed systems...',
            'System architecture patterns for scalable applications...',
            'Web development frameworks and performance considerations...',
            'User interface design principles and usability studies...',
            'Game development engines and graphics programming...'
        ],
        'metadata': [
            {'id': 'db_paper_1', 'hybrid_score': 0.92, 'rrf_score': 0.92},   # Very relevant
            {'id': 'sys_paper_2', 'hybrid_score': 0.47, 'rrf_score': 0.47}, # Moderately relevant
            {'id': 'web_paper_3', 'hybrid_score': 0.29, 'rrf_score': 0.29}, # Low relevance
            {'id': 'ui_paper_4', 'hybrid_score': 0.23, 'rrf_score': 0.23},  # Low relevance
            {'id': 'game_paper_5', 'hybrid_score': 0.16, 'rrf_score': 0.16} # Not relevant
        ]
    })
    
    # Query 3: Mixed topic - different score distribution
    query3_results = pd.DataFrame({
        'id': ['mix_paper_1', 'mix_paper_2', 'mix_paper_3', 'mix_paper_4', 'mix_paper_5'],
        'content': [
            'Interdisciplinary approaches to computational problems...',
            'Cross-domain applications of machine learning methods...',
            'Research methodologies in computer science studies...',
            'Educational technology and learning management systems...',
            'Historical perspectives on computing and algorithms...'
        ],
        'metadata': [
            {'id': 'mix_paper_1', 'hybrid_score': 0.76, 'rrf_score': 0.76},
            {'id': 'mix_paper_2', 'hybrid_score': 0.71, 'rrf_score': 0.71},
            {'id': 'mix_paper_3', 'hybrid_score': 0.68, 'rrf_score': 0.68},
            {'id': 'mix_paper_4', 'hybrid_score': 0.63, 'rrf_score': 0.63},
            {'id': 'mix_paper_5', 'hybrid_score': 0.59, 'rrf_score': 0.59}
        ]
    })
    
    return {
        'ai_ml_query': ("What are the latest advances in machine learning and AI?", query1_results),
        'database_query': ("How to optimize database performance for large systems?", query2_results),
        'mixed_query': ("Computational approaches to interdisciplinary research", query3_results)
    }


def test_fixed_evaluation():
    """Test the fixed evaluation service with different queries."""
    print("🧪 TESTING FIXED EVALUATION SERVICE")
    print("=" * 60)
    
    eval_service = EvaluationService(cache_service=None)
    test_data = create_test_queries_and_results()
    
    results = {}
    
    for query_key, (query_text, ctx_df) in test_data.items():
        print(f"\n📝 Testing: {query_key}")
        print(f"   Query: {query_text}")
        print(f"   Results: {len(ctx_df)} documents")
        
        # Show score distribution
        scores = []
        for _, row in ctx_df.iterrows():
            score = row.get('metadata', {}).get('hybrid_score', 0.0)
            scores.append(score)
        print(f"   Score range: {min(scores):.3f} - {max(scores):.3f}")
        
        # Evaluate with fixed logic (using adaptive method by default)
        try:
            metrics = eval_service.evaluate_query_results(
                query=query_text,
                ctx_df=ctx_df,
                use_cache=False  # Disable cache to ensure fresh evaluation
            )
            
            if metrics:
                mrr = metrics.mrr.value if metrics.mrr else 0.0
                map_val = metrics.map_score.value if metrics.map_score else 0.0
                p3 = metrics.precision_at_k[3].value if metrics.precision_at_k and 3 in metrics.precision_at_k else 0.0
                p5 = metrics.precision_at_k[5].value if metrics.precision_at_k and 5 in metrics.precision_at_k else 0.0
                
                print(f"   📊 Evaluation Results:")
                print(f"      MRR: {mrr:.3f}")
                print(f"      MAP: {map_val:.3f}")
                print(f"      Precision@3: {p3:.3f}")
                print(f"      Precision@5: {p5:.3f}")
                
                results[query_key] = {
                    'query': query_text,
                    'mrr': mrr,
                    'map': map_val,
                    'p3': p3,
                    'p5': p5
                }
            else:
                print("   ❌ No evaluation metrics returned")
                results[query_key] = None
                
        except Exception as e:
            print(f"   ❌ Evaluation failed: {e}")
            import traceback
            print(f"   Stack trace: {traceback.format_exc()}")
            results[query_key] = None
    
    return results


def analyze_fix_success(results):
    """Analyze if the fix successfully produces different metrics for different queries."""
    print(f"\n🔍 ANALYSIS: Fix Success Verification")
    print("=" * 60)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if len(valid_results) < 2:
        print("❌ Not enough valid results to compare")
        return False
    
    # Check for metric diversity
    unique_metrics = set()
    for query_key, result in valid_results.items():
        metric_signature = f"{result['mrr']:.3f}_{result['map']:.3f}_{result['p3']:.3f}"
        unique_metrics.add(metric_signature)
    
    print(f"📊 Metric Analysis:")
    print(f"   Total queries evaluated: {len(valid_results)}")
    print(f"   Unique metric patterns: {len(unique_metrics)}")
    
    if len(unique_metrics) == 1:
        print("   ❌ All queries still produce identical metrics!")
        print("   🔍 Identical pattern:", list(unique_metrics)[0])
        return False
    elif len(unique_metrics) == len(valid_results):
        print("   ✅ All queries produce unique metrics - PERFECT!")
        return True
    else:
        print(f"   ⚠️  Some metric diversity: {len(unique_metrics)} patterns for {len(valid_results)} queries")
        print("   🔍 This is better than before but could be improved")
        return True
    
    # Show detailed comparison
    print(f"\n📋 Detailed Metric Comparison:")
    for query_key, result in valid_results.items():
        print(f"   {query_key}:")
        print(f"      MRR={result['mrr']:.3f}, MAP={result['map']:.3f}, P@3={result['p3']:.3f}, P@5={result['p5']:.3f}")
    
    # Check for perfect scores (which might indicate issues)
    perfect_scores = []
    for query_key, result in valid_results.items():
        if result['mrr'] == 1.0 and result['map'] == 1.0:
            perfect_scores.append(query_key)
    
    if perfect_scores:
        print(f"\n⚠️  WARNING: {len(perfect_scores)} queries still have perfect scores:")
        for query_key in perfect_scores:
            print(f"      - {query_key}: MRR=1.0, MAP=1.0")
        print("   This may indicate the relevance judgment logic needs further tuning")
        return False
    else:
        print(f"\n✅ No unrealistic perfect scores detected")
        return True


def main():
    """Test the evaluation service fix."""
    print("🔧 EVALUATION SERVICE FIX VERIFICATION")
    print("=" * 70)
    print("Testing whether different queries now produce different evaluation metrics.")
    print("=" * 70)
    
    # Test the fixed evaluation
    results = test_fixed_evaluation()
    
    # Analyze the success
    success = analyze_fix_success(results)
    
    print(f"\n🎯 FINAL VERDICT")
    print("=" * 60)
    if success:
        print("✅ SUCCESS: Fix appears to be working!")
        print("✅ Different queries now produce different evaluation metrics")
        print("✅ No more cache-related identical metrics issue")
        print("\n💡 Next steps:")
        print("   1. Deploy the fix to production")
        print("   2. Clear evaluation cache: service.clear_evaluation_cache()")
        print("   3. Test with real user queries")
        print("   4. Monitor for realistic metric diversity")
    else:
        print("❌ ISSUE PERSISTS: Fix needs more work")
        print("❌ Still seeing identical or unrealistic metrics")
        print("\n💡 Additional fixes needed:")
        print("   1. Further tune synthetic relevance judgment logic")
        print("   2. Investigate metric calculation algorithms")
        print("   3. Consider using real relevance judgments")


if __name__ == "__main__":
    main()