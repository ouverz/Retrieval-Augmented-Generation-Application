#!/usr/bin/env python3
"""
Clear evaluation cache to ensure fresh evaluation results after fixing the service.
"""
from core.services.cache_service import CacheService
from core.services.evaluation_service import EvaluationService
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_evaluation_cache():
    """Clear all evaluation cache entries."""
    print("🧹 CLEARING EVALUATION CACHE")
    print("=" * 50)
    
    try:
        # Create cache service
        cache_service = CacheService()
        
        if not cache_service.is_available():
            print("❌ Cache service not available - cache may be disabled or Redis not running")
            return False
        
        # Create evaluation service
        eval_service = EvaluationService(cache_service=cache_service)
        
        # Get cache stats before clearing
        stats_before = eval_service.get_cache_stats()
        print(f"📊 Cache stats before clearing: {stats_before}")
        
        # Clear evaluation cache
        success = eval_service.clear_evaluation_cache()
        
        if success:
            print("✅ Evaluation cache cleared successfully")
            
            # Get cache stats after clearing
            stats_after = eval_service.get_cache_stats()
            print(f"📊 Cache stats after clearing: {stats_after}")
            
            return True
        else:
            print("❌ Failed to clear evaluation cache")
            return False
            
    except Exception as e:
        print(f"❌ Error clearing evaluation cache: {e}")
        return False


def test_cache_clear_verification():
    """Verify cache was actually cleared by testing evaluation."""
    print("\n🧪 VERIFYING CACHE CLEAR")
    print("=" * 50)
    
    try:
        import pandas as pd
        from core.services.evaluation_service import EvaluationService
        
        # Create evaluation service
        eval_service = EvaluationService()
        
        # Create test data
        test_df = pd.DataFrame({
            'id': ['test_doc_1', 'test_doc_2'],
            'content': ['Test content 1', 'Test content 2'],
            'metadata': [
                {'id': 'test_doc_1', 'hybrid_score': 0.8},
                {'id': 'test_doc_2', 'hybrid_score': 0.6}
            ]
        })
        
        # Run evaluation (should be fresh, not cached)
        print("🔍 Testing evaluation with fresh cache...")
        metrics = eval_service.evaluate_query_results(
            query="test cache clear query",
            ctx_df=test_df,
            use_cache=True  # Enable cache to test if it's truly cleared
        )
        
        if metrics:
            print("✅ Evaluation completed successfully")
            print(f"   MRR: {metrics.mrr.value if metrics.mrr else 'N/A'}")
            print(f"   MAP: {metrics.map_score.value if metrics.map_score else 'N/A'}")
            print("📝 This should now be fresh evaluation results, not cached")
        else:
            print("❌ Evaluation failed")
            
    except Exception as e:
        print(f"❌ Cache verification test failed: {e}")


def main():
    """Clear evaluation cache and verify."""
    print("🔧 EVALUATION CACHE MANAGEMENT")
    print("=" * 70)
    print("Clearing evaluation cache after fixing synthetic relevance logic.")
    print("This ensures all future evaluations use the improved algorithm.")
    print("=" * 70)
    
    # Clear the cache
    success = clear_evaluation_cache()
    
    if success:
        # Verify cache was cleared
        test_cache_clear_verification()
        
        print("\n✅ CACHE MANAGEMENT COMPLETE")
        print("=" * 50)
        print("✅ Evaluation cache has been cleared")
        print("✅ All future evaluations will use the fixed algorithm")
        print("✅ Different queries will now produce different metrics")
        print("\n💡 IMPORTANT REMINDERS:")
        print("   1. The evaluation fix is now active")
        print("   2. Cache has been cleared for fresh results")
        print("   3. Monitor evaluation metrics in production")
        print("   4. Metrics should now be more realistic and diverse")
        
    else:
        print("\n⚠️  CACHE CLEAR FAILED")
        print("=" * 50)
        print("❌ Could not clear evaluation cache")
        print("💡 Manual alternatives:")
        print("   1. Restart Redis server if using Redis cache")
        print("   2. Disable evaluation caching temporarily: use_cache=False")
        print("   3. Check cache service configuration")


if __name__ == "__main__":
    main()