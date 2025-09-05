#!/usr/bin/env python3
"""
Diagnostic script to test cache key generation for evaluation service.
This script helps identify why different queries might be getting identical 
evaluation metrics due to cache key collisions.
"""
import pandas as pd
import hashlib
import json
from typing import Dict, Any, Set
from core.services.evaluation_service import EvaluationService
from core.services.cache_service import CacheService


def create_sample_dataframes():
    """Create sample context DataFrames that might cause cache collisions."""
    
    # Case 1: Different queries, same documents and scores
    df1 = pd.DataFrame({
        'id': ['doc1', 'doc2', 'doc3'],
        'content': [
            'Machine learning is a subset of artificial intelligence...',
            'Deep learning uses neural networks with multiple layers...',
            'Natural language processing enables computers to understand text...'
        ],
        'metadata': [
            {'hybrid_score': 0.85, 'bm25_score': 0.7, 'vector_score': 0.9, 'id': 'doc1'},
            {'hybrid_score': 0.75, 'bm25_score': 0.8, 'vector_score': 0.7, 'id': 'doc2'}, 
            {'hybrid_score': 0.65, 'bm25_score': 0.6, 'vector_score': 0.7, 'id': 'doc3'}
        ]
    })
    
    df2 = pd.DataFrame({
        'id': ['doc1', 'doc2', 'doc3'],
        'content': [
            'Machine learning is a subset of artificial intelligence...',
            'Deep learning uses neural networks with multiple layers...',
            'Natural language processing enables computers to understand text...'
        ],
        'metadata': [
            {'hybrid_score': 0.85, 'bm25_score': 0.7, 'vector_score': 0.9, 'id': 'doc1'},
            {'hybrid_score': 0.75, 'bm25_score': 0.8, 'vector_score': 0.7, 'id': 'doc2'}, 
            {'hybrid_score': 0.65, 'bm25_score': 0.6, 'vector_score': 0.7, 'id': 'doc3'}
        ]
    })
    
    # Case 2: Different queries, same documents but different scores
    df3 = pd.DataFrame({
        'id': ['doc1', 'doc2', 'doc3'],
        'content': [
            'Machine learning is a subset of artificial intelligence...',
            'Deep learning uses neural networks with multiple layers...',
            'Natural language processing enables computers to understand text...'
        ],
        'metadata': [
            {'hybrid_score': 0.95, 'bm25_score': 0.9, 'vector_score': 0.9, 'id': 'doc1'},
            {'hybrid_score': 0.85, 'bm25_score': 0.8, 'vector_score': 0.9, 'id': 'doc2'}, 
            {'hybrid_score': 0.75, 'bm25_score': 0.7, 'vector_score': 0.8, 'id': 'doc3'}
        ]
    })
    
    # Case 3: Different queries, different documents  
    df4 = pd.DataFrame({
        'id': ['doc4', 'doc5', 'doc6'],
        'content': [
            'Python programming language for data science applications...',
            'JavaScript frameworks for web development projects...',
            'Database optimization techniques for performance...'
        ],
        'metadata': [
            {'hybrid_score': 0.85, 'bm25_score': 0.7, 'vector_score': 0.9, 'id': 'doc4'},
            {'hybrid_score': 0.75, 'bm25_score': 0.8, 'vector_score': 0.7, 'id': 'doc5'}, 
            {'hybrid_score': 0.65, 'bm25_score': 0.6, 'vector_score': 0.7, 'id': 'doc6'}
        ]
    })
    
    return {
        'query1_df1': df1,
        'query2_df2': df2,  # Identical to df1
        'query3_df3': df3,  # Same docs, different scores
        'query4_df4': df4   # Different docs
    }


def test_cache_key_generation():
    """Test cache key generation for different query/context combinations."""
    print("🔍 DIAGNOSTIC: Cache Key Generation Analysis")
    print("=" * 60)
    
    # Create evaluation service
    eval_service = EvaluationService(cache_service=None)
    
    # Test queries
    queries = {
        'query1': "What is machine learning?",
        'query2': "How does artificial intelligence work?", 
        'query3': "Explain deep learning concepts",
        'query4': "Best programming languages for data science"
    }
    
    # Get sample DataFrames
    dataframes = create_sample_dataframes()
    
    # Test all combinations
    results = {}
    print("Testing cache key generation for different query/context combinations:")
    print()
    
    for query_key, query_text in queries.items():
        print(f"📝 Query: {query_key}")
        print(f"   Text: {query_text}")
        
        for df_key, ctx_df in dataframes.items():
            # Generate context hash
            ctx_hash = eval_service._hash_context_df(ctx_df)
            
            # Generate cache key
            cache_key = eval_service._get_cache_key(query_text, ctx_hash)
            
            # Store result
            combo_key = f"{query_key}_{df_key}"
            results[combo_key] = {
                'query': query_text,
                'ctx_hash': ctx_hash,
                'cache_key': cache_key,
                'df_key': df_key
            }
            
            print(f"   {df_key}: ctx_hash={ctx_hash}, cache_key={cache_key[:12]}...")
        print()
    
    return results


def analyze_cache_collisions(results: Dict[str, Dict[str, Any]]):
    """Analyze results for potential cache key collisions."""
    print("🚨 COLLISION ANALYSIS")
    print("=" * 60)
    
    # Group by cache key
    cache_key_groups = {}
    for combo_key, result in results.items():
        cache_key = result['cache_key']
        if cache_key not in cache_key_groups:
            cache_key_groups[cache_key] = []
        cache_key_groups[cache_key].append((combo_key, result))
    
    # Find collisions
    collisions = {k: v for k, v in cache_key_groups.items() if len(v) > 1}
    
    if collisions:
        print(f"❌ Found {len(collisions)} cache key collisions!")
        for cache_key, items in collisions.items():
            print(f"\n🔥 COLLISION: Cache key {cache_key[:12]}...")
            for combo_key, result in items:
                query_short = result['query'][:30] + "..." if len(result['query']) > 30 else result['query']
                print(f"   - {combo_key}: '{query_short}' (ctx_hash: {result['ctx_hash']})")
    else:
        print("✅ No cache key collisions detected!")
    
    # Group by context hash  
    ctx_hash_groups = {}
    for combo_key, result in results.items():
        ctx_hash = result['ctx_hash']
        if ctx_hash not in ctx_hash_groups:
            ctx_hash_groups[ctx_hash] = []
        ctx_hash_groups[ctx_hash].append((combo_key, result))
    
    print(f"\n📊 Context Hash Distribution:")
    for ctx_hash, items in ctx_hash_groups.items():
        print(f"   ctx_hash {ctx_hash}: {len(items)} combinations")
        if len(items) > 1:
            print(f"      ⚠️  Multiple queries share this context hash:")
            for combo_key, result in items:
                query_short = result['query'][:25] + "..." if len(result['query']) > 25 else result['query']
                print(f"         - {combo_key}: '{query_short}'")


def test_context_df_hashing():
    """Test the _hash_context_df method in detail."""
    print("\n🔬 DETAILED CONTEXT HASHING ANALYSIS")
    print("=" * 60)
    
    eval_service = EvaluationService(cache_service=None)
    dataframes = create_sample_dataframes()
    
    for df_name, ctx_df in dataframes.items():
        print(f"\n📋 DataFrame: {df_name}")
        print(f"   Shape: {ctx_df.shape}")
        
        # Show what goes into the hash
        content_parts = []
        for i, (_, row) in enumerate(ctx_df.iterrows()):
            doc_id = str(row.get("id", ""))
            metadata = row.get("metadata", {})
            
            if isinstance(metadata, str):
                try:
                    metadata = json.loads(metadata)
                except:
                    metadata = {}
            
            score = metadata.get("hybrid_score", metadata.get("score", 0.0))
            part = f"{doc_id}:{score}"
            content_parts.append(part)
            
            print(f"   Row {i}: id='{doc_id}', score={score} -> '{part}'")
        
        # Generate hash
        content = "|".join(content_parts)
        hash_value = hashlib.md5(content.encode('utf-8')).hexdigest()[:16]
        
        print(f"   Hash Input: '{content}'")
        print(f"   Hash Output: {hash_value}")


def simulate_evaluation_caching():
    """Simulate the evaluation caching process to identify issues."""
    print("\n🎯 EVALUATION CACHING SIMULATION")
    print("=" * 60)
    
    eval_service = EvaluationService(cache_service=None)
    
    queries = [
        "What is machine learning?",
        "How does artificial intelligence work?", 
        "Explain neural networks",
    ]
    
    # Create context DataFrames that might cause issues
    dataframes = create_sample_dataframes()
    
    # Simulate caching behavior
    cache_simulation = {}
    
    for i, query in enumerate(queries):
        for df_key, ctx_df in dataframes.items():
            ctx_hash = eval_service._hash_context_df(ctx_df)
            cache_key = eval_service._get_cache_key(query, ctx_hash)
            
            cache_entry_key = f"query{i+1}_{df_key}"
            
            # Check if this cache key was seen before
            if cache_key in cache_simulation:
                print(f"🚨 CACHE HIT SIMULATION for {cache_entry_key}:")
                print(f"   New query: '{query}'")
                print(f"   Previous query: '{cache_simulation[cache_key]['query']}'")
                print(f"   Cache key: {cache_key}")
                print(f"   ❌ These different queries would share cached evaluation results!")
                print()
            else:
                cache_simulation[cache_key] = {
                    'query': query,
                    'ctx_df_key': df_key,
                    'entry_key': cache_entry_key
                }
    
    print(f"Total unique cache keys: {len(cache_simulation)}")
    print(f"Total query/context combinations: {len(queries) * len(dataframes)}")


def main():
    """Run all diagnostic tests."""
    print("🔧 EVALUATION SERVICE CACHE DIAGNOSTICS")
    print("=" * 70)
    print("This script diagnoses cache key generation issues that could")
    print("cause identical evaluation metrics for different queries.")
    print("=" * 70)
    
    # Test 1: Cache key generation
    results = test_cache_key_generation()
    
    # Test 2: Analyze collisions
    analyze_cache_collisions(results)
    
    # Test 3: Detailed hashing analysis
    test_context_df_hashing()
    
    # Test 4: Cache behavior simulation
    simulate_evaluation_caching()
    
    print("\n💡 RECOMMENDATIONS")
    print("=" * 60)
    print("1. If cache collisions found: The issue is confirmed")
    print("2. If no collisions: Look at the actual data being processed")
    print("3. Consider adding more context to cache keys (e.g., timestamp, config)")
    print("4. Add debug logging to track cache key generation in production")
    print("5. Consider clearing evaluation cache: service.clear_evaluation_cache()")


if __name__ == "__main__":
    main()