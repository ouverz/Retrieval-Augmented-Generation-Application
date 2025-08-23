#!/usr/bin/env python3
"""
Demo script showing cache management capabilities.
"""
import requests
import json
from time import sleep

def demo_cache_management():
    """Demonstrate cache management endpoints."""
    base_url = "http://localhost:8000"
    
    print("=== Cache Management Demo ===")
    
    # 1. Check initial cache status
    print("\n1. Initial cache health check...")
    response = requests.get(f"{base_url}/cache/health")
    if response.status_code == 200:
        health = response.json()
        print(f"   Cache available: {health.get('available', False)}")
        print(f"   Degraded mode: {health.get('degraded_mode', True)}")
    
    # 2. Get initial stats
    print("\n2. Initial cache stats...")
    response = requests.get(f"{base_url}/cache/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"   Total keys: {stats.get('total_keys', 0)}")
        print(f"   Embedding cache: {stats.get('embedding_cache_count', 0)}")
        print(f"   Query cache: {stats.get('query_cache_count', 0)}")
        print(f"   Active sessions: {stats.get('active_sessions', 0)}")
    
    # 3. Create some test queries to populate cache
    print("\n3. Creating test queries to populate cache...")
    test_queries = [
        "What are bedtime routines?",
        "How do bedtime routines affect sleep?",
        "What is the impact on maternal mood?"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"   Query {i}: {query[:30]}...")
        response = requests.post(f"{base_url}/query", json={"query": query, "top_k": 3})
        if response.status_code == 200:
            result = response.json()
            print(f"      Cache hit: {result.get('cache_hit', 'N/A')}")
        sleep(0.5)  # Brief pause between queries
    
    # 4. Check stats after queries
    print("\n4. Cache stats after queries...")
    response = requests.get(f"{base_url}/cache/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"   Total keys: {stats.get('total_keys', 0)}")
        print(f"   Query cache: {stats.get('query_cache_count', 0)}")
    
    # 5. Test cache clearing
    print("\n5. Testing cache clearing...")
    
    # Clear query cache
    print("   Clearing query cache...")
    response = requests.post(f"{base_url}/cache/clear", params={"cache_type": "queries"})
    if response.status_code == 200:
        result = response.json()
        print(f"   ✓ {result.get('message', 'Cache cleared')}")
    
    # Check stats after clearing
    response = requests.get(f"{base_url}/cache/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"   Query cache after clear: {stats.get('query_cache_count', 0)}")
    
    # 6. Test session management
    print("\n6. Testing session management...")
    
    # Create a session
    response = requests.post(f"{base_url}/cache/session", json={"user_id": "demo_user"})
    if response.status_code == 200:
        session = response.json()
        session_id = session.get('session_id')
        print(f"   Created session: {session_id}")
        
        # Make queries with session
        for i, query in enumerate(test_queries[:2], 1):
            query_data = {
                "query": query,
                "top_k": 3,
                "session_id": session_id
            }
            response = requests.post(f"{base_url}/query", json=query_data)
            if response.status_code == 200:
                print(f"   Query {i} with session completed")
        
        # Get session data
        response = requests.get(f"{base_url}/cache/session/{session_id}")
        if response.status_code == 200:
            session_data = response.json()
            print(f"   Session queries: {session_data.get('total_queries', 0)}")
            print(f"   History length: {len(session_data.get('query_history', []))}")
    
    # 7. Final stats
    print("\n7. Final cache stats...")
    response = requests.get(f"{base_url}/cache/stats")
    if response.status_code == 200:
        stats = response.json()
        print(f"   Total keys: {stats.get('total_keys', 0)}")
        print(f"   Active sessions: {stats.get('active_sessions', 0)}")
        if stats.get('redis_connected'):
            print(f"   Memory used: {stats.get('memory_used_mb', 0)} MB")
    
    print("\n=== Demo Complete ===")
    print("\nCache management endpoints available:")
    print("  GET  /cache/health  - Check cache health")
    print("  GET  /cache/stats   - Get cache statistics")
    print("  POST /cache/clear   - Clear cache (all|queries|embeddings|sessions)")
    print("  POST /cache/session - Create new session")
    print("  GET  /cache/session/{id} - Get session data")

if __name__ == "__main__":
    demo_cache_management()