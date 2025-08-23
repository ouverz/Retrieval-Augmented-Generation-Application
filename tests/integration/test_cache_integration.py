#!/usr/bin/env python3
"""
Quick test to validate cache integration in API endpoints.
Tests both cache hit/miss scenarios and graceful degradation.
"""
import json
import requests
from time import sleep

def test_api_endpoints():
    """Test cache integration in API endpoints."""
    base_url = "http://localhost:8000"
    
    print("=== Testing Cache Integration ===")
    
    # 1. Test cache health endpoint
    print("\n1. Testing cache health endpoint...")
    try:
        response = requests.get(f"{base_url}/cache/health")
        if response.status_code == 200:
            health = response.json()
            print(f"   ✓ Cache health check successful: available={health.get('available', False)}")
        else:
            print(f"   ⚠ Cache health check returned {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Cache health check failed: {e}")
    
    # 2. Test cache stats endpoint
    print("\n2. Testing cache stats endpoint...")
    try:
        response = requests.get(f"{base_url}/cache/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"   ✓ Cache stats retrieved: {stats}")
        else:
            print(f"   ⚠ Cache stats returned {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Cache stats failed: {e}")
    
    # 3. Test query without session (should work with graceful degradation)
    print("\n3. Testing query endpoint (cache miss expected)...")
    query_data = {
        "query": "What are the benefits of bedtime routines?",
        "top_k": 5
    }
    
    try:
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Query successful:")
            print(f"     - Cache hit: {result.get('cache_hit', 'N/A')}")
            print(f"     - Latency: {result.get('latency_ms', 'N/A')}ms")
            print(f"     - Answer length: {len(result.get('answer', ''))}")
        else:
            print(f"   ⚠ Query returned {response.status_code}: {response.text}")
    except Exception as e:
        print(f"   ⚠ Query failed: {e}")
    
    # 4. Test the same query again (should be cache hit if Redis is available)
    print("\n4. Testing same query again (cache hit expected if Redis available)...")
    try:
        response = requests.post(f"{base_url}/query", json=query_data)
        if response.status_code == 200:
            result = response.json()
            print(f"   ✓ Second query successful:")
            print(f"     - Cache hit: {result.get('cache_hit', 'N/A')}")
            print(f"     - Latency: {result.get('latency_ms', 'N/A')}ms")
            if result.get('cache_hit'):
                print("   ✓ Cache is working correctly!")
        else:
            print(f"   ⚠ Second query returned {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Second query failed: {e}")
    
    # 5. Test session creation
    print("\n5. Testing session creation...")
    try:
        response = requests.post(f"{base_url}/cache/session", json={"user_id": "test_user"})
        if response.status_code == 200:
            session = response.json()
            session_id = session.get('session_id')
            print(f"   ✓ Session created: {session_id}")
            
            # 6. Test query with session
            print("\n6. Testing query with session...")
            query_with_session = query_data.copy()
            query_with_session['session_id'] = session_id
            
            response = requests.post(f"{base_url}/query", json=query_with_session)
            if response.status_code == 200:
                result = response.json()
                print(f"   ✓ Query with session successful:")
                print(f"     - Session ID in response: {result.get('session_id')}")
                print(f"     - Cache hit: {result.get('cache_hit', 'N/A')}")
            else:
                print(f"   ⚠ Query with session returned {response.status_code}")
                
        else:
            print(f"   ⚠ Session creation returned {response.status_code}")
    except Exception as e:
        print(f"   ⚠ Session creation failed: {e}")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_api_endpoints()