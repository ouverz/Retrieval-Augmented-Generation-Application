#!/usr/bin/env python3
"""
Final test to verify RRF metrics are updating properly.
"""
import requests
import time

BASE_URL = "http://localhost:8000"

def test_metrics_variation():
    """Test that different queries produce different evaluation metrics."""
    
    queries = [
        "How does sleep affect infant cognitive development?",
        "What are effective bedtime routines for toddlers?",
        "How does maternal mood relate to children's sleep patterns?"
    ]
    
    print("🔍 Testing RRF Metrics Updates")
    print("=" * 50)
    
    results = []
    
    for i, query in enumerate(queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        
        payload = {
            "query": query,
            "top_k": 5,
            "rrf_k": 60,
            "enable_evaluation": True,
            "session_id": f"test_{int(time.time())}_{i}"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/query", json=payload, timeout=120)
            response.raise_for_status()
            data = response.json()
            
            metrics = data.get("evaluation_metrics")
            if metrics:
                mrr = metrics.get("mrr", {}).get("value")
                map_score = metrics.get("map_score", {}).get("value") 
                p5 = metrics.get("precision_at_k", {}).get("5", {}).get("value") if metrics.get("precision_at_k") else None
                
                result = {
                    "query": query,
                    "mrr": mrr,
                    "map": map_score,
                    "precision_5": p5,
                    "cache_hit": data.get("cache_hit", False)
                }
                
                results.append(result)
                
                print(f"  MRR: {mrr}")
                print(f"  MAP: {map_score}")
                print(f"  P@5: {p5}")
                print(f"  Cache: {data.get('cache_hit', False)}")
                
            else:
                print(f"  ❌ No metrics returned")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        time.sleep(2)
    
    # Analysis
    print("\n" + "=" * 50)
    print("📊 ANALYSIS")
    print("=" * 50)
    
    if len(results) < 2:
        print("❌ Not enough successful queries to compare")
        return False
    
    print("\nResults Summary:")
    for i, result in enumerate(results, 1):
        print(f"Query {i}: MRR={result['mrr']}, MAP={result['map']}, P@5={result['precision_5']}")
    
    # Check for variation
    mrr_values = [r['mrr'] for r in results if r['mrr'] is not None]
    map_values = [r['map'] for r in results if r['map'] is not None]
    p5_values = [r['precision_5'] for r in results if r['precision_5'] is not None]
    
    mrr_varies = len(set(mrr_values)) > 1 if mrr_values else False
    map_varies = len(set(map_values)) > 1 if map_values else False
    p5_varies = len(set(p5_values)) > 1 if p5_values else False
    
    print(f"\nVariation Analysis:")
    print(f"  MRR varies: {mrr_varies} (values: {set(mrr_values) if mrr_values else 'None'})")
    print(f"  MAP varies: {map_varies} (values: {set(map_values) if map_values else 'None'})")
    print(f"  P@5 varies: {p5_varies} (values: {set(p5_values) if p5_values else 'None'})")
    
    any_variation = mrr_varies or map_varies or p5_varies
    
    print(f"\n🏁 FINAL RESULT:")
    if any_variation:
        print("✅ SUCCESS: Metrics vary between queries!")
        print("   The RRF metrics update bug has been FIXED.")
        return True
    else:
        print("❌ ISSUE PERSISTS: Metrics remain identical")
        print("   The bug still exists - metrics don't update.")
        return False

if __name__ == "__main__":
    success = test_metrics_variation()
    print(f"\n{'✅ PASS' if success else '❌ FAIL'}")
    exit(0 if success else 1)