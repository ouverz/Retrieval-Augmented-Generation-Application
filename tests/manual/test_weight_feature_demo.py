#!/usr/bin/env python3
"""
Manual demo script to test the weight control feature.

Usage:
1. Start the FastAPI server: uvicorn backend.main:app --reload
2. Ensure system is initialized via /init endpoint
3. Run this script: python tests/manual/test_weight_feature_demo.py
"""

import requests
import json
import time
from typing import Dict, Any


def make_query(query: str, vector_weight: float = None, top_k: int = 5) -> Dict[str, Any]:
    """Make a query to the API with optional weight parameter."""
    url = "http://localhost:8000/query"
    payload = {"query": query, "top_k": top_k}
    
    if vector_weight is not None:
        payload["vector_weight"] = vector_weight
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def demo_weight_feature():
    """Demonstrate the weight control feature."""
    print("🔍 Weight Control Feature Demo")
    print("=" * 50)
    
    test_query = "machine learning"
    
    print(f"Query: '{test_query}'")
    print()
    
    # Test different weight configurations
    weight_configs = [
        {"name": "Vector Heavy", "vector_weight": 0.9, "description": "90% semantic, 10% keyword"},
        {"name": "Balanced", "vector_weight": 0.5, "description": "50% semantic, 50% keyword"},  
        {"name": "BM25 Heavy", "vector_weight": 0.1, "description": "10% semantic, 90% keyword"},
        {"name": "Default", "vector_weight": None, "description": "System default (70% semantic, 30% keyword)"}
    ]
    
    results = []
    
    for config in weight_configs:
        print(f"Testing {config['name']} ({config['description']})...")
        
        start_time = time.time()
        result = make_query(test_query, vector_weight=config['vector_weight'], top_k=3)
        end_time = time.time()
        
        if result["success"]:
            data = result["data"]
            latency = end_time - start_time
            
            # Extract key metrics
            answer_preview = data["answer"][:100] + "..." if len(data["answer"]) > 100 else data["answer"]
            confidence = data.get("confidence", 0)
            backend_latency = data.get("latency_ms", 0)
            
            # Extract top result scores
            top_results = []
            if data.get("results_table"):
                for i, result_row in enumerate(data["results_table"][:3]):
                    top_results.append({
                        "rank": i + 1,
                        "id": result_row.get("source_id", f"result_{i+1}"),
                        "hybrid_score": result_row.get("hybrid_score", 0),
                        "bm25_score": result_row.get("bm25_score", 0),
                        "vector_score": result_row.get("vector_score", 0),
                        "engines": result_row.get("engines", "unknown")
                    })
            
            config_result = {
                "config": config,
                "answer_preview": answer_preview,
                "confidence": confidence,
                "frontend_latency_ms": round(latency * 1000, 1),
                "backend_latency_ms": backend_latency,
                "top_results": top_results
            }
            
            results.append(config_result)
            
            print(f"  ✅ Success - Confidence: {confidence:.2f}, Latency: {latency*1000:.1f}ms")
            print(f"  📝 Answer: {answer_preview}")
            print(f"  🏆 Top result: {top_results[0]['id'] if top_results else 'None'} (score: {top_results[0]['hybrid_score'] if top_results else 'N/A'})")
            print()
            
        else:
            print(f"  ❌ Failed: {result['error']}")
            print()
    
    # Summary analysis
    print("\n📊 RESULTS COMPARISON")
    print("=" * 50)
    
    if len(results) >= 2:
        print("Weight Configuration Effects:")
        
        for result in results:
            config_name = result["config"]["name"]
            vector_weight = result["config"]["vector_weight"]
            if vector_weight is not None:
                bm25_weight = 1.0 - vector_weight
                print(f"\n{config_name} (V:{vector_weight:.1f}, BM25:{bm25_weight:.1f}):")
            else:
                print(f"\n{config_name}:")
                
            print(f"  • Confidence: {result['confidence']:.2f}")
            print(f"  • Latency: {result['backend_latency_ms']}ms")
            
            if result["top_results"]:
                print(f"  • Top result scores:")
                for r in result["top_results"]:
                    print(f"    - #{r['rank']}: hybrid={r['hybrid_score']:.4f}, bm25={r['bm25_score']:.4f}, vector={r['vector_score']:.4f}")
        
        # Check if results are different
        top_ids = [r["top_results"][0]["id"] if r["top_results"] else None for r in results]
        unique_top_results = len(set(filter(None, top_ids)))
        
        print(f"\n🔍 Analysis:")
        print(f"  • Unique top results across configurations: {unique_top_results}/{len(results)}")
        
        if unique_top_results > 1:
            print("  ✅ Different weights produce different rankings - feature working correctly!")
        else:
            print("  ⚠️ Same top result across all configurations - results may be query/dataset dependent")
    
    print(f"\n🎉 Demo completed! Tested {len(results)} weight configurations.")
    
    return results


if __name__ == "__main__":
    # Check if server is available
    try:
        health_response = requests.get("http://localhost:8000/healthz", timeout=5)
        if health_response.status_code != 200:
            print("❌ Server not healthy. Please start with: uvicorn backend.main:app --reload")
            exit(1)
    except requests.ConnectionError:
        print("❌ Cannot connect to server. Please start with: uvicorn backend.main:app --reload")
        exit(1)
    
    # Check if system is initialized
    try:
        status_response = requests.get("http://localhost:8000/init/status")
        if status_response.status_code == 200:
            status = status_response.json().get("status")
            if status != "completed":
                print(f"⚠️ System not fully initialized (status: {status}). Please initialize via /init endpoint first.")
                print("You can still run the demo, but queries may fail.")
                print()
        else:
            print("⚠️ Cannot check initialization status. Proceeding with demo...")
            print()
    except Exception as e:
        print(f"⚠️ Error checking status: {e}. Proceeding with demo...")
        print()
    
    # Run the demo
    demo_weight_feature()