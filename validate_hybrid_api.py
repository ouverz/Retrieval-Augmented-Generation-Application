#!/usr/bin/env python3
"""
API-based validation script to test hybrid search through the FastAPI endpoints.
This will show you the detailed hybrid search logging in the FastAPI server console.
"""
import requests
import json
import time
from typing import Dict, Any


# Configuration
API_BASE_URL = "http://localhost:8000"


def make_request(endpoint: str, method: str = "GET", data: Dict[str, Any] = None) -> Dict[str, Any]:
    """Make a request to the FastAPI backend"""
    url = f"{API_BASE_URL}{endpoint}"
    
    try:
        if method == "POST":
            response = requests.post(url, json=data)
        else:
            response = requests.get(url)
            
        if response.status_code == 200:
            return {"success": True, "data": response.json()}
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
    except requests.ConnectionError:
        return {"success": False, "error": "Cannot connect to FastAPI server"}
    except Exception as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}


def check_server_status():
    """Check if server is running and initialized"""
    print("🏥 Checking server status...")
    
    # Health check
    health = make_request("/healthz")
    if not health["success"]:
        print(f"❌ Server not healthy: {health['error']}")
        return False
    print("✅ Server is healthy")
    
    # Initialization status
    status = make_request("/init/status")
    if not status["success"]:
        print(f"❌ Cannot get init status: {status['error']}")
        return False
    
    init_data = status["data"]
    if init_data["status"] != "completed":
        print(f"⚠️  System not ready: {init_data['status']} - {init_data['message']}")
        return False
    
    print(f"✅ System initialized: {init_data['documents_processed']} documents processed")
    return True


def test_hybrid_search(query: str, top_k: int = 8):
    """Test a single query and analyze response"""
    print(f"\n🔍 Testing query: '{query}'")
    print("-" * 60)
    
    start_time = time.time()
    
    result = make_request("/query", "POST", {"query": query, "top_k": top_k})
    
    elapsed = time.time() - start_time
    
    if not result["success"]:
        print(f"❌ Query failed: {result['error']}")
        return None
    
    data = result["data"]
    
    print(f"⏱️  Query completed in {elapsed:.2f}s (API latency: {data.get('latency_ms', 0)}ms)")
    print(f"🎯 Answer: {data['answer'][:200]}...")
    print(f"📊 Confidence: {data['confidence']:.2f}")
    print(f"📚 Context sufficient: {data['enough_context']}")
    print(f"🎖️  Precision: {data['precision']:.2f} ({data['evidence_precision']})")
    
    # Show citations to see if hybrid worked
    citations = data.get("citations", [])
    print(f"📖 Citations: {len(citations)}")
    for i, cite in enumerate(citations[:3], 1):  # Show first 3
        source = cite.get('source_url') or cite.get('doc_id') or f"Citation {i}"
        score = cite.get('score', 'N/A')
        print(f"  {i}. {source} (Score: {score})")
    
    return data


def run_validation_suite():
    """Run comprehensive validation tests"""
    print("🧪 HYBRID SEARCH API VALIDATION")
    print("=" * 60)
    
    # Check server
    if not check_server_status():
        print("❌ Server not ready. Please ensure FastAPI is running and initialized.")
        return False
    
    # Test queries designed to highlight hybrid behavior
    test_cases = [
        {
            "query": "baby sleep schedule",
            "description": "Keyword-heavy query (should favor BM25)",
            "top_k": 8
        },
        {
            "query": "how to improve infant rest quality during nighttime",
            "description": "Semantic query (should favor vector search)",
            "top_k": 8
        },
        {
            "query": "bedtime routine for toddlers",
            "description": "Mixed query (both engines should contribute)",
            "top_k": 8
        }
    ]
    
    print(f"\n📋 Running {len(test_cases)} test queries...")
    print("\n⚠️  Check your FastAPI server console for detailed hybrid search logging!")
    print("    You should see logs like:")
    print("    🔍 Hybrid Search Query: '...'")
    print("    ⚖️  Weights: BM25=0.5, Vector=0.5")
    print("    📝 BM25 found X documents")
    print("    🎯 Vector found Y documents")
    print("    🏆 Top Z results by hybrid score")
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n{'='*20} TEST {i}/{len(test_cases)} {'='*20}")
        print(f"Description: {test_case['description']}")
        
        result = test_hybrid_search(test_case["query"], test_case["top_k"])
        if result:
            results.append(result)
        
        if i < len(test_cases):
            print("\nWaiting 2 seconds before next test...")
            time.sleep(2)
    
    # Summary
    print(f"\n{'='*20} VALIDATION SUMMARY {'='*20}")
    
    if not results:
        print("❌ No successful queries - validation failed")
        return False
    
    print(f"✅ {len(results)} queries completed successfully")
    
    avg_confidence = sum(r['confidence'] for r in results) / len(results)
    avg_citations = sum(len(r.get('citations', [])) for r in results) / len(results)
    sufficient_context = sum(1 for r in results if r['enough_context'])
    
    print(f"📊 Average confidence: {avg_confidence:.2f}")
    print(f"📚 Average citations per query: {avg_citations:.1f}")
    print(f"✅ Queries with sufficient context: {sufficient_context}/{len(results)}")
    
    print(f"\n🎯 WHAT TO LOOK FOR IN SERVER LOGS:")
    print("  ✅ Both BM25 and Vector searches executed for each query")
    print("  ✅ Documents found by each engine (numbers > 0)")
    print("  ✅ Combined pool of documents created")
    print("  ✅ Deduplication applied (if applicable)")
    print("  ✅ Top results showing mix of BM25# and VECTOR# sources")
    print("  ✅ Different scores for different engines")
    
    print(f"\n🔍 HYBRID BEHAVIOR INDICATORS:")
    print("  • Keyword queries should show more BM25# results at top")
    print("  • Semantic queries should show more VECTOR# results at top")
    print("  • Mixed queries should show both types in results")
    print("  • Scores should reflect 0.5/0.5 weighting (BM25: rank-based, Vector: similarity-based)")
    
    return True


if __name__ == "__main__":
    print("🚀 Starting API-based hybrid search validation...")
    print("📡 Make sure your FastAPI server is running: uvicorn app.main:app --reload")
    print("🔍 Watch the server console for detailed hybrid search logs!")
    
    success = run_validation_suite()
    
    if success:
        print("\n🎉 Validation completed! Check server logs for hybrid search details.")
    else:
        print("\n❌ Validation failed. Check server status and try again.")
    
    exit(0 if success else 1)