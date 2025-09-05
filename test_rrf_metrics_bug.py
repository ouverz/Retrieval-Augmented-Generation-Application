#!/usr/bin/env python3
"""
Test script to verify if RRF metrics change between different queries.
Tests the claim that metrics in the "📊 How Well Did We Find What You Were Looking For?" 
section do not change for subsequent queries.
"""
import time
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

def test_rrf_metrics_consistency():
    """Test if RRF metrics change for different queries."""
    
    # Test queries - deliberately different to expect different results
    test_queries = [
        "What is machine learning?",
        "How does solar energy work?", 
        "What are the benefits of exercise?",
        "Explain quantum computing",
        "What is climate change?"
    ]
    
    results = {}
    
    logger.info("🔍 Testing RRF metrics consistency across multiple queries...")
    
    for i, query in enumerate(test_queries):
        logger.info(f"\n{'='*50}")
        logger.info(f"🔍 Query {i+1}: {query}")
        logger.info(f"{'='*50}")
        
        # Send query request with evaluation enabled
        payload = {
            "query": query,
            "top_k": 5,
            "rrf_k": 60,
            "enable_evaluation": True,
            "session_id": f"test_session_{int(time.time())}"
        }
        
        try:
            response = requests.post(f"{BASE_URL}/query", json=payload)
            response.raise_for_status()
            data = response.json()
            
            # Extract evaluation metrics
            eval_metrics = data.get("evaluation_metrics")
            
            if eval_metrics:
                # Store key metrics for comparison
                key_metrics = {
                    "mrr": eval_metrics.get("mrr", {}).get("value") if eval_metrics.get("mrr") else None,
                    "map_score": eval_metrics.get("map_score", {}).get("value") if eval_metrics.get("map_score") else None,
                    "precision_at_1": eval_metrics.get("precision_at_k", {}).get(1, {}).get("value") if eval_metrics.get("precision_at_k") else None,
                    "precision_at_3": eval_metrics.get("precision_at_k", {}).get(3, {}).get("value") if eval_metrics.get("precision_at_k") else None,
                    "precision_at_5": eval_metrics.get("precision_at_k", {}).get(5, {}).get("value") if eval_metrics.get("precision_at_k") else None,
                }
                
                results[query] = {
                    "metrics": key_metrics,
                    "results_count": len(data.get("results_table", [])),
                    "answer_preview": data.get("answer", "")[:100],
                    "latency_ms": data.get("latency_ms"),
                    "cache_hit": data.get("cache_hit", False)
                }
                
                logger.info(f"✅ Query processed successfully")
                logger.info(f"   Cache hit: {data.get('cache_hit', False)}")
                logger.info(f"   Latency: {data.get('latency_ms')}ms")
                logger.info(f"   Results: {len(data.get('results_table', []))}")
                logger.info(f"   MRR: {key_metrics.get('mrr')}")
                logger.info(f"   MAP: {key_metrics.get('map_score')}")
                logger.info(f"   P@1: {key_metrics.get('precision_at_1')}")
                logger.info(f"   P@3: {key_metrics.get('precision_at_3')}")
                logger.info(f"   P@5: {key_metrics.get('precision_at_5')}")
                
            else:
                logger.warning(f"❌ No evaluation metrics returned for query: {query}")
                results[query] = {"metrics": None, "error": "No evaluation metrics"}
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Request failed for query '{query}': {e}")
            results[query] = {"error": str(e)}
        except Exception as e:
            logger.error(f"❌ Unexpected error for query '{query}': {e}")
            results[query] = {"error": str(e)}
            
        # Small delay between queries
        time.sleep(1)
    
    # Analyze results
    logger.info(f"\n{'='*60}")
    logger.info("🔍 ANALYSIS: Checking for identical metrics across queries")
    logger.info(f"{'='*60}")
    
    # Extract all metric values for comparison
    all_metrics = []
    successful_queries = []
    
    for query, result in results.items():
        if "metrics" in result and result["metrics"] and not result.get("error"):
            all_metrics.append(result["metrics"])
            successful_queries.append(query)
    
    if len(all_metrics) < 2:
        logger.error("❌ Not enough successful queries to compare metrics")
        return False
    
    # Compare metrics across queries
    identical_metrics = True
    reference_metrics = all_metrics[0]
    
    logger.info(f"📊 Reference metrics from query 1: '{successful_queries[0]}'")
    for key, value in reference_metrics.items():
        logger.info(f"   {key}: {value}")
    
    for i, metrics in enumerate(all_metrics[1:], 2):
        logger.info(f"\n📊 Comparing with query {i}: '{successful_queries[i-1]}'")
        
        for key, value in metrics.items():
            ref_value = reference_metrics.get(key)
            logger.info(f"   {key}: {value} (ref: {ref_value})")
            
            if value != ref_value:
                identical_metrics = False
                logger.info(f"   ✅ DIFFERENT: {key} changed from {ref_value} to {value}")
            else:
                logger.info(f"   ⚠️  SAME: {key} identical across queries")
    
    # Final verdict
    logger.info(f"\n{'='*60}")
    logger.info("🏁 FINAL VERDICT")
    logger.info(f"{'='*60}")
    
    if identical_metrics:
        logger.error("❌ BUG CONFIRMED: All RRF metrics are IDENTICAL across different queries!")
        logger.error("   This indicates the metrics are not being recalculated properly.")
        logger.error("   The user's claim appears to be CORRECT.")
        return True
    else:
        logger.info("✅ METRICS VARY: RRF metrics change appropriately between queries.")
        logger.info("   The user's claim appears to be INCORRECT.")
        return False

if __name__ == "__main__":
    # Check if server is running
    try:
        health_response = requests.get(f"{BASE_URL}/health")
        logger.info(f"✅ Server is running (status: {health_response.status_code})")
    except requests.exceptions.ConnectionError:
        logger.error("❌ Server not running. Please start the application first.")
        logger.info("   Run: python start_app.py")
        exit(1)
    
    bug_confirmed = test_rrf_metrics_consistency()
    
    if bug_confirmed:
        logger.error("\n🚨 BUG CONFIRMED: RRF metrics do not change between queries")
        exit(1)
    else:
        logger.info("\n✅ NO BUG: RRF metrics change appropriately between queries") 
        exit(0)