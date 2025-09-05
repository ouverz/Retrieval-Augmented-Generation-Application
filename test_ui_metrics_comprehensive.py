#!/usr/bin/env python3
"""
Comprehensive test to verify if RRF metrics update in the UI between different queries.
Tests both backend API responses and simulates UI behavior.
"""
import time
import requests
import json
import logging
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8000"

def test_backend_api_metrics():
    """Test if backend API returns different metrics for different queries."""
    
    test_queries = [
        "What is artificial intelligence?",
        "How do solar panels generate electricity?", 
        "What are the health benefits of regular exercise?",
        "Explain blockchain technology",
        "What causes climate change?"
    ]
    
    logger.info("🔍 STEP 1: Testing Backend API Metrics")
    logger.info("="*60)
    
    api_results = {}
    
    for i, query in enumerate(test_queries, 1):
        logger.info(f"\n📊 API Test Query {i}: {query}")
        
        # Use unique session to avoid any session-based caching
        session_id = f"ui_test_session_{int(time.time())}_{i}"
        
        payload = {
            "query": query,
            "top_k": 5,
            "rrf_k": 60,
            "enable_evaluation": True,
            "session_id": session_id
        }
        
        try:
            response = requests.post(f"{BASE_URL}/query", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            # Extract evaluation metrics that would appear in UI
            eval_metrics = data.get("evaluation_metrics")
            
            if eval_metrics:
                # Focus on metrics that appear in "Additional Performance Metrics"
                ui_metrics = {
                    "MRR": eval_metrics.get("mrr", {}).get("value") if eval_metrics.get("mrr") else None,
                    "MAP": eval_metrics.get("map_score", {}).get("value") if eval_metrics.get("map_score") else None,
                    "Precision@1": eval_metrics.get("precision_at_k", {}).get("1", {}).get("value") if eval_metrics.get("precision_at_k") else None,
                    "Precision@3": eval_metrics.get("precision_at_k", {}).get("3", {}).get("value") if eval_metrics.get("precision_at_k") else None,
                    "Precision@5": eval_metrics.get("precision_at_k", {}).get("5", {}).get("value") if eval_metrics.get("precision_at_k") else None,
                    "Recall@3": eval_metrics.get("recall_at_k", {}).get("3", {}).get("value") if eval_metrics.get("recall_at_k") else None,
                    "NDCG@5": eval_metrics.get("ndcg_at_k", {}).get("5", {}).get("value") if eval_metrics.get("ndcg_at_k") else None,
                }
                
                api_results[query] = {
                    "metrics": ui_metrics,
                    "cache_hit": data.get("cache_hit", False),
                    "latency_ms": data.get("latency_ms"),
                    "results_count": len(data.get("results_table", [])),
                    "session_id": session_id
                }
                
                logger.info(f"   ✅ Query processed successfully")
                logger.info(f"   Cache hit: {data.get('cache_hit', False)}")
                logger.info(f"   Session: {session_id}")
                logger.info(f"   📈 UI Metrics:")
                for metric, value in ui_metrics.items():
                    if value is not None:
                        logger.info(f"     {metric}: {value:.4f}")
                    else:
                        logger.info(f"     {metric}: None")
                        
            else:
                logger.error(f"   ❌ No evaluation metrics returned")
                api_results[query] = {"error": "No evaluation metrics"}
                
        except Exception as e:
            logger.error(f"   ❌ API request failed: {e}")
            api_results[query] = {"error": str(e)}
        
        # Small delay between requests
        time.sleep(2)
    
    return api_results

def analyze_api_results(results: Dict[str, Any]) -> bool:
    """Analyze API results to check for metric variation."""
    
    logger.info(f"\n🔍 STEP 2: Analyzing API Results")
    logger.info("="*60)
    
    successful_results = {}
    for query, result in results.items():
        if "metrics" in result and not result.get("error"):
            successful_results[query] = result["metrics"]
    
    if len(successful_results) < 2:
        logger.error("❌ Not enough successful API responses to compare")
        return False
    
    # Check if all metrics are identical
    queries = list(successful_results.keys())
    reference_metrics = successful_results[queries[0]]
    
    logger.info(f"📊 Reference Query: '{queries[0][:50]}...'")
    for metric, value in reference_metrics.items():
        if value is not None:
            logger.info(f"   {metric}: {value:.4f}")
    
    metrics_vary = False
    varying_metrics = []
    
    for i, (query, metrics) in enumerate(list(successful_results.items())[1:], 2):
        logger.info(f"\n📊 Query {i}: '{query[:50]}...'")
        
        for metric, value in metrics.items():
            ref_value = reference_metrics.get(metric)
            if value is not None:
                ref_str = f"{ref_value:.4f}" if ref_value is not None else "None"
                logger.info(f"   {metric}: {value:.4f} (ref: {ref_str})")
                
                if ref_value is not None and abs(value - ref_value) > 0.0001:  # Allow for floating point precision
                    metrics_vary = True
                    if metric not in varying_metrics:
                        varying_metrics.append(metric)
                    logger.info(f"     ✅ DIFFERENT from reference")
                else:
                    logger.info(f"     ⚠️  SAME as reference")
    
    logger.info(f"\n📈 Analysis Summary:")
    logger.info(f"   Queries tested: {len(successful_results)}")
    logger.info(f"   Metrics that vary: {varying_metrics}")
    logger.info(f"   Overall variation detected: {metrics_vary}")
    
    return metrics_vary

def test_streamlit_ui_integration():
    """Test metrics as they would appear in Streamlit UI."""
    
    logger.info(f"\n🔍 STEP 3: Simulating Streamlit UI Integration")
    logger.info("="*60)
    
    # Read the Streamlit app to understand how metrics are displayed
    try:
        with open("/Users/oferk/Data Tutorials/RAG/RAG Application-Timescale/frontend/streamlit_app.py", "r") as f:
            streamlit_content = f.read()
        
        # Check if metrics are cached/stored in session state
        if "st.session_state" in streamlit_content and "evaluation_metrics" in streamlit_content:
            logger.warning("⚠️  POTENTIAL ISSUE: Streamlit app may be caching metrics in session_state")
            logger.info("   This could cause metrics to not update in the UI even if backend changes")
        
        # Look for metrics display patterns
        if "📊" in streamlit_content and ("MRR" in streamlit_content or "MAP" in streamlit_content):
            logger.info("✅ Found metrics display section in Streamlit app")
        else:
            logger.warning("⚠️  Could not locate metrics display in Streamlit app")
            
    except Exception as e:
        logger.error(f"❌ Could not analyze Streamlit app: {e}")
    
    # Simulate multiple queries with different session states
    ui_simulation_results = {}
    
    test_scenarios = [
        ("First query - fresh session", "What is machine learning?", f"ui_sim_{int(time.time())}_1"),
        ("Second query - same session", "How does photosynthesis work?", f"ui_sim_{int(time.time())}_1"),  # Same session
        ("Third query - new session", "What is quantum computing?", f"ui_sim_{int(time.time())}_2"),  # New session
    ]
    
    for scenario, query, session_id in test_scenarios:
        logger.info(f"\n🎯 UI Scenario: {scenario}")
        logger.info(f"   Query: {query}")
        logger.info(f"   Session: {session_id}")
        
        payload = {
            "query": query,
            "top_k": 5,
            "rrf_k": 60,
            "enable_evaluation": True,
            "session_id": session_id
        }
        
        try:
            response = requests.post(f"{BASE_URL}/query", json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            
            eval_metrics = data.get("evaluation_metrics")
            if eval_metrics:
                key_ui_metrics = {
                    "MRR": eval_metrics.get("mrr", {}).get("value") if eval_metrics.get("mrr") else None,
                    "MAP": eval_metrics.get("map_score", {}).get("value") if eval_metrics.get("map_score") else None,
                    "P@5": eval_metrics.get("precision_at_k", {}).get("5", {}).get("value") if eval_metrics.get("precision_at_k") else None,
                }
                
                ui_simulation_results[scenario] = key_ui_metrics
                
                logger.info(f"   📊 UI would show:")
                for metric, value in key_ui_metrics.items():
                    if value is not None:
                        logger.info(f"     {metric}: {value:.4f}")
                        
            else:
                logger.error(f"   ❌ No metrics for UI display")
                
        except Exception as e:
            logger.error(f"   ❌ UI simulation failed: {e}")
        
        time.sleep(1)
    
    return ui_simulation_results

def main():
    """Main test function."""
    
    logger.info("🧪 COMPREHENSIVE UI METRICS UPDATE TEST")
    logger.info("="*80)
    logger.info("Testing whether RRF metrics actually update in UI between queries")
    logger.info("="*80)
    
    # Check server availability
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        logger.info(f"✅ Server available (status: {health.status_code})")
    except:
        logger.error("❌ Server not available. Please start the application.")
        return False
    
    # Step 1: Test backend API
    api_results = test_backend_api_metrics()
    backend_varies = analyze_api_results(api_results)
    
    # Step 2: Test UI integration
    ui_results = test_streamlit_ui_integration()
    
    # Step 3: Final analysis
    logger.info(f"\n🏁 FINAL COMPREHENSIVE ANALYSIS")
    logger.info("="*60)
    
    if backend_varies:
        logger.info("✅ BACKEND: Metrics vary correctly between queries")
    else:
        logger.error("❌ BACKEND: Metrics do NOT vary between queries")
    
    # Check UI simulation results
    if len(ui_results) >= 2:
        ui_values = list(ui_results.values())
        ui_varies = False
        
        for metric in ["MRR", "MAP", "P@5"]:
            values = [result.get(metric) for result in ui_values if result.get(metric) is not None]
            if len(set(values)) > 1:
                ui_varies = True
                break
        
        if ui_varies:
            logger.info("✅ UI SIMULATION: Metrics would vary in UI")
        else:
            logger.error("❌ UI SIMULATION: Metrics would NOT vary in UI")
    else:
        logger.warning("⚠️  UI SIMULATION: Insufficient data to determine")
    
    # Final verdict
    logger.info(f"\n{'='*60}")
    if not backend_varies:
        logger.error("🚨 USER CLAIM CONFIRMED: Metrics do not update")
        logger.error("   The issue persists - metrics remain static")
        logger.error("   Recommendation: Investigate cache invalidation and session management")
        return False
    else:
        logger.info("✅ METRICS UPDATE CORRECTLY")
        logger.info("   Backend varies appropriately")
        logger.info("   If UI still shows static values, check Streamlit session state")
        return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)