#!/usr/bin/env python3
"""
Debug script to trace the complete evaluation metrics flow.
This simulates what happens when a user submits a query with evaluation enabled.
"""
import requests
import json
import time
from typing import Dict, Any

def debug_api_flow():
    """Debug the complete API flow for evaluation metrics."""
    print("🔍 DEBUGGING EVALUATION METRICS FLOW")
    print("=" * 70)
    
    base_url = "http://localhost:8000"
    
    # Step 1: Check if server is running
    print("📡 Step 1: Checking server health...")
    try:
        health_response = requests.get(f"{base_url}/health", timeout=5)
        if health_response.status_code == 200:
            print("✅ Server is running")
        else:
            print(f"⚠️  Server returned {health_response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Server not running! Start with: uvicorn backend.main:app --reload")
        return False
    except Exception as e:
        print(f"❌ Error checking server: {e}")
        return False
    
    # Step 2: Check initialization status
    print("\n🔧 Step 2: Checking initialization status...")
    try:
        init_status = requests.get(f"{base_url}/init/status", timeout=5)
        if init_status.status_code == 200:
            data = init_status.json()
            if data.get("initialized", False):
                print("✅ System is initialized")
            else:
                print("⚠️  System not initialized - this might be the issue!")
                print("   Components status:")
                for component, status in data.items():
                    print(f"     {component}: {status}")
                return False
        else:
            print(f"❌ Cannot check init status: {init_status.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error checking init status: {e}")
        return False
    
    # Step 3: Make query with evaluation enabled
    print("\n📊 Step 3: Testing query with evaluation metrics...")
    test_payload = {
        "query": "What is machine learning and how does it work?",
        "top_k": 5,
        "enable_evaluation": True,
        "session_id": "debug_session"
    }
    
    print(f"   Payload: {json.dumps(test_payload, indent=2)}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{base_url}/query",
            json=test_payload,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"\n📥 Response received in {elapsed:.2f}s")
        print(f"   Status code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("✅ Query successful!")
            
            # Analyze response structure
            print("\n📋 Response Analysis:")
            print(f"   Top-level keys: {list(data.keys())}")
            
            # Check for evaluation_metrics
            if "evaluation_metrics" in data:
                eval_metrics = data["evaluation_metrics"]
                print(f"   📊 evaluation_metrics present: {eval_metrics is not None}")
                
                if eval_metrics:
                    print(f"   📊 evaluation_metrics type: {type(eval_metrics)}")
                    print(f"   📊 evaluation_metrics keys: {list(eval_metrics.keys())}")
                    
                    # Check individual metrics
                    if eval_metrics.get("mrr"):
                        mrr = eval_metrics["mrr"]
                        print(f"   🎯 MRR: {mrr['value']:.3f} ({mrr.get('interpretation', 'Unknown')})")
                    
                    if eval_metrics.get("map_score"):
                        map_score = eval_metrics["map_score"]
                        print(f"   🎯 MAP: {map_score['value']:.3f} ({map_score.get('interpretation', 'Unknown')})")
                    
                    if eval_metrics.get("precision_at_k"):
                        precision_at_k = eval_metrics["precision_at_k"]
                        print(f"   🎯 Precision@K: {list(precision_at_k.keys())}")
                        for k, precision in precision_at_k.items():
                            print(f"      Precision@{k}: {precision['value']:.3f} ({precision.get('interpretation', 'Unknown')})")
                    
                    print("\n🎉 EVALUATION METRICS ARE WORKING IN THE API!")
                    print("   The issue must be in the frontend display logic.")
                    return True
                else:
                    print("   ❌ evaluation_metrics is None - this is the problem!")
                    return False
            else:
                print("   ❌ evaluation_metrics not in response - this is the problem!")
                return False
                
        elif response.status_code == 412:
            print("❌ Service not initialized")
            return False
        else:
            print(f"❌ Query failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error making query: {e}")
        return False

def debug_frontend_integration():
    """Debug the frontend integration issues."""
    print("\n🎨 DEBUGGING FRONTEND INTEGRATION")
    print("=" * 50)
    
    print("Checking frontend code for evaluation metrics display...")
    
    # Check if display function exists
    try:
        with open('frontend/streamlit_app.py', 'r') as f:
            content = f.read()
        
        checks = [
            ('display_evaluation_metrics function', 'def display_evaluation_metrics' in content),
            ('show_evaluation checkbox', 'show_evaluation = st.checkbox' in content),
            ('enable_evaluation in query_system', 'enable_evaluation=show_evaluation' in content),
            ('evaluation_metrics display condition', 'if evaluation_metrics and show_evaluation:' in content),
            ('evaluation metrics section', '📊 How Well Did We Find What You Were Looking For?' in content),
        ]
        
        print("\nFrontend code checks:")
        for check_name, result in checks:
            status = "✅" if result else "❌"
            print(f"   {status} {check_name}: {result}")
            
        # Check for the specific display location
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if 'display_evaluation_metrics(evaluation_metrics' in line:
                print(f"\n📍 Found display call at line {i+1}:")
                print(f"   {line.strip()}")
                
                # Check surrounding context
                start = max(0, i-3)
                end = min(len(lines), i+4)
                print("\n   Context:")
                for j in range(start, end):
                    marker = " >>>>" if j == i else "     "
                    print(f"{marker} {j+1:4d}: {lines[j]}")
        
    except Exception as e:
        print(f"❌ Error reading frontend code: {e}")

def print_debugging_summary():
    """Print a summary of debugging steps."""
    print("\n" + "=" * 70)
    print("🎯 DEBUGGING SUMMARY & NEXT STEPS")
    print("=" * 70)
    
    print("\n1. 📡 SERVER STATUS:")
    print("   ✓ Check if FastAPI server is running: uvicorn backend.main:app --reload")
    print("   ✓ Check if system is initialized: curl http://localhost:8000/init/status")
    
    print("\n2. 🔧 API FLOW:")
    print("   ✓ Verify enable_evaluation=True is sent in request payload")
    print("   ✓ Check server logs for DEBUG messages about evaluation metrics")
    print("   ✓ Verify evaluation_metrics is not None in API response")
    
    print("\n3. 🎨 FRONTEND FLOW:")
    print("   ✓ Verify show_evaluation checkbox is checked (default: True)")
    print("   ✓ Verify query_system() passes enable_evaluation=show_evaluation")
    print("   ✓ Verify condition: if evaluation_metrics and show_evaluation:")
    print("   ✓ Verify display_evaluation_metrics() is called")
    
    print("\n4. 🔍 DEBUGGING TOOLS:")
    print("   ✓ Check browser developer console for JavaScript errors")
    print("   ✓ Check FastAPI server logs for DEBUG messages")
    print("   ✓ Manually test API with curl or requests")
    
    print("\n🎯 MOST LIKELY ISSUES:")
    print("   1. System not initialized (POST /init not called)")
    print("   2. evaluation_service returning None (check server logs)")
    print("   3. Frontend condition logic not triggering display")
    print("   4. Data format mismatch in display_evaluation_metrics()")

def main():
    """Run the complete debugging flow."""
    success = debug_api_flow()
    
    if success:
        debug_frontend_integration()
    
    print_debugging_summary()

if __name__ == "__main__":
    main()