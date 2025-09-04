#!/usr/bin/env python3
"""
Test script to verify evaluation metrics appear correctly in the UI.
Run this alongside the Streamlit app to verify the implementation.
"""
import requests
import json
from typing import Dict, Any

def test_api_evaluation_integration():
    """Test that the API returns evaluation metrics when requested."""
    print("🧪 TESTING API EVALUATION INTEGRATION")
    print("=" * 60)
    
    # Test query request with evaluation enabled
    test_payload = {
        "query": "What is machine learning?",
        "top_k": 5,
        "enable_evaluation": True,
        "session_id": "test_session"
    }
    
    try:
        print("📤 Sending test query to API...")
        response = requests.post(
            "http://localhost:8000/query",
            json=test_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API responded successfully")
            
            # Check for evaluation metrics
            if "evaluation_metrics" in data:
                print("✅ evaluation_metrics found in response")
                metrics = data["evaluation_metrics"]
                
                # Check metric structure
                if metrics:
                    print(f"✅ Metrics structure: {list(metrics.keys())}")
                    
                    if "mrr" in metrics and metrics["mrr"]:
                        mrr = metrics["mrr"]
                        print(f"✅ MRR: {mrr['value']:.3f} ({mrr['interpretation']})")
                        print(f"   Description: {mrr['description']}")
                    
                    if "map_score" in metrics and metrics["map_score"]:
                        map_score = metrics["map_score"]
                        print(f"✅ MAP: {map_score['value']:.3f} ({map_score['interpretation']})")
                        print(f"   Description: {map_score['description']}")
                    
                    if "precision_at_k" in metrics and metrics["precision_at_k"]:
                        precision_at_k = metrics["precision_at_k"]
                        print(f"✅ Precision@K available for: {list(precision_at_k.keys())}")
                        for k, p in precision_at_k.items():
                            print(f"   Precision@{k}: {p['value']:.3f} ({p['interpretation']})")
                    
                    print("\n🎯 THESE METRICS SHOULD APPEAR IN THE UI!")
                    
                else:
                    print("❌ evaluation_metrics is empty")
            else:
                print("❌ evaluation_metrics not found in response")
                print("Available keys:", list(data.keys()))
        else:
            print(f"❌ API error: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API at http://localhost:8000")
        print("   Make sure to start the FastAPI server: uvicorn backend.main:app --reload")
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def print_ui_instructions():
    """Print instructions for finding evaluation metrics in the UI."""
    print("\n" + "=" * 60)
    print("📍 WHERE TO FIND EVALUATION METRICS IN THE UI")
    print("=" * 60)
    
    print("\n1. START THE APPLICATION:")
    print("   streamlit run frontend/streamlit_app.py")
    
    print("\n2. MAKE SURE FASTAPI IS RUNNING:")
    print("   uvicorn backend.main:app --reload")
    
    print("\n3. IN THE STREAMLIT INTERFACE:")
    print("   ✓ Go to 'Advanced Settings'")
    print("   ✓ Make sure '☑️ Show evaluation metrics' is checked (default: enabled)")
    print("   ✓ Submit any query")
    
    print("\n4. LOOK FOR THIS SEQUENCE IN THE RESULTS:")
    print("""
   ✅ Query completed successfully!
   
   [Search Results Summary section with counts]
   
   📝 Answer
   [The AI's answer text]
   
   📊 Query Performance
   ┌─────────────┬─────────────┬─────────────┐
   │ Confidence  │   Context   │ Query Time  │
   │    0.85     │   ✅ Yes    │   1.2s      │
   └─────────────┴─────────────┴─────────────┘
   
   ──────────────────────────────────────────
   
   📊 How Well Did We Find What You Were Looking For?  ← THIS IS NEW!
   
   ┌─────────────┬─────────────┬─────────────┐
   │🟢 0.875     │🔵 0.742     │🟡 0.600     │
   │Answer       │Overall      │Top 5 Results│
   │Ranking      │Search       │Accuracy     │
   │Quality      │Quality      │             │
   │First relevant│Overall ranking│3 out of 5│
   │found at pos 1│quality is good│relevant   │
   └─────────────┴─────────────┴─────────────┘
   
   🔍 [Expand] What Do These Metrics Mean?
   [Detailed explanations section]
   
   [Additional details sections continue...]
   """)
    
    print("\n5. IF METRICS DON'T APPEAR:")
    print("   ❌ Check 'Show evaluation metrics' is enabled")
    print("   ❌ Check FastAPI server is running (http://localhost:8000)")
    print("   ❌ Check browser console for errors")
    print("   ❌ Make sure system is initialized (POST /init)")

def main():
    """Run the complete test and instructions."""
    print("🎯 EVALUATION METRICS UI VERIFICATION")
    print("This script helps verify that MRR, Precision@K, and MAP")
    print("metrics are properly displayed in the Streamlit UI.")
    print()
    
    test_api_evaluation_integration()
    print_ui_instructions()
    
    print("\n" + "=" * 60)
    print("🎉 SUMMARY")
    print("=" * 60)
    print("The evaluation metrics should now appear in the UI after:")
    print("1. ✅ Fixed query_system() to pass enable_evaluation=True")
    print("2. ✅ Moved metrics display to correct location (after Query Performance)")
    print("3. ✅ Fixed enum handling for MetricQuality interpretations") 
    print("4. ✅ Added proper descriptions and styling")
    print("5. ✅ Connected UI toggle to API parameter")
    print()
    print("🔍 Look for the '📊 How Well Did We Find...' section!")

if __name__ == "__main__":
    main()