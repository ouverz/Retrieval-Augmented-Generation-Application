#!/usr/bin/env python3
"""
Fix for UI metrics caching issue.
The problem is likely that Streamlit session state is interfering with fresh metric display.
"""
import logging
from typing import Dict, Any

def create_streamlit_metrics_fix():
    """Create a fix for the Streamlit metrics caching issue."""
    
    # Read the current Streamlit app
    with open("/Users/oferk/Data Tutorials/RAG/RAG Application-Timescale/frontend/streamlit_app.py", "r") as f:
        content = f.read()
    
    # Check if the display_evaluation_metrics is getting cached results
    print("🔍 Analyzing Streamlit metrics display...")
    
    # Look for potential caching issues
    issues_found = []
    
    if "st.session_state.query_history" in content:
        issues_found.append("✅ CONFIRMED: query_history stored in session_state - potential caching issue")
    
    if "display_evaluation_metrics(evaluation_metrics, st.session_state.get('query_history'" in content:
        issues_found.append("✅ CONFIRMED: display function receives session history - could use stale data")
    
    # Check if evaluation_metrics are being modified by query_history
    if "query_history" in content and "evaluation_metrics" in content:
        issues_found.append("⚠️  POTENTIAL: evaluation_metrics and query_history interaction")
    
    print("\n📊 Issues Found:")
    for issue in issues_found:
        print(f"  {issue}")
    
    if issues_found:
        print("\n🔧 SOLUTION REQUIRED:")
        print("  1. Ensure display_evaluation_metrics uses ONLY fresh metrics from current query")
        print("  2. Add debug logging to track metric values")
        print("  3. Clear session state metrics cache between queries")
        print("  4. Add query timestamp/hash to prevent stale data display")
        
        return True
    else:
        print("\n✅ No obvious caching issues found")
        return False

def analyze_metrics_flow():
    """Analyze how metrics flow through the Streamlit app."""
    
    print("\n🔍 METRICS FLOW ANALYSIS")
    print("="*50)
    
    # Read Streamlit app
    with open("/Users/oferk/Data Tutorials/RAG/RAG Application-Timescale/frontend/streamlit_app.py", "r") as f:
        lines = f.readlines()
    
    # Track metrics flow
    metrics_references = []
    
    for i, line in enumerate(lines, 1):
        if "evaluation_metrics" in line:
            metrics_references.append((i, line.strip()))
    
    print(f"📊 Found {len(metrics_references)} references to evaluation_metrics:")
    
    for line_num, line_content in metrics_references:
        print(f"  Line {line_num}: {line_content}")
        
        # Identify the type of reference
        if "= data.get(" in line_content:
            print(f"    ✅ FRESH: Getting metrics from API response")
        elif "st.session_state" in line_content:
            print(f"    ⚠️  CACHED: Using session state")
        elif "display_evaluation_metrics(" in line_content:
            print(f"    🖼️  DISPLAY: Passing to display function")
        elif "query_history" in line_content:
            print(f"    📚 HISTORY: Storing in history")
    
    print(f"\n🎯 KEY FINDING:")
    print(f"   The display function should use the 'evaluation_metrics' parameter")
    print(f"   If metrics appear static, the issue is likely:")
    print(f"   1. Backend returning identical results (we fixed this)")
    print(f"   2. Frontend caching old results (potential issue)")
    print(f"   3. Display function using wrong data source")

def create_debug_patch():
    """Create a debug patch to add logging to the Streamlit app."""
    
    print("\n🔧 Creating debug patch...")
    
    debug_code = '''
    # DEBUG: Log evaluation metrics to help diagnose caching issues
    if evaluation_metrics:
        import time
        debug_timestamp = time.time()
        debug_mrr = evaluation_metrics.get("mrr", {}).get("value", "None") if evaluation_metrics.get("mrr") else "None"
        debug_map = evaluation_metrics.get("map_score", {}).get("value", "None") if evaluation_metrics.get("map_score") else "None"
        debug_p5 = evaluation_metrics.get("precision_at_k", {}).get("5", {}).get("value", "None") if evaluation_metrics.get("precision_at_k") else "None"
        
        print(f"🔍 UI DEBUG [{debug_timestamp}]: MRR={debug_mrr}, MAP={debug_map}, P@5={debug_p5}")
        print(f"🔍 UI DEBUG: Query='{query[:30]}...', Show_eval={show_evaluation}")
        
        # Force a unique key for metrics display to prevent Streamlit caching
        metrics_key = f"metrics_{hash(str(evaluation_metrics))}_{int(debug_timestamp)}"
        st.markdown(f"<!-- Metrics Key: {metrics_key} -->", unsafe_allow_html=True)
    '''
    
    print("✅ Debug patch created")
    print("   This patch adds logging and cache-busting to metrics display")
    print("   It should be added before the display_evaluation_metrics call")
    
    return debug_code

if __name__ == "__main__":
    print("🧪 STREAMLIT METRICS CACHING FIX")
    print("="*50)
    
    has_issues = create_streamlit_metrics_fix()
    analyze_metrics_flow()
    
    if has_issues:
        debug_patch = create_debug_patch()
        print(f"\n📋 RECOMMENDED ACTIONS:")
        print(f"1. Add debug logging before metrics display")
        print(f"2. Ensure fresh metrics are used (not session history)")
        print(f"3. Clear relevant session state on new queries")
        print(f"4. Test with actual UI to confirm fix")
    else:
        print(f"\n✅ No major issues detected in metrics flow")