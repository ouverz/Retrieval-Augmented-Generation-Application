"""
Streamlit Frontend for RAG Application
Pure frontend that communicates with FastAPI backend
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any, List
import time


# Configuration
API_BASE_URL = "http://localhost:8000"  # FastAPI server URL


def make_api_request(
    endpoint: str, method: str = "GET", data: Dict[str, Any] = None
) -> Dict[str, Any]:
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
            return {
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}",
            }
    except requests.ConnectionError:
        return {
            "success": False,
            "error": "Cannot connect to FastAPI server. Make sure it's running on http://localhost:8000",
        }
    except Exception as e:
        return {"success": False, "error": f"Request failed: {str(e)}"}


def check_health() -> bool:
    """Check if FastAPI server is healthy"""
    result = make_api_request("/healthz")
    return result.get("success", False)


def initialize_system(force: bool = False) -> Dict[str, Any]:
    """Initialize the RAG system"""
    return make_api_request("/init", "POST", {"force": force})


def get_init_status() -> Dict[str, Any]:
    """Get initialization status"""
    return make_api_request("/init/status")


def query_system(query: str, top_k: int = 8, rrf_k: int = None, session_id: str = None, enable_evaluation: bool = True) -> Dict[str, Any]:
    """Query the RAG system"""
    payload = {"query": query, "top_k": top_k, "enable_evaluation": enable_evaluation}
    if rrf_k is not None:
        payload["rrf_k"] = rrf_k
    if session_id is not None:
        payload["session_id"] = session_id
    return make_api_request("/query", "POST", payload)


def create_session(user_id: str = "streamlit_user") -> Dict[str, Any]:
    """Create a new session"""
    return make_api_request("/cache/session", "POST", {"user_id": user_id})


def get_session(session_id: str) -> Dict[str, Any]:
    """Get session data"""
    return make_api_request(f"/cache/session/{session_id}")


def upload_document(file) -> Dict[str, Any]:
    """Upload a document to the system"""
    # Note: This would need to be implemented differently for file uploads
    # For now, just return a placeholder
    return {"success": False, "error": "File upload not implemented in frontend yet"}


def get_quality_color(quality: str) -> str:
    """Get color based on quality level"""
    color_map = {
        "Excellent": "#22c55e",  # Green
        "Good": "#3b82f6",      # Blue  
        "Fair": "#f59e0b",      # Amber
        "Poor": "#ef4444"       # Red
    }
    return color_map.get(quality, "#6b7280")  # Default gray


def get_quality_emoji(quality: str) -> str:
    """Get emoji based on quality level"""
    emoji_map = {
        "Excellent": "🟢",
        "Good": "🔵",
        "Fair": "🟡", 
        "Poor": "🔴"
    }
    return emoji_map.get(quality, "⚪")


def display_evaluation_metrics(evaluation_metrics: Dict[str, Any], query_history: List[Dict[str, Any]] = None):
    """Display evaluation metrics in user-friendly format"""
    st.markdown('<div class="evaluation-header">📊 How Well Did We Find What You Were Looking For?</div>', unsafe_allow_html=True)
    
    # Extract metrics
    mrr = evaluation_metrics.get("mrr")
    precision_at_k = evaluation_metrics.get("precision_at_k", {})
    map_score = evaluation_metrics.get("map_score")
    recall_at_k = evaluation_metrics.get("recall_at_k", {})
    ndcg_at_k = evaluation_metrics.get("ndcg_at_k", {})
    
    # Main metrics cards
    cols = st.columns(3)
    
    # MRR Card
    if mrr:
        with cols[0]:
            # Handle both enum and string interpretation values
            interpretation = mrr.get("interpretation", "Fair")
            if hasattr(interpretation, 'value'):
                interpretation = interpretation.value
            quality_color = get_quality_color(interpretation)
            quality_emoji = get_quality_emoji(interpretation)
            value = mrr.get('value', 0)
            progress_width = int(value * 100)
            description = mrr.get('description', 'Mean Reciprocal Rank')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {quality_color};">
                    {quality_emoji} {value:.3f}
                </div>
                <div class="metric-label">Answer Ranking Quality</div>
                <div class="metric-description">{description}</div>
                <div class="metric-progress">
                    <div class="metric-progress-bar" style="width: {progress_width}%; background-color: {quality_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
    # MAP Card
    if map_score:
        with cols[1]:
            # Handle both enum and string interpretation values
            interpretation = map_score.get("interpretation", "Fair")
            if hasattr(interpretation, 'value'):
                interpretation = interpretation.value
            quality_color = get_quality_color(interpretation)
            quality_emoji = get_quality_emoji(interpretation)
            value = map_score.get('value', 0)
            progress_width = int(value * 100)
            description = map_score.get('description', 'Mean Average Precision')
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="color: {quality_color};">
                    {quality_emoji} {value:.3f}
                </div>
                <div class="metric-label">Overall Search Quality</div>
                <div class="metric-description">{description}</div>
                <div class="metric-progress">
                    <div class="metric-progress-bar" style="width: {progress_width}%; background-color: {quality_color};"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # Precision@K Card (use most common K value)
    if precision_at_k:
        with cols[2]:
            # Find the most relevant K value (prefer 5, then 3, then 1, then first available)
            preferred_ks = [5, 3, 1]
            display_k = None
            display_precision = None
            
            for k in preferred_ks:
                if k in precision_at_k:  # Check for integer keys first
                    display_k = k
                    display_precision = precision_at_k[k]
                    break
                elif str(k) in precision_at_k:  # Then check for string keys
                    display_k = k
                    display_precision = precision_at_k[str(k)]
                    break
            
            if not display_precision and precision_at_k:
                # Use first available if preferred not found
                first_k = list(precision_at_k.keys())[0]
                display_k = int(first_k) if isinstance(first_k, str) and first_k.isdigit() else first_k
                display_precision = precision_at_k[first_k]
                
            if display_precision:
                # Handle both enum and string interpretation values
                interpretation = display_precision.get("interpretation", "Fair")
                if hasattr(interpretation, 'value'):
                    interpretation = interpretation.value
                quality_color = get_quality_color(interpretation)
                quality_emoji = get_quality_emoji(interpretation)
                value = display_precision.get('value', 0)
                progress_width = int(value * 100)
                description = display_precision.get('description', f'Precision at {display_k}')
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value" style="color: {quality_color};">
                        {quality_emoji} {value:.3f}
                    </div>
                    <div class="metric-label">Top {display_k} Results Accuracy</div>
                    <div class="metric-description">{description}</div>
                    <div class="metric-progress">
                        <div class="metric-progress-bar" style="width: {progress_width}%; background-color: {quality_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    # Expandable detailed explanations
    with st.expander("🔍 What Do These Metrics Mean?"):
        if mrr:
            st.markdown(f"""
            **Answer Ranking Quality (MRR: {mrr.get('value', 0):.3f})** {get_quality_emoji(mrr.get('interpretation', 'Fair'))}
            
            {mrr.get('description', 'Measures how well we rank the most relevant results first.')}
            
            **Your Result:** {mrr.get('interpretation', 'Fair')} - This means the system is performing {mrr.get('interpretation', 'fair').lower()} at putting the best answers first.
            """)
            st.markdown("---")
            
        if map_score:
            st.markdown(f"""
            **Overall Search Quality (MAP: {map_score.get('value', 0):.3f})** {get_quality_emoji(map_score.get('interpretation', 'Fair'))}
            
            {map_score.get('description', 'Measures the overall quality of search results across all positions.')}
            
            **Your Result:** {map_score.get('interpretation', 'Fair')} - The system's overall search performance is {map_score.get('interpretation', 'fair').lower()}.
            """)
            st.markdown("---")
            
        if precision_at_k and display_precision:
            st.markdown(f"""
            **Top {display_k} Results Accuracy (P@{display_k}: {display_precision.get('value', 0):.3f})** {get_quality_emoji(display_precision.get('interpretation', 'Fair'))}
            
            {display_precision.get('description', f'Measures what fraction of the top {display_k} results are actually relevant to your question.')}
            
            **Your Result:** {display_precision.get('interpretation', 'Fair')} - {display_precision.get('interpretation', 'Fair')} accuracy in the top results.
            """)
    
    # Additional metrics in collapsible section
    additional_metrics_available = bool(recall_at_k or ndcg_at_k or len(precision_at_k) > 1)
    
    if additional_metrics_available:
        with st.expander("📈 Additional Performance Metrics"):
            
            # Show all Precision@K values
            if len(precision_at_k) > 1:
                st.markdown("**Precision at Different Result Counts:**")
                precision_cols = st.columns(min(len(precision_at_k), 4))
                for i, (k, p_result) in enumerate(precision_at_k.items()):
                    if i < len(precision_cols):
                        with precision_cols[i]:
                            quality_color = get_quality_color(p_result.get("interpretation", "Fair"))
                            st.markdown(f"""
                            <div style="text-align: center; padding: 0.5rem;">
                                <div style="color: {quality_color}; font-size: 1.2rem; font-weight: bold;">
                                    {p_result.get('value', 0):.3f}
                                </div>
                                <div style="font-size: 0.8rem; color: #a0aec0;">
                                    Top {k} Results
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Show Recall@K if available
            if recall_at_k:
                st.markdown("---")
                st.markdown("**Recall at Different Result Counts:**")
                recall_cols = st.columns(min(len(recall_at_k), 4))
                for i, (k, r_result) in enumerate(recall_at_k.items()):
                    if i < len(recall_cols):
                        with recall_cols[i]:
                            quality_color = get_quality_color(r_result.get("interpretation", "Fair"))
                            st.markdown(f"""
                            <div style="text-align: center; padding: 0.5rem;">
                                <div style="color: {quality_color}; font-size: 1.2rem; font-weight: bold;">
                                    {r_result.get('value', 0):.3f}
                                </div>
                                <div style="font-size: 0.8rem; color: #a0aec0;">
                                    Recall@{k}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
            
            # Show NDCG@K if available  
            if ndcg_at_k:
                st.markdown("---")
                st.markdown("**Normalized Discounted Cumulative Gain:**")
                ndcg_cols = st.columns(min(len(ndcg_at_k), 4))
                for i, (k, ndcg_result) in enumerate(ndcg_at_k.items()):
                    if i < len(ndcg_cols):
                        with ndcg_cols[i]:
                            quality_color = get_quality_color(ndcg_result.get("interpretation", "Fair"))
                            st.markdown(f"""
                            <div style="text-align: center; padding: 0.5rem;">
                                <div style="color: {quality_color}; font-size: 1.2rem; font-weight: bold;">
                                    {ndcg_result.get('value', 0):.3f}
                                </div>
                                <div style="font-size: 0.8rem; color: #a0aec0;">
                                    NDCG@{k}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Historical metrics tracking
    if query_history and len(query_history) > 1:
        with st.expander("📈 Session Performance Trends"):
            st.markdown("**Performance Over Your Recent Queries:**")
            
            # Extract historical data
            queries = [q['query'][:30] + "..." if len(q['query']) > 30 else q['query'] for q in query_history]
            confidences = [q.get('confidence', 0) for q in query_history]
            
            # Create simple metrics overview
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            queries_with_context = sum(1 for q in query_history if q.get('enough_context', False))
            context_percentage = (queries_with_context / len(query_history)) * 100 if query_history else 0
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg Confidence", f"{avg_confidence:.2f}", f"{confidences[-1] - avg_confidence:+.2f}" if len(confidences) > 1 else None)
            with col2:
                st.metric("Queries This Session", len(query_history))
            with col3:
                st.metric("Context Success Rate", f"{context_percentage:.0f}%")
            
            # Show recent query list
            if len(query_history) > 1:
                st.markdown("**Recent Queries:**")
                for i, q in enumerate(reversed(query_history[-5:]), 1):  # Show last 5 queries
                    quality_emoji = "✅" if q.get('enough_context') else "⚠️"
                    confidence = q.get('confidence', 0)
                    st.write(f"{i}. {quality_emoji} {q['query'][:50]}... (Confidence: {confidence:.2f})")
    
    st.markdown("---")


# Streamlit UI
def main():
    st.set_page_config(
        page_title="RAG System", 
        page_icon="🔍", 
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for dark theme with orange accents
    st.markdown("""
    <style>
        /* Main theme colors */
        .stApp {
            background-color: #1a1d29;
            color: #ffffff;
        }
        
        /* Header styling */
        .main-header {
            background: linear-gradient(90deg, #2d3748 0%, #1a1d29 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            border: 1px solid #4a5568;
        }
        
        .main-title {
            color: #ffffff;
            font-size: 2.5rem;
            font-weight: 700;
            margin: 0;
            text-align: center;
        }
        
        .description {
            color: #a0aec0;
            font-size: 1.1rem;
            line-height: 1.6;
            margin-top: 1rem;
            text-align: center;
        }
        
        /* Subtle accent elements */
        .stButton > button {
            background-color: #4a5568 !important;
            color: white !important;
            border: 1px solid #6b7280 !important;
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.3s ease !important;
        }
        
        .stButton > button:hover {
            background-color: #5a6173 !important;
            border-color: #9ca3af !important;
            transform: translateY(-1px) !important;
        }
        
        /* Slider styling */
        .stSlider > div > div > div > div {
            background-color: #ff6b35 !important;
        }
        
        /* Card styling */
        .metric-card {
            background-color: #2d3748;
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #4a5568;
            text-align: center;
            margin: 0.5rem;
        }
        
        .metric-value {
            color: #ff6b35;
            font-size: 2rem;
            font-weight: 700;
        }
        
        .metric-label {
            color: #a0aec0;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .metric-description {
            color: #cbd5e0;
            font-size: 0.8rem;
            margin-top: 0.5rem;
            font-style: italic;
        }
        
        /* Weight display styling */
        .weight-display {
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #4a5568;
            margin-top: 1rem;
        }
        
        .weight-title {
            color: #ff6b35;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        
        .weight-item {
            color: #ffffff;
            margin: 0.2rem 0;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #2d3748;
        }
        
        /* Text input styling */
        .stTextInput > div > div > input {
            background-color: #4a5568 !important;
            color: white !important;
            border: 1px solid #6b7280 !important;
            border-radius: 8px !important;
        }
        
        /* Expander styling */
        .streamlit-expanderHeader {
            background-color: #2d3748 !important;
            color: white !important;
        }
        
        /* Success/error message styling */
        .stAlert > div {
            background-color: #2d3748 !important;
            border: 1px solid #4a5568 !important;
        }
        
        /* Metric containers */
        .stMetric {
            background-color: #2d3748;
            padding: 1rem;
            border-radius: 8px;
            border: 1px solid #4a5568;
        }
        
        .stMetric [data-testid="metric-container"] {
            background-color: transparent;
        }
        
        /* Evaluation metrics specific styling */
        .evaluation-header {
            color: #ffffff;
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 1rem;
            text-align: center;
        }
        
        .quality-excellent { color: #22c55e !important; }
        .quality-good { color: #3b82f6 !important; }
        .quality-fair { color: #f59e0b !important; }
        .quality-poor { color: #ef4444 !important; }
        
        /* Mobile responsive adjustments */
        @media (max-width: 768px) {
            .metric-card {
                margin: 0.25rem;
                padding: 1rem;
            }
            
            .metric-value {
                font-size: 1.5rem;
            }
            
            .main-title {
                font-size: 2rem;
            }
            
            .description {
                font-size: 1rem;
            }
        }
        
        /* Progress bar styling for metrics */
        .metric-progress {
            width: 100%;
            background-color: #4a5568;
            border-radius: 4px;
            height: 8px;
            margin-top: 0.5rem;
        }
        
        .metric-progress-bar {
            height: 100%;
            border-radius: 4px;
            transition: width 0.3s ease;
        }
    </style>
    """, unsafe_allow_html=True)

    # Main header with description
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🔍 RAG System</h1>
        <div class="description">
            <strong>Retrieval-Augmented Generation (RAG)</strong> combines the power of large language models with your document knowledge base. 
            This system intelligently searches through your uploaded documents using both semantic (vector) and keyword (BM25) search 
            combined with Reciprocal Rank Fusion (RRF), then synthesizes accurate, contextual answers with proper citations. 
            <br><br>
            <strong>How it works:</strong> Upload PDFs → System processes and indexes content → Ask questions → Get AI-powered answers backed by your documents.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar for system controls
    with st.sidebar:
        st.header("⚙️ System Controls")

        # Health check
        if st.button("🏥 Check Server Health"):
            if check_health():
                st.success("✅ Server is healthy")
            else:
                st.error("❌ Server is not responding")

        st.markdown("---")

        # Initialization
        st.subheader("📚 System Initialization")
        force_init = st.checkbox(
            "Force re-initialization",
            help="Re-process all documents even if already processed",
        )

        if st.button("🚀 Initialize System"):
            result = initialize_system(force=force_init)

            if result["success"]:
                data = result["data"]
                if data["initialized"]:
                    st.success(f"✅ {data['reason']}")
                elif data["reason"] == "initialization-started":
                    st.info("🚀 Initialization started! Check progress below.")
                else:
                    st.warning(f"⚠️ {data.get('reason', 'Unknown status')}")
            else:
                st.error(f"❌ Initialization failed: {result['error']}")

        # Progress tracking
        st.markdown("---")
        st.subheader("📊 Initialization Status")

        status_result = get_init_status()
        if status_result["success"]:
            status_data = status_result["data"]
            status = status_data["status"]
            progress = status_data["progress"]
            message = status_data["message"]
            docs_processed = status_data["documents_processed"]
            total_docs = status_data["total_documents"]

            # Status indicator
            if status == "idle":
                st.info("💤 System idle - not initialized")
            elif status == "running":
                st.warning("🔄 Initialization in progress...")

                # Progress bar
                progress_bar = st.progress(progress)

                # Progress details
                if total_docs > 0:
                    st.write(f"📄 Documents: {docs_processed}/{total_docs}")

                st.write(f"📋 Status: {message}")

                # Auto-refresh every 2 seconds when running
                time.sleep(2)
                st.rerun()

            elif status == "completed":
                st.success(f"✅ {message}")
                if total_docs > 0:
                    st.write(f"📄 Processed {docs_processed} documents")
            elif status == "error":
                st.error(f"❌ Error: {message}")
        else:
            st.error("Failed to get initialization status")

        # Move API endpoints to sidebar
        st.markdown("---")
        st.subheader("🔧 API Endpoints")
        st.code(f"""
        Health: {API_BASE_URL}/healthz
        Init: {API_BASE_URL}/init
        Query: {API_BASE_URL}/query
        """)

    # Check system readiness first
    status_result = get_init_status()
    system_ready = False
    if status_result["success"]:
        system_ready = status_result["data"]["status"] == "completed"

    if not system_ready:
        st.warning(
            "⚠️ System not ready. Please initialize the system first using the sidebar."
        )
        st.stop()

    # Initialize session for historical metrics tracking
    if 'session_id' not in st.session_state:
        session_result = create_session()
        if session_result["success"]:
            st.session_state.session_id = session_result["data"]["session_id"]
            st.session_state.query_history = []
        else:
            st.session_state.session_id = None

    # 2. Query input
    query = st.text_input(
        "Enter your question:",
        placeholder="Ask something about baby sleep research...",
        key="query_input",
    )

    # 3. Search button (initial - will be updated with logic later)
    search_button_placeholder = st.empty()

    # 4. Advanced Query Settings
    with st.expander("⚙️ Advanced Query Settings"):
        col_topk, col_eval = st.columns(2)
        
        with col_topk:
            top_k = st.slider(
                "Number of results to retrieve", min_value=1, max_value=20, value=8
            )
        
        with col_eval:
            show_evaluation = st.checkbox(
                "Show evaluation metrics",
                value=True,
                help="Display performance metrics like MRR, Precision@K, and MAP for query results"
            )

    # 5. RRF Configuration - COMMENTED OUT FOR NOW
    # st.markdown("### ⚖️ RRF Configuration")
    # 
    # col_rrf, col_display = st.columns([3, 2])
    # 
    # with col_rrf:
    #     rrf_k = st.slider(
    #         "RRF K Parameter",
    #         min_value=1,
    #         max_value=200,
    #         value=60,  # Default value matching backend
    #         step=5,
    #         help="Reciprocal Rank Fusion k parameter. Higher values make rank fusion more conservative (less influence from lower-ranked results)."
    #     )
    # 
    # with col_display:
    #     st.markdown(f"""
    #     <div class="weight-display">
    #         <div class="weight-title">RRF Settings</div>
    #         <div class="weight-item">🔢 K Value: {rrf_k}</div>
    #         <div class="weight-item">📊 Fusion: Reciprocal Rank</div>
    #     </div>
    #     """, unsafe_allow_html=True)
    
    # Set default RRF k value when configuration is commented out
    rrf_k = 60  # Default backend value

    # Update search button with actual button
    with search_button_placeholder.container():
        if st.button("🔍 Search", disabled=not query or not query.strip()):
            start_time = time.time()

            with st.spinner("Searching and synthesizing answer..."):
                result = query_system(query, top_k=top_k, rrf_k=rrf_k, session_id=st.session_state.get('session_id'), enable_evaluation=show_evaluation)

            elapsed_time = time.time() - start_time

            if result["success"]:
                data = result["data"]
                st.success("✅ Query completed successfully!")

                # 6. Store metrics in session history
                evaluation_metrics = data.get("evaluation_metrics")
                if evaluation_metrics and show_evaluation:
                    # Store metrics in session history
                    if 'query_history' not in st.session_state:
                        st.session_state.query_history = []
                    
                    # Add current query metrics to history
                    query_entry = {
                        'query': query,
                        'timestamp': time.time(),
                        'evaluation_metrics': evaluation_metrics,
                        'confidence': data.get("confidence", 0),
                        'enough_context': data.get("enough_context", False)
                    }
                    st.session_state.query_history.append(query_entry)
                    
                    # Keep only last 10 queries for performance
                    if len(st.session_state.query_history) > 10:
                        st.session_state.query_history = st.session_state.query_history[-10:]

                # 7. Search Results Summary (moved from breakdown expander)
                if data.get("results_table"):
                    results_data = data["results_table"]
                    total_results = len(results_data)
                    bm25_found = len([
                        r for r in results_data
                        if "bm25" in r.get("engines", "")
                    ])
                    vector_found = len([
                        r for r in results_data
                        if "vector" in r.get("engines", "")
                    ])
                    both_engines = len([
                        r for r in results_data
                        if "+" in r.get("engines", "")
                    ])

                    st.markdown("### 🔍 Search Results Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{total_results}</div>
                            <div class="metric-label">Total Results</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{bm25_found}</div>
                            <div class="metric-label">Found by BM25</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{vector_found}</div>
                            <div class="metric-label">Found by Vector</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-value">{both_engines}</div>
                            <div class="metric-label">Found by Both</div>
                        </div>
                        """, unsafe_allow_html=True)

                # 8. Answer
                st.markdown("### 📝 Answer")
                st.write(data["answer"])

                # 9. Query Performance (removed API Latency column)
                st.markdown("### 📊 Query Performance")
                col_conf, col_ctx, col_time = st.columns(3)
                
                with col_conf:
                    confidence_val = data['confidence']
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{confidence_val:.2f}</div>
                        <div class="metric-label">Confidence</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_ctx:
                    context_status = "✅ Yes" if data["enough_context"] else "⚠️ No"
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value" style="font-size: 1.5rem;">{context_status}</div>
                        <div class="metric-label">Context</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col_time:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{elapsed_time:.2f}s</div>
                        <div class="metric-label">Query Time</div>
                    </div>
                    """, unsafe_allow_html=True)

                # 10. Evaluation Metrics Display (after Query Performance)
                evaluation_metrics = data.get("evaluation_metrics")
                
                if evaluation_metrics and show_evaluation:
                    st.markdown("---")  # Separator
                    display_evaluation_metrics(evaluation_metrics, st.session_state.get('query_history', []))

                # 11. Additional details (controlled by show_evaluation toggle)
                if show_evaluation:
                    with st.expander("🧠 Thought Process"):
                        for i, thought in enumerate(data.get("thought_process", []), 1):
                            st.write(f"{i}. {thought}")

                    if data.get("citations"):
                        with st.expander("📚 Citations"):
                            for i, citation in enumerate(data["citations"], 1):
                                # Try different fields for source identification
                                source = (
                                    citation.get("source_url")
                                    or citation.get("source")
                                    or citation.get("doc_id")
                                    or f"Citation {i}"
                                )

                                score = citation.get("score", "N/A")
                                chunk_id = citation.get("chunk_id", "")
                                doc_id = citation.get("doc_id", "")

                                # Create a detailed citation display
                                citation_text = f"**{source}**"
                                if chunk_id:
                                    citation_text += f" (Chunk: {chunk_id})"
                                if doc_id and doc_id != source:
                                    citation_text += f" [Doc: {doc_id}]"
                                citation_text += f" - Score: {score}"

                                st.write(f"{i}. {citation_text}")

                                # Debug info in expandable section
                                if st.checkbox(
                                    f"Debug info for citation {i}", key=f"debug_cite_{i}"
                                ):
                                    st.json(citation)

                    # Search Results Breakdown (controlled by show_evaluation toggle)
                    if data.get("results_table"):
                        with st.expander("📊 Search Results Breakdown"):
                            results_df = pd.DataFrame(data["results_table"])

                            # Filter to only the requested columns
                            filtered_columns = [
                                "rank", "source_id", "content_preview", "engines", 
                                "bm25_rank", "vector_rank", "vector_similarity", "distance"
                            ]
                            
                            # Only include columns that exist in the dataframe
                            available_columns = [col for col in filtered_columns if col in results_df.columns]
                            filtered_df = results_df[available_columns]

                            # Style the dataframe for better readability
                            st.dataframe(
                                filtered_df,
                                column_config={
                                    "rank": st.column_config.NumberColumn(
                                        "Rank", width="small"
                                    ),
                                    "source_id": st.column_config.TextColumn(
                                        "Source ID", width="medium"
                                    ),
                                    "content_preview": st.column_config.TextColumn(
                                        "Content Preview", width="large"
                                    ),
                                    "engines": st.column_config.TextColumn(
                                        "Found By", width="small"
                                    ),
                                    "bm25_rank": st.column_config.NumberColumn(
                                        "BM25 Rank", width="small"
                                    ),
                                    "vector_rank": st.column_config.NumberColumn(
                                        "Vector Rank", width="small"
                                    ),
                                    "vector_similarity": st.column_config.NumberColumn(
                                        "Vector Similarity", format="%.4f", width="small"
                                    ),
                                    "distance": st.column_config.NumberColumn(
                                        "Vector Distance", format="%.4f", width="small"
                                    ),
                                },
                                hide_index=True,
                                use_container_width=True,
                            )

                    # Technical details (controlled by show_evaluation toggle)
                    with st.expander("🔬 Technical Details"):
                        st.json(
                            {
                                "precision": data.get("precision", 0),
                                "evidence_precision": data.get(
                                    "evidence_precision", "unknown"
                                ),
                                "query_settings": {
                                    "top_k": top_k,
                                    "rrf_k": rrf_k  # Using default value when UI config is disabled
                                },
                            }
                        )

            else:
                st.error(f"❌ Query failed: {result['error']}")

    # 11. Server Status (moved from sidebar to main content)
    st.markdown("---")  # Add separator
    st.markdown("### 📊 System Status")
    
    # Quick health check display
    if check_health():
        st.markdown("""
        <div style="background-color: #22543d; color: #68d391; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
            🟢 <strong>Server Online</strong>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background-color: #742a2a; color: #fc8181; padding: 1rem; border-radius: 8px; margin: 1rem 0; text-align: center;">
            🔴 <strong>Server Offline</strong>
        </div>
        """, unsafe_allow_html=True)

    # 12. Quick Start Guide (moved from sidebar to main content)
    st.markdown("### ℹ️ Quick Start Guide")
    
    # Use info container for better formatting
    with st.container():
        st.info("""
        **1. Start FastAPI Server:**
        ```
        uvicorn backend.main:app --reload
        ```
        
        **2. Initialize System:**
        Use the sidebar controls to process your documents
        
        **3. Ask Questions:**
        Get AI-powered answers with citations from your documents
        """)


if __name__ == "__main__":
    main()
