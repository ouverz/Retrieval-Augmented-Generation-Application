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


def query_system(query: str, top_k: int = 8, rrf_k: int = None, session_id: str = None) -> Dict[str, Any]:
    """Query the RAG system"""
    payload = {"query": query, "top_k": top_k, "enable_evaluation": False}  # Disabled in lean branch
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


# REMOVED: display_evaluation_metrics function - evaluation system removed in lean branch


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
        
        /* REMOVED: Evaluation metrics specific styling - evaluation system removed in lean branch */
        
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
            # REMOVED: Evaluation metrics checkbox - evaluation system removed in lean branch
            show_evaluation = False

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
                result = query_system(query, top_k=top_k, rrf_k=rrf_k, session_id=st.session_state.get('session_id'))

            elapsed_time = time.time() - start_time

            if result["success"]:
                data = result["data"]
                st.success("✅ Query completed successfully!")

                # REMOVED: Evaluation metrics session history - evaluation system removed in lean branch

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

                # REMOVED: Evaluation Metrics Display - evaluation system removed in lean branch

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
