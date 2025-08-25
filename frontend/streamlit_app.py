"""
Streamlit Frontend for RAG Application
Pure frontend that communicates with FastAPI backend
"""

import streamlit as st
import requests
import pandas as pd
from typing import Dict, Any
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


def query_system(query: str, top_k: int = 8, vector_weight: float = None) -> Dict[str, Any]:
    """Query the RAG system"""
    payload = {"query": query, "top_k": top_k}
    if vector_weight is not None:
        payload["vector_weight"] = vector_weight
    return make_api_request("/query", "POST", payload)


def upload_document(file) -> Dict[str, Any]:
    """Upload a document to the system"""
    # Note: This would need to be implemented differently for file uploads
    # For now, just return a placeholder
    return {"success": False, "error": "File upload not implemented in frontend yet"}


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
    </style>
    """, unsafe_allow_html=True)

    # Main header with description
    st.markdown("""
    <div class="main-header">
        <h1 class="main-title">🔍 RAG System</h1>
        <div class="description">
            <strong>Retrieval-Augmented Generation (RAG)</strong> combines the power of large language models with your document knowledge base. 
            This system intelligently searches through your uploaded documents using both semantic (vector) and keyword (BM25) search, 
            then synthesizes accurate, contextual answers with proper citations. 
            <br><br>
            <strong>How it works:</strong> Upload PDFs → System processes and indexes content → Ask questions → Get AI-powered answers backed by your documents.
            Adjust the search weights below to fine-tune between semantic understanding and exact keyword matching for optimal results.
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

    # Main interface
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("💬 Query Interface")

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

        # Query input
        query = st.text_input(
            "Enter your question:",
            placeholder="Ask something about baby sleep research...",
            key="query_input",
        )

        # Weight controls - styled with custom components
        st.markdown("### ⚖️ Search Weight Configuration")
        
        col_weight, col_display = st.columns([3, 2])
        
        with col_weight:
            vector_weight = st.slider(
                "Vector Search Weight",
                min_value=0.0,
                max_value=1.0,
                value=0.7,  # Default value matching backend
                step=0.05,
                help="Adjust the balance between semantic (vector) and keyword (BM25) search. Higher values favor semantic similarity."
            )
        
        with col_display:
            bm25_weight = 1.0 - vector_weight
            st.markdown(f"""
            <div class="weight-display">
                <div class="weight-title">Current Weights</div>
                <div class="weight-item">🔍 Vector: {vector_weight:.2f}</div>
                <div class="weight-item">📝 BM25: {bm25_weight:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # Query settings  
        with st.expander("⚙️ Advanced Query Settings"):
            top_k = st.slider(
                "Number of results to retrieve", min_value=1, max_value=20, value=8
            )

        if st.button("🔍 Search", disabled=not query or not query.strip()):
            start_time = time.time()

            with st.spinner("Searching and synthesizing answer..."):
                result = query_system(query, top_k=top_k, vector_weight=vector_weight)

            elapsed_time = time.time() - start_time

            if result["success"]:
                data = result["data"]
                st.success("✅ Query completed successfully!")

                # Display answer
                st.subheader("📝 Answer")
                st.write(data["answer"])

                # Metrics - Custom styled cards
                st.markdown("### 📊 Query Performance")
                col_conf, col_ctx, col_time, col_lat = st.columns(4)
                
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
                
                with col_lat:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-value">{data.get('latency_ms', 0)}ms</div>
                        <div class="metric-label">API Latency</div>
                    </div>
                    """, unsafe_allow_html=True)

                # Additional details
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

                # Search results breakdown table
                if data.get("results_table"):
                    with st.expander("📊 Search Results Breakdown"):
                        results_df = pd.DataFrame(data["results_table"])

                        # Style the dataframe for better readability
                        st.dataframe(
                            results_df,
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
                                "hybrid_score": st.column_config.NumberColumn(
                                    "True Hybrid Score", format="%.4f", width="small"
                                ),
                                "bm25_score": st.column_config.NumberColumn(
                                    "BM25 Component", format="%.4f", width="small"
                                ),
                                "vector_score": st.column_config.NumberColumn(
                                    "Vector Component", format="%.4f", width="small"
                                ),
                                "vector_similarity": st.column_config.NumberColumn(
                                    "Vector Similarity", format="%.4f", width="small"
                                ),
                                "quality_penalty": st.column_config.NumberColumn(
                                    "Quality Penalty", format="%.2f", width="small"
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
                                "distance": st.column_config.NumberColumn(
                                    "Vector Distance", format="%.4f", width="small"
                                ),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )

                        # Summary stats for true hybrid results
                        total_results = len(data["results_table"])
                        bm25_found = len(
                            [
                                r
                                for r in data["results_table"]
                                if "bm25" in r.get("engines", "")
                            ]
                        )
                        vector_found = len(
                            [
                                r
                                for r in data["results_table"]
                                if "vector" in r.get("engines", "")
                            ]
                        )
                        both_engines = len(
                            [
                                r
                                for r in data["results_table"]
                                if "+" in r.get("engines", "")
                            ]
                        )

                        # Search results summary with custom styling
                        st.markdown("#### 🔍 Search Results Summary")
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

                # Technical details
                with st.expander("🔬 Technical Details"):
                    st.json(
                        {
                            "precision": data.get("precision", 0),
                            "evidence_precision": data.get(
                                "evidence_precision", "unknown"
                            ),
                            "query_settings": {
                                "top_k": top_k,
                                "vector_weight": vector_weight,
                                "bm25_weight": 1.0 - vector_weight
                            },
                        }
                    )

            else:
                st.error(f"❌ Query failed: {result['error']}")

    with col2:
        # System Status with custom styling
        st.markdown("""
        <div style="background-color: #2d3748; padding: 1.5rem; border-radius: 10px; border: 1px solid #4a5568; margin-bottom: 1rem;">
            <h3 style="color: #ff6b35; margin-top: 0;">📊 System Status</h3>
        """, unsafe_allow_html=True)

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

        st.markdown("</div>", unsafe_allow_html=True)

        # Instructions with custom styling
        st.markdown("""
        <div style="background-color: #2d3748; padding: 1.5rem; border-radius: 10px; border: 1px solid #4a5568;">
            <h3 style="color: #ff6b35; margin-top: 0;">ℹ️ Quick Start Guide</h3>
            <div style="color: #a0aec0; line-height: 1.6;">
                <p><strong style="color: #ffffff;">1. Start FastAPI Server:</strong><br>
                <code style="background-color: #1a1d29; padding: 0.5rem; border-radius: 4px; display: block; margin: 0.5rem 0;">
                uvicorn backend.main:app --reload
                </code></p>
                
                <p><strong style="color: #ffffff;">2. Initialize System:</strong><br>
                Use the sidebar controls to process your documents</p>
                
                <p><strong style="color: #ffffff;">3. Configure Search:</strong><br>
                Adjust the weight slider to balance semantic vs keyword search</p>
                
                <p><strong style="color: #ffffff;">4. Ask Questions:</strong><br>
                Get AI-powered answers with citations from your documents</p>
            </div>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
