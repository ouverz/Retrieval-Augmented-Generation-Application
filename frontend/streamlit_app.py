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


def query_system(query: str, top_k: int = 8) -> Dict[str, Any]:
    """Query the RAG system"""
    return make_api_request("/query", "POST", {"query": query, "top_k": top_k})


def upload_document(file) -> Dict[str, Any]:
    """Upload a document to the system"""
    # Note: This would need to be implemented differently for file uploads
    # For now, just return a placeholder
    return {"success": False, "error": "File upload not implemented in frontend yet"}


# Streamlit UI
def main():
    st.set_page_config(page_title="RAG System", page_icon="🔍", layout="wide")

    st.title("🔍 RAG System")
    st.markdown("---")

    # Sidebar for system controls
    with st.sidebar:
        st.header("System Controls")

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

        # Query settings
        with st.expander("⚙️ Query Settings"):
            top_k = st.slider(
                "Number of results to retrieve", min_value=1, max_value=20, value=8
            )

        if st.button("🔍 Search", disabled=not query or not query.strip()):
            start_time = time.time()

            with st.spinner("Searching and synthesizing answer..."):
                result = query_system(query, top_k=top_k)

            elapsed_time = time.time() - start_time

            if result["success"]:
                data = result["data"]
                st.success("✅ Query completed successfully!")

                # Display answer
                st.subheader("📝 Answer")
                st.write(data["answer"])

                # Metrics
                col_conf, col_ctx, col_time, col_lat = st.columns(4)
                with col_conf:
                    st.metric("Confidence", f"{data['confidence']:.2f}")
                with col_ctx:
                    context_icon = "✅" if data["enough_context"] else "⚠️"
                    st.metric(
                        "Context",
                        f"{context_icon} {'Yes' if data['enough_context'] else 'No'}",
                    )
                with col_time:
                    st.metric("Query Time", f"{elapsed_time:.2f}s")
                with col_lat:
                    st.metric("API Latency", f"{data.get('latency_ms', 0)}ms")

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

                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Results", total_results)
                        with col2:
                            st.metric("Found by BM25", bm25_found)
                        with col3:
                            st.metric("Found by Vector", vector_found)
                        with col4:
                            st.metric("Found by Both", both_engines)

                # Technical details
                with st.expander("🔬 Technical Details"):
                    st.json(
                        {
                            "precision": data.get("precision", 0),
                            "evidence_precision": data.get(
                                "evidence_precision", "unknown"
                            ),
                            "query_settings": {"top_k": top_k},
                        }
                    )

            else:
                st.error(f"❌ Query failed: {result['error']}")

    with col2:
        st.header("📊 System Status")

        # Quick health check display
        health_placeholder = st.empty()

        if check_health():
            health_placeholder.success("🟢 Server Online")
        else:
            health_placeholder.error("🔴 Server Offline")

        st.markdown("---")

        st.subheader("ℹ️ Instructions")
        st.markdown("""
        1. **Start the FastAPI server**: 
           ```bash
           uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
           ```
        
        2. **Initialize the system** using the sidebar controls
        
        3. **Ask questions** about the documents in your knowledge base
        
        4. **Review results** with confidence scores and citations
        """)


if __name__ == "__main__":
    main()
