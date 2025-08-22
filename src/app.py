import os
import json
import pandas as pd
import streamlit as st
from config.settings import HybridSearchConfig
from Processing_Documents import (
    DocumentProcessor,
    RAGApplication,
)

# ---------- CONFIG ----------
DATA_DIR = os.environ.get(
    "RAG_DATA_DIR",
    "/Users/oferk/Data Tutorials/RAG/RAG Application-Timescale/data",  # <-- change if needed
)

processor = DocumentProcessor(
    DATA_DIR,
    bm25_engine=st.session_state["rag_app"].bm25_engine,
    vector_engine=st.session_state["rag_app"].vector_engine,
)


# 1) Create/return one processor for the whole app lifetime
@st.cache_resource
def get_processor():
    """
    Build and cache the processor object once for the server process.
    Adjust this to however you instantiate your processor.
    """
    # Example:
    # return Processor(settings=get_settings(), ...)
    return Processor()


# 2) Initialize the app exactly once (process PDFs, chunk, embed, upsert, init retrievers)
@st.cache_resource
def ensure_app_initialized():
    """
    Heavy, one-time initialization gate.
    This function is intentionally cached so Streamlit won't re-run it
    on every script re-execution. It must be triggered explicitly by the user.
    """
    processor = get_processor()
    # This is your heavy function that used to run at import time
    processor.run_application()
    # If you want to return handles (bm25, hybrid, etc.), you can return them here.
    # For now we return True as a sentinel.
    return True


# 3) Simple UI gating: user must click the button to initialize
st.title("RAG App")

# Optional banner if not initialized
if "initialized" not in st.session_state:
    st.session_state.initialized = False

# Buttons
col1, col2 = st.columns(2)
with col1:
    init_clicked = st.button("Initialize / Build Index")
with col2:
    rebuild_clicked = st.button("Force Rebuild (clear cache)")

# Force rebuild clears the cached resources and resets the flag
if rebuild_clicked:
    ensure_app_initialized.clear()  # clear the heavy initializer cache
    get_processor.clear()  # clear the processor too if it keeps state/conns
    st.session_state.initialized = False
    st.success(
        "Initialization cache cleared. Click 'Initialize / Build Index' to rebuild."
    )
    st.stop()

# Run initialization only when requested, and only once due to cache
if init_clicked and not st.session_state.initialized:
    with st.spinner(
        "Initializing application: processing PDFs, chunking, embeddings, upserts, and retrievers..."
    ):
        _ = ensure_app_initialized()
    st.session_state.initialized = True
    st.success("Initialization complete.")
    # Stop here so the rest of the script renders cleanly on the next run
    st.stop()

# If not initialized yet, block the rest of the UI
if not st.session_state.initialized:
    st.info(
        "Click **Initialize / Build Index** to process documents and build the index before querying."
    )
    st.stop()

# 4) --- From here down, your app is READY (BM25 + Vector + Hybrid already set up) ---


# Example: build or fetch your hybrid engine/retrievers (these should also be cached)
@st.cache_resource
def get_hybrid_engine():
    """
    Build your BM25 index and HybridSearchEngine once, using the already-processed/ingested data.
    If your processor exposes these, you can fetch them from there instead.
    """
    # Example pattern:
    # processor = get_processor()
    # return processor.get_hybrid_engine()
    return build_hybrid_engine_from_db()  # replace with your function


hybrid = get_hybrid_engine()

# User input & answer
query = st.text_input("Ask a question")
if query:
    with st.spinner("Searching & synthesizing..."):
        ctx_df, response = hybrid.search(query)
    st.subheader("Answer")
    st.write(response.answer)
    st.caption(
        f"Confidence: {response.confidence:.2f}  |  Enough context: {response.enough_context}"
    )
    with st.expander("Citations"):
        st.write(response.citations)
    with st.expander("Context (top-k)"):
        st.dataframe(ctx_df)
