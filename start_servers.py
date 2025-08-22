#!/usr/bin/env python3
"""
Development server startup script
Starts both FastAPI backend and Streamlit frontend
"""
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def start_fastapi():
    """Start FastAPI backend server"""
    print("🚀 Starting FastAPI backend on http://localhost:8000")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn", 
        "app.main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"
    ])

def start_streamlit():
    """Start Streamlit frontend"""
    print("🎨 Starting Streamlit frontend on http://localhost:8501")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit", "run", "app.py", 
        "--server.port", "8501", 
        "--server.headless", "true"  # Prevent auto-opening browser
    ])

def main():
    print("🔧 RAG Application Development Servers")
    print("=" * 50)
    
    # Start FastAPI
    fastapi_process = start_fastapi()
    
    # Give FastAPI time to start
    print("⏳ Waiting for FastAPI to start...")
    time.sleep(3)
    
    # Start Streamlit
    streamlit_process = start_streamlit()
    
    # Give Streamlit time to start
    time.sleep(3)
    
    print("\n✅ Servers started!")
    print("📍 FastAPI Backend: http://localhost:8000")
    print("📍 Streamlit Frontend: http://localhost:8501")
    print("📍 API Documentation: http://localhost:8000/docs")
    print("\n💡 Press Ctrl+C to stop both servers")
    
    # Open browser
    try:
        webbrowser.open("http://localhost:8501")
    except:
        pass
    
    try:
        # Wait for processes
        fastapi_process.wait()
        streamlit_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Shutting down servers...")
        fastapi_process.terminate()
        streamlit_process.terminate()
        
        # Wait for graceful shutdown
        fastapi_process.wait(timeout=5)
        streamlit_process.wait(timeout=5)
        print("✅ Servers stopped")

if __name__ == "__main__":
    main()