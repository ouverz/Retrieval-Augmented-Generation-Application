#!/usr/bin/env python3
"""
Main application launcher for the RAG Application
Starts both FastAPI backend and Streamlit frontend
"""

import os
import sys
import subprocess
import time
import signal
from pathlib import Path

def start_backend():
    """Start the FastAPI backend server"""
    print("🚀 Starting FastAPI backend server...")
    return subprocess.Popen([
        sys.executable, "-m", "uvicorn",
        "backend.main:app",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--reload"
    ])

def start_frontend():
    """Start the Streamlit frontend server"""
    print("🎨 Starting Streamlit frontend server...")
    return subprocess.Popen([
        sys.executable, "-m", "streamlit",
        "run", "frontend/streamlit_app.py",
        "--server.port", "8501",
        "--server.address", "0.0.0.0"
    ])

def check_dependencies():
    """Check if required services are running"""
    print("🔍 Checking dependencies...")
    
    # Check if Docker services are running
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if "timescaledb" not in result.stdout:
            print("⚠️  TimescaleDB container not found. Please start Docker services:")
            print("   cd infrastructure/docker && docker-compose up -d")
            return False
        if "rag_redis" not in result.stdout:
            print("⚠️  Redis container not found. Please start Docker services:")
            print("   cd infrastructure/docker && docker-compose up -d")
            return False
    except FileNotFoundError:
        print("⚠️  Docker not found. Please install Docker and start services.")
        return False
    
    print("✅ Dependencies look good!")
    return True

def main():
    """Main application launcher"""
    print("="*60)
    print("🤖 RAG Application Launcher")
    print("="*60)
    
    # Check if we're in the right directory
    if not Path("backend/main.py").exists():
        print("❌ Please run this script from the project root directory")
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Start servers
    backend_process = None
    frontend_process = None
    
    try:
        # Start backend
        backend_process = start_backend()
        time.sleep(3)  # Give backend time to start
        
        # Start frontend  
        frontend_process = start_frontend()
        
        print("\n" + "="*60)
        print("✅ Services started successfully!")
        print("📡 FastAPI Backend: http://localhost:8000")
        print("🖥️  Streamlit Frontend: http://localhost:8501")
        print("📚 API Documentation: http://localhost:8000/docs")
        print("="*60)
        print("Press Ctrl+C to stop all services")
        
        # Wait for processes
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
        if backend_process:
            backend_process.terminate()
            backend_process.wait()
            
        if frontend_process:
            frontend_process.terminate()
            frontend_process.wait()
            
        print("✅ All services stopped.")

if __name__ == "__main__":
    main()