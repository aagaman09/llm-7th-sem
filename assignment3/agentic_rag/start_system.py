#!/usr/bin/env python3
"""
Startup script for the RAG Document Processing System
Runs both FastAPI backend and Streamlit frontend
"""

import subprocess
import sys
import time
import requests
import threading
from loguru import logger

def check_port_available(port):
    """Check if a port is available"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) != 0

def wait_for_api(max_attempts=30):
    """Wait for FastAPI to be ready"""
    for attempt in range(max_attempts):
        try:
            response = requests.get("http://localhost:8000/agents/status", timeout=2)
            if response.status_code == 200:
                logger.success("FastAPI server is ready!")
                return True
        except requests.exceptions.RequestException:
            pass
        
        logger.info(f"Waiting for FastAPI server... ({attempt + 1}/{max_attempts})")
        time.sleep(2)
    
    return False

def run_fastapi():
    """Run FastAPI backend"""
    logger.info("Starting FastAPI backend...")
    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "rag:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"FastAPI failed to start: {e}")
    except KeyboardInterrupt:
        logger.info("FastAPI server stopped by user")

def run_streamlit():
    """Run Streamlit frontend"""
    logger.info("Starting Streamlit frontend...")
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", 
            "run", "streamlit_ui.py", 
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Streamlit failed to start: {e}")
    except KeyboardInterrupt:
        logger.info("Streamlit server stopped by user")

def main():
    """Main startup function"""
    print("🚀 Starting RAG Document Processing System")
    print("=" * 50)
    
    # Check if ports are available
    if not check_port_available(8000):
        logger.error("Port 8000 is already in use. Please stop any existing FastAPI instances.")
        return
    
    if not check_port_available(8501):
        logger.error("Port 8501 is already in use. Please stop any existing Streamlit instances.")
        return
    
    # Start FastAPI in a separate thread
    api_thread = threading.Thread(target=run_fastapi, daemon=True)
    api_thread.start()
    
    # Wait for FastAPI to be ready
    if not wait_for_api():
        logger.error("FastAPI server failed to start within the timeout period")
        return
    
    # Start Streamlit in the main thread
    logger.info("🌟 System is ready!")
    print("\n" + "=" * 50)
    print("📚 RAG Document Processing System")
    print("FastAPI Backend: http://localhost:8000")
    print("Streamlit Frontend: http://localhost:8501")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 50 + "\n")
    
    try:
        run_streamlit()
    except KeyboardInterrupt:
        logger.info("System shutdown initiated")
        print("\n👋 RAG System stopped. Thank you!")

if __name__ == "__main__":
    main()
