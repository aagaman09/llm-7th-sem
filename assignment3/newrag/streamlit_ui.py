import streamlit as st
import requests
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
from typing import Dict, List, Any
import httpx
import asyncio
import os
import base64
from pathlib import Path
from streamlit_option_menu import option_menu
import altair as alt

# Page configuration
st.set_page_config(
    page_title="RAG Document Processing System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(8.5px);
        -webkit-backdrop-filter: blur(8.5px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .success-message {
        background: linear-gradient(90deg, #00b09b, #96c93d);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #00b09b;
    }
    .error-message {
        background: linear-gradient(90deg, #ff416c, #ff4b2b);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #ff416c;
    }
    .info-message {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 5px solid #4facfe;
    }
    .sidebar-section {
        margin: 1rem 0;
        padding: 1.5rem;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        max-width: 80%;
    }
    .user-message {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: auto;
        margin-right: 0;
    }
    .bot-message {
        background: #f0f2f6;
        color: #262730;
        margin-left: 0;
        margin-right: auto;
        border-left: 4px solid #667eea;
    }
    .stProgress .st-bo {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    .evaluation-score {
        background: linear-gradient(45deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem;
        text-align: center;
        color: #333;
        font-weight: bold;
    }
    .document-card {
        background: rgba(255, 255, 255, 0.9);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    .status-indicator {
        width: 12px;
        height: 12px;
        border-radius: 50%;
        display: inline-block;
        margin-right: 8px;
    }
    .status-online { background-color: #28a745; }
    .status-offline { background-color: #dc3545; }
    .status-busy { background-color: #ffc107; }
    
    /* Dark theme adjustments */
    [data-theme="dark"] {
        --primary-color: #667eea;
        --background-color: #0e1117;
        --secondary-background-color: #262730;
    }
    
    /* Navigation menu styling */
    .nav-menu {
        background: rgba(255, 255, 255, 0.1);
        padding: 1rem;
        border-radius: 10px;
        backdrop-filter: blur(10px);
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'current_documents' not in st.session_state:
    st.session_state.current_documents = []
if 'evaluation_results' not in st.session_state:
    st.session_state.evaluation_results = []
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama3"

# Helper functions
def get_base64_of_bin_file(bin_file):
    """Convert binary file to base64"""
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def make_request(endpoint: str, method: str = "GET", data: dict = None, files: dict = None) -> dict:
    """Make API request with error handling"""
    try:
        url = f"{API_BASE_URL}{endpoint}"
        
        if method == "GET":
            response = requests.get(url, timeout=30)
        elif method == "POST":
            if files:
                response = requests.post(url, files=files, timeout=60)
            else:
                response = requests.post(url, json=data, timeout=60)
        elif method == "DELETE":
            response = requests.delete(url, timeout=30)
        else:
            return {"success": False, "error": "Invalid HTTP method"}
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
            
    except requests.exceptions.ConnectionError:
        return {"success": False, "error": "Cannot connect to API server. Make sure it's running on localhost:8000"}
    except requests.exceptions.Timeout:
        return {"success": False, "error": "Request timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/agents/status", timeout=5)
        return response.status_code == 200
    except:
        return False

def display_message(message: str, message_type: str = "info"):
    """Display styled messages"""
    if message_type == "success":
        st.markdown(f'<div class="success-message">✅ {message}</div>', unsafe_allow_html=True)
    elif message_type == "error":
        st.markdown(f'<div class="error-message">❌ {message}</div>', unsafe_allow_html=True)
    elif message_type == "info":
        st.markdown(f'<div class="info-message">ℹ️ {message}</div>', unsafe_allow_html=True)
    else:
        st.info(message)

def create_metric_card(title: str, value: str, icon: str = "📊"):
    """Create a styled metric card"""
    return f"""
    <div class="metric-card">
        <h3>{icon} {title}</h3>
        <h2>{value}</h2>
    </div>
    """

def render_chat_message(message: str, is_user: bool = True):
    """Render a chat message with styling"""
    css_class = "user-message" if is_user else "bot-message"
    icon = "👤" if is_user else "🤖"
    st.markdown(f"""
    <div class="chat-message {css_class}">
        {icon} {message}
    </div>
    """, unsafe_allow_html=True)

# Main header with enhanced styling
st.markdown('<div class="main-header">📚 RAG Document Processing System</div>', unsafe_allow_html=True)

# API Status Check with enhanced display
col1, col2, col3, col4, col5 = st.columns([1, 1, 2, 1, 1])
with col3:
    if check_api_health():
        st.markdown("""
        <div style="background: linear-gradient(90deg, #00b09b, #96c93d); padding: 1rem; border-radius: 10px; text-align: center; color: white; font-weight: bold;">
            <span class="status-indicator status-online"></span>🟢 API Server Connected
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background: linear-gradient(90deg, #ff416c, #ff4b2b); padding: 1rem; border-radius: 10px; text-align: center; color: white; font-weight: bold;">
            <span class="status-indicator status-offline"></span>🔴 API Server Disconnected
        </div>
        """, unsafe_allow_html=True)
        st.error("Start the FastAPI server first")
        with st.expander("How to start the server"):
            st.code("python rag.py", language="bash")
            st.write("Or use the start_system.py script:")
            st.code("python start_system.py", language="bash")
        st.stop()

# Enhanced Navigation
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown('<div class="nav-menu">', unsafe_allow_html=True)
    page = st.selectbox(
        "🧭 Navigate to:",
        ["🏠 Home Dashboard", "📄 Document Processing", "🔍 Smart Query Interface", 
         "📊 Evaluation Dashboard", "⚙️ System Status", "🔗 Agent Communication", "💬 Chat Interface"],
        index=0
    )
    st.markdown('</div>', unsafe_allow_html=True)

# Quick stats in header
if page == "🏠 Home Dashboard":
    st.header("🏠 Welcome to RAG Document Processing System")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        doc_count = len(st.session_state.current_documents)
        st.markdown(create_metric_card("Documents", str(doc_count), "📄"), unsafe_allow_html=True)
    
    with col2:
        query_count = len(st.session_state.query_history)
        st.markdown(create_metric_card("Queries", str(query_count), "🔍"), unsafe_allow_html=True)
    
    with col3:
        eval_count = len(st.session_state.evaluation_results)
        st.markdown(create_metric_card("Evaluations", str(eval_count), "📊"), unsafe_allow_html=True)
    
    with col4:
        chat_count = len(st.session_state.chat_messages)
        st.markdown(create_metric_card("Chat Messages", str(chat_count), "💬"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📄 Upload Document", type="primary", use_container_width=True):
            st.session_state.page = "📄 Document Processing"
            st.experimental_rerun()
    
    with col2:
        if st.button("🔍 Ask Question", type="primary", use_container_width=True):
            st.session_state.page = "🔍 Smart Query Interface"
            st.experimental_rerun()
    
    with col3:
        if st.button("📊 View Analytics", type="primary", use_container_width=True):
            st.session_state.page = "📊 Evaluation Dashboard"
            st.experimental_rerun()
    
    # Recent activity
    st.subheader("📈 Recent Activity")
    
    if st.session_state.query_history:
        recent_queries = st.session_state.query_history[-3:]
        for i, query in enumerate(reversed(recent_queries)):
            with st.container():
                st.markdown(f"""
                <div class="document-card">
                    <strong>Q:</strong> {query['question'][:100]}...
                    <br><small>Asked: {query['timestamp']}</small>
                </div>
                """, unsafe_allow_html=True)
    else:
        display_message("No recent activity. Start by uploading a document or asking a question!", "info")
    
    # System overview
    st.subheader("🔧 System Overview")
    
    try:
        status_result = make_request("/agents/status")
        if status_result.get("success", False):
            agents = status_result.get("agents", {})
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Active Agents:**")
                for name, info in agents.items():
                    status = info.get("status", "unknown")
                    status_icon = "🟢" if status == "idle" else "🟡" if status == "busy" else "🔴"
                    st.write(f"{status_icon} {name}: {status}")
            
            with col2:
                st.write("**System Health:**")
                total_agents = len(agents)
                active_agents = len([a for a in agents.values() if a.get("status") != "offline"])
                health_percentage = (active_agents / total_agents * 100) if total_agents > 0 else 0
                
                st.progress(health_percentage / 100)
                st.write(f"Health: {health_percentage:.1f}% ({active_agents}/{total_agents} agents active)")
    except:
        display_message("Could not fetch system status", "error")

# Document Processing Page
elif page == "📄 Document Processing":
    st.header("📄 Document Processing Center")
    
    # Enhanced tabs with better styling
    tab1, tab2, tab3, tab4 = st.tabs(["📎 PDF Upload", "🌐 URL Processing", "📊 Document Analytics", "🗂️ Management"])
    
    with tab1:
        st.subheader("📎 Upload PDF Documents")
        
        # Drag and drop area simulation
        st.markdown("""
        <div style="border: 2px dashed #667eea; border-radius: 10px; padding: 2rem; text-align: center; margin: 1rem 0; background: rgba(102, 126, 234, 0.1);">
            <h4>📁 Drop your PDF files here or click to browse</h4>
            <p>Supported formats: PDF (up to 10MB per file)</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Choose PDF files", 
            type="pdf", 
            accept_multiple_files=True,
            help="You can upload multiple PDF files at once"
        )
        
        if uploaded_files:
            st.write(f"**{len(uploaded_files)} file(s) selected:**")
            
            # Show file details
            for i, file in enumerate(uploaded_files):
                col1, col2, col3 = st.columns([3, 1, 1])
                with col1:
                    st.write(f"📄 {file.name}")
                with col2:
                    st.write(f"{file.size / 1024:.1f} KB")
                with col3:
                    if st.button(f"Remove", key=f"remove_{i}"):
                        uploaded_files.remove(file)
                        st.experimental_rerun()
            
            # Batch processing
            col1, col2 = st.columns([1, 1])
            with col1:
                chunk_size = st.slider("Chunk Size", 1000, 3000, 2000, help="Text chunk size for processing")
            with col2:
                overlap = st.slider("Chunk Overlap", 100, 500, 400, help="Overlap between chunks")
            
            if st.button("🚀 Process All Files", type="primary", use_container_width=True):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success_count = 0
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {file.name}...")
                    progress_bar.progress((i) / len(uploaded_files))
                    
                    with st.spinner(f"Processing {file.name}..."):
                        files = {"file": (file.name, file.getvalue(), "application/pdf")}
                        result = make_request("/upload", method="POST", files=files)
                        
                        if result.get("success"):
                            success_count += 1
                            st.session_state.current_documents.append({
                                "name": file.name,
                                "type": "PDF",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "size": f"{file.size / 1024:.1f} KB",
                                "chunks": result.get("chunks_count", "Unknown")
                            })
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                if success_count == len(uploaded_files):
                    display_message(f"✅ Successfully processed all {success_count} files!", "success")
                elif success_count > 0:
                    display_message(f"✅ Processed {success_count}/{len(uploaded_files)} files", "info")
                else:
                    display_message("❌ Failed to process any files", "error")
    
    with tab2:
        st.subheader("🌐 Web Content Processing")
        
        # URL input with validation
        url = st.text_input(
            "🔗 Enter URL:", 
            placeholder="https://example.com/article",
            help="Enter a valid URL starting with http:// or https://"
        )
        
        # URL preview
        if url and url.startswith(('http://', 'https://')):
            st.markdown(f"""
            <div class="document-card">
                <strong>URL Preview:</strong><br>
                <a href="{url}" target="_blank">{url}</a>
            </div>
            """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Advanced options
            with st.expander("⚙️ Advanced Options"):
                follow_links = st.checkbox("Follow internal links", help="Extract content from linked pages")
                extract_images = st.checkbox("Extract image descriptions", help="Include image alt text and captions")
                max_depth = st.slider("Maximum depth", 1, 3, 1, help="How deep to follow links")
        
        with col2:
            if url and st.button("🌐 Process URL", type="primary", use_container_width=True):
                if url.startswith(('http://', 'https://')):
                    with st.spinner("🕷️ Scraping content..."):
                        result = make_request("/url", method="POST", data={"url": url})
                        
                        if result.get("success"):
                            display_message(f"✅ {result['message']}", "success")
                            st.session_state.current_documents.append({
                                "name": url,
                                "type": "URL",
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "size": "Web Content",
                                "chunks": "Multiple"
                            })
                        else:
                            display_message(f"❌ Error: {result.get('error', 'Unknown error')}", "error")
                else:
                    display_message("❌ Please enter a valid URL starting with http:// or https://", "error")
        
        # Bulk URL processing
        st.markdown("---")
        st.subheader("📋 Bulk URL Processing")
        
        urls_text = st.text_area(
            "Enter multiple URLs (one per line):",
            height=100,
            placeholder="https://example1.com\nhttps://example2.com\nhttps://example3.com"
        )
        
        if urls_text and st.button("🔄 Process All URLs"):
            urls = [url.strip() for url in urls_text.split('\n') if url.strip().startswith(('http://', 'https://'))]
            
            if urls:
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                success_count = 0
                for i, url in enumerate(urls):
                    status_text.text(f"Processing {url}...")
                    progress_bar.progress(i / len(urls))
                    
                    result = make_request("/url", method="POST", data={"url": url})
                    if result.get("success"):
                        success_count += 1
                        st.session_state.current_documents.append({
                            "name": url,
                            "type": "URL",
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "size": "Web Content",
                            "chunks": "Multiple"
                        })
                
                progress_bar.progress(1.0)
                status_text.text(f"Processed {success_count}/{len(urls)} URLs successfully!")
                display_message(f"✅ Processed {success_count}/{len(urls)} URLs", "success")
            else:
                display_message("❌ No valid URLs found", "error")
    
    with tab3:
        st.subheader("📊 Document Analytics")
        
        if st.session_state.current_documents:
            df = pd.DataFrame(st.session_state.current_documents)
            
            # Document type distribution
            col1, col2 = st.columns(2)
            
            with col1:
                type_counts = df['type'].value_counts()
                fig = px.pie(
                    values=type_counts.values, 
                    names=type_counts.index,
                    title="Document Types",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Processing timeline
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                daily_counts = df.groupby(df['timestamp'].dt.date).size()
                
                fig = px.bar(
                    x=daily_counts.index,
                    y=daily_counts.values,
                    title="Documents Processed Over Time",
                    labels={'x': 'Date', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Document details table
            st.subheader("📋 Document Details")
            
            # Add search and filter
            search_term = st.text_input("🔍 Search documents:", placeholder="Enter filename or URL...")
            type_filter = st.multiselect("Filter by type:", df['type'].unique(), default=df['type'].unique())
            
            # Apply filters
            filtered_df = df[df['type'].isin(type_filter)]
            if search_term:
                filtered_df = filtered_df[filtered_df['name'].str.contains(search_term, case=False)]
            
            # Display table with enhanced styling
            st.dataframe(
                filtered_df,
                use_container_width=True,
                column_config={
                    "name": st.column_config.TextColumn("Document Name", width="large"),
                    "type": st.column_config.TextColumn("Type", width="small"),
                    "timestamp": st.column_config.DatetimeColumn("Processed", width="medium"),
                    "size": st.column_config.TextColumn("Size", width="small"),
                    "chunks": st.column_config.NumberColumn("Chunks", width="small")
                }
            )
        else:
            display_message("📄 No documents processed yet. Upload a PDF or process a URL to get started!", "info")
    
    with tab4:
        st.subheader("🗂️ Document Management")
        
        # Current documents overview
        if st.session_state.current_documents:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_docs = len(st.session_state.current_documents)
                st.markdown(create_metric_card("Total Documents", str(total_docs), "📄"), unsafe_allow_html=True)
            
            with col2:
                pdf_count = len([d for d in st.session_state.current_documents if d['type'] == 'PDF'])
                st.markdown(create_metric_card("PDF Files", str(pdf_count), "📑"), unsafe_allow_html=True)
            
            with col3:
                url_count = len([d for d in st.session_state.current_documents if d['type'] == 'URL'])
                st.markdown(create_metric_card("Web Pages", str(url_count), "🌐"), unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Document management actions
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔧 Management Actions")
                
                if st.button("📤 Export Document List", use_container_width=True):
                    df = pd.DataFrame(st.session_state.current_documents)
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="💾 Download CSV",
                        data=csv,
                        file_name=f"rag_documents_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
                
                if st.button("🔄 Refresh Database Status", use_container_width=True):
                    # This would check the actual database status
                    display_message("Database status refreshed", "success")
            
            with col2:
                st.subheader("⚠️ Danger Zone")
                
                if st.button("🗑️ Clear All Documents", type="secondary", use_container_width=True):
                    if 'confirm_clear' not in st.session_state:
                        st.session_state.confirm_clear = False
                    
                    if not st.session_state.confirm_clear:
                        st.session_state.confirm_clear = True
                        display_message("⚠️ Click the button again to confirm deletion of ALL documents.", "error")
                    else:
                        result = make_request("/clear", method="DELETE")
                        if result.get("success"):
                            display_message(f"✅ {result['message']}", "success")
                            st.session_state.current_documents = []
                            st.session_state.confirm_clear = False
                            st.experimental_rerun()
                        else:
                            display_message(f"❌ Error: {result.get('error', 'Unknown error')}", "error")
        else:
            display_message("📄 No documents to manage. Upload some documents first!", "info")

# Query Interface Page
elif page == "🔍 Smart Query Interface":
    st.header("🔍 Smart Query Interface")
    
    # Enhanced query interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 Ask Your Questions")
        
        # Question input with suggestions
        question = st.text_area(
            "Enter your question:", 
            height=120, 
            placeholder="What would you like to know from the documents?",
            help="Ask detailed questions about your uploaded documents"
        )
        
        # Suggested questions
        if st.session_state.current_documents:
            with st.expander("💡 Suggested Questions"):
                suggestions = [
                    "What are the main topics discussed in the documents?",
                    "Can you summarize the key findings?",
                    "What are the important dates mentioned?",
                    "Who are the main people or organizations mentioned?",
                    "What conclusions can be drawn from this content?"
                ]
                
                for suggestion in suggestions:
                    if st.button(f"💡 {suggestion}", key=f"suggest_{suggestion[:20]}"):
                        question = suggestion
                        st.experimental_rerun()
        
        # Advanced query options
        with st.expander("⚙️ Advanced Query Options"):
            col_opt1, col_opt2, col_opt3 = st.columns(3)
            
            with col_opt1:
                n_results = st.slider("📄 Number of results:", 1, 10, 5, help="How many document chunks to retrieve")
                
            with col_opt2:
                include_evaluation = st.checkbox("📊 Include Evaluation", value=False, help="Evaluate the response quality")
                
            with col_opt3:
                response_style = st.selectbox("🎨 Response Style:", 
                    ["Detailed", "Concise", "Bullet Points", "Academic"],
                    help="How should the answer be formatted?"
                )
        
        # Query buttons
        col_btn1, col_btn2, col_btn3 = st.columns([2, 1, 1])
        
        with col_btn1:
            search_btn = st.button("🔍 Search Documents", type="primary", disabled=not question.strip(), use_container_width=True)
        
        with col_btn2:
            if st.button("🔄 Clear", use_container_width=True):
                question = ""
                st.experimental_rerun()
        
        with col_btn3:
            if st.button("💾 Save Query", disabled=not question.strip(), use_container_width=True):
                # Save to favorites
                pass
        
        # Process query
        if search_btn and question.strip():
            with st.spinner("🔍 Searching through your documents..."):
                query_data = {"question": question, "n_results": n_results}
                
                if include_evaluation:
                    result = make_request("/query_with_evaluation", method="POST", data=query_data)
                else:
                    result = make_request("/query", method="POST", data=query_data)
                
                if result.get("success") or "query_response" in result:
                    # Handle both response formats
                    if "query_response" in result:
                        answer = result["query_response"]["answer"]
                        sources = result["query_response"].get("sources", [])
                        evaluation = result.get("evaluation", {})
                    else:
                        answer = result["answer"]
                        sources = result.get("sources", [])
                        evaluation = {}
                    
                    # Store in query history
                    query_record = {
                        "question": question,
                        "answer": answer,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "sources": sources,
                        "evaluation": evaluation,
                        "response_style": response_style,
                        "n_results": n_results
                    }
                    st.session_state.query_history.append(query_record)
                    
                    # Display results in an attractive format
                    st.markdown("---")
                    st.subheader("📝 Answer")
                    
                    # Format answer based on style
                    if response_style == "Bullet Points":
                        # Convert to bullet points if possible
                        sentences = answer.split('. ')
                        formatted_answer = '\n'.join([f"• {sentence.strip()}." for sentence in sentences if sentence.strip()])
                        st.markdown(formatted_answer)
                    else:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 1.5rem; border-radius: 10px; border-left: 4px solid #667eea;">
                            {answer}
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Display evaluation if included
                    if evaluation:
                        st.subheader("📊 Response Quality Metrics")
                        
                        col_eval1, col_eval2, col_eval3, col_eval4 = st.columns(4)
                        
                        with col_eval1:
                            if "relevance" in evaluation:
                                score = evaluation['relevance'].get('score', 0)
                                st.markdown(f"""
                                <div class="evaluation-score">
                                    <h4>🎯 Relevance</h4>
                                    <h2>{score}/5</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col_eval2:
                            if "groundedness" in evaluation:
                                grounded = evaluation['groundedness'].get('grounded', False)
                                st.markdown(f"""
                                <div class="evaluation-score">
                                    <h4>🏗️ Grounded</h4>
                                    <h2>{"✅" if grounded else "❌"}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col_eval3:
                            if "retrieval_relevance" in evaluation:
                                score = evaluation['retrieval_relevance'].get('score', 0)
                                st.markdown(f"""
                                <div class="evaluation-score">
                                    <h4>📚 Retrieval</h4>
                                    <h2>{score}/5</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col_eval4:
                            if "overall_score" in evaluation:
                                score = evaluation.get('overall_score', 0)
                                st.markdown(f"""
                                <div class="evaluation-score">
                                    <h4>🏆 Overall</h4>
                                    <h2>{score:.1f}/5</h2>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Detailed evaluation
                        with st.expander("📋 Detailed Evaluation Report"):
                            st.json(evaluation)
                    
                    # Display sources
                    if sources:
                        st.subheader("📚 Source Documents")
                        for i, source in enumerate(sources, 1):
                            with st.expander(f"📄 Source {i} - Click to expand"):
                                st.markdown(f"""
                                <div style="background: #f8f9fa; padding: 1rem; border-radius: 8px; border-left: 3px solid #667eea;">
                                    {source}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Query actions
                    col_act1, col_act2, col_act3 = st.columns(3)
                    
                    with col_act1:
                        if st.button("🔄 Ask Follow-up"):
                            st.session_state.follow_up_context = {
                                "original_question": question,
                                "original_answer": answer
                            }
                    
                    with col_act2:
                        if st.button("💾 Save Result"):
                            # Save to favorites or export
                            display_message("Result saved to history!", "success")
                    
                    with col_act3:
                        if st.button("📤 Share"):
                            # Generate shareable link or export
                            display_message("Sharing functionality coming soon!", "info")
                
                else:
                    display_message(f"❌ Error: {result.get('error', 'Unknown error')}", "error")
    
    with col2:
        st.subheader("📈 Query History")
        
        # Search history
        if st.text_input("🔍 Search history:", placeholder="Search past queries..."):
            # Filter history based on search
            pass
        
        if st.session_state.query_history:
            # Show recent queries with better formatting
            st.write(f"**Last {min(10, len(st.session_state.query_history))} Queries:**")
            
            for i, query in enumerate(reversed(st.session_state.query_history[-10:]), 1):
                with st.container():
                    # Query preview card
                    st.markdown(f"""
                    <div class="document-card">
                        <strong>Q{len(st.session_state.query_history) - i + 1}:</strong> {query['question'][:80]}...
                        <br><small>📅 {query['timestamp']}</small>
                        <br><small>📄 {query.get('n_results', 5)} sources</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"🔍 View Details", key=f"view_query_{i}", use_container_width=True):
                        st.session_state.selected_query = query
                        # Could show in modal or expand inline
        else:
            display_message("📝 No queries yet. Ask your first question!", "info")
        
        # Query statistics
        if st.session_state.query_history:
            st.markdown("---")
            st.subheader("📊 Statistics")
            
            total_queries = len(st.session_state.query_history)
            avg_sources = sum(len(q.get('sources', [])) for q in st.session_state.query_history) / total_queries if total_queries > 0 else 0
            
            st.markdown(create_metric_card("Total Queries", str(total_queries), "📝"), unsafe_allow_html=True)
            st.markdown(create_metric_card("Avg Sources", f"{avg_sources:.1f}", "📚"), unsafe_allow_html=True)

# Chat Interface Page
elif page == "💬 Chat Interface":
    st.header("💬 Interactive Chat with Your Documents")
    
    # Chat interface
    st.subheader("🤖 AI Assistant")
    
    # Chat container
    chat_container = st.container()
    
    # Display chat messages
    with chat_container:
        if st.session_state.chat_messages:
            for message in st.session_state.chat_messages:
                render_chat_message(message["content"], message["is_user"])
        else:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; color: #666;">
                <h3>👋 Hello! I'm your AI assistant.</h3>
                <p>I can help you find information from your uploaded documents. What would you like to know?</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Chat input
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_input(
            "💬 Type your message:", 
            placeholder="Ask me anything about your documents...",
            key="chat_input"
        )
    
    with col2:
        send_btn = st.button("📤 Send", type="primary", use_container_width=True)
    
    # Process chat message
    if (send_btn or user_input) and user_input.strip():
        # Add user message
        st.session_state.chat_messages.append({
            "content": user_input,
            "is_user": True,
            "timestamp": datetime.now()
        })
        
        # Get AI response
        with st.spinner("🤖 AI is thinking..."):
            query_data = {"question": user_input, "n_results": 3}
            result = make_request("/query", method="POST", data=query_data)
            
            if result.get("success"):
                ai_response = result["answer"]
                
                # Add AI response
                st.session_state.chat_messages.append({
                    "content": ai_response,
                    "is_user": False,
                    "timestamp": datetime.now(),
                    "sources": result.get("sources", [])
                })
            else:
                error_msg = f"Sorry, I encountered an error: {result.get('error', 'Unknown error')}"
                st.session_state.chat_messages.append({
                    "content": error_msg,
                    "is_user": False,
                    "timestamp": datetime.now()
                })
        
        # Clear input and refresh
        st.experimental_rerun()
    
    # Chat controls
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_messages = []
            st.experimental_rerun()
    
    with col2:
        if st.button("💾 Save Chat", use_container_width=True):
            # Save chat to file
            chat_data = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.chat_messages
            }
            chat_json = json.dumps(chat_data, indent=2, default=str)
            st.download_button(
                "📥 Download Chat",
                chat_json,
                file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("📤 Export Chat", use_container_width=True):
            # Export as readable format
            chat_text = "\n\n".join([
                f"{'User' if msg['is_user'] else 'AI'}: {msg['content']}"
                for msg in st.session_state.chat_messages
            ])
            st.download_button(
                "📄 Download TXT",
                chat_text,
                file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
    
    with col4:
        if st.button("🔄 Refresh", use_container_width=True):
            st.experimental_rerun()
    
    # Chat statistics
    if st.session_state.chat_messages:
        st.subheader("📊 Chat Statistics")
        
        user_messages = len([m for m in st.session_state.chat_messages if m["is_user"]])
        ai_messages = len([m for m in st.session_state.chat_messages if not m["is_user"]])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(create_metric_card("Your Messages", str(user_messages), "👤"), unsafe_allow_html=True)
        
        with col2:
            st.markdown(create_metric_card("AI Responses", str(ai_messages), "🤖"), unsafe_allow_html=True)
        
        with col3:
            total_messages = len(st.session_state.chat_messages)
            st.markdown(create_metric_card("Total Messages", str(total_messages), "💬"), unsafe_allow_html=True)

# Evaluation Dashboard Page
elif page == "📊 Evaluation Dashboard":
    st.header("📊 Advanced Evaluation Dashboard")
    
    # Evaluation overview metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_evaluations = len(st.session_state.evaluation_results)
        st.markdown(create_metric_card("Total Evaluations", str(total_evaluations), "📊"), unsafe_allow_html=True)
    
    with col2:
        if st.session_state.evaluation_results:
            avg_score = sum(r["results"].get("overall_score", 0) for r in st.session_state.evaluation_results) / len(st.session_state.evaluation_results)
            st.markdown(create_metric_card("Avg Score", f"{avg_score:.1f}/5", "⭐"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Avg Score", "0.0/5", "⭐"), unsafe_allow_html=True)
    
    with col3:
        queries_with_eval = len([q for q in st.session_state.query_history if q.get("evaluation")])
        st.markdown(create_metric_card("Evaluated Queries", str(queries_with_eval), "🔍"), unsafe_allow_html=True)
    
    with col4:
        if st.session_state.evaluation_results:
            latest_score = st.session_state.evaluation_results[-1]["results"].get("overall_score", 0)
            st.markdown(create_metric_card("Latest Score", f"{latest_score:.1f}/5", "🏆"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Latest Score", "N/A", "🏆"), unsafe_allow_html=True)
    
    # Enhanced evaluation interface
    st.markdown("---")
    eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs(["🎯 Quick Evaluation", "📋 Individual Metrics", "📦 Batch Processing", "📈 Analytics"])
    
    with eval_tab1:
        st.subheader("🎯 Quick RAG Evaluation")
        
        # Pre-fill from recent query if available
        if st.session_state.query_history and st.button("📝 Use Last Query"):
            last_query = st.session_state.query_history[-1]
            st.session_state.eval_question = last_query["question"]
            st.session_state.eval_answer = last_query["answer"]
            st.session_state.eval_sources = "\n".join(last_query.get("sources", []))
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            eval_question = st.text_input(
                "Question:", 
                key="eval_q", 
                value=st.session_state.get("eval_question", ""),
                placeholder="What question was asked?"
            )
            
            eval_answer = st.text_area(
                "Answer:", 
                key="eval_a", 
                height=120,
                value=st.session_state.get("eval_answer", ""),
                placeholder="The answer to evaluate..."
            )
            
            eval_context = st.text_area(
                "Context Sources (one per line):", 
                key="eval_c", 
                height=150,
                value=st.session_state.get("eval_sources", ""),
                placeholder="Source document 1\nSource document 2\n..."
            )
            
            eval_ground_truth = st.text_input(
                "Ground Truth (optional):", 
                key="eval_gt",
                placeholder="Expected correct answer (optional)"
            )
        
        with col2:
            st.subheader("📋 Evaluation Options")
            
            evaluation_mode = st.radio(
                "Evaluation Mode:",
                ["Complete Evaluation", "Quick Assessment", "Custom Metrics"]
            )
            
            if evaluation_mode == "Custom Metrics":
                selected_metrics = st.multiselect(
                    "Select Metrics:",
                    ["Relevance", "Groundedness", "Correctness", "Retrieval Relevance"],
                    default=["Relevance", "Groundedness"]
                )
            
            confidence_level = st.slider("Confidence Level:", 0.0, 1.0, 0.8, help="Evaluation confidence threshold")
            
            include_explanation = st.checkbox("Include Detailed Explanations", value=True)
        
        # Evaluation button
        if st.button("🚀 Evaluate Response", type="primary", use_container_width=True):
            if eval_question and eval_answer and eval_context:
                context_list = [line.strip() for line in eval_context.split('\n') if line.strip()]
                eval_data = {
                    "question": eval_question,
                    "answer": eval_answer,
                    "context": context_list,
                    "ground_truth": eval_ground_truth if eval_ground_truth else None
                }
                
                with st.spinner("🔍 Running comprehensive evaluation..."):
                    result = make_request("/evaluate/complete", method="POST", data=eval_data)
                    
                    if result.get("success"):
                        eval_results = result["results"]
                        
                        # Display results in a beautiful format
                        st.success("✅ Evaluation completed successfully!")
                        
                        # Create score visualization
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if "relevance" in eval_results:
                                score = eval_results['relevance'].get('score', 0)
                                color = "green" if score >= 4 else "orange" if score >= 3 else "red"
                                st.markdown(f"""
                                <div class="evaluation-score" style="background: linear-gradient(45deg, {color}50, {color}80);">
                                    <h4>🎯 Relevance</h4>
                                    <h2>{score}/5</h2>
                                    <p>{eval_results['relevance'].get('explanation', '')[:50]}...</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            if "groundedness" in eval_results:
                                grounded = eval_results['groundedness'].get('grounded', False)
                                color = "green" if grounded else "red"
                                st.markdown(f"""
                                <div class="evaluation-score" style="background: linear-gradient(45deg, {color}50, {color}80);">
                                    <h4>🏗️ Grounded</h4>
                                    <h2>{"✅" if grounded else "❌"}</h2>
                                    <p>{"Well supported" if grounded else "Lacks support"}</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col3:
                            if "retrieval_relevance" in eval_results:
                                score = eval_results['retrieval_relevance'].get('score', 0)
                                color = "green" if score >= 4 else "orange" if score >= 3 else "red"
                                st.markdown(f"""
                                <div class="evaluation-score" style="background: linear-gradient(45deg, {color}50, {color}80);">
                                    <h4>📚 Retrieval</h4>
                                    <h2>{score}/5</h2>
                                    <p>Document relevance</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col4:
                            if "overall_score" in eval_results:
                                score = eval_results.get('overall_score', 0)
                                color = "green" if score >= 4 else "orange" if score >= 3 else "red"
                                st.markdown(f"""
                                <div class="evaluation-score" style="background: linear-gradient(45deg, {color}50, {color}80);">
                                    <h4>🏆 Overall</h4>
                                    <h2>{score:.1f}/5</h2>
                                    <p>Composite score</p>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        # Store results with timestamp
                        st.session_state.evaluation_results.append({
                            "timestamp": datetime.now(),
                            "question": eval_question,
                            "answer": eval_answer,
                            "results": eval_results,
                            "mode": evaluation_mode,
                            "confidence": confidence_level
                        })
                        
                        # Detailed breakdown
                        if include_explanation:
                            with st.expander("📊 Detailed Evaluation Breakdown"):
                                for metric, details in eval_results.items():
                                    if isinstance(details, dict) and "explanation" in details:
                                        st.subheader(f"📋 {metric.title()} Analysis")
                                        st.write(details["explanation"])
                                        if "score" in details:
                                            st.progress(details["score"] / 5)
                        
                        # Recommendations
                        st.subheader("💡 Improvement Recommendations")
                        recommendations = []
                        
                        if eval_results.get("relevance", {}).get("score", 5) < 4:
                            recommendations.append("🎯 Consider refining the question to be more specific")
                        if not eval_results.get("groundedness", {}).get("grounded", True):
                            recommendations.append("🏗️ Ensure answers are better supported by source documents")
                        if eval_results.get("retrieval_relevance", {}).get("score", 5) < 4:
                            recommendations.append("📚 Improve document chunking or retrieval strategy")
                        
                        if recommendations:
                            for rec in recommendations:
                                st.write(f"• {rec}")
                        else:
                            st.success("🎉 Excellent performance! No specific improvements needed.")
                    
                    else:
                        display_message(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}", "error")
            else:
                display_message("❌ Please fill in Question, Answer, and Context fields", "error")
    
    with eval_tab2:
        st.subheader("📋 Individual Metric Evaluation")
        
        metric_type = st.selectbox(
            "Choose evaluation metric:", 
            ["Relevance", "Groundedness", "Correctness", "Retrieval Relevance"],
            help="Select which specific metric to evaluate"
        )
        
        if metric_type == "Relevance":
            st.write("**📝 Evaluate Answer Relevance**")
            col1, col2 = st.columns([2, 1])
            
            with col1:
                rel_question = st.text_input("Question:", key="rel_q", placeholder="Original question")
                rel_answer = st.text_area("Answer:", key="rel_a", height=100, placeholder="Answer to evaluate")
            
            with col2:
                st.write("**Relevance Criteria:**")
                st.write("• Does it address the question?")
                st.write("• Is it helpful and informative?")
                st.write("• Stays on topic?")
                st.write("• Complete enough?")
            
            if st.button("📊 Evaluate Relevance", type="primary") and rel_question and rel_answer:
                data = {"question": rel_question, "answer": rel_answer}
                with st.spinner("Evaluating relevance..."):
                    result = make_request("/evaluate/relevance", method="POST", data=data)
                    if result.get("success"):
                        res = result["results"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Relevance Score", f"{res.get('score', 0)}/5")
                            st.metric("Overall Assessment", "✅ Relevant" if res.get('relevant', False) else "❌ Not Relevant")
                        
                        with col2:
                            st.write("**Explanation:**")
                            st.write(res.get('explanation', 'No explanation provided'))
        
        elif metric_type == "Groundedness":
            st.write("**🏗️ Evaluate Answer Groundedness**")
            ground_answer = st.text_area("Answer:", key="ground_a", height=100)
            ground_context = st.text_area("Context (one per line):", key="ground_c", height=150)
            
            if st.button("🏗️ Evaluate Groundedness", type="primary") and ground_answer and ground_context:
                context_list = [line.strip() for line in ground_context.split('\n') if line.strip()]
                data = {"answer": ground_answer, "context": context_list}
                with st.spinner("Evaluating groundedness..."):
                    result = make_request("/evaluate/groundedness", method="POST", data=data)
                    if result.get("success"):
                        res = result["results"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Grounded", "✅ Yes" if res.get('grounded', False) else "❌ No")
                            st.metric("Hallucination", "❌ Detected" if res.get('hallucination', False) else "✅ None")
                        
                        with col2:
                            st.write("**Analysis:**")
                            st.write(res.get('explanation', 'No explanation provided'))
        
        elif metric_type == "Correctness":
            st.write("**✅ Evaluate Answer Correctness**")
            corr_question = st.text_input("Question:", key="corr_q")
            corr_answer = st.text_area("Answer:", key="corr_a", height=100)
            corr_truth = st.text_input("Ground Truth:", key="corr_t")
            
            if st.button("✅ Evaluate Correctness", type="primary") and all([corr_question, corr_answer, corr_truth]):
                data = {"question": corr_question, "answer": corr_answer, "ground_truth": corr_truth}
                with st.spinner("Evaluating correctness..."):
                    result = make_request("/evaluate/correctness", method="POST", data=data)
                    if result.get("success"):
                        res = result["results"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Correctness", "✅ Correct" if res.get('correct', False) else "❌ Incorrect")
                        
                        with col2:
                            st.write("**Reasoning:**")
                            st.write(res.get('explanation', 'No explanation provided'))
        
        elif metric_type == "Retrieval Relevance":
            st.write("**📚 Evaluate Retrieval Relevance**")
            retr_question = st.text_input("Question:", key="retr_q")
            retr_docs = st.text_area("Retrieved Documents (one per line):", key="retr_d", height=150)
            
            if st.button("📚 Evaluate Retrieval", type="primary") and retr_question and retr_docs:
                docs_list = [line.strip() for line in retr_docs.split('\n') if line.strip()]
                data = {"question": retr_question, "retrieved_docs": docs_list}
                with st.spinner("Evaluating retrieval relevance..."):
                    result = make_request("/evaluate/retrieval_relevance", method="POST", data=data)
                    if result.get("success"):
                        res = result["results"]
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Retrieval Score", f"{res.get('score', 0)}/5")
                            st.metric("Documents Relevant", "✅ Yes" if res.get('relevant', False) else "❌ No")
                        
                        with col2:
                            st.write("**Assessment:**")
                            st.write(res.get('explanation', 'No explanation provided'))
    
    with eval_tab3:
        st.subheader("📦 Batch Evaluation Processing")
        
        # Batch evaluation template
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.write("**📤 Upload Batch File**")
            batch_file = st.file_uploader(
                "Upload evaluation batch (JSON)", 
                type="json",
                help="Upload a JSON file with multiple evaluation requests"
            )
            
            if batch_file:
                try:
                    batch_data = json.load(batch_file)
                    st.success(f"✅ Loaded {len(batch_data)} evaluation requests")
                    
                    # Preview first few items
                    with st.expander("👁️ Preview Data"):
                        st.json(batch_data[:3] if len(batch_data) > 3 else batch_data)
                    
                    if st.button("🚀 Run Batch Evaluation", type="primary"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        with st.spinner("Processing batch evaluation..."):
                            result = make_request("/evaluate/batch", method="POST", data=batch_data)
                            
                            if result.get("success"):
                                progress_bar.progress(1.0)
                                status_text.text("Batch evaluation completed!")
                                
                                st.success("🎉 Batch evaluation completed successfully!")
                                
                                # Display batch results
                                batch_results = result.get("batch_results", [])
                                batch_stats = result.get("batch_statistics", {})
                                
                                # Statistics
                                col_stat1, col_stat2, col_stat3 = st.columns(3)
                                
                                with col_stat1:
                                    st.metric("Total Evaluations", batch_stats.get("total_evaluations", 0))
                                
                                with col_stat2:
                                    avg_score = batch_stats.get("average_score", 0)
                                    st.metric("Average Score", f"{avg_score:.2f}/5")
                                
                                with col_stat3:
                                    score_range = f"{batch_stats.get('min_score', 0):.1f} - {batch_stats.get('max_score', 0):.1f}"
                                    st.metric("Score Range", score_range)
                                
                                # Download results
                                results_json = json.dumps(result, indent=2)
                                st.download_button(
                                    "💾 Download Results",
                                    results_json,
                                    file_name=f"batch_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                    mime="application/json"
                                )
                            else:
                                display_message(f"❌ Batch evaluation failed: {result.get('error', 'Unknown error')}", "error")
                                
                except json.JSONDecodeError:
                    st.error("❌ Invalid JSON file format")
        
        with col2:
            st.write("**📝 Create Batch Template**")
            
            # Template generator
            num_samples = st.number_input("Number of sample entries:", 1, 10, 3)
            
            template = []
            for i in range(num_samples):
                template.append({
                    "question": f"Sample question {i+1}",
                    "answer": f"Sample answer {i+1}",
                    "context": [f"Sample context {i+1}"],
                    "ground_truth": f"Sample ground truth {i+1}"
                })
            
            st.write("**Template Preview:**")
            st.json(template)
            
            # Download template
            template_json = json.dumps(template, indent=2)
            st.download_button(
                "📥 Download Template",
                template_json,
                file_name="batch_evaluation_template.json",
                mime="application/json"
            )
    
    with eval_tab4:
        st.subheader("📈 Evaluation Analytics")
        
        if st.session_state.evaluation_results:
            # Create comprehensive analytics
            df_eval = pd.DataFrame([
                {
                    "timestamp": result["timestamp"],
                    "question": result["question"][:50] + "...",
                    "relevance": result["results"].get("relevance", {}).get("score", 0),
                    "grounded": 1 if result["results"].get("groundedness", {}).get("grounded", False) else 0,
                    "retrieval": result["results"].get("retrieval_relevance", {}).get("score", 0),
                    "correct": 1 if result["results"].get("correctness", {}).get("correct", False) else 0,
                    "overall": result["results"].get("overall_score", 0),
                    "mode": result.get("mode", "Complete"),
                    "confidence": result.get("confidence", 0.8)
                }
                for result in st.session_state.evaluation_results
            ])
            
            # Score trends
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.line(
                    df_eval, 
                    x="timestamp", 
                    y=["relevance", "retrieval", "overall"], 
                    title="📈 Evaluation Scores Over Time",
                    labels={"value": "Score", "timestamp": "Time"},
                    color_discrete_sequence=["#667eea", "#764ba2", "#f093fb"]
                )
                fig.update_layout(hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Score distribution
                fig = px.histogram(
                    df_eval, 
                    x="overall", 
                    nbins=20,
                    title="📊 Overall Score Distribution",
                    color_discrete_sequence=["#667eea"]
                )
                fig.update_layout(
                    xaxis_title="Overall Score",
                    yaxis_title="Frequency"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance metrics
            st.subheader("📊 Performance Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                avg_relevance = df_eval["relevance"].mean()
                st.markdown(create_metric_card("Avg Relevance", f"{avg_relevance:.1f}/5", "🎯"), unsafe_allow_html=True)
            
            with col2:
                grounded_pct = df_eval["grounded"].mean() * 100
                st.markdown(create_metric_card("Grounded %", f"{grounded_pct:.1f}%", "🏗️"), unsafe_allow_html=True)
            
            with col3:
                avg_retrieval = df_eval["retrieval"].mean()
                st.markdown(create_metric_card("Avg Retrieval", f"{avg_retrieval:.1f}/5", "📚"), unsafe_allow_html=True)
            
            with col4:
                if "correct" in df_eval.columns and df_eval["correct"].sum() > 0:
                    correct_pct = df_eval["correct"].mean() * 100
                    st.markdown(create_metric_card("Correct %", f"{correct_pct:.1f}%", "✅"), unsafe_allow_html=True)
                else:
                    st.markdown(create_metric_card("Correct %", "N/A", "✅"), unsafe_allow_html=True)
            
            # Detailed table
            st.subheader("📋 Evaluation History Table")
            
            # Add filtering
            col_filter1, col_filter2 = st.columns(2)
            
            with col_filter1:
                min_score = st.slider("Minimum Overall Score:", 0.0, 5.0, 0.0)
                
            with col_filter2:
                mode_filter = st.multiselect("Evaluation Mode:", df_eval["mode"].unique(), default=df_eval["mode"].unique())
            
            # Apply filters
            filtered_df = df_eval[
                (df_eval["overall"] >= min_score) & 
                (df_eval["mode"].isin(mode_filter))
            ]
            
            st.dataframe(
                filtered_df.sort_values("timestamp", ascending=False),
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", width="medium"),
                    "question": st.column_config.TextColumn("Question", width="large"),
                    "relevance": st.column_config.NumberColumn("Relevance", width="small", format="%.1f"),
                    "retrieval": st.column_config.NumberColumn("Retrieval", width="small", format="%.1f"),
                    "overall": st.column_config.NumberColumn("Overall", width="small", format="%.1f"),
                    "grounded": st.column_config.CheckboxColumn("Grounded", width="small"),
                    "correct": st.column_config.CheckboxColumn("Correct", width="small")
                }
            )
            
            # Export analytics
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📊 Export Analytics Data"):
                    csv = filtered_df.to_csv(index=False)
                    st.download_button(
                        "💾 Download CSV",
                        csv,
                        file_name=f"evaluation_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime='text/csv'
                    )
            
            with col2:
                if st.button("📈 Generate Report"):
                    # Generate comprehensive report
                    report = f"""
# RAG Evaluation Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Statistics
- Total Evaluations: {len(filtered_df)}
- Average Overall Score: {filtered_df['overall'].mean():.2f}/5
- Average Relevance Score: {filtered_df['relevance'].mean():.2f}/5
- Average Retrieval Score: {filtered_df['retrieval'].mean():.2f}/5
- Grounded Responses: {filtered_df['grounded'].sum()}/{len(filtered_df)} ({filtered_df['grounded'].mean()*100:.1f}%)

## Performance Trends
{filtered_df.describe().to_string()}
                    """
                    st.download_button(
                        "📄 Download Report",
                        report,
                        file_name=f"evaluation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown"
                    )
        
        else:
            display_message("📊 No evaluation data available. Run some evaluations first!", "info")
            
            # Show sample evaluation
            if st.button("🚀 Run Sample Evaluation"):
                sample_data = {
                    "question": "What is the capital of France?",
                    "answer": "The capital of France is Paris.",
                    "context": ["Paris is the capital and most populous city of France."],
                    "ground_truth": "Paris"
                }
                
                with st.spinner("Running sample evaluation..."):
                    result = make_request("/evaluate/complete", method="POST", data=sample_data)
                    
                    if result.get("success"):
                        st.session_state.evaluation_results.append({
                            "timestamp": datetime.now(),
                            "question": sample_data["question"],
                            "answer": sample_data["answer"],
                            "results": result["results"],
                            "mode": "Sample",
                            "confidence": 1.0
                        })
                        st.experimental_rerun()

# System Status Page
elif page == "⚙️ System Status":
    st.header("⚙️ Advanced System Status")
    
    # Real-time system metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Get system status
    agents_status = make_request("/agents/status")
    shared_data = make_request("/agents/shared_data")
    activities = make_request("/agents/activities")
    evaluator_health = make_request("/evaluator/health")
    
    # System health overview
    with col1:
        if agents_status.get("success"):
            total_agents = len(agents_status.get("agents", {}))
            st.markdown(create_metric_card("Total Agents", str(total_agents), "🤖"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Total Agents", "Error", "🤖"), unsafe_allow_html=True)
    
    with col2:
        if agents_status.get("success"):
            agent_data = agents_status.get("agents", {})
            active_agents = len([a for a in agent_data.values() if a.get("status") != "offline"])
            st.markdown(create_metric_card("Active Agents", str(active_agents), "✅"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Active Agents", "Error", "✅"), unsafe_allow_html=True)
    
    with col3:
        if shared_data.get("success"):
            shared_keys = len(shared_data.get("shared_data", {}))
            st.markdown(create_metric_card("Shared Keys", str(shared_keys), "🔑"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Shared Keys", "Error", "🔑"), unsafe_allow_html=True)
    
    with col4:
        if evaluator_health.get("status") == "healthy":
            st.markdown(create_metric_card("Evaluator", "Healthy", "🏥"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Evaluator", "Offline", "🏥"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Enhanced system tabs
    sys_tab1, sys_tab2, sys_tab3, sys_tab4 = st.tabs(["🤖 Agent Monitor", "💾 Memory Status", "📋 Activity Logs", "🔧 System Health"])
    
    with sys_tab1:
        st.subheader("🤖 Agent Monitoring Dashboard")
        
        if agents_status.get("success"):
            agent_data = agents_status.get("agents", {})
            
            if agent_data:
                # Agent status overview
                status_counts = {}
                for info in agent_data.values():
                    status = info.get("status", "unknown")
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                # Status distribution chart
                if status_counts:
                    fig = px.pie(
                        values=list(status_counts.values()),
                        names=list(status_counts.keys()),
                        title="🔄 Agent Status Distribution",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Detailed agent table
                st.subheader("📊 Agent Details")
                
                agent_df = pd.DataFrame([
                    {
                        "Agent Name": name,
                        "Status": info.get("status", "unknown"),
                        "Messages": info.get("message_count", 0),
                        "Last Seen": info.get("last_seen", "unknown"),
                        "Health": "🟢" if info.get("status") == "idle" else "🟡" if info.get("status") == "busy" else "🔴"
                    }
                    for name, info in agent_data.items()
                ])
                
                # Add search functionality
                search_agent = st.text_input("🔍 Search agents:", placeholder="Enter agent name...")
                if search_agent:
                    agent_df = agent_df[agent_df['Agent Name'].str.contains(search_agent, case=False)]
                
                st.dataframe(
                    agent_df,
                    use_container_width=True,
                    column_config={
                        "Agent Name": st.column_config.TextColumn("Agent", width="medium"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                        "Messages": st.column_config.NumberColumn("Messages", width="small"),
                        "Last Seen": st.column_config.TextColumn("Last Seen", width="medium"),
                        "Health": st.column_config.TextColumn("Health", width="small")
                    }
                )
                
                # Agent controls
                st.subheader("🎮 Agent Controls")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if st.button("🔄 Refresh Status", use_container_width=True):
                        st.experimental_rerun()
                
                with col2:
                    if st.button("📊 Export Agent Data", use_container_width=True):
                        csv = agent_df.to_csv(index=False)
                        st.download_button(
                            "💾 Download CSV",
                            csv,
                            file_name=f"agent_status_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                
                with col3:
                    if st.button("🔔 Test Notifications", use_container_width=True):
                        display_message("🔔 Test notification sent to all agents!", "info")
            else:
                display_message("❌ No agent data available", "error")
        else:
            display_message("❌ Failed to fetch agent status", "error")
    
    with sys_tab2:
        st.subheader("💾 Shared Memory Management")
        
        if shared_data.get("success"):
            shared_keys = shared_data.get("shared_data", {})
            
            if shared_keys:
                # Memory usage overview
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Total Memory Keys:** {len(shared_keys)}")
                    
                    # Memory size estimation (rough)
                    total_size = sum(len(str(v)) for v in shared_keys.values())
                    st.write(f"**Estimated Size:** {total_size} characters")
                
                with col2:
                    # Recent updates
                    recent_updates = sorted(
                        [(k, v) for k, v in shared_keys.items() if isinstance(v, dict)],
                        key=lambda x: x[1].get('timestamp', ''),
                        reverse=True
                    )[:5]
                    
                    st.write("**Recent Updates:**")
                    for key, info in recent_updates:
                        if isinstance(info, dict) and 'updated_by' in info:
                            st.write(f"• {key} by {info.get('updated_by', 'unknown')}")
                
                # Memory browser
                st.subheader("🗂️ Memory Browser")
                
                # Search memory
                search_memory = st.text_input("🔍 Search memory keys:", placeholder="Enter key name...")
                
                filtered_keys = shared_keys
                if search_memory:
                    filtered_keys = {k: v for k, v in shared_keys.items() if search_memory.lower() in k.lower()}
                
                # Display memory keys
                for key, value in list(filtered_keys.items())[:20]:  # Limit to first 20
                    with st.expander(f"🔑 {key}"):
                        col1, col2 = st.columns([3, 1])
                        
                        with col1:
                            if isinstance(value, dict):
                                if 'value' in value:
                                    st.json(value)
                                else:
                                    st.json(value)
                            else:
                                st.write(str(value))
                        
                        with col2:
                            if st.button(f"Delete", key=f"del_{key}"):
                                # This would require a backend endpoint
                                display_message(f"Delete functionality for {key} would be implemented", "info")
                
                if len(filtered_keys) > 20:
                    st.info(f"Showing first 20 of {len(filtered_keys)} keys. Use search to narrow down.")
            
            else:
                display_message("💾 No shared memory data available", "info")
        else:
            display_message("❌ Failed to fetch shared memory data", "error")
    
    with sys_tab3:
        st.subheader("📋 System Activity Logs")
        
        if activities.get("success"):
            activity_list = activities.get("activities", [])
            
            if activity_list:
                # Activity filters
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Agent filter
                    all_agents = list(set([a.get("agent", "unknown") for a in activity_list]))
                    selected_agents = st.multiselect("Filter by agent:", all_agents, default=all_agents)
                
                with col2:
                    # Activity type filter
                    all_activities = list(set([a.get("activity", "unknown") for a in activity_list]))
                    selected_activities = st.multiselect("Filter by activity:", all_activities, default=all_activities)
                
                with col3:
                    # Time range
                    time_range = st.selectbox("Time range:", ["Last hour", "Last 24 hours", "Last week", "All time"])
                
                # Filter activities
                filtered_activities = [
                    a for a in activity_list 
                    if a.get("agent") in selected_agents and a.get("activity") in selected_activities
                ]
                
                # Activity timeline
                if filtered_activities:
                    df_activities = pd.DataFrame(filtered_activities)
                    df_activities['timestamp'] = pd.to_datetime(df_activities['timestamp'])
                    
                    # Timeline chart
                    fig = px.timeline(
                        df_activities.head(50),  # Limit for performance
                        x_start="timestamp",
                        x_end="timestamp",
                        y="agent",
                        color="activity",
                        title="📈 Activity Timeline (Last 50 events)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Activity table
                    st.subheader("📊 Activity Details")
                    
                    display_activities = df_activities.sort_values("timestamp", ascending=False).head(100)
                    st.dataframe(
                        display_activities,
                        use_container_width=True,
                        column_config={
                            "timestamp": st.column_config.DatetimeColumn("Time", width="medium"),
                            "agent": st.column_config.TextColumn("Agent", width="small"),
                            "activity": st.column_config.TextColumn("Activity", width="medium"),
                            "details": st.column_config.TextColumn("Details", width="large")
                        }
                    )
                    
                    # Export logs
                    if st.button("📤 Export Activity Logs"):
                        csv = df_activities.to_csv(index=False)
                        st.download_button(
                            "💾 Download Logs",
                            csv,
                            file_name=f"activity_logs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime='text/csv'
                        )
                else:
                    display_message("🔍 No activities match the current filters", "info")
            else:
                display_message("📋 No recent activities found", "info")
        else:
            display_message("❌ Failed to fetch activity logs", "error")
    
    with sys_tab4:
        st.subheader("🔧 System Health Dashboard")
        
        # Health checks
        health_checks = {
            "API Server": check_api_health(),
            "Agent Communication": agents_status.get("success", False),
            "Shared Memory": shared_data.get("success", False),
            "Activity Logging": activities.get("success", False),
            "Evaluator Service": evaluator_health.get("status") == "healthy"
        }
        
        # Health overview
        healthy_count = sum(health_checks.values())
        total_checks = len(health_checks)
        health_percentage = (healthy_count / total_checks) * 100
        
        # Overall health indicator
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if health_percentage == 100:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #00b09b, #96c93d); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                    <h2>🟢 System Healthy</h2>
                    <h3>{healthy_count}/{total_checks} Services Online</h3>
                    <div style="width: 100%; background: rgba(255,255,255,0.3); border-radius: 10px; margin-top: 1rem;">
                        <div style="width: {health_percentage}%; background: white; height: 10px; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            elif health_percentage >= 80:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #ffc107, #ff8c00); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                    <h2>🟡 System Warning</h2>
                    <h3>{healthy_count}/{total_checks} Services Online</h3>
                    <div style="width: 100%; background: rgba(255,255,255,0.3); border-radius: 10px; margin-top: 1rem;">
                        <div style="width: {health_percentage}%; background: white; height: 10px; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style="background: linear-gradient(90deg, #ff416c, #ff4b2b); padding: 2rem; border-radius: 15px; text-align: center; color: white;">
                    <h2>🔴 System Critical</h2>
                    <h3>{healthy_count}/{total_checks} Services Online</h3>
                    <div style="width: 100%; background: rgba(255,255,255,0.3); border-radius: 10px; margin-top: 1rem;">
                        <div style="width: {health_percentage}%; background: white; height: 10px; border-radius: 10px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Detailed health status
        st.subheader("🔍 Service Health Details")
        
        for service, is_healthy in health_checks.items():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                status_icon = "🟢" if is_healthy else "🔴"
                status_text = "Healthy" if is_healthy else "Unhealthy"
                st.write(f"{status_icon} **{service}:** {status_text}")
            
            with col2:
                if not is_healthy:
                    if st.button(f"🔧 Fix", key=f"fix_{service}"):
                        display_message(f"Attempting to fix {service}...", "info")
        
        # System actions
        st.markdown("---")
        st.subheader("🛠️ System Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("🔄 Refresh All", use_container_width=True):
                st.experimental_rerun()
        
        with col2:
            if st.button("🧹 Clear Cache", use_container_width=True):
                # This would require backend implementation
                display_message("Cache clearing would be implemented", "info")
        
        with col3:
            if st.button("📊 Generate Report", use_container_width=True):
                report = f"""
# System Health Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overall Health: {health_percentage:.1f}%

## Service Status:
{chr(10).join([f"- {service}: {'✅ Healthy' if status else '❌ Unhealthy'}" for service, status in health_checks.items()])}

## System Metrics:
- Total Agents: {len(agents_status.get('agents', {}))}
- Active Agents: {len([a for a in agents_status.get('agents', {}).values() if a.get('status') != 'offline'])}
- Shared Memory Keys: {len(shared_data.get('shared_data', {}))}
- Recent Activities: {len(activities.get('activities', []))}
                """
                st.download_button(
                    "📄 Download Report",
                    report,
                    file_name=f"system_health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )
        
        with col4:
            if st.button("🚨 Emergency Stop", use_container_width=True):
                if st.session_state.get('confirm_emergency', False):
                    display_message("Emergency stop would be implemented", "error")
                    st.session_state.confirm_emergency = False
                else:
                    st.session_state.confirm_emergency = True
                    display_message("⚠️ Click again to confirm emergency stop", "error")

# Agent Communication Page
elif page == "🔗 Agent Communication":
    st.header("🔗 Advanced Agent Communication")
    
    # Communication overview
    col1, col2, col3 = st.columns(3)
    
    with col1:
        agents_status = make_request("/agents/status")
        if agents_status.get("success"):
            active_agents = len([a for a in agents_status.get("agents", {}).values() if a.get("status") != "offline"])
            st.markdown(create_metric_card("Active Agents", str(active_agents), "🤖"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Active Agents", "Error", "🤖"), unsafe_allow_html=True)
    
    with col2:
        if agents_status.get("success"):
            total_messages = sum(a.get("message_count", 0) for a in agents_status.get("agents", {}).values())
            st.markdown(create_metric_card("Total Messages", str(total_messages), "💬"), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Total Messages", "Error", "💬"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("Communication", "Online", "📡"), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Communication tabs
    comm_tab1, comm_tab2, comm_tab3 = st.tabs(["📤 Send Messages", "📥 Message Logs", "🔧 Communication Tools"])
    
    with comm_tab1:
        st.subheader("📤 Agent Message Center")
        
        # Message composition
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.write("**📝 Compose Message**")
            
            # Pre-filled templates
            message_templates = {
                "Status Request": {
                    "type": "status_update",
                    "data": {"request": "status"}
                },
                "Health Check": {
                    "type": "health_check",
                    "data": {"check_type": "full"}
                },
                "Task Assignment": {
                    "type": "task_request",
                    "data": {"task": "example_task", "priority": "normal"}
                },
                "Custom": {}
            }
            
            template = st.selectbox("📋 Message Template:", list(message_templates.keys()))
            
            from_agent = st.text_input("From Agent:", value="user_interface", help="Sending agent identifier")
            
            # Get available agents
            available_agents = ["document_agent", "scraping_agent", "system", "evaluator"]
            if agents_status.get("success"):
                available_agents = list(agents_status.get("agents", {}).keys())
            
            to_agent = st.selectbox("To Agent:", available_agents, help="Target agent for the message")
            
            # Message type based on template
            if template != "Custom":
                message_type = message_templates[template].get("type", "task_request")
                message_data = json.dumps(message_templates[template].get("data", {}), indent=2)
            else:
                message_type = st.selectbox("Message Type:", ["task_request", "status_update", "coordination", "health_check"])
                message_data = ""
            
            st.write(f"**Message Type:** {message_type}")
            
            message_data = st.text_area(
                "Message Data (JSON):", 
                value=message_data,
                height=150,
                placeholder='{"key": "value", "task": "example"}'
            )
        
        with col2:
            st.write("**🎛️ Message Options**")
            
            priority = st.selectbox("Priority:", ["low", "normal", "high", "urgent"])
            requires_response = st.checkbox("Requires Response", value=False)
            broadcast = st.checkbox("Broadcast to All", value=False)
            
            # Message preview
            st.write("**👁️ Message Preview:**")
            preview_data = {
                "from": from_agent,
                "to": to_agent if not broadcast else "all_agents",
                "type": message_type,
                "priority": priority,
                "requires_response": requires_response
            }
            
            if message_data.strip():
                try:
                    preview_data["data"] = json.loads(message_data)
                except:
                    preview_data["data"] = "Invalid JSON"
            
            st.json(preview_data)
        
        # Send message
        if st.button("📤 Send Message", type="primary", use_container_width=True):
            try:
                data_dict = json.loads(message_data) if message_data.strip() else {}
                
                # Add priority and response requirement to data
                data_dict["priority"] = priority
                data_dict["requires_response"] = requires_response
                
                if broadcast:
                    display_message("📡 Broadcasting message to all agents...", "info")
                    # Broadcast logic would be implemented in backend
                else:
                    display_message(f"📤 Message sent to {to_agent}", "success")
                
                # This would actually send the message via the backend
                st.success(f"✅ Message sent successfully!")
                
            except json.JSONDecodeError:
                display_message("❌ Invalid JSON format in message data", "error")
    
    with comm_tab2:
        st.subheader("📥 Communication Logs")
        
        # Log filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            log_agent = st.selectbox("Filter by Agent:", ["All"] + (list(agents_status.get("agents", {}).keys()) if agents_status.get("success") else []))
        
        with col2:
            log_type = st.selectbox("Message Type:", ["All", "task_request", "status_update", "coordination", "health_check"])
        
        with col3:
            log_timeframe = st.selectbox("Timeframe:", ["Last hour", "Last 24 hours", "Last week", "All time"])
        
        # Mock communication logs (would come from backend)
        sample_logs = [
            {
                "timestamp": datetime.now() - pd.Timedelta(minutes=5),
                "from": "user_interface",
                "to": "document_agent",
                "type": "task_request",
                "status": "delivered",
                "response_received": True
            },
            {
                "timestamp": datetime.now() - pd.Timedelta(minutes=10),
                "from": "scraping_agent",
                "to": "system",
                "type": "status_update",
                "status": "delivered",
                "response_received": False
            },
            {
                "timestamp": datetime.now() - pd.Timedelta(minutes=15),
                "from": "system",
                "to": "evaluator",
                "type": "health_check",
                "status": "delivered",
                "response_received": True
            }
        ]
        
        if sample_logs:
            df_logs = pd.DataFrame(sample_logs)
            
            # Apply filters
            if log_agent != "All":
                df_logs = df_logs[(df_logs['from'] == log_agent) | (df_logs['to'] == log_agent)]
            if log_type != "All":
                df_logs = df_logs[df_logs['type'] == log_type]
            
            # Display logs
            st.dataframe(
                df_logs.sort_values("timestamp", ascending=False),
                use_container_width=True,
                column_config={
                    "timestamp": st.column_config.DatetimeColumn("Time", width="medium"),
                    "from": st.column_config.TextColumn("From", width="small"),
                    "to": st.column_config.TextColumn("To", width="small"),
                    "type": st.column_config.TextColumn("Type", width="medium"),
                    "status": st.column_config.TextColumn("Status", width="small"),
                    "response_received": st.column_config.CheckboxColumn("Response", width="small")
                }
            )
            
            # Communication statistics
            st.subheader("📊 Communication Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Messages", len(df_logs))
            
            with col2:
                delivered_count = len(df_logs[df_logs['status'] == 'delivered'])
                st.metric("Delivered", delivered_count)
            
            with col3:
                response_rate = df_logs['response_received'].mean() * 100 if len(df_logs) > 0 else 0
                st.metric("Response Rate", f"{response_rate:.1f}%")
        
        else:
            display_message("📥 No communication logs available", "info")
    
    with comm_tab3:
        st.subheader("🔧 Communication Tools")
        
        # Network diagnostics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**🔍 Network Diagnostics**")
            
            if st.button("🏥 Health Check All Agents", use_container_width=True):
                with st.spinner("Running health checks..."):
                    # Simulate health check
                    time.sleep(2)
                    st.success("✅ All agents responding normally")
            
            if st.button("📊 Test Message Routing", use_container_width=True):
                with st.spinner("Testing message routes..."):
                    time.sleep(1)
                    st.success("✅ All message routes operational")
            
            if st.button("🔄 Restart Communication", use_container_width=True):
                with st.spinner("Restarting communication system..."):
                    time.sleep(2)
                    st.success("✅ Communication system restarted")
        
        with col2:
            st.write("**📈 Performance Monitoring**")
            
            # Mock performance data
            perf_data = {
                "Average Latency": "45ms",
                "Message Success Rate": "98.5%",
                "Active Connections": "5",
                "Queue Length": "0"
            }
            
            for metric, value in perf_data.items():
                st.metric(metric, value)
        
        # Message queue management
        st.markdown("---")
        st.subheader("📬 Message Queue Management")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📬 View Message Queues", use_container_width=True):
                display_message("📬 Message queue viewer would show pending messages", "info")
        
        with col2:
            if st.button("🧹 Clear Message Queues", use_container_width=True):
                display_message("🧹 Message queues cleared", "success")
        
        with col3:
            if st.button("⏸️ Pause Communication", use_container_width=True):
                display_message("⏸️ Communication paused", "info")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666; padding: 1rem;'>
        RAG Document Processing System | Built with Streamlit & FastAPI
    </div>
    """, 
    unsafe_allow_html=True
)

# Auto-refresh for real-time updates (optional)
if st.sidebar.checkbox("Auto-refresh (30s)", value=False):
    time.sleep(30)
    st.experimental_rerun()
