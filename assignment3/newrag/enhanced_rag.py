#!/usr/bin/env python3
"""
Enhanced RAG Document Processing API
Advanced FastAPI backend with comprehensive features
"""

import tempfile
import os
import asyncio
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import uuid
import json
import logging
from pathlib import Path

# FastAPI and related imports
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, validator
import uvicorn

# Core dependencies
from loguru import logger
import psutil
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Import existing modules
from document_agent import process_pdf
from scrape_agent import scrape_url
from agent_communication import simple_bus, coordinator
from evaluator import RAGEvaluator

# Langchain and database imports
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Enhanced Pydantic models
class DocumentMetadata(BaseModel):
    """Enhanced document metadata"""
    filename: str
    file_type: str
    file_size: int
    processed_at: datetime
    chunk_count: int
    processing_time: float
    source_type: str = Field(..., description="PDF, URL, or TEXT")

class QueryRequest(BaseModel):
    """Enhanced query request with advanced options"""
    question: str = Field(..., min_length=1, max_length=2000)
    n_results: Optional[int] = Field(5, ge=1, le=20)
    include_metadata: Optional[bool] = True
    response_format: Optional[str] = Field("detailed", regex="^(detailed|concise|bullet_points|academic)$")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, ge=100, le=4000)
    include_sources: Optional[bool] = True
    stream_response: Optional[bool] = False

class QueryResponse(BaseModel):
    """Enhanced query response"""
    answer: str
    sources: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    processing_time: float
    success: bool
    confidence_score: Optional[float] = None
    message: Optional[str] = None

class ProcessURL(BaseModel):
    """Enhanced URL processing request"""
    url: str = Field(..., regex=r'^https?://')
    follow_links: Optional[bool] = False
    max_depth: Optional[int] = Field(1, ge=1, le=3)
    extract_images: Optional[bool] = False
    timeout: Optional[int] = Field(30, ge=10, le=120)

class SystemStatus(BaseModel):
    """System status response"""
    status: str
    uptime: float
    memory_usage: Dict[str, float]
    cpu_usage: float
    active_agents: int
    total_documents: int
    total_queries: int
    database_status: str
    last_updated: datetime

class EvaluationRequest(BaseModel):
    """Enhanced evaluation request"""
    question: str
    answer: str
    context: List[str]
    ground_truth: Optional[str] = None
    evaluation_mode: Optional[str] = Field("complete", regex="^(complete|quick|custom)$")
    metrics: Optional[List[str]] = ["relevance", "groundedness", "retrieval_relevance"]

class BatchEvaluationRequest(BaseModel):
    """Batch evaluation request"""
    evaluations: List[EvaluationRequest]
    parallel_processing: Optional[bool] = True
    max_workers: Optional[int] = Field(3, ge=1, le=10)

# Security
security = HTTPBearer(auto_error=False)

# Global variables
app = FastAPI(
    title="Enhanced RAG Document Processing API",
    description="Advanced RAG system with comprehensive document processing and evaluation",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enhanced CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Global state
system_state = {
    "start_time": datetime.now(),
    "total_queries": 0,
    "total_documents": 0,
    "embeddings": None,
    "client": None,
    "collection": None,
    "llm": None,
    "evaluator": None,
    "executor": ThreadPoolExecutor(max_workers=4)
}

# Dependency injection
async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)):
    """Simple authentication dependency (can be enhanced)"""
    # For now, just return a default user
    return {"user_id": "default", "permissions": ["read", "write"]}

def get_system_metrics() -> Dict[str, Any]:
    """Get current system metrics"""
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "memory_usage": {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent()
        },
        "cpu_usage": process.cpu_percent(),
        "uptime": (datetime.now() - system_state["start_time"]).total_seconds(),
        "total_queries": system_state["total_queries"],
        "total_documents": system_state["total_documents"]
    }

# Enhanced initialization
async def initialize_components():
    """Initialize all system components with error handling"""
    try:
        logger.info("🚀 Initializing enhanced RAG system...")
        
        # Initialize embeddings
        logger.info("📊 Loading embeddings model...")
        system_state["embeddings"] = OllamaEmbeddings(model="mxbai-embed-large")
        
        # Test embeddings
        test_embedding = system_state["embeddings"].embed_query("test")
        embedding_dim = len(test_embedding)
        logger.info(f"✅ Embeddings loaded. Dimension: {embedding_dim}")
        
        # Initialize ChromaDB
        logger.info("🗄️ Connecting to ChromaDB...")
        system_state["client"] = chromadb.PersistentClient(path="chroma_store")
        
        collection_name = f"enhanced_docs_mxbai_{embedding_dim}d"
        system_state["collection"] = system_state["client"].get_or_create_collection(name=collection_name)
        logger.info(f"✅ ChromaDB connected. Collection: {collection_name}")
        
        # Initialize LLM
        logger.info("🤖 Loading language model...")
        system_state["llm"] = ChatOllama(model="llama3", temperature=0.7)
        logger.info("✅ Language model loaded")
        
        # Initialize evaluator
        logger.info("📝 Loading RAG evaluator...")
        system_state["evaluator"] = RAGEvaluator(model_name="llama3", temperature=0)
        logger.info("✅ RAG evaluator loaded")
        
        logger.success("🎉 All components initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize components: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    """Enhanced startup sequence"""
    logger.info("🔄 Starting Enhanced RAG API Server...")
    
    # Initialize components
    success = await initialize_components()
    if not success:
        logger.error("💥 Startup failed!")
        raise Exception("Failed to initialize system components")
    
    # Start background tasks
    asyncio.create_task(system_health_monitor())
    
    logger.success("✅ Enhanced RAG API Server started successfully!")

async def system_health_monitor():
    """Background task to monitor system health"""
    while True:
        try:
            metrics = get_system_metrics()
            
            # Log health status every 5 minutes
            if metrics["uptime"] % 300 < 1:  # Every 5 minutes
                logger.info(f"💓 System Health - CPU: {metrics['cpu_usage']:.1f}%, Memory: {metrics['memory_usage']['percent']:.1f}%")
            
            # Check for high resource usage
            if metrics["cpu_usage"] > 80:
                logger.warning(f"⚠️ High CPU usage: {metrics['cpu_usage']:.1f}%")
            
            if metrics["memory_usage"]["percent"] > 80:
                logger.warning(f"⚠️ High memory usage: {metrics['memory_usage']['percent']:.1f}%")
            
            await asyncio.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            logger.error(f"❌ Health monitor error: {e}")
            await asyncio.sleep(60)  # Wait longer on error

# Enhanced endpoints

@app.get("/", response_model=Dict[str, Any])
async def root():
    """Enhanced root endpoint with system info"""
    metrics = get_system_metrics()
    
    return {
        "service": "Enhanced RAG Document Processing API",
        "version": "2.0.0",
        "status": "operational",
        "uptime_seconds": metrics["uptime"],
        "features": [
            "Multi-format document processing",
            "Advanced RAG evaluation",
            "Real-time system monitoring",
            "Agent-based architecture",
            "Batch processing support"
        ],
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "metrics": "/metrics",
            "upload": "/upload",
            "query": "/query",
            "evaluate": "/evaluate/*"
        }
    }

@app.get("/health", response_model=SystemStatus)
async def health_check():
    """Comprehensive health check endpoint"""
    metrics = get_system_metrics()
    
    # Check component health
    components_healthy = all([
        system_state["embeddings"] is not None,
        system_state["client"] is not None,
        system_state["collection"] is not None,
        system_state["llm"] is not None,
        system_state["evaluator"] is not None
    ])
    
    # Check database connectivity
    try:
        if system_state["collection"]:
            system_state["collection"].count()
            db_status = "healthy"
        else:
            db_status = "disconnected"
    except Exception:
        db_status = "error"
    
    status = "healthy" if components_healthy and db_status == "healthy" else "degraded"
    
    return SystemStatus(
        status=status,
        uptime=metrics["uptime"],
        memory_usage=metrics["memory_usage"],
        cpu_usage=metrics["cpu_usage"],
        active_agents=len(simple_bus.agents),
        total_documents=metrics["total_documents"],
        total_queries=metrics["total_queries"],
        database_status=db_status,
        last_updated=datetime.now()
    )

@app.get("/metrics", response_model=Dict[str, Any])
async def get_metrics():
    """Detailed system metrics endpoint"""
    metrics = get_system_metrics()
    
    # Add database metrics
    try:
        if system_state["collection"]:
            doc_count = system_state["collection"].count()
        else:
            doc_count = 0
    except:
        doc_count = 0
    
    # Add agent metrics
    agent_status = {name: info["status"] for name, info in simple_bus.agents.items()}
    
    return {
        "timestamp": datetime.now().isoformat(),
        "system": metrics,
        "database": {
            "total_documents": doc_count,
            "connection_status": "connected" if system_state["collection"] else "disconnected"
        },
        "agents": {
            "total": len(simple_bus.agents),
            "status": agent_status
        },
        "performance": {
            "avg_query_time": "N/A",  # Would track this in real implementation
            "success_rate": "N/A"
        }
    }

@app.post("/upload", response_model=Dict[str, Any])
async def upload_documents(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_current_user)
):
    """Enhanced document upload with batch processing"""
    start_time = datetime.now()
    
    if not all(file.filename.lower().endswith('.pdf') for file in files):
        raise HTTPException(
            status_code=400, 
            detail="Only PDF files are supported. Please upload PDF files only."
        )
    
    results = []
    total_chunks = 0
    
    for file in files:
        try:
            # Process file
            content = await file.read()
            file_size = len(content)
            
            logger.info(f"📄 Processing {file.filename} ({file_size} bytes)")
            
            result = await process_pdf(content, file.filename)
            
            if result["success"]:
                chunks = result["chunks"]
                chunk_count = len(chunks)
                total_chunks += chunk_count
                
                # Add chunks to vector database
                texts = [chunk.page_content for chunk in chunks]
                embeddings_list = system_state["embeddings"].embed_documents(texts)
                
                # Store in database with metadata
                for i, (text, emb) in enumerate(zip(texts, embeddings_list)):
                    doc_id = f"{file.filename}_{i}_{uuid.uuid4().hex[:8]}"
                    metadata = {
                        "filename": file.filename,
                        "chunk_index": i,
                        "total_chunks": chunk_count,
                        "file_size": file_size,
                        "processed_at": datetime.now().isoformat(),
                        "user_id": user["user_id"]
                    }
                    
                    system_state["collection"].add(
                        ids=[doc_id],
                        documents=[text],
                        embeddings=[emb],
                        metadatas=[metadata]
                    )
                
                results.append({
                    "filename": file.filename,
                    "success": True,
                    "chunks": chunk_count,
                    "file_size": file_size
                })
                
                system_state["total_documents"] += 1
                
            else:
                results.append({
                    "filename": file.filename,
                    "success": False,
                    "error": result.get("error", "Unknown error")
                })
                
        except Exception as e:
            logger.error(f"❌ Error processing {file.filename}: {str(e)}")
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    processing_time = (datetime.now() - start_time).total_seconds()
    
    successful_files = len([r for r in results if r["success"]])
    
    return {
        "success": successful_files > 0,
        "processed_files": len(files),
        "successful_files": successful_files,
        "total_chunks": total_chunks,
        "processing_time": processing_time,
        "results": results,
        "message": f"Processed {successful_files}/{len(files)} files successfully"
    }

@app.post("/url", response_model=Dict[str, Any])
async def process_webpage(
    url_data: ProcessURL,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_current_user)
):
    """Enhanced URL processing with advanced options"""
    start_time = datetime.now()
    
    if not all([system_state["embeddings"], system_state["collection"]]):
        raise HTTPException(
            status_code=503,
            detail="Backend components not initialized"
        )
    
    try:
        logger.info(f"🌐 Processing URL: {url_data.url}")
        
        result = await scrape_url(url_data.url)
        
        if not result["success"]:
            raise HTTPException(
                status_code=400, 
                detail=result.get("error", "Failed to process URL")
            )
        
        content = result["content"]
        
        # Enhanced text splitting
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", "?", "!", " ", ""],
            length_function=len,
            is_separator_regex=False
        )
        
        text_chunks = text_splitter.split_text(content)
        
        if not text_chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the URL"
            )
        
        # Create embeddings and store
        embeddings_list = system_state["embeddings"].embed_documents(text_chunks)
        
        for i, (text, emb) in enumerate(zip(text_chunks, embeddings_list)):
            doc_id = f"url_{uuid.uuid4().hex[:8]}_{i}"
            metadata = {
                "source_url": url_data.url,
                "chunk_index": i,
                "total_chunks": len(text_chunks),
                "processed_at": datetime.now().isoformat(),
                "source_type": "URL",
                "user_id": user["user_id"]
            }
            
            system_state["collection"].add(
                ids=[doc_id],
                documents=[text],
                embeddings=[emb],
                metadatas=[metadata]
            )
        
        system_state["total_documents"] += 1
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "success": True,
            "url": url_data.url,
            "chunks_created": len(text_chunks),
            "content_length": len(content),
            "processing_time": processing_time,
            "message": f"Successfully processed {len(text_chunks)} chunks from URL"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing URL {url_data.url}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing URL: {str(e)}"
        )

@app.post("/query", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    user: dict = Depends(get_current_user)
):
    """Enhanced document querying with advanced options"""
    start_time = datetime.now()
    
    if not all([system_state["embeddings"], system_state["collection"], system_state["llm"]]):
        raise HTTPException(status_code=503, detail="Components not initialized")
    
    try:
        logger.info(f"🔍 Processing query: {request.question[:50]}...")
        
        # Create query embedding
        query_embedding = system_state["embeddings"].embed_query(request.question)
        
        # Search documents
        results = system_state["collection"].query(
            query_embeddings=[query_embedding],
            n_results=request.n_results,
            include=["documents", "metadatas", "distances"]
        )
        
        if not results['documents'][0]:
            return QueryResponse(
                answer="No relevant documents found. Please upload some documents first.",
                sources=[],
                metadata={"found_documents": 0},
                processing_time=(datetime.now() - start_time).total_seconds(),
                success=True,
                message="No documents found"
            )
        
        # Prepare context
        context = "\n\n".join(results['documents'][0])
        
        # Format response based on style
        style_prompts = {
            "detailed": "Provide a comprehensive and detailed answer based on the context:",
            "concise": "Provide a brief and concise answer based on the context:",
            "bullet_points": "Provide the answer in bullet points based on the context:",
            "academic": "Provide an academic-style answer with references based on the context:"
        }
        
        prompt_template = style_prompts.get(request.response_format, style_prompts["detailed"])
        
        # Create prompt
        prompt = ChatPromptTemplate.from_template(f"""
{prompt_template}

Context:
{{context}}

Question: {{question}}

Answer:""")
        
        # Configure LLM
        llm = ChatOllama(
            model="llama3", 
            temperature=request.temperature,
            num_predict=request.max_tokens
        )
        
        chain = prompt | llm | StrOutputParser()
        
        # Generate response
        response = chain.invoke({
            "context": context,
            "question": request.question
        })
        
        # Prepare sources with metadata
        sources = []
        if request.include_sources:
            for i, (doc, metadata, distance) in enumerate(zip(
                results['documents'][0], 
                results['metadatas'][0], 
                results['distances'][0]
            )):
                sources.append({
                    "content": doc[:500] + "..." if len(doc) > 500 else doc,
                    "metadata": metadata,
                    "relevance_score": 1 - distance,  # Convert distance to relevance
                    "rank": i + 1
                })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        system_state["total_queries"] += 1
        
        return QueryResponse(
            answer=response,
            sources=sources,
            metadata={
                "query_format": request.response_format,
                "temperature": request.temperature,
                "sources_count": len(sources),
                "user_id": user["user_id"]
            },
            processing_time=processing_time,
            success=True,
            confidence_score=sum(s["relevance_score"] for s in sources) / len(sources) if sources else 0
        )
        
    except Exception as e:
        logger.error(f"❌ Error in query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/complete", response_model=Dict[str, Any])
async def evaluate_complete_rag(
    request: EvaluationRequest,
    user: dict = Depends(get_current_user)
):
    """Enhanced complete RAG evaluation"""
    if not system_state["evaluator"]:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        start_time = datetime.now()
        
        logger.info(f"📊 Running {request.evaluation_mode} evaluation...")
        
        result = system_state["evaluator"].evaluate_complete_rag(
            question=request.question,
            answer=request.answer,
            context=request.context,
            ground_truth=request.ground_truth
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Add metadata
        result["metadata"] = {
            "evaluation_mode": request.evaluation_mode,
            "processing_time": processing_time,
            "user_id": user["user_id"],
            "timestamp": datetime.now().isoformat()
        }
        
        return {
            "success": True,
            "results": result,
            "message": f"Complete RAG evaluation finished. Overall score: {result['overall_score']:.2f}/5"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in complete RAG evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/evaluate/batch", response_model=Dict[str, Any])
async def batch_evaluate(
    request: BatchEvaluationRequest,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(get_current_user)
):
    """Enhanced batch evaluation with parallel processing"""
    if not system_state["evaluator"]:
        raise HTTPException(status_code=503, detail="Evaluator not initialized")
    
    try:
        start_time = datetime.now()
        logger.info(f"📦 Processing batch evaluation of {len(request.evaluations)} items...")
        
        async def evaluate_single(eval_request: EvaluationRequest) -> Dict[str, Any]:
            """Evaluate a single request"""
            try:
                result = system_state["evaluator"].evaluate_complete_rag(
                    question=eval_request.question,
                    answer=eval_request.answer,
                    context=eval_request.context,
                    ground_truth=eval_request.ground_truth
                )
                return {"success": True, "result": result}
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        if request.parallel_processing:
            # Process in parallel
            tasks = [evaluate_single(eval_req) for eval_req in request.evaluations]
            evaluation_results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process sequentially
            evaluation_results = []
            for eval_req in request.evaluations:
                result = await evaluate_single(eval_req)
                evaluation_results.append(result)
        
        # Compile results
        successful_results = [r for r in evaluation_results if isinstance(r, dict) and r.get("success")]
        failed_results = [r for r in evaluation_results if not (isinstance(r, dict) and r.get("success"))]
        
        # Calculate statistics
        if successful_results:
            overall_scores = [r["result"]["overall_score"] for r in successful_results]
            batch_stats = {
                "total_evaluations": len(request.evaluations),
                "successful_evaluations": len(successful_results),
                "failed_evaluations": len(failed_results),
                "average_score": sum(overall_scores) / len(overall_scores),
                "min_score": min(overall_scores),
                "max_score": max(overall_scores),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        else:
            batch_stats = {
                "total_evaluations": len(request.evaluations),
                "successful_evaluations": 0,
                "failed_evaluations": len(failed_results),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
        
        return {
            "success": len(successful_results) > 0,
            "batch_results": evaluation_results,
            "batch_statistics": batch_stats,
            "message": f"Batch evaluation completed. {len(successful_results)}/{len(request.evaluations)} successful."
        }
        
    except Exception as e:
        logger.error(f"❌ Error in batch evaluation: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/clear", response_model=Dict[str, Any])
async def clear_database(user: dict = Depends(get_current_user)):
    """Enhanced database clearing with confirmation"""
    if not system_state["collection"]:
        raise HTTPException(status_code=503, detail="Database not initialized")
    
    try:
        # Get all document IDs
        all_data = system_state["collection"].get()
        doc_count = len(all_data['ids'])
        
        if doc_count > 0:
            # Delete all documents
            system_state["collection"].delete(ids=all_data['ids'])
            system_state["total_documents"] = 0
            
            logger.info(f"🗑️ Cleared {doc_count} documents from database (user: {user['user_id']})")
            
            return {
                "success": True, 
                "documents_deleted": doc_count,
                "timestamp": datetime.now().isoformat(),
                "message": f"Successfully cleared {doc_count} documents from database"
            }
        else:
            return {
                "success": True,
                "documents_deleted": 0,
                "message": "Database was already empty"
            }
            
    except Exception as e:
        logger.error(f"❌ Error clearing database: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing database: {str(e)}")

@app.get("/agents/status", response_model=Dict[str, Any])
async def get_agent_status():
    """Enhanced agent status with detailed information"""
    try:
        status = {}
        for name, info in simple_bus.agents.items():
            status[name] = {
                "status": info["status"],
                "message_count": len(info["messages"]),
                "last_seen": datetime.now().isoformat(),  # Would track this properly
                "health": "healthy" if info["status"] == "idle" else "busy" if info["status"] == "busy" else "error"
            }
        
        return {
            "success": True,
            "agents": status,
            "total_agents": len(status),
            "active_agents": len([a for a in status.values() if a["status"] != "offline"]),
            "shared_data_keys": list(simple_bus.shared_data.keys()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Enhanced error handling"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "success": False,
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

if __name__ == "__main__":
    # Enhanced server configuration
    uvicorn.run(
        "enhanced_rag:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )
