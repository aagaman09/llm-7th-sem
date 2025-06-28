# app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
from loguru import logger
import uvicorn
import asyncio # Import asyncio for direct run

# Import your pipeline functions and RAG components
from ingestion import run_ingestion_pipeline
from chat_interface import chat_with_scraped_data
# --- FIX: Update imports to use the new getter functions ---
from rag_components import (
    initialize_rag_components,
    get_chroma_client,      # NEW: Import get_chroma_client
    get_chroma_collection,  # NEW: Import get_chroma_collection
    get_embeddings_instance # This one was already correct
)

# Request/Response Models (Assuming these are complete and correct)
class SearchRequest(BaseModel):
    query: str
    num_results: Optional[int] = 3

class SearchResponse(BaseModel):
    initial_answer: str
    success: bool
    message: str
    scraped_urls: List[str]
    retrieved_sources: List[str]
    evaluation_results: Optional[Dict[str, Any]] = None

class ChatRequest(BaseModel):
    question: str
    num_results: Optional[int] = 3

class ChatResponse(BaseModel):
    answer: str
    success: bool
    message: str
    retrieved_sources: List[str]
    evaluation_results: Optional[Dict[str, Any]] = None

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chat API",
    description="API for a RAG system with search, scraping, and conversational chat.",
    version="0.1.0",
)

# Startup event: Initialize RAG components
@app.on_event("startup")
async def startup_event():
    logger.info("FastAPI startup event: Initializing RAG components...")
    success = initialize_rag_components() # This will now initialize _embeddings_instance within rag_components.py
    if not success:
        logger.critical("Failed to initialize RAG components on startup. Exiting.")
        raise HTTPException(status_code=500, detail="Failed to initialize RAG components on startup.")
    else:
        logger.success("FastAPI startup: RAG components initialized.")

    logger.info("FastAPI startup event: Clearing existing database content...")
    try:
        # --- FIX: Use get_chroma_client and get_chroma_collection ---
        current_client = get_chroma_client()
        current_collection = get_chroma_collection()
        collection_name = current_collection.name
        
        # Delete the existing collection
        current_client.delete_collection(name=collection_name)
        
        # --- THE FIX: Get the initialized embeddings instance via the new getter function ---
        initialized_embeddings_instance = get_embeddings_instance()
        if initialized_embeddings_instance is None:
            raise RuntimeError("Embeddings instance is None after initialization, cannot re-initialize collection.")
            
        # Re-create the collection - the get_chroma_collection() function will now 
        # automatically handle getting a fresh reference
        new_collection = current_client.get_or_create_collection(
            name=collection_name # No embedding_function needed here for raw chromadb client
        )
        
        logger.success(f"ChromaDB collection '{collection_name}' cleared and re-initialized on startup.")
    except Exception as e:
        logger.error(f"Error clearing/re-initializing database on startup: {e}")
        raise HTTPException(status_code=500, detail=f"Database initialization failed: {e}")

@app.post("/search", response_model=SearchResponse)
async def search_endpoint(request: SearchRequest):
    logger.info(f"Received search request for query: '{request.query}'")
    try:
        # Pass num_results from the request
        result = await run_ingestion_pipeline(request.query, n_results=request.num_results)
        return SearchResponse(
            initial_answer=result["initial_answer"],
            success=result["success"],
            message=result["message"],
            scraped_urls=result["scraped_urls"],
            retrieved_sources=result["retrieved_sources"],
            evaluation_results=result["evaluation_results"]
        )
    except Exception as e:
        logger.error(f"Error processing search request: {e}")
        raise HTTPException(status_code=500, detail=f"Search operation failed: {e}")

@app.post("/query", response_model=ChatResponse)
async def query_endpoint(request: ChatRequest):
    logger.info(f"Received chat query: '{request.question}'")
    try:
        # Pass num_results from the request
        result = await chat_with_scraped_data(request.question, n_results=request.num_results)
        return ChatResponse(
            answer=result["answer"],
            success=result["success"],
            message=result["message"],
            retrieved_sources=result["retrieved_sources"],
            evaluation_results=result["evaluation_results"]
        )
    except Exception as e:
        logger.error(f"Error processing chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Chat operation failed: {e}")


@app.post("/clear_database", response_model=dict)
async def clear_database_endpoint():
    """
    Endpoint to manually clear all data from the ChromaDB database.
    Note: The database is also cleared on app startup.
    """
    logger.info("Received request to clear database.")
    try:
        # --- FIX: Use get_chroma_client and get_chroma_collection ---
        client = get_chroma_client()
        collection = get_chroma_collection()
        collection_name = collection.name
        
        # Delete and re-create the collection
        client.delete_collection(name=collection_name)
        
        # --- THE FIX: Get the initialized embeddings instance via the new getter function ---
        # This part is only necessary if you're recreating the collection with an embedding_function argument
        # for the *low-level* client. As per our refined rag_components.py, the low-level client's
        # get_or_create_collection does NOT take embedding_function. LangChain's Chroma wrapper does.
        # So, we remove `embedding_function` here.
        
        # Re-create the collection (without embedding_function for raw chromadb client)
        new_collection = client.get_or_create_collection(
            name=collection_name
        )
        
        logger.success(f"ChromaDB collection '{collection_name}' successfully cleared and re-initialized.")
        return {"success": True, "message": "Database cleared and re-initialized."}
    except Exception as e:
        logger.error(f"Error clearing database: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear database: {e}")

# This block allows running the app directly without `uvicorn` command
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)