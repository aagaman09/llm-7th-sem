import os
from typing import List, Dict, TypedDict, Literal, Optional, Any

import chromadb
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from loguru import logger

# --- RAG Component Initialization ---
# Renamed globals for clarity and to fit the getter pattern
_embeddings_instance: Optional[OllamaEmbeddings] = None
_chroma_client: Optional[chromadb.PersistentClient] = None
_chroma_collection: Optional[chromadb.Collection] = None
_collection_name: Optional[str] = None  # Store the collection name for fresh lookups
_embedding_dim: Optional[int] = None
_llm: Optional[ChatOllama] = None
_rag_initialized: bool = False # Flag to check if RAG components are ready

# Define a consistent collection name base
CHROMA_COLLECTION_NAME_BASE = "rag_documents"

# System prompt for the LLM
SYSTEM_PROMPT = """
You are an intelligent assistant that answers questions based on provided context from documents. Your role is to:

1. **Analyze the provided context carefully** and extract relevant information to answer the user's question
2. **Answer based ONLY on the information provided** in the context - do not use external knowledge
3. **Be accurate and precise** - if the context doesn't contain enough information to answer the question, clearly state this
4. **Quote directly from the context** when appropriate, using quotation marks
5. **Maintain the same tone and style** as the source material when possible

## Instructions:
- If the answer is clearly stated in the context, provide a direct answer
- If the context contains partial information, explain what you can determine and what is unclear
- If the context doesn't contain relevant information, respond with: "The provided context doesn't contain enough information to answer this question."
- Always be honest about the limitations of the provided context

## Context:
{context}

## Question:
{question}

## Answer:
"""

def initialize_rag_components(embedding_model: str = "mxbai-embed-large", llm_model: str = "llama3.2:1b") -> bool:
    """
    Initialize embeddings, Chroma client, and LLM.
    This function should be called once at the start of the application.
    It now accepts model names as arguments for flexibility.
    """
    global _embeddings_instance, _chroma_client, _chroma_collection, _collection_name, _embedding_dim, _llm, _rag_initialized
    
    if _rag_initialized:
        logger.info("RAG components already initialized and appear to be valid.")
        return True

    try:
        # Initialize Ollama Embeddings
        _embeddings_instance = OllamaEmbeddings(model=embedding_model)
        logger.info(f"Ollama embeddings initialized with model: {embedding_model}")

        # Initialize ChromaDB Persistent Client
        _chroma_client = chromadb.PersistentClient(path="./chroma_store")
        logger.info("ChromaDB client initialized with persistent storage.")

        # Test embedding dimension consistency
        test_single = _embeddings_instance.embed_query("test query")
        test_batch = _embeddings_instance.embed_documents(["test document one", "test document two"])
        
        single_dim = len(test_single)
        batch_dim = len(test_batch[0]) if test_batch else 0
        
        if single_dim != batch_dim:
            logger.warning(f"⚠️ Embedding dimension inconsistency detected!")
            logger.warning(f"Single query: {single_dim}d, Batch: {batch_dim}d")
            logger.info("Using batch embedding dimension for consistency.")
            _embedding_dim = batch_dim
        else:
            _embedding_dim = single_dim
        
        # Get or create ChromaDB collection based on embedding dimension
        # The collection name now explicitly includes the model and dimension
        collection_name_with_dim = f"{CHROMA_COLLECTION_NAME_BASE}_{embedding_model.replace('-', '_')}_{_embedding_dim}d"
        _collection_name = collection_name_with_dim  # Store the collection name
        _chroma_collection = _chroma_client.get_or_create_collection(name=collection_name_with_dim)
        logger.info(f"Using ChromaDB collection: {collection_name_with_dim} (embedding dimension: {_embedding_dim})")

        # Initialize Ollama Chat LLM
        _llm = ChatOllama(model=llm_model, temperature=0.7)
        logger.info(f"Ollama LLM initialized with model: {llm_model}")
        
        logger.success("All RAG components initialized successfully!")
        _rag_initialized = True
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG components: {str(e)}")
        logger.error("Make sure Ollama is running and the required models are installed (e.g., `ollama pull mxbai-embed-large`, `ollama pull llama3.2:1b`).")
        _rag_initialized = False
        return False

# --- Getter functions for RAG components ---

def get_llm():
    global _llm
    if not _rag_initialized:
        initialize_rag_components() # Ensure initialization before returning
    if _llm is None:
        raise RuntimeError("LLM not initialized. Check RAG component initialization.")
    return _llm

def get_embeddings_instance():
    global _embeddings_instance
    if not _rag_initialized:
        initialize_rag_components() # Ensure initialization before returning
    if _embeddings_instance is None:
        raise RuntimeError("Embeddings not initialized. Check RAG component initialization.")
    return _embeddings_instance

def get_chroma_client():
    global _chroma_client
    if not _rag_initialized:
        initialize_rag_components() # Ensure initialization before returning
    if _chroma_client is None:
        raise RuntimeError("Chroma client not initialized. Check RAG component initialization.")
    return _chroma_client

def get_chroma_collection():
    global _chroma_client, _collection_name
    if not _rag_initialized:
        initialize_rag_components() # Ensure initialization before returning
    if _chroma_client is None or _collection_name is None:
        raise RuntimeError("Chroma client or collection name not initialized. Check RAG component initialization.")
    
    # Always get a fresh reference to handle cases where collection was recreated
    try:
        fresh_collection = _chroma_client.get_collection(name=_collection_name)
        return fresh_collection
    except Exception as e:
        logger.error(f"Failed to get collection '{_collection_name}': {e}")
        # Fallback: try to create the collection if it doesn't exist
        try:
            fresh_collection = _chroma_client.get_or_create_collection(name=_collection_name)
            logger.info(f"Created missing collection '{_collection_name}'")
            return fresh_collection
        except Exception as create_error:
            logger.error(f"Failed to create collection '{_collection_name}': {create_error}")
            raise RuntimeError(f"Chroma collection '{_collection_name}' not accessible and cannot be created.")

def is_rag_initialized() -> bool:
    global _rag_initialized
    return _rag_initialized

# --- Langgraph State ---
class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        query: The user's initial search query or chat question.
        urls: List of URLs extracted by the search agent (used in ingestion).
        scraped_content: Dictionary mapping URLs to their scraped Markdown content (used in ingestion).
        generation: The final answer or generated response.
        error: Any error messages encountered during the flow.
        retrieved_sources: List of sources (URLs) from retrieved chunks.
        n_results: Number of results for ChromaDB retrieval (optional).
        retrieved_elements: Raw retrieved documents and metadatas from ChromaDB.
        evaluation_results: Optional[Dict[str, Any]] # To store the overall evaluation output
        ground_truth: Optional[str] # If you have a ground truth for evaluation (e.g., in a dataset)
    """
    query: str
    urls: List[str]
    scraped_content: Dict[str, str]
    generation: str
    error: str
    retrieved_sources: List[str]
    n_results: Optional[int]
    retrieved_elements: Optional[Dict[str, Any]]
    evaluation_results: Optional[Dict[str, Any]]
    ground_truth: Optional[str]


# --- Common Langgraph Nodes ---
async def generate_node(state: GraphState) -> GraphState:
    """
    Node to generate a response based on the query by retrieving context from ChromaDB
    and passing it to the LLM. This node is used by both ingestion and chat.
    It now uses getter functions to ensure it has the latest ChromaDB objects.
    """
    logger.info("---GENERATE NODE: Retrieving context and generating final response---")
    query = state["query"]

    # Use getter functions to ensure components are initialized and retrieved safely
    current_embeddings_instance = get_embeddings_instance()
    current_llm = get_llm()
    current_collection = get_chroma_collection()

    if current_embeddings_instance is None or current_llm is None or current_collection is None:
        error_msg = "RAG components not fully initialized or accessible for generation."
        logger.error(error_msg)
        return {"generation": "Failed to initialize generation components.", "query": query, "error": error_msg}

    try:
        query_embedding = current_embeddings_instance.embed_query(query)
    except Exception as e:
        logger.error(f"Error embedding query: {e}")
        return {"generation": f"Could not embed query: {str(e)}", "query": query, "error": str(e)}
        
    retrieved_elements_data = {"documents": [], "metadatas": [], "ids": [], "distances": []} # Initialize
    try:
        n_results_from_state = state.get("n_results", 5) # Default to 5 if not provided in state
        results = current_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results_from_state,
            include=['documents', 'metadatas', 'distances'] # 'ids' is returned by default
        )
        retrieved_elements_data["documents"] = results.get('documents', [[]])[0]
        retrieved_elements_data["metadatas"] = results.get('metadatas', [[]])[0]
        retrieved_elements_data["ids"] = results.get('ids', [[]])[0]
        retrieved_elements_data["distances"] = results.get('distances', [[]])[0]

    except Exception as e:
        logger.error(f"Error querying ChromaDB: {e}")
        return {"generation": f"Could not retrieve context from database: {str(e)}", "query": query, "error": str(e), "retrieved_elements": retrieved_elements_data}

    retrieved_documents = retrieved_elements_data["documents"]
    retrieved_metadatas = retrieved_elements_data["metadatas"]
    
    if not retrieved_documents:
        generation = f"The provided context doesn't contain enough information to answer this question. No relevant documents found in the database for '{query}'."
        retrieved_sources = []
    else:
        context = "\n\n".join(retrieved_documents)
        retrieved_sources = [m.get("source", "N/A") for m in retrieved_metadatas if m]
        
        chat_prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT)
        output_parser = StrOutputParser()
        chain = chat_prompt | current_llm | output_parser # Use current_llm
        
        try:
            response = await chain.ainvoke({
                "context": context,
                "question": query
            })
            generation = response
        except Exception as e:
            logger.error(f"Error during LLM generation: {e}")
            generation = f"An error occurred during LLM generation: {str(e)}"
            return {"generation": generation, "query": query, "urls": state.get("urls", []), "scraped_content": state.get("scraped_content", {}), "retrieved_sources": retrieved_sources, "error": str(e), "retrieved_elements": retrieved_elements_data}

    logger.info("Generation complete.")
    return {"generation": generation, "query": query, "urls": state.get("urls", []), "scraped_content": state.get("scraped_content", {}), "retrieved_sources": list(set(retrieved_sources)), "error": "", "retrieved_elements": retrieved_elements_data}


def decide_to_generate(state: GraphState) -> Literal["generate", "end_with_error"]:
    """
    Determines whether to proceed to generation or end due to an error/no results.
    """
    if state.get("error"):
        logger.info("---DECISION: Error detected, ending with error.---")
        return "end_with_error"
    
    logger.info("---DECISION: Proceeding to generate.---")
    return "generate"

# Ensure RAG components are initialized when this module is imported.
# This ensures global variables are set up initially.
# It's good practice to call this, but FastAPI's on_event("startup") is more robust.
# Keep this here for standalone testing/execution of this module.
initialize_rag_components()