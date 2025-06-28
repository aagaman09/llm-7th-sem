from typing import Optional
from loguru import logger

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma # Also need to import Chroma for the retrieve_node

# Import shared RAG components and state.
# --- FIX: Update imports to use the getter functions and new global names from rag_components ---
from rag_components import GraphState, is_rag_initialized, generate_node, decide_to_generate, get_chroma_client, get_chroma_collection, get_llm, get_embeddings_instance
# Import agents
from scraping_agent import ScrapingAgent
from search_agent import GoogleSearchLinkAgent

from evaluation_agent import evaluate_rag_node # Keep this, as evaluation is part of the ingestion flow too

# --- Nodes for Ingestion Workflow ---

async def search_node(state: GraphState) -> GraphState:
    """
    Node to perform a Google search and extract links.
    """
    logger.info("---SEARCH NODE: Extracting links from Google Search---")
    query = state["query"]
    
    # --- FIX: Use the new getter for rag_initialized status ---
    if not is_rag_initialized():
        logger.error("RAG components not initialized. Cannot proceed with search.")
        return {"urls": [], "query": query, "error": "RAG components not initialized for search."}

    search_extractor = GoogleSearchLinkAgent(num_results=3) # Get top 3 links
    try:
        urls = search_extractor.get_links(query)
        logger.info(f"Search Agent found {len(urls)} URLs.")
        return {"urls": urls, "query": query, "error": ""}
    except Exception as e:
        logger.error(f"Error in Search Node: {e}")
        return {"urls": [], "query": query, "error": str(e)}

# --- NEW: Added/Corrected retrieve_node for ingestion pipeline ---
# This node should exist in your ingestion.py if you intend to retrieve for the initial answer
def retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieves documents from the vector database based on the user's query.
    """
    logger.info("---RETRIEVE NODE in INGESTION---")
    query = state["query"]

    try:
        # --- FIX: Use new getter functions for client and collection ---
        current_client = get_chroma_client()
        current_collection = get_chroma_collection()
        
        # Initialize the retriever using the *initialized embeddings instance*
        retriever = Chroma(
            client=current_client, # Use current_client (chromadb.PersistentClient)
            collection_name=current_collection.name, # Access name from the collection object
            embedding_function=get_embeddings_instance() # <-- Use the getter function here
        ).as_retriever(search_kwargs={"k": state["n_results"]})

        retrieved_elements = retriever.invoke(query)
        retrieved_sources = [doc.metadata.get('source', 'N/A') for doc in retrieved_elements]

        logger.info(f"Retrieved {len(retrieved_elements)} documents.")
        logger.debug(f"Retrieved sources: {retrieved_sources}")

        return {**state, "retrieved_elements": retrieved_elements, "retrieved_sources": retrieved_sources}
    except Exception as e:
        logger.error(f"Error in retrieve_node: {e}")
        return {**state, "error": f"Retrieval failed: {e}"}
# --- END NEW: Added/Corrected retrieve_node ---


async def scrape_and_store_node(state: GraphState) -> GraphState:
    """
    Node to scrape the content from the extracted URLs, chunk it, embed it,
    and store it in the ChromaDB collection.
    It now uses `get_chroma_collection()` to ensure it has the latest ChromaDB objects.
    """
    logger.info("---SCRAPE & STORE NODE: Scraping content and storing in DB---")
    urls = state["urls"]
    query = state["query"]

    # Get the latest client and collection from rag_components dynamically
    try:
        # --- FIX: Only need the collection for adding data ---
        current_collection = get_chroma_collection()
    except RuntimeError as e:
        error_msg = f"Failed to get ChromaDB components for scrape and store: {str(e)}"
        logger.error(error_msg)
        return {"scraped_content": {}, "urls": urls, "query": query, "error": error_msg}

    # --- FIX: Use the getter function for embeddings here ---
    initialized_embeddings_instance = get_embeddings_instance()
    if initialized_embeddings_instance is None:
        error_msg = "Embeddings not initialized for scraping and storage."
        logger.error(error_msg)
        return {"scraped_content": {}, "urls": urls, "query": query, "error": error_msg}
    # --- END FIX ---

    if not urls:
        logger.info("No URLs to scrape. Skipping scraping and storage.")
        return {"scraped_content": {}, "urls": urls, "query": query, "error": state.get("error", "No URLs to scrape.")}

    scraping_agent = ScrapingAgent()
    scraped_data = {}
    total_chunks_processed = 0

    try:
        scraped_data = await scraping_agent.scrape_urls(urls)

        for url, markdown_content in scraped_data.items():
            if markdown_content:
                logger.info(f"Processing scraped content from: {url}")
                doc = Document(page_content=markdown_content, metadata={"source": url, "type": "web_scrape"})

                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=2000,
                    chunk_overlap=400,
                    separators=["\n\n", "\n", ".", "?", "!", " ", ""],
                    length_function=len,
                    is_separator_regex=False
                )
                splits = text_splitter.split_documents([doc])

                if not splits:
                    logger.warning(f"No chunks generated for {url}. Content might be too short or problematic.")
                    continue

                logger.info(f"Generated {len(splits)} chunks for {url}.")

                valid_texts = []
                valid_metadatas = []
                for i, split_doc in enumerate(splits): # Iterate through Document objects
                    if split_doc.page_content and split_doc.page_content.strip():
                        valid_texts.append(split_doc.page_content)
                        valid_metadatas.append(split_doc.metadata)
                    else:
                        logger.warning(f"Skipping empty chunk from {url} (chunk index {i}).")

                if valid_texts:
                    # --- FIX: Use the initialized_embeddings_instance here ---
                    embeddings_list = initialized_embeddings_instance.embed_documents(valid_texts)

                    for i, (text_chunk, emb, metadata) in enumerate(zip(valid_texts, embeddings_list, valid_metadatas)):
                        # Create a unique ID for each chunk
                        chunk_id = f"{url.replace('.', '_').replace('/', '_').replace(':', '')}_chunk_{i}"
                        
                        # Use current_collection instead of global collection
                        current_collection.add(
                            ids=[chunk_id],
                            documents=[text_chunk],
                            embeddings=[emb], # embeddings are directly provided here
                            metadatas=[metadata]
                        )
                        total_chunks_processed += 1
                else:
                    logger.warning(f"No valid texts to embed for {url} after stripping empty chunks.")

        logger.success(f"Successfully processed and stored {total_chunks_processed} chunks from scraped URLs.")
        return {"scraped_content": scraped_data, "urls": urls, "query": query, "error": ""}

    except Exception as e:
        logger.error(f"Error in Scrape & Store Node: {e}")
        return {"scraped_content": scraped_data, "urls": urls, "query": query, "error": str(e)}


### Build the Graph for Initial Ingestion ###
workflow_ingestion = StateGraph(GraphState)

workflow_ingestion.add_node("search", search_node)
workflow_ingestion.add_node("scrape_and_store", scrape_and_store_node)
workflow_ingestion.add_node("retrieve", retrieve_node) # Add retrieve node
workflow_ingestion.add_node("generate_initial", generate_node) 
workflow_ingestion.add_node("evaluate_rag", evaluate_rag_node)

workflow_ingestion.set_entry_point("search")

workflow_ingestion.add_edge("search", "scrape_and_store")
workflow_ingestion.add_conditional_edges(
    "scrape_and_store",
    decide_to_generate,
    {
        "generate": "retrieve", # Go to retrieve after scrape and store
        "end_with_error": END,
    },
)

workflow_ingestion.add_edge("retrieve", "generate_initial") # Add edge from retrieve to generate
workflow_ingestion.add_edge("generate_initial", "evaluate_rag")
workflow_ingestion.add_edge("evaluate_rag", END)

rag_ingestion_app = workflow_ingestion.compile()

async def run_ingestion_pipeline(query: str, n_results: Optional[int] = 5, ground_truth: Optional[str] = None):
    """
    Runs the RAG pipeline to search, scrape, store, generate an initial response,
    and optionally evaluate the RAG performance.
    """
    logger.info(f"\n--- Starting Data Ingestion Pipeline for query: '{query}' ---")
    
    inputs = {
        "query": query,
        "urls": [],
        "scraped_content": {},
        "generation": "",
        "error": "",
        "retrieved_sources": [],
        "n_results": n_results,
        "retrieved_elements": None,
        "evaluation_results": None, # Initialize evaluation results
        "ground_truth": ground_truth # Pass ground truth if provided
    }
    
    final_state = await rag_ingestion_app.ainvoke(inputs)
    
    logger.info("\n--- Data Ingestion Pipeline Finished ---")
    
    response_data = {
        "initial_answer": final_state.get("generation", "No initial answer generated."),
        "success": True if not final_state.get("error") else False,
        "message": final_state.get("error") if final_state.get("error") else "Ingestion and initial generation completed successfully.",
        "scraped_urls": final_state.get("urls", []),
        "retrieved_sources": final_state.get("retrieved_sources", []),
        "evaluation_results": final_state.get("evaluation_results", None) # Include evaluation results
    }

    logger.debug(f"Final ingestion pipeline response: {response_data}")
    return response_data