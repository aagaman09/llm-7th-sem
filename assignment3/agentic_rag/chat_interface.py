import asyncio # Add this if not present for async functionality
from typing import Optional, Literal # Keep Literal if used for decision nodes
from loguru import logger
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma # Import Chroma if used for retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter # If needed for any splitting in chat

# Import shared RAG components and state.
# --- FIX: Update imports to use the new getter functions and initialization flag ---
from rag_components import (
    GraphState,
    generate_node,
    decide_to_generate, # Often used with generate_node in conditional edges
    get_chroma_client,
    get_chroma_collection,
    get_llm, # For accessing the LLM instance
    get_embeddings_instance, # For accessing the embeddings instance
    initialize_rag_components, # Useful to ensure startup init if not handled by FastAPI lifespan
    is_rag_initialized # For checking initialization status
)
# Import evaluation node
from evaluation_agent import evaluate_rag_node

### Build the Graph for Conversational Chat ###
workflow_chat = StateGraph(GraphState)

# Note: generate_node from rag_components already handles retrieval
# if you want a separate retrieve_node for chat, define it and add it to the graph.
# If generate_node handles both, then you don't need a separate chat_retrieve_node in the graph flow.
# Given your current graph only has generate_chat_response and evaluate_chat_response,
# it implies generate_node handles retrieval internally, which is what our `rag_components.py`
# `generate_node` function does.

workflow_chat.add_node("generate_chat_response", generate_node) # This `generate_node` internally does retrieval
workflow_chat.add_node("evaluate_chat_response", evaluate_rag_node)

workflow_chat.set_entry_point("generate_chat_response") # Assuming generate_node starts the process

workflow_chat.add_edge("generate_chat_response", "evaluate_chat_response")
workflow_chat.add_edge("evaluate_chat_response", END)

rag_chat_app = workflow_chat.compile()

# --- IMPORTANT: If you want a separate retrieve_node *in this chat graph*,
# you need to add it to the workflow and define its edges.
# The chat_retrieve_node function itself is correct in its usage of getters.
# However, your current `workflow_chat` does NOT include `chat_retrieve_node`.
# If `generate_node` already performs retrieval (as per `rag_components.py`),
# then this `chat_retrieve_node` function is currently unused in this graph.
# I'll keep the definition here for completeness, but note its non-inclusion in the graph.
def chat_retrieve_node(state: GraphState) -> GraphState:
    """
    Retrieves documents from the vector database based on the user's chat question.
    This is often integrated into generate_node, but could be a separate node if needed.
    """
    logger.info("---CHAT RETRIEVE NODE---")
    question = state["query"] # The chat question is the query

    try:
        # --- FIX: Use get_chroma_client and get_chroma_collection ---
        current_client = get_chroma_client()
        current_collection = get_chroma_collection()
        
        # Initialize the retriever using the *initialized embeddings instance*
        retriever = Chroma(
            client=current_client, # Use the client from the getter
            collection_name=current_collection.name, # Use the collection name from the getter
            embedding_function=get_embeddings_instance() # <-- Use the getter function here
        ).as_retriever(search_kwargs={"k": state.get("n_results", 3)}) # Use n_results from state, default to 3

        retrieved_elements = retriever.invoke(question)
        retrieved_sources = [doc.metadata.get('source', 'N/A') for doc in retrieved_elements]

        logger.info(f"Retrieved {len(retrieved_elements)} documents for chat.")
        logger.debug(f"Chat retrieved sources: {retrieved_sources}")

        return {**state, "retrieved_elements": retrieved_elements, "retrieved_sources": retrieved_sources}
    except Exception as e:
        logger.error(f"Error in chat_retrieve_node: {e}")
        return {**state, "error": f"Chat retrieval failed: {e}"}

async def chat_with_scraped_data(question: str, n_results: Optional[int] = 3) -> dict:
    """
    Handles conversational chat by querying the stored data and returning an answer,
    along with evaluation results for the generated response.
    """
    logger.info(f"\n--- Starting Chat Pipeline for question: '{question}' ---")

    # --- FIX: Ensure RAG components are initialized ---
    if not is_rag_initialized():
        logger.warning("RAG components not initialized. Attempting initialization now for chat...")
        if not initialize_rag_components(): # Call the initialization function
            error_msg = "Failed to initialize RAG components for chat. Cannot proceed."
            logger.error(error_msg)
            return {"answer": "Error: Failed to set up chat environment.", "success": False, "error": error_msg}

    inputs = {
        "query": question,
        "n_results": n_results,
        "urls": [],
        "scraped_content": {},
        "generation": "",
        "error": "",
        "retrieved_sources": [],
        "retrieved_elements": None,
        "evaluation_results": None,
        "ground_truth": None # Ground truth is typically not available for live chat
    }

    final_state = await rag_chat_app.ainvoke(inputs)

    logger.info("\n--- Chat Pipeline Finished ---")

    answer = final_state.get("generation", "Sorry, I could not generate a response.")
    sources = final_state.get("retrieved_sources", [])
    chat_error = final_state.get("error", "")
    evaluation_results = final_state.get("evaluation_results", None)

    response_data = {
        "answer": answer,
        "success": True if not chat_error else False,
        "message": chat_error if chat_error else "Response generated successfully.",
        "retrieved_sources": sources,
        "evaluation_results": evaluation_results
    }

    logger.debug(f"Final chat pipeline response: {response_data}")
    return response_data

# Example of how to run this file directly for testing
if __name__ == "__main__":
    async def main_chat_test():
        logger.info("Running chat interface test in standalone mode.")
        # Make sure you have some data in './chroma_store' for this to retrieve
        
        # Example 1: Basic query
        test_query_1 = "What is the capital of Nepal?" # Replace with query relevant to your ingested data
        print(f"\n--- Chatting about: '{test_query_1}' ---")
        result_1 = await chat_with_scraped_data(test_query_1)
        print(f"Answer: {result_1['answer']}")
        print(f"Sources: {result_1['retrieved_sources']}")
        print(f"Success: {result_1['success']}")
        print(f"Error: {result_1['message']}")

        # Example 2: Another query
        test_query_2 = "Tell me about the history of the Kathmandu Durbar Square."
        print(f"\n--- Chatting about: '{test_query_2}' ---")
        result_2 = await chat_with_scraped_data(test_query_2)
        print(f"Answer: {result_2['answer']}")
        print(f"Sources: {result_2['retrieved_sources']}")
        print(f"Success: {result_2['success']}")
        print(f"Error: {result_2['message']}")

    import asyncio
    asyncio.run(main_chat_test())