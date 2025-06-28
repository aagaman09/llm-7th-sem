# main.py

import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

# Import the main functions from our modularized files
from ingestion import run_ingestion_pipeline
from chat_interface import chat_with_scraped_data
from rag_components import initialize_rag_components # Make sure this is imported if used directly

async def main():
    """
    Orchestrates the RAG data ingestion pipeline and then allows for conversational chat.
    """
    # Load environment variables for Google Search API
    load_dotenv()

    # --- Ensure RAG components are initialized before starting any pipeline ---
    logger.info("\n" + "="*80)
    logger.info("PHASE 0: Initializing RAG components (for direct script execution)")
    logger.info("="*80 + "\n")
    if not initialize_rag_components():
        logger.error("RAG components failed to initialize. Cannot proceed. "
                     "Ensure Ollama is running and models are installed.")
        return

    # --- PHASE 1: Data Ingestion (Search, Scrape, Store, Initial Answer, Evaluation) ---
    logger.info("\n" + "="*80)
    logger.info("PHASE 1: Running Data Ingestion Pipeline with Evaluation")
    logger.info("="*80 + "\n")
    
    # Define the initial query for data ingestion
    ingestion_query = "Who is Shah Rukh Khan and what are his famous movies?"
    # Add a ground truth for correctness evaluation (optional, but good for testing)
    # This ground truth would ideally come from a curated dataset.
    ingestion_ground_truth = "Shah Rukh Khan is an Indian actor, film producer, and television personality. He is known for his work in over 80 Hindi films. Some of his famous movies include Dilwale Dulhania Le Jayenge, Kuch Kuch Hota Hai, My Name Is Khan, and Chennai Express."

    ingestion_response = await run_ingestion_pipeline(
        ingestion_query, 
        n_results=5,
        ground_truth=ingestion_ground_truth # Pass ground truth here
    )
    
    logger.info("\n--- Ingestion Pipeline Summary ---")
    logger.info(f"Initial Answer: {ingestion_response['initial_answer']}")
    logger.info(f"Success: {ingestion_response['success']}")
    logger.info(f"Message: {ingestion_response['message']}")
    logger.info(f"Scraped URLs: {ingestion_response['scraped_urls']}")
    logger.info(f"Sources Used for Initial Answer: {ingestion_response['retrieved_sources']}")
    
    # --- NEW: Display Evaluation Results ---
    if ingestion_response.get("evaluation_results"):
        logger.info("\n--- RAG Evaluation Results for Ingestion ---")
        eval_res = ingestion_response["evaluation_results"]
        logger.info(f"Overall Score: {eval_res.get('overall_score', 'N/A'):.2f}/5")
        if "correctness" in eval_res:
            logger.info(f"Correctness: {eval_res['correctness']['correct']} - {eval_res['correctness']['explanation']}")
        logger.info(f"Relevance: {eval_res['relevance']['score']}/5 - {eval_res['relevance']['explanation']}")
        logger.info(f"Groundedness: {eval_res['groundedness']['grounded']} (Hallucination: {eval_res['groundedness']['hallucination']}) - {eval_res['groundedness']['explanation']}")
        logger.info(f"Retrieval Relevance: {eval_res['retrieval_relevance']['score']}/5 - {eval_res['retrieval_relevance']['explanation']}")
    else:
        logger.warning("No evaluation results available for ingestion.")
    # --- END NEW ---

    logger.info("\n" + "="*80 + "\n")

    if not ingestion_response['success']:
        logger.error("Ingestion failed, cannot proceed to chat. Check logs for errors.")
        return

    # --- PHASE 2: Conversational Chat with Stored Data ---
    logger.info("\n" + "="*80)
    logger.info("PHASE 2: Starting Conversational Chat with Stored Data")
    logger.info("="*80 + "\n")

    print("You can now ask questions based on the ingested content.")
    print("Type 'exit' to end the chat.")

    while True:
        user_question = input("\nYour question (or 'exit'): ")
        if user_question.lower() == 'exit':
            print("Exiting chat. Goodbye!")
            break
        
        chat_response = await chat_with_scraped_data(user_question, n_results=3) 
        
        print("\n--- Chat Response ---")
        print(f"Answer: {chat_response['answer']}")
        print(f"Success: {chat_response['success']}")
        print(f"Message: {chat_response['message']}")
        print(f"Sources Used: {chat_response.get('retrieved_sources', [])}")
        print("-" * 30)

if __name__ == "__main__":
    asyncio.run(main())