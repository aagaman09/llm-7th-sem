# evaluation_agent.py
from typing import Dict, Any
from rag_components import GraphState
from evaluator import RAGEvaluator
from loguru import logger

# Initialize the evaluator globally or pass it.
# Global initialization is simpler for a single instance.
evaluator = RAGEvaluator(model_name="llama3.2:1b") # Use llama3.2:3b or specific model for evaluation

async def evaluate_rag_node(state: GraphState) -> GraphState:
    """
    Node to evaluate the RAG pipeline's performance based on the generated answer
    and retrieved context.
    """
    logger.info("---EVALUATION NODE: Evaluating RAG performance---")

    query = state["query"]
    generated_answer = state["generation"]
    
    # Retrieved documents are available in retrieved_elements['documents']
    retrieved_context_docs = state.get("retrieved_elements", {}).get("documents", [])
    
    ground_truth = state.get("ground_truth", None) # Get ground truth if available in state

    if not generated_answer:
        logger.warning("No answer generated, skipping RAG evaluation.")
        return {"evaluation_results": {"error": "No answer to evaluate."}}

    if not retrieved_context_docs:
        logger.warning("No context retrieved, skipping full RAG evaluation (will still evaluate relevance).")
        # You might want to handle this more gracefully, perhaps by only running relevance eval
        # For now, evaluate with empty context for groundedness/retrieval relevance
        # or skip those if context is strictly required.
    
    try:
        # Perform the complete RAG evaluation
        evaluation_results = evaluator.evaluate_complete_rag(
            question=query,
            answer=generated_answer,
            context=retrieved_context_docs,
            ground_truth=ground_truth # Pass ground_truth if available
        )
        logger.success(f"RAG evaluation completed with overall score: {evaluation_results.get('overall_score', 'N/A'):.2f}")
        return {"evaluation_results": evaluation_results}

    except Exception as e:
        logger.error(f"Error during RAG evaluation: {e}")
        return {"evaluation_results": {"error": f"Evaluation failed: {str(e)}"}}