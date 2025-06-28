from typing_extensions import Annotated, TypedDict
from typing import List, Dict, Any, Optional
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field
import json
from loguru import logger

# Initialize Ollama LLM for evaluation with a larger, more capable model
evaluator_llm = ChatOllama(model="llama3.2:1b", temperature=0)

# Grade schemas for structured output
class CorrectnessGrade(BaseModel):
    """Schema for correctness evaluation"""
    explanation: str = Field(description="Explain your reasoning for the score")
    correct: bool = Field(description="True if the answer is correct, False otherwise")

class RelevanceGrade(BaseModel):
    """Schema for relevance evaluation"""
    explanation: str = Field(description="Explain your reasoning for the score")
    relevant: bool = Field(description="True if the answer is relevant, False otherwise")
    score: int = Field(description="Relevance score from 1-5", ge=1, le=5)

class GroundednessGrade(BaseModel):
    """Schema for groundedness evaluation"""
    explanation: str = Field(description="Explain your reasoning for the score")
    grounded: bool = Field(description="True if the answer is grounded in context, False otherwise")
    hallucination: bool = Field(description="True if answer contains hallucinations, False otherwise")

class RetrievalRelevanceGrade(BaseModel):
    """Schema for retrieval relevance evaluation"""
    explanation: str = Field(description="Explain your reasoning for the score")
    relevant: bool = Field(description="True if retrieved docs are relevant, False otherwise")
    score: int = Field(description="Relevance score from 1-5", ge=1, le=5)

# Evaluation prompts
CORRECTNESS_INSTRUCTIONS = """You are evaluating if a student's answer is factually correct compared to the ground truth.

Compare the student answer to the ground truth answer:
- Correct (true): The student answer is factually accurate and aligns with the ground truth
- Incorrect (false): The student answer contains factual errors or contradicts the ground truth

The student answer can have additional information as long as it's accurate."""

RELEVANCE_INSTRUCTIONS = """You are evaluating how well a generated response addresses the user's question.

Does the response directly address the question? Is it helpful and on-topic?

Rate 1-5:
- 5: Perfectly answers the question
- 4: Good answer with minor gaps  
- 3: Partially answers the question
- 2: Somewhat related but incomplete
- 1: Does not answer the question

Be concise in your explanation."""

GROUNDEDNESS_INSTRUCTIONS = """You are checking if a response is supported by the provided context.

Key question: Does the response contain information that is NOT in the context?

- Grounded (true): All claims in the response can be found in the context
- Not grounded (false): The response contains claims not supported by the context

A response is grounded if it only uses information from the context, even if it doesn't use all the context.

Be strict: if ANY claim cannot be verified from the context, mark as not grounded."""

RETRIEVAL_RELEVANCE_INSTRUCTIONS = """You are evaluating if the retrieved documents can help answer the question.

Do the documents contain useful information for answering the question?

Rate 1-5:
- 5: Documents directly address the question topic
- 4: Documents contain very useful information
- 3: Documents have some relevant information
- 2: Documents are somewhat related
- 1: Documents are not helpful for the question

Consider all retrieved documents together."""

class RAGEvaluator:
    """Comprehensive RAG evaluation system using Ollama"""
    
    def __init__(self, model_name: str = "llama3.2:3b", temperature: float = 0):
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        logger.info(f"Initialized RAG Evaluator with model: {model_name}")
    
    def _parse_structured_output(self, response: str, schema_class) -> Dict:
        """Parse LLM response into structured format"""
        try:
            # Try to extract JSON from the response
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                validated_data = schema_class(**data)
                return validated_data.dict()
        except Exception as e:
            logger.warning(f"Failed to parse structured output: {e}")
        
        # Fallback: manual parsing
        result = self._manual_parse(response, schema_class)
        
        # Validate the result using the schema
        try:
            validated_result = schema_class(**result)
            return validated_result.dict()
        except Exception as e:
            logger.warning(f"Manual parsing result validation failed: {e}, using result as-is")
            return result
    
    def _manual_parse(self, response: str, schema_class) -> Dict:
        """Manual parsing fallback with improved logic"""
        result = {}
        response_lower = response.lower()
        
        # Extract explanation - take the full response as explanation
        result["explanation"] = response.strip()
        
        # Improved boolean detection
        if "correct" in schema_class.__fields__:
            # Look for explicit true/false or positive/negative indicators
            if "correct: true" in response_lower or "\"correct\": true" in response_lower:
                result["correct"] = True
            elif "correct: false" in response_lower or "\"correct\": false" in response_lower:
                result["correct"] = False
            else:
                # Fallback: look for positive indicators
                positive_words = ["correct", "accurate", "right", "yes"]
                negative_words = ["incorrect", "wrong", "false", "no", "inaccurate"]
                pos_count = sum(1 for word in positive_words if word in response_lower)
                neg_count = sum(1 for word in negative_words if word in response_lower)
                result["correct"] = pos_count > neg_count
        
        if "relevant" in schema_class.__fields__:
            if "relevant: true" in response_lower or "\"relevant\": true" in response_lower:
                result["relevant"] = True
            elif "relevant: false" in response_lower or "\"relevant\": false" in response_lower:
                result["relevant"] = False
            else:
                # Check for relevance indicators
                result["relevant"] = "relevant" in response_lower and "not relevant" not in response_lower
        
        if "grounded" in schema_class.__fields__:
            if "grounded: true" in response_lower or "\"grounded\": true" in response_lower:
                result["grounded"] = True
            elif "grounded: false" in response_lower or "\"grounded\": false" in response_lower:
                result["grounded"] = False
            else:
                # Look for grounding indicators
                grounded_indicators = ["grounded", "supported", "backed by context"]
                ungrounded_indicators = ["not grounded", "unsupported", "hallucination", "fabricated"]
                
                grounded_count = sum(1 for indicator in grounded_indicators if indicator in response_lower)
                ungrounded_count = sum(1 for indicator in ungrounded_indicators if indicator in response_lower)
                result["grounded"] = grounded_count > ungrounded_count
        
        if "hallucination" in schema_class.__fields__:
            if "hallucination: true" in response_lower or "\"hallucination\": true" in response_lower:
                result["hallucination"] = True
            elif "hallucination: false" in response_lower or "\"hallucination\": false" in response_lower:
                result["hallucination"] = False
            else:
                # Hallucination is opposite of grounded
                hallucination_indicators = ["hallucination", "fabricated", "not supported", "made up"]
                result["hallucination"] = any(indicator in response_lower for indicator in hallucination_indicators)
        
        # Extract score with better regex
        if "score" in schema_class.__fields__:
            import re
            # Look for score patterns like "score: 3", "score": 3, or just standalone numbers
            score_patterns = [
                r'score[:\s]*(\d)',
                r'"score":\s*(\d)',
                r'(\d)/5',
                r'rate[:\s]*(\d)',
                r'rating[:\s]*(\d)'
            ]
            
            for pattern in score_patterns:
                score_match = re.search(pattern, response_lower)
                if score_match:
                    score = int(score_match.group(1))
                    if 1 <= score <= 5:
                        result["score"] = score
                        break
            
            # Default score if none found
            if "score" not in result:
                result["score"] = 3
        
        return result
    
    def evaluate_correctness(self, question: str, student_answer: str, ground_truth: str) -> Dict:
        """Evaluate answer correctness against ground truth"""
        prompt = f"""
{CORRECTNESS_INSTRUCTIONS}

QUESTION: {question}
GROUND TRUTH ANSWER: {ground_truth}
STUDENT ANSWER: {student_answer}

Is the student answer factually correct compared to the ground truth?

Format your response as:
Correct: [true/false]
Explanation: [Your reasoning]
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_structured_output(response.content, CorrectnessGrade)
            logger.info(f"Correctness evaluation completed: {result['correct']}")
            return result
        except Exception as e:
            logger.error(f"Error in correctness evaluation: {e}")
            return {"explanation": f"Evaluation error: {e}", "correct": False}
    
    def evaluate_relevance(self, question: str, answer: str) -> Dict:
        """Evaluate how well the answer addresses the question"""
        prompt = f"""
{RELEVANCE_INSTRUCTIONS}

QUESTION: {question}

ANSWER: {answer}

Rate the relevance (1-5) and explain why.

Format your response as:
Score: [1-5]
Relevant: [true/false]
Explanation: [Your reasoning]
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_structured_output(response.content, RelevanceGrade)
            
            # Ensure consistency between score and relevant boolean
            if result["score"] >= 3:
                result["relevant"] = True
            else:
                result["relevant"] = False
                
            logger.info(f"Relevance evaluation completed: {result['score']}/5, relevant={result['relevant']}")
            return result
        except Exception as e:
            logger.error(f"Error in relevance evaluation: {e}")
            return {"explanation": f"Evaluation error: {e}", "relevant": False, "score": 1}
    
    def evaluate_groundedness(self, answer: str, context: List[str]) -> Dict:
        """Evaluate if the answer is grounded in the retrieved context"""
        context_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])
        
        prompt = f"""
{GROUNDEDNESS_INSTRUCTIONS}

CONTEXT:
{context_text}

GENERATED ANSWER: {answer}

Check each claim in the answer against the context. Is the answer grounded?

Format your response as:
Grounded: [true/false]
Hallucination: [true/false]
Explanation: [Your reasoning]
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_structured_output(response.content, GroundednessGrade)
            
            # Ensure logical consistency: if grounded is true, hallucination should be false
            if result["grounded"]:
                result["hallucination"] = False
            elif not result["grounded"]:
                result["hallucination"] = True
                
            logger.info(f"Groundedness evaluation completed: grounded={result['grounded']}, hallucination={result['hallucination']}")
            return result
        except Exception as e:
            logger.error(f"Error in groundedness evaluation: {e}")
            return {"explanation": f"Evaluation error: {e}", "grounded": False, "hallucination": True}
    
    def evaluate_retrieval_relevance(self, question: str, retrieved_docs: List[str]) -> Dict:
        """Evaluate relevance of retrieved documents to the question"""
        docs_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(retrieved_docs)])
        
        prompt = f"""
{RETRIEVAL_RELEVANCE_INSTRUCTIONS}

QUESTION: {question}

RETRIEVED DOCUMENTS:
{docs_text}

Rate how well these documents can help answer the question.

Format your response as:
Score: [1-5]
Relevant: [true/false]
Explanation: [Your reasoning]
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = self._parse_structured_output(response.content, RetrievalRelevanceGrade)
            
            # Ensure consistency between score and relevant boolean
            if result["score"] >= 3:
                result["relevant"] = True
            else:
                result["relevant"] = False
                
            logger.info(f"Retrieval relevance evaluation completed: {result['score']}/5, relevant={result['relevant']}")
            return result
        except Exception as e:
            logger.error(f"Error in retrieval relevance evaluation: {e}")
            return {"explanation": f"Evaluation error: {e}", "relevant": False, "score": 1}
    
    def evaluate_complete_rag(self, question: str, answer: str, context: List[str], 
                             ground_truth: Optional[str] = None) -> Dict:
        """Perform complete RAG evaluation with all metrics"""
        results = {}
        
        # Always evaluate these three
        results["relevance"] = self.evaluate_relevance(question, answer)
        results["groundedness"] = self.evaluate_groundedness(answer, context)
        results["retrieval_relevance"] = self.evaluate_retrieval_relevance(question, context)
        
        # Only evaluate correctness if ground truth is provided
        if ground_truth:
            results["correctness"] = self.evaluate_correctness(question, answer, ground_truth)
        
        # Calculate overall score with better weighting
        scores = []
        weights = []
        
        # Relevance score (high weight - most important)
        scores.append(results["relevance"]["score"])
        weights.append(0.4)
        
        # Groundedness score (high weight - very important for RAG)
        groundedness_score = 5 if results["groundedness"]["grounded"] else 1
        scores.append(groundedness_score)
        weights.append(0.4)
        
        # Retrieval relevance score (medium weight)
        scores.append(results["retrieval_relevance"]["score"])
        weights.append(0.2)
        
        # Correctness score (high weight if available)
        if ground_truth and "correctness" in results:
            correctness_score = 5 if results["correctness"]["correct"] else 1
            scores.append(correctness_score)
            weights.append(0.3)
            # Adjust other weights to accommodate correctness
            weights[0] = 0.3  # relevance
            weights[1] = 0.3  # groundedness  
            weights[2] = 0.1  # retrieval relevance
        
        # Calculate weighted average
        total_weight = sum(weights)
        weighted_scores = [score * weight for score, weight in zip(scores, weights)]
        results["overall_score"] = sum(weighted_scores) / total_weight
        
        results["summary"] = {
            "total_evaluations": len(scores),
            "has_ground_truth": ground_truth is not None,
            "overall_score": results["overall_score"],
            "component_scores": {
                "relevance": results["relevance"]["score"],
                "groundedness": groundedness_score,
                "retrieval_relevance": results["retrieval_relevance"]["score"]
            }
        }
        
        if ground_truth and "correctness" in results:
            results["summary"]["component_scores"]["correctness"] = correctness_score
        
        logger.info(f"Complete RAG evaluation finished. Overall score: {results['overall_score']:.2f}/5")
        return results

    def debug_evaluation(self, question: str, answer: str, context: List[str], 
                        ground_truth: Optional[str] = None) -> Dict:
        """
        Debug version of evaluation that shows the raw LLM responses
        to help understand evaluation issues
        """
        debug_results = {
            "question": question,
            "answer": answer,
            "context_summary": f"{len(context)} documents retrieved",
            "raw_responses": {}
        }
        
        # Test relevance with raw response
        relevance_prompt = f"""
{RELEVANCE_INSTRUCTIONS}

QUESTION: {question}

ANSWER: {answer}

Rate the relevance (1-5) and explain why.

Format your response as:
Score: [1-5]
Relevant: [true/false]
Explanation: [Your reasoning]
"""
        try:
            relevance_response = self.llm.invoke(relevance_prompt)
            debug_results["raw_responses"]["relevance"] = relevance_response.content
            debug_results["relevance_parsed"] = self._parse_structured_output(relevance_response.content, RelevanceGrade)
        except Exception as e:
            debug_results["raw_responses"]["relevance"] = f"Error: {e}"
        
        # Test groundedness with raw response
        context_text = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])
        groundedness_prompt = f"""
{GROUNDEDNESS_INSTRUCTIONS}

CONTEXT:
{context_text}

GENERATED ANSWER: {answer}

Check each claim in the answer against the context. Is the answer grounded?

Format your response as:
Grounded: [true/false]
Hallucination: [true/false]
Explanation: [Your reasoning]
"""
        try:
            groundedness_response = self.llm.invoke(groundedness_prompt)
            debug_results["raw_responses"]["groundedness"] = groundedness_response.content
            debug_results["groundedness_parsed"] = self._parse_structured_output(groundedness_response.content, GroundednessGrade)
        except Exception as e:
            debug_results["raw_responses"]["groundedness"] = f"Error: {e}"
        
        return debug_results

# Example usage and testing
def test_evaluator():
    """Test the RAG evaluator with sample data"""
    evaluator = RAGEvaluator()
    
    # Sample data
    question = "What is the capital of France?"
    answer = "The capital of France is Paris. It is located in the north-central part of the country."
    ground_truth = "Paris is the capital of France."
    context = [
        "Paris is the capital and most populous city of France.",
        "Located in northern France, Paris is known for its culture and history.",
        "The city has been the capital since the 12th century."
    ]
    
    # Test individual evaluations
    print("=== Testing Individual Evaluations ===")
    
    correctness_result = evaluator.evaluate_correctness(question, answer, ground_truth)
    print(f"Correctness: {correctness_result}")
    
    relevance_result = evaluator.evaluate_relevance(question, answer)
    print(f"Relevance: {relevance_result}")
    
    groundedness_result = evaluator.evaluate_groundedness(answer, context)
    print(f"Groundedness: {groundedness_result}")
    
    retrieval_result = evaluator.evaluate_retrieval_relevance(question, context)
    print(f"Retrieval Relevance: {retrieval_result}")
    
    # Test complete evaluation
    print("\n=== Testing Complete RAG Evaluation ===")
    complete_result = evaluator.evaluate_complete_rag(question, answer, context, ground_truth)
    print(f"Complete Evaluation: {json.dumps(complete_result, indent=2)}")

if __name__ == "__main__":
    test_evaluator()