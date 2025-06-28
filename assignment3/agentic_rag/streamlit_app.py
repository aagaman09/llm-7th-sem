import streamlit as st
import requests
import json

# Configuration for the FastAPI backend
FASTAPI_BASE_URL = "http://localhost:5000"

# --- Page Configuration ---
st.set_page_config(page_title="RAG Chat System", layout="wide") # Changed layout to 'wide' for full width

st.title("📚 RAG Chat System")
st.markdown("---") # Separator below the main title

# --- Session State Initialization ---
# Initialize session state variables if they don't exist
if 'ingestion_status' not in st.session_state:
    st.session_state.ingestion_status = ""
if 'ingestion_error_message' not in st.session_state: # To store specific ingestion error
    st.session_state.ingestion_error_message = ""
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'ingestion_successful' not in st.session_state:
    st.session_state.ingestion_successful = False

# New session state variable to control the text input value for chat
if 'current_chat_message' not in st.session_state:
    st.session_state.current_chat_message = ""

# --- Helper Function for API Calls ---
def call_api(endpoint: str, data: dict) -> dict:
    """
    Helper function to make POST requests to the FastAPI backend.
    Handles connection errors, HTTP errors, and other exceptions.
    """
    try:
        response = requests.post(f"{FASTAPI_BASE_URL}/{endpoint}", json=data)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.ConnectionError:
        st.session_state.ingestion_error_message = (
            f"Connection error: Could not connect to the backend at {FASTAPI_BASE_URL}. "
            "Please ensure the FastAPI app (app.py) is running in a separate terminal."
        )
        st.error(st.session_state.ingestion_error_message)
        return {"success": False, "message": st.session_state.ingestion_error_message}
    except requests.exceptions.HTTPError as e:
        error_detail = e.response.text
        st.session_state.ingestion_error_message = (
            f"HTTP error: {e.response.status_code} - {error_detail}"
        )
        st.error(st.session_state.ingestion_error_message)
        return {"success": False, "message": st.session_state.ingestion_error_message}
    except Exception as e:
        st.session_state.ingestion_error_message = f"An unexpected error occurred during API call: {e}"
        st.error(st.session_state.ingestion_error_message)
        return {"success": False, "message": st.session_state.ingestion_error_message}

# Helper function to format evaluation details for display
def format_eval_detail(title, result, explanation_key="explanation", score_key=None, bool_key=None):
    detail_str = f"**{title}:**\n"
    if bool_key is not None:
        detail_str += f"- **Result:** {'✅ Yes' if result.get(bool_key) else '❌ No'}\n"
    if score_key:
        detail_str += f"- **Score:** {result.get(score_key, 'N/A')}\n"
    if result.get(explanation_key):
        detail_str += f"- **Explanation:** {result.get(explanation_key)}\n"
    return detail_str

# --- Section for Ingestion ---
if not st.session_state.ingestion_successful:
    st.header("1. Data Ingestion (Search, Scrape, Store)")
    st.write("Enter an initial query to search Google, scrape top results, and store the content in the vector database.")

    ingestion_query = st.text_input(
        "Ingestion Query:",
        placeholder="e.g., latest AI advancements in 2024",
        key="ingestion_input"
    )

    if st.button("Start Searching 🚀", use_container_width=True):
        if ingestion_query:
            st.session_state.ingestion_status = "Starting ingestion... This might take a while."
            st.session_state.ingestion_successful = False # Reset in case of re-ingestion
            st.session_state.ingestion_error_message = "" # Clear previous errors
            st.session_state.chat_messages = [] # Clear previous chat messages on new ingestion
            
            with st.spinner('Ingesting data...'):
                response = call_api("search", {"query": ingestion_query, "n_results": 5}) 
                
                if response and response.get("success"):
                    st.session_state.ingestion_status = (
                        f"Ingestion successful! Initial answer: {response.get('initial_answer', 'N/A')}\n"
                        f"Scraped URLs: {', '.join(response.get('scraped_urls', []))}\n"
                        f"Sources Used: {', '.join(response.get('retrieved_sources', []))}"
                    )
                    st.session_state.ingestion_successful = True
                    st.success("Data ingestion completed successfully! The chat interface is now available below.")
                else:
                    st.session_state.ingestion_status = (
                        f"Ingestion failed: {response.get('message', 'Unknown error.')}"
                    )
                    st.session_state.ingestion_successful = False
                    st.error("Data ingestion failed!")
            st.rerun() # Force rerun to switch to chat interface or show error immediately
        else:
            st.warning("Please enter an ingestion query.")
    
    st.markdown("### Ingestion Status:")
    st.info(st.session_state.ingestion_status or "Status will appear here after starting search...")

    if st.session_state.ingestion_error_message:
        st.error(st.session_state.ingestion_error_message)

else:
    # Section 2: Conversational Chat (shown only if ingestion was successful)
    st.header("2. Conversational Chat (Query Stored Data)")
    st.write("Ask questions based on the content previously ingested into the database.")

    # Text input for chat question, controlled by 'current_chat_message' in session state
    chat_question = st.text_input(
        "Your Question:",
        placeholder="e.g., What did the articles say about Llama 3.1?",
        key="chat_input",
        value=st.session_state.current_chat_message
    )

    if st.button("Ask 💬", use_container_width=True):
        if chat_question:
            # Append user message to chat history immediately
            st.session_state.chat_messages.append({"role": "user", "content": chat_question})
            
            with st.spinner('Getting response...'):
                response = call_api("query", {"question": chat_question, "n_results": 3})
                if response and response.get("success"):
                    answer = response.get("answer", "No answer generated.")
                    sources = response.get("retrieved_sources", [])
                    evaluation_results = response.get("evaluation_results") # Get eval results
                    
                    source_text = ""
                    if sources:
                        source_text = "\n\n**Sources:**\n" + "\n".join([f"- {s}" for s in sorted(list(set(sources)))])
                    
                    # Store evaluation results along with the assistant's message
                    st.session_state.chat_messages.append({
                        "role": "assistant", 
                        "content": answer + source_text,
                        "evaluation": evaluation_results # Attach evaluation results here
                    })
                else:
                    st.session_state.chat_messages.append(
                        {"role": "assistant", "content": f"Error: {response.get('message', 'Unknown error.')}"}
                    )
            
            # Clear the input box
            st.session_state.current_chat_message = "" 
        else:
            st.warning("Please enter a question to chat.")

    # Display chat history in a scrollable container
    st.markdown("### Chat History:")
    chat_container = st.container(height=800, border=True)
    for msg in st.session_state.chat_messages:
        if msg["role"] == "user":
            chat_container.chat_message("user").write(msg["content"])
        else:
            with chat_container.chat_message("assistant"):
                st.write(msg["content"]) # Display the assistant's answer and sources

                # --- NEW: Display Evaluation Results for this chat message ---
                if msg.get("evaluation"):
                    eval_res = msg["evaluation"]
                    st.expander("Show RAG Evaluation for this response 📊").write("") # Create an expander
                    
                    with st.expander("Show RAG Evaluation for this response 📊"): # Re-using the expander context
                        if eval_res.get("overall_score"):
                            st.markdown(f"**Overall RAG Score:** `{eval_res['overall_score']:.2f} / 5`")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if "correctness" in eval_res: # Correctness only if ground truth was passed (unlikely in live chat)
                                st.markdown(format_eval_detail("Correctness", eval_res["correctness"], bool_key="correct"))
                            st.markdown(format_eval_detail("Relevance", eval_res["relevance"], score_key="score", bool_key="relevant"))
                        
                        with col2:
                            grounded_detail = format_eval_detail("Groundedness", eval_res["groundedness"], bool_key="grounded")
                            if eval_res["groundedness"].get("hallucination"):
                                grounded_detail += "- **Hallucination Detected:** ❗ True\n"
                            else:
                                grounded_detail += "- **Hallucination Detected:** ✅ False\n"
                            st.markdown(grounded_detail)
                            st.markdown(format_eval_detail("Retrieval Relevance", eval_res["retrieval_relevance"], score_key="score", bool_key="relevant"))
                # --- END NEW ---