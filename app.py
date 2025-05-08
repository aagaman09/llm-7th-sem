# streamlit_app.py
import streamlit as st
import requests
import pandas as pd

# --- Configuration for the FastAPI backend ---
API_URL = "http://127.0.0.1:8000" # Make sure this matches your FastAPI server address

# --- Helper function to call the API ---
def call_api(endpoint, text):
    """Sends a POST request to the specified API endpoint."""
    try:
        response = requests.post(f"{API_URL}/{endpoint}/", json={"text": text})
        response.raise_for_status()  # Raises an exception for HTTP errors (4XX, 5XX)
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error connecting to API: {e}")
        return None

def fetch_comparison_data():
    """Fetches the stemming vs. lemmatization comparison data."""
    try:
        response = requests.get(f"{API_URL}/compare_stem_lemma/")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching comparison data: {e}")
        return None

# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="NLP Preprocessing Demo")

st.title("🔬 NLP Preprocessing Interactive Demo")
st.markdown("""
Welcome to the NLP Preprocessing Demo! Enter some text below and choose an NLP task
to see it in action. This application uses a FastAPI backend with spaCy and NLTK.
""")

# --- Sidebar for About and Instructions ---
st.sidebar.header("About")
st.sidebar.info(
    "This application demonstrates common Natural Language Processing (NLP) "
    "preprocessing techniques. It uses a FastAPI backend for processing."
)
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1.  **Ensure the FastAPI backend is running.**
    (Open your terminal, navigate to the directory with `main.py`, and run `uvicorn main:app --reload`)
2.  **Enter text** in the text area below.
3.  **Select an NLP operation** from the dropdown menu.
4.  Click the **"Process Text"** button.
5.  The results will be displayed below the button.
6.  You can also view a **comparison of Stemming and Lemmatization**.
""")


# --- Main Area ---
st.header("📝 Input Text")
default_text = "Apple Inc. is looking at buying U.K. startup for $1 billion. The quick brown foxes are jumping over the lazy dogs. Historically, studies have shown interesting patterns."
user_text = st.text_area("Enter your text here:", default_text, height=150)

nlp_operations = {
    "Tokenize": "tokenize",
    "Lemmatize": "lemmatize",
    "Stem (NLTK Porter)": "stem",
    "Part-of-Speech Tagging": "pos_tag",
    "Named Entity Recognition": "ner"
}

selected_operation_display = st.selectbox(
    "Choose an NLP operation:",
    options=list(nlp_operations.keys())
)

if st.button("🚀 Process Text", type="primary"):
    if not user_text.strip():
        st.warning("Please enter some text to process.")
    elif selected_operation_display:
        endpoint = nlp_operations[selected_operation_display]
        st.info(f"Processing with: {selected_operation_display}...")
        result = call_api(endpoint, user_text)

        if result:
            st.subheader("✨ Results:")
            if endpoint == "tokenize":
                st.write("Tokens:", result.get("tokens"))
            elif endpoint == "lemmatize":
                st.write("Lemmas (Token: Lemma):")
                df_lemmas = pd.DataFrame(result.get("lemmas", []))
                st.dataframe(df_lemmas, use_container_width=True)
            elif endpoint == "stem":
                st.write("Stems (Token: Stem):")
                df_stems = pd.DataFrame(result.get("stems", []))
                st.dataframe(df_stems, use_container_width=True)
            elif endpoint == "pos_tag":
                st.write("Part-of-Speech Tags (Token: POS Tag):")
                df_pos = pd.DataFrame(result.get("pos_tags", []))
                st.dataframe(df_pos, use_container_width=True)
            elif endpoint == "ner":
                st.write("Named Entities (Entity: Label):")
                if result.get("entities"):
                    df_ner = pd.DataFrame(result.get("entities", []))
                    st.dataframe(df_ner, use_container_width=True)
                else:
                    st.write("No named entities found.")
            else:
                st.json(result) # Fallback for any other response

st.divider()

# --- Stemming vs Lemmatization Comparison ---
st.header("🆚 Stemming vs. Lemmatization Comparison")
st.markdown("""
This section shows the difference between stemming (using NLTK's Porter Stemmer)
and lemmatization (using spaCy) for a predefined set of words.
""")

if st.button("🔍 Show Comparison Table"):
    comparison_data = fetch_comparison_data()
    if comparison_data and "comparison" in comparison_data:
        df_comparison = pd.DataFrame(comparison_data["comparison"])
        st.dataframe(df_comparison, use_container_width=True)
        st.markdown(
            """
            **Key Differences Observed:**
            * **Stemming** often produces non-dictionary words (e.g., "runn" for "running"). It's a cruder, rule-based process.
            * **Lemmatization** aims to return the actual dictionary form of a word (lemma). It's more sophisticated and often uses a vocabulary and morphological analysis.
            * Notice how "better" is lemmatized to "good" (its root meaning) but stemmed to "better".
            * "mice" is correctly lemmatized to "mouse", while stemming might struggle or produce "mice".
            """
        )
    else:
        st.warning("Could not fetch comparison data. Ensure the API is running.")

st.sidebar.markdown("---")
st.sidebar.markdown("Dataset used for example text: `abisee/cnn_dailymail` (a small snippet).")
st.sidebar.markdown("NLP Libraries: `spaCy`, `NLTK`")
st.sidebar.markdown("API Framework: `FastAPI`")
st.sidebar.markdown("Web Framework: `Streamlit`")


# --- To run this Streamlit app: ---
# 1. Save this code as `streamlit_app.py` in the SAME directory as `main.py`.
# 2. Make sure your FastAPI server (`main.py`) is running:
#    `uvicorn main:app --reload`
# 3. Open a NEW terminal window/tab, navigate to the directory, and run:
#    `streamlit run streamlit_app.py`
