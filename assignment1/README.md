# NLP Preprocessing Suite

A comprehensive Natural Language Processing (NLP) toolkit with an interactive web interface for text preprocessing, analysis, and visualization.

## Features

- **Text Preprocessing**: Tokenization, lemmatization, stemming, POS tagging, and named entity recognition
- **Stem vs Lemma Comparison**: Compare stemming and lemmatization results for words
- **TF-IDF Embeddings**: Generate and visualize TF-IDF embeddings for multiple documents
- **Interactive Visualizations**: 2D and 3D visualizations of document embeddings using PCA

## Architecture

- **FastAPI Backend**: Handles NLP processing tasks and provides API endpoints
- **Streamlit Frontend**: Provides an interactive web interface for users
- **SpaCy & NLTK**: Core NLP libraries for text processing

## Installation


### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <https://github.com/aagaman09/llm-7th-sem.git>
   ```


4. **Install dependencies:**
   ```bash
   pip install fastapi uvicorn streamlit spacy nltk scikit-learn pandas numpy plotly
   ```



## Running the Application

1. **Start the FastAPI backend server:**
   ```bash
   uvicorn main:app --reload
   ```

2. **In a new terminal, start the Streamlit frontend:**
   ```bash
   streamlit run app.py
   ```

3. **Open your web browser and navigate to:**
   ```
   http://localhost:8501
   ```

## API Endpoints

- **POST /preprocess**: Process text and return tokens, lemmas, stems, POS tags, and named entities
- **POST /compare_stem_lemma**: Compare stemming and lemmatization for a list of words
- **POST /tfidf**: Generate TF-IDF embeddings for multiple documents and reduce dimensions for visualization

