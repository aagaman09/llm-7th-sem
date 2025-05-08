# nlp_utils.py
import spacy
from typing import Dict, List, Union, Tuple
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

nlp = spacy.load("en_core_web_sm")
stemmer = PorterStemmer()

def preprocess_text(text: str) -> Dict:
    doc = nlp(text)

    return {
        "tokens": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "stems": [stemmer.stem(token.text) for token in doc],
        "pos_tags": [(token.text, token.pos_) for token in doc],
        "named_entities": [(ent.text, ent.label_) for ent in doc.ents]
    }

def compare_stem_lemma(words: List[str]) -> List[Dict[str, str]]:
    return [{
        "word": w,
        "stem": stemmer.stem(w),
        "lemma": nlp(w)[0].lemma_
    } for w in words]

def create_tfidf_embeddings(documents: List[str]) -> Tuple[np.ndarray, List[str]]:
    """Create TF-IDF embeddings for a list of documents."""
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix.toarray(), feature_names

def reduce_dimensions(embeddings: np.ndarray, method: str = 'pca', n_components: int = 2) -> np.ndarray:
    """Reduce dimensionality of embeddings using PCA."""
    if embeddings.shape[0] < 2:
        # If we have only one document, we can't reduce dimensions
        # Return a placeholder array with zeros based on requested components
        return np.zeros((1, n_components))
    
    # Only use PCA regardless of method parameter
    n_components = min(n_components, embeddings.shape[0], embeddings.shape[1])
    reducer = PCA(n_components=n_components)
    
    return reducer.fit_transform(embeddings)

def get_top_tfidf_terms(tfidf_array: np.ndarray, feature_names: List[str], top_n: int = 10) -> List[List[Tuple[str, float]]]:
    """Get the top N terms with highest TF-IDF scores for each document."""
    top_terms = []
    
    for doc_vector in tfidf_array:
        # Get indices of top N values
        top_indices = doc_vector.argsort()[-top_n:][::-1]
        # Get the corresponding terms and scores
        doc_terms = [(feature_names[i], doc_vector[i]) for i in top_indices if doc_vector[i] > 0]
        top_terms.append(doc_terms)
    
    return top_terms
