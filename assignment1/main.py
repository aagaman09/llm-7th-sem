# main.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import List, Optional
from utils import preprocess_text, compare_stem_lemma, create_tfidf_embeddings, reduce_dimensions, get_top_tfidf_terms

app = FastAPI()

class TextRequest(BaseModel):
    text: str

class CompareRequest(BaseModel):
    words: List[str]

class TfidfRequest(BaseModel):
    documents: List[str]
    reduction_method: Optional[str] = "pca"  # 'pca' will be used regardless
    n_components: Optional[int] = 2

@app.post("/preprocess")
def preprocess(req: TextRequest):
    return preprocess_text(req.text)

@app.post("/compare_stem_lemma")
def compare(req: CompareRequest):
    return compare_stem_lemma(req.words)

@app.post("/tfidf")
def tfidf(req: TfidfRequest):
    # Create TF-IDF embeddings
    tfidf_matrix, feature_names = create_tfidf_embeddings(req.documents)
    
    # Get top TF-IDF terms for each document
    top_terms = get_top_tfidf_terms(tfidf_matrix, feature_names)
    
    # Calculate max possible components
    max_components = min(tfidf_matrix.shape[0], tfidf_matrix.shape[1])
    n_components = min(req.n_components, max_components)
    
    # Reduce dimensions for visualization using only PCA
    reduced_embeddings = reduce_dimensions(
        tfidf_matrix, 
        method="pca", 
        n_components=n_components
    ).tolist()
    
    return {
        "embeddings": tfidf_matrix.tolist(),
        "feature_names": feature_names.tolist(),
        "top_terms": top_terms,
        "reduced_embeddings": reduced_embeddings,
        "reduction_method": "pca",
        "n_components": n_components
    }
