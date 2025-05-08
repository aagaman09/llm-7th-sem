# main.py
import spacy
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from nltk.stem import PorterStemmer
import nltk

# --- Download necessary NLTK data (run this once if you haven't) ---
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')

# --- Load spaCy model ---
# Make sure you have the model downloaded: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    print("'en_core_web_sm' downloaded and loaded successfully.")


# --- Initialize FastAPI app ---
app = FastAPI(
    title="NLP Preprocessing API",
    description="An API for basic NLP tasks like tokenization, lemmatization, stemming, POS tagging, and NER.",
    version="1.0.0"
)

# --- Pydantic model for request body ---
class TextIn(BaseModel):
    text: str

# --- Pydantic models for responses (optional but good practice) ---
class TokenizationOut(BaseModel):
    tokens: list[str]

class LemmatizationOut(BaseModel):
    lemmas: list[dict[str, str]] # {"token": "lemma"}

class StemmingOut(BaseModel):
    stems: list[dict[str, str]] # {"token": "stem"}

class PosTaggingOut(BaseModel):
    pos_tags: list[dict[str, str]] # {"token": "pos_tag"}

class NerOut(BaseModel):
    entities: list[dict[str, str]] # {"entity": "label"}

class StemLemCompareOut(BaseModel):
    comparison: list[dict[str, str]] # {"word": "original", "stem": "stemmed", "lemma": "lemmatized"}


# --- Initialize NLTK Porter Stemmer ---
porter = PorterStemmer()

# --- Helper function for stemming with NLTK ---
def get_stems_nltk(text: str) -> list[dict[str, str]]:
    """
    Performs stemming on the input text using NLTK's PorterStemmer.
    Returns a list of dictionaries, each containing the original token and its stem.
    """
    words = nltk.word_tokenize(text)
    stemmed_words = []
    for word in words:
        stemmed_words.append({"token": word, "stem": porter.stem(word)})
    return stemmed_words

# --- API Endpoints ---

@app.post("/tokenize/", response_model=TokenizationOut, tags=["NLP Operations"])
async def tokenize_text(text_in: TextIn):
    """
    Tokenizes the input text.
    Splits the text into individual words or punctuation marks.
    """
    doc = nlp(text_in.text)
    tokens = [token.text for token in doc]
    return {"tokens": tokens}

@app.post("/lemmatize/", response_model=LemmatizationOut, tags=["NLP Operations"])
async def lemmatize_text(text_in: TextIn):
    """
    Lemmatizes the input text.
    Reduces words to their base or dictionary form (lemma).
    """
    doc = nlp(text_in.text)
    lemmas = [{"token": token.text, "lemma": token.lemma_} for token in doc if not token.is_punct and not token.is_space]
    return {"lemmas": lemmas}

@app.post("/stem/", response_model=StemmingOut, tags=["NLP Operations"])
async def stem_text(text_in: TextIn):
    """
    Stems the input text using NLTK's PorterStemmer.
    Reduces words to their root form, which may not always be a valid word.
    """
    if not text_in.text.strip():
        return {"stems": []}
    stems = get_stems_nltk(text_in.text)
    return {"stems": stems}

@app.post("/pos_tag/", response_model=PosTaggingOut, tags=["NLP Operations"])
async def pos_tag_text(text_in: TextIn):
    """
    Performs Part-of-Speech (POS) tagging on the input text.
    Assigns a grammatical category (e.g., noun, verb, adjective) to each token.
    """
    doc = nlp(text_in.text)
    pos_tags = [{"token": token.text, "pos_tag": token.tag_ + " (" + spacy.explain(token.tag_) + ")"} for token in doc if not token.is_space]
    return {"pos_tags": pos_tags}

@app.post("/ner/", response_model=NerOut, tags=["NLP Operations"])
async def ner_text(text_in: TextIn):
    """
    Performs Named Entity Recognition (NER) on the input text.
    Identifies and categorizes named entities (e.g., persons, organizations, locations).
    """
    doc = nlp(text_in.text)
    entities = [{"entity": ent.text, "label": ent.label_ + " (" + spacy.explain(ent.label_) + ")"} for ent in doc.ents]
    return {"entities": entities}

@app.get("/compare_stem_lemma/", response_model=StemLemCompareOut, tags=["Comparison"])
async def compare_stemming_lemmatization():
    """
    Provides a comparison of stemming and lemmatization for a predefined list of words.
    """
    words = [
        "running", "ran", "runs",
        "better", "good",
        "mice", "mouse",
        "corpora", "corpus",
        "studies", "studying",
        "meeting", "meets",
        "apples", "apple's",
        "automobiles", "automobile",
        "programming", "program",
        "historical", "history"
    ]
    comparison_results = []
    for word in words:
        doc = nlp(word) # spaCy for lemmatization
        lemma = doc[0].lemma_ if doc else word # Get lemma of the first token
        stem = porter.stem(word) # NLTK for stemming
        comparison_results.append({"word": word, "stem": stem, "lemma": lemma})
    return {"comparison": comparison_results}

# --- To run this FastAPI app: ---
# 1. Save this code as `main.py`.
# 2. Make sure you have FastAPI and Uvicorn installed:
#    pip install fastapi uvicorn spacy nltk
# 3. Download the spaCy model:
#    python -m spacy download en_core_web_sm
# 4. Run the server from your terminal:
#    uvicorn main:app --reload
#
# You can then access the API docs at http://127.0.0.1:8000/docs
