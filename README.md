# Large Language Models (LLM) - 7th Semester 

A comprehensive collection of assignments demonstrating various aspects of Large Language Models, Natural Language Processing, and AI development techniques.


This repository contains all assignments completed for the 7th semester LLM course, showcasing practical implementations ranging from basic NLP preprocessing to advanced transformer architectures and multi-agent systems.

## Assignment Portfolio

### Assignment 1: NLP Preprocessing Suite
**Directory:** `assignment1/`

A comprehensive NLP toolkit with interactive web interface for text preprocessing and analysis.

**Key Features:**
- Text preprocessing pipeline (tokenization, lemmatization, stemming)
- POS tagging and Named Entity Recognition
- TF-IDF embeddings with interactive visualizations
- FastAPI backend + Streamlit frontend architecture

**Technologies:** FastAPI, Streamlit, SpaCy, NLTK, Plotly

---

### Assignment 2: Transformers From Scratch
**Directory:** `assignment2_transformers/`

Complete implementation of transformer architecture from fundamental building blocks.

**Key Components:**
- Byte Pair Encoding (BPE) tokenization from scratch
- Self-attention mechanism and multi-head attention
- Positional encoding (sinusoidal and learned)
- Full GPT-2 architecture implementation
- Training on OpenWebText dataset

**Achievements:**
- 120M parameter GPT-2 model trained successfully
- Custom tokenizer with 50K+ vocabulary
- Text generation capabilities with temperature control

**Technologies:** PyTorch, PyTorch Lightning, NumPy

---

### Assignment 3A: Multi-Agent RAG System
**Directory:** `assignment3/AGENTIC_RAG/`

Sophisticated Retrieval-Augmented Generation system with specialized agents.

**Agent Architecture:**
- **Search Agent**: Google Custom Search API integration
- **Scraping Agent**: Asynchronous web crawling with crawl4ai
- **Evaluation Agent**: Comprehensive RAG evaluation metrics

**Key Features:**
- LangGraph orchestration for multi-agent workflows
- ChromaDB vector storage with Ollama embeddings
- Real-time web search and intelligent content extraction
- Comprehensive evaluation framework (relevance, groundedness, correctness)

**Technologies:** LangGraph, ChromaDB, Ollama, Streamlit, FastAPI

---

### Assignment 3B: BERT Fine-tuning
**Directory:** `assignment3/fine_tuning/`

Fine-tuning pre-trained BERT for disaster tweet classification.

**Implementation:**
- Binary text classification for disaster-related tweets
- Comprehensive data preprocessing pipeline
- BERT tokenization and sequence classification
- Training with validation and performance monitoring

**Results:**
- 83-84% validation accuracy achieved
- Robust text cleaning and preprocessing
- Model checkpointing and prediction pipeline

**Technologies:** PyTorch, Transformers, BERT, NLTK

---

### Assignment 4: Prompting Strategies Experiment
**Directory:** `assignment4/`

Comparative analysis of different prompting strategies for LLM applications.

**Prompting Strategies:**
1. **Direct Prompting**: Straightforward task instructions
2. **Few-Shot Prompting**: Learning from examples
3. **Chain-of-Thought**: Step-by-step reasoning

**Domain:** Formula 1 race prediction with multiple variables
- Qualifying results analysis
- Track characteristics consideration
- Weather condition impact
- Historical data integration

**Key Findings:**
- Chain-of-Thought performed best for complex analysis
- Systematic evaluation of prompt effectiveness
- Real-world applicability to sports analytics

---

### Assignment 5: BLIP Image Captioning
**Directory:** `assignment5(blip_captioning)/`

Advanced image captioning system using Salesforce's BLIP model.

**Key Components:**
- **FastAPI Backend**: RESTful API for image captioning
- **Streamlit Frontend**: User-friendly web interface  
- **Jupyter Notebook**: Model experimentation and testing

**Features:**
- Multiple input methods (file upload, URL)
- State-of-the-art BLIP large model (900M parameters)
- Real-time processing with cross-platform support
- RESTful API for easy integration

**Technologies:** Transformers, FastAPI, Streamlit, PIL, PyTorch

## 🛠️ Technologies Stack

### Core ML/AI Frameworks
- **PyTorch**: Deep learning framework for transformer implementations
- **Transformers**: Hugging Face library for pre-trained models
- **LangGraph**: Multi-agent workflow orchestration
- **ChromaDB**: Vector database for RAG systems

### Web Development
- **FastAPI**: High-performance API development
- **Streamlit**: Interactive web applications
- **Uvicorn**: ASGI server for FastAPI

### NLP Libraries
- **SpaCy**: Advanced NLP processing
- **NLTK**: Natural language toolkit
- **Ollama**: Local LLM inference

### Data & Visualization
- **NumPy/Pandas**: Data manipulation and analysis
- **Plotly**: Interactive visualizations
- **Scikit-learn**: Machine learning utilities

## 🚀 Getting Started

### Prerequisites
```bash
# Core requirements
Python 3.8+
CUDA-compatible GPU (recommended)

# For Assignment 3A (Agentic RAG)
Ollama installation
Google Custom Search API credentials

# For Assignment 2 (Transformers)
PyTorch with CUDA support
```

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd llm

# Install common dependencies
pip install torch transformers fastapi streamlit nltk spacy

# For specific assignments, check individual README files
```

## 📊 Project Highlights

### 🏆 Technical Achievements

1. **Custom Transformer Implementation**
   - Built GPT-2 from scratch with 120M parameters
   - Achieved competitive text generation quality
   - Custom BPE tokenizer implementation

2. **Multi-Agent RAG System**
   - Sophisticated agent orchestration with LangGraph
   - Real-time web search and intelligent scraping
   - Comprehensive evaluation framework

3. **Production-Ready Applications**
   - Multiple web interfaces with FastAPI + Streamlit
   - RESTful APIs for model serving
   - Scalable architecture patterns

### 📈 Learning Outcomes

- **Deep Understanding**: Transformer architecture from first principles
- **Practical Skills**: Production deployment of ML models
- **System Design**: Multi-agent architectures and RAG systems
- **Evaluation**: Comprehensive model assessment techniques
- **Web Development**: Full-stack AI application development

## 🔍 Key Insights

### Transformer Architecture
- Self-attention enables parallel processing and long-range dependencies
- Positional encoding is crucial for sequence understanding
- Layer normalization and residual connections stabilize training

### RAG Systems
- Multi-agent approaches improve system modularity and reliability
- Comprehensive evaluation is essential for production deployment
- Real-time web search enhances knowledge freshness

### Prompting Strategies
- Chain-of-thought reasoning improves complex task performance
- Domain-specific examples enhance model understanding
- Systematic evaluation reveals strategy effectiveness


## 📁 Repository Structure

```
llm/
├── assignment1/                    # NLP Preprocessing Suite
│   ├── app.py                     # Streamlit frontend
│   ├── main.py                    # FastAPI backend
│   └── README.md
├── assignment2_transformers/       # Transformers From Scratch
│   ├── transformers_from_scratch.ipynb
│   └── README.md
├── assignment3/
│   ├── AGENTIC_RAG/              # Multi-Agent RAG System
│   │   ├── streamlit_app.py
│   │   ├── app.py
│   │   └── README.md
│   └── fine_tuning/              # BERT Fine-tuning
│       └── README.md
├── assignment4/                   # Prompting Strategies
│   ├── f1_prompts.py
│   ├── experiment_runner.py
│   └── README.md
├── assignment5(blip_captioning)/  # BLIP Image Captioning
│   ├── app.py
│   ├── Blip.py
│   ├── BLIP.ipynb
│   └── README.md
└── README.md                      # This file
```


## 📝 Documentation

Each assignment contains detailed documentation including:
- Architecture diagrams and system flows
- Implementation details and code explanations
- Usage instructions and API documentation
- Results analysis and performance metrics
- Screenshots and visual examples

## 🤝 Contributing

This repository serves as a learning portfolio. For improvements or suggestions:
1. Review individual assignment documentation
2. Test implementations locally
3. Submit issues for bugs or enhancement ideas

