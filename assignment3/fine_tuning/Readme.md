# Fine-tuning BERT for Text Classification

A comprehensive implementation of fine-tuning a pre-trained BERT model for binary text classification of disaster-related tweets.

## Project Overview

This project demonstrates the complete pipeline for fine-tuning BERT (Bidirectional Encoder Representations from Transformers) to classify tweets into two categories - determining whether a tweet is about a real disaster or not. The implementation covers data preprocessing, model training, validation, and prediction generation.

## Objectives

- Fine-tune a pre-trained BERT model for binary text classification
- Achieve robust performance on disaster tweet classification


## Environment Setup

### Hardware Requirements

- GPU support recommended (CUDA-compatible)
- Automatic device detection: `torch.device("cuda:0" if torch.cuda.is_available() else "cpu")`

## Data Structure

### Input Data Format
- **train.csv**: Training data with columns:
  - `id`: Unique identifier
  - `keyword`: Keywords related to disaster
  - `location`: Location information
  - `text`: Tweet content (main feature)
  - `target`: Binary classification label (0/1)

## 🔧 Data Preprocessing Pipeline

### Text Cleaning Function

The `clean_text()` function performs comprehensive text preprocessing:

1. **Lowercasing**: Ensures text consistency
2. **URL Removal**: Strips URLs using regex patterns
3. **HTML Tag Removal**: Eliminates HTML markup
4. **Punctuation Removal**: Removes predefined punctuation marks
5. **Stopword Removal**: Filters common English stopwords using NLTK
6. **Emoji Removal**: Removes Unicode emoji characters

### BERT Tokenization Process

1. **Tokenizer Initialization**: `BertTokenizer.from_pretrained('bert-base-uncased')`
2. **Special Tokens**: Adds `[CLS]` and `[SEP]` tokens
3. **Sequence Padding**: Ensures uniform length across all inputs
4. **Attention Masks**: Indicates which tokens to attend to
5. **Tensor Conversion**: Converts to PyTorch tensors

## Model Architecture

### BERT Configuration
- **Base Model**: `bert-base-uncased`
- **Task**: Sequence Classification
- **Number of Labels**: 2 (binary classification)
- **Special Features**: 
  - `output_attentions=False`
  - `output_hidden_states=False`

### Training Setup
- **Optimizer**: AdamW with learning rate 2e-5 and epsilon 1e-8
- **Scheduler**: Linear warmup and decay
- **Batch Size**: 32
- **Epochs**: 4
- **Data Split**: 80% training, 20% validation

## Training Process

### Training Loop Features
- **Gradient Clipping**: Prevents exploding gradients
- **Model Checkpointing**: Saves best performing model
- **Progress Tracking**: Monitors loss and accuracy
- **Reproducibility**: Fixed random seeds

### Validation Strategy
- **Evaluation Mode**: Disables dropout and batch normalization
- **No Gradient Computation**: Uses `torch.no_grad()` for efficiency
- **Performance Metrics**: Tracks accuracy and loss

## 📊 Training Results

```
======== Training Summary ========
Total Training Time: 0:03:57 (h:mm:ss)

Epoch 1: Training Loss: 0.47 | Validation Accuracy: 0.83
Epoch 2: Training Loss: 0.36 | Validation Accuracy: 0.84
Epoch 3: Training Loss: 0.29 | Validation Accuracy: 0.83
Epoch 4: Training Loss: 0.25 | Validation Accuracy: 0.83

Final Validation Accuracy: 83-84%
```

## Prediction Pipeline

### Test Data Processing
1. **Data Loading**: Load test.csv
2. **Text Preprocessing**: Apply same cleaning function
3. **Tokenization**: Use identical tokenizer settings
4. **Model Loading**: Load best checkpoint
5. **Batch Prediction**: Process in batches for efficiency
6. **Output Generation**: Create submission.csv

### Prediction Features
- **Consistent Preprocessing**: Same pipeline as training data
- **Efficient Batching**: Memory-optimized prediction
- **Submission Ready**: Direct CSV output for competitions


