# Transformers From Scratch

A comprehensive implementation of transformer architecture from the ground up, including tokenization, positional encoding, self-attention mechanisms, and a complete GPT-2 model trained on OpenWebText dataset.

## Table of Contents

1. [Overview](#overview)
2. [Implementation Details](#implementation-details)
3. [Key Components](#key-components)
4. [Training Process](#training-process)
5. [Results](#results)
6. [Usage](#usage)
7. [Dependencies](#dependencies)

## Overview

This project implements the complete transformer architecture from scratch using PyTorch, focusing on understanding the fundamental building blocks that make modern language models work. The implementation covers:

- **Byte Pair Encoding (BPE) Tokenization**: Custom tokenizer built from scratch
- **Positional Encoding**: Both sinusoidal and learned positional embeddings
- **Self-Attention Mechanism**: Scaled dot-product attention with multi-head implementation
- **Transformer Blocks**: Complete transformer decoder blocks with layer normalization
- **GPT-2 Architecture**: Full implementation compatible with GPT-2 specifications
- **Training Pipeline**: End-to-end training on large-scale text data

## Implementation Details

### 1. Byte Pair Encoding (BPE) Tokenization

The tokenization process is implemented from scratch using the BPE algorithm:

**Step 1: Vocabulary Building**
- Separates characters in words with spaces
- Adds end-of-word tokens (`</w>`)
- Counts frequency of tokens in the corpus

**Step 2: Pair Statistics**
- Calculates frequency of consecutive symbol pairs
- Identifies most common bigrams for merging

**Step 3: Vocabulary Merging**
- Iteratively merges the most frequent pairs
- Creates sub-word tokens through 4000 merge operations
- Results in 3813 unique tokens from Shakespeare corpus

**Key Functions:**
- `build_vocab()`: Initial vocabulary creation
- `get_stats()`: Pair frequency calculation
- `merge_vocab()`: Vocabulary merging operations
- `tokenize()`: Text to token conversion with attention masks

### 2. Positional Encoding

Two approaches to positional encoding are implemented:

**Sinusoidal Positional Encoding:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Learned Positional Embeddings:**
- Trainable embedding matrix for positions
- More flexible but requires more parameters

**Features:**
- Supports sequences up to maximum length
- Preserves relative position information
- Enables model to understand token order

### 3. Self-Attention Mechanism

The core innovation of transformers, implemented with:

**Scaled Dot-Product Attention:**
```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

**Multi-Head Attention:**
- 12 attention heads for GPT-2 configuration
- Parallel processing of different representation subspaces
- Concatenation and linear projection of head outputs

**Attention Masking:**
- Causal masking for autoregressive generation
- Padding mask for variable-length sequences
- Prevents information leakage from future tokens

**Key Features:**
- Efficient matrix operations
- Support for variable sequence lengths
- Visualization capabilities for attention patterns

### 4. Transformer Architecture

**Transformer Block Components:**
- Multi-head self-attention layer
- Position-wise feed-forward network (MLP)
- Residual connections around each sub-layer
- Layer normalization (pre-norm configuration)

**GPT-2 Model Configurations:**
- **GPT-Mini**: 6 layers, 6 heads, 192 hidden dimensions (1.2M parameters)
- **GPT-2**: 12 layers, 12 heads, 768 hidden dimensions (120M parameters)
- **GPT-3**: 96 layers, 96 heads, 2048 hidden dimensions (175B parameters)

### 5. Advanced Components

**Layer Normalization:**
- Normalizes activations across the feature dimension
- Stabilizes training and improves convergence
- Applied before attention and MLP layers (pre-norm)

**Dropout Regularization:**
- Prevents overfitting during training
- Applied to attention weights and MLP outputs

**GELU Activation:**
- Gaussian Error Linear Unit activation function
- Smoother alternative to ReLU for transformer models

## Key Components

### Tokenization Pipeline

1. **Text Preprocessing**: Character-level separation and end-of-word marking
2. **BPE Training**: Iterative merging of most frequent character pairs
3. **Vocabulary Creation**: Mapping between tokens and integer IDs
4. **Tokenization**: Converting text to sequences of token IDs with attention masks

### Attention Visualization

The implementation includes attention pattern visualization capabilities:
- Heatmap visualization of attention weights
- Analysis of which tokens attend to which other tokens
- Demonstration of how attention creates semantic relationships

### Training Infrastructure

**Dataset Handling:**
- Support for Shakespeare and OpenWebText datasets
- Efficient data loading with PyTorch DataLoader
- Memory-mapped file reading for large datasets

**Optimization:**
- AdamW optimizer with weight decay
- Cosine learning rate scheduling with warmup
- Gradient accumulation for large effective batch sizes

**Learning Rate Scheduling:**
- 2000 step warmup period
- Cosine decay over 600,000 training steps
- Minimum learning rate factor of 0.1

## Training Process

### Dataset Preparation

**OpenWebText Dataset:**
- Large-scale web text corpus
- Pre-tokenized using GPT-2 tokenizer
- Stored as memory-mapped binary files for efficiency

**Training Configuration:**
- Sequence length: 1024 tokens
- Batch size: 12 (with 10x gradient accumulation = effective batch size 120)
- Total training steps: 600,000
- Learning rate: 6e-4 with cosine scheduling

### Model Architecture

**GPT-2 Configuration Used:**
- Vocabulary size: 50,257 tokens
- Hidden dimensions: 768
- Number of layers: 12
- Attention heads: 12
- Maximum sequence length: 1024
- Dropout: 0.0 (no dropout for this training)

### Training Infrastructure

**PyTorch Lightning Integration:**
- Automatic optimization and logging
- Multi-GPU training support
- Gradient accumulation handling
- Checkpoint saving and loading

## Results

### Text Generation Capabilities

The trained model demonstrates:
- Coherent text generation from prompts
- Understanding of grammatical structures
- Contextual word prediction
- Temperature-controlled randomness in generation

### Model Performance

**Training Metrics:**
- Cross-entropy loss minimization
- Perplexity evaluation on validation data
- Gradient norm monitoring
- Learning rate scheduling effectiveness

### Generated Examples

The model can generate human-like text continuations from prompts like:
```
Prompt: "Hello my name is "
Generated: [30 tokens of contextually relevant continuation]
```

## Usage

### Training a New Model

```python
# Configure model architecture
config = dict(
    max_len=1024,
    emb_dim=768,
    num_layers=12,
    num_heads=12,
    model_dim=768,
    vocab_size=50257,
    weight_decay=0.1,
    dropout=0.0
)

# Initialize model
model = GPT2(**config)

# Setup training
trainer = pl.Trainer(
    max_steps=600000,
    accumulate_grad_batches=10,
    accelerator="gpu",
    devices=-1,
    strategy="dp"
)

# Train model
trainer.fit(model, train_loader)
```

### Text Generation

```python
# Load trained model
model = GPT2.load_from_checkpoint("gpt2-openwebtext.ckpt")

# Generate text
generation = generate(
    model, 
    prompt="Your prompt here", 
    max_new_tokens=50, 
    temperature=0.8
)
```

### Custom Tokenization

```python
# Build vocabulary from corpus
vocab = build_vocab(corpus)

# Train BPE for specified number of merges
num_merges = 4000
for i in range(num_merges):
    pairs = get_stats(vocab)
    best = max(pairs, key=pairs.get)
    vocab = merge_vocab(best, vocab)

# Tokenize text
tokens = tokenize(text, sorted_tokens, token2id, return_strings=True)
```

## Dependencies

### Core Libraries
- `torch`: PyTorch deep learning framework
- `torch.nn`: Neural network modules
- `torch.nn.functional`: Functional interface for neural networks
- `pytorch_lightning`: High-level PyTorch wrapper for training

### Data Processing
- `numpy`: Numerical computing
- `scipy`: Scientific computing utilities
- `tiktoken`: GPT tokenizer implementation
- `datasets`: Hugging Face datasets library

### Visualization and Analysis
- `matplotlib.pyplot`: Plotting and visualization
- `seaborn`: Statistical data visualization
- `tqdm`: Progress bars for iterative processes

### Utilities
- `re`: Regular expressions for text processing
- `math`: Mathematical functions
- `collections`: Specialized container datatypes

## Project Structure

```
aagam/
├── transformers_from_scratch.ipynb  # Main implementation notebook
├── README.md                        # This documentation file
├── data/
│   ├── shakespeare/
│   │   └── train.txt               # Shakespeare corpus
│   └── openwebtext/
│       └── train.bin               # OpenWebText binary data
└── checkpoints/
    └── gpt2-openwebtext.ckpt      # Trained model checkpoint
```

## Key Insights and Learnings

### Attention Mechanism Benefits
- **Parallelization**: Unlike RNNs, attention allows parallel processing of sequences
- **Long-range Dependencies**: Direct connections between distant tokens
- **Interpretability**: Attention weights provide insight into model decisions

### Positional Encoding Importance
- **Order Preservation**: Essential for maintaining sequence order information
- **Sinusoidal vs Learned**: Both approaches work, with learned embeddings being more flexible

### Training Considerations
- **Learning Rate Scheduling**: Critical for stable training and convergence
- **Gradient Accumulation**: Enables large effective batch sizes on limited hardware
- **Layer Normalization**: Pre-norm configuration provides better training stability

### Scaling Laws
- **Model Size**: Larger models generally perform better but require more resources
- **Data Requirements**: Large models need substantial training data for optimal performance
- **Compute Trade-offs**: Training time scales with model size and sequence length

## Future Improvements

1. **Efficiency Optimizations**: Flash Attention, gradient checkpointing
2. **Architecture Variants**: Encoder-decoder models, sparse attention patterns
3. **Training Techniques**: Better optimization strategies, curriculum learning
4. **Evaluation Metrics**: Comprehensive benchmarking on downstream tasks
5. **Model Compression**: Quantization, pruning, knowledge distillation

## Conclusion

This implementation provides a complete understanding of transformer architecture from fundamental principles. By building each component from scratch, we gain insights into:

- How modern language models process and generate text
- The importance of attention mechanisms in capturing semantic relationships
- The role of positional encoding in sequence modeling
- Training dynamics and optimization challenges for large models

The resulting GPT-2 model demonstrates the power of the transformer architecture and serves as a foundation for understanding more advanced language models like GPT-3, GPT-4, and beyond.
