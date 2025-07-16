# Building a GPT from Scratch

A complete implementation of a GPT (Generative Pre-trained Transformer) model built from scratch using PyTorch, following the "Zero To Hero" approach. This project demonstrates the fundamental concepts of transformer architecture and language modeling.

## 🎯 Overview

This project implements a character-level GPT model trained on Shakespeare's text. The implementation covers all essential components of a transformer, including:

- **Self-Attention Mechanism**: Multi-head attention with scaled dot-product attention
- **Transformer Blocks**: Complete encoder blocks with residual connections
- **Positional Encoding**: Learnable position embeddings
- **Layer Normalization**: For training stability
- **Feed-Forward Networks**: Position-wise fully connected layers

## 🚀 Features

- **From Scratch Implementation**: No high-level transformer libraries used
- **Educational Focus**: Clear, well-commented code explaining each component
- **Character-Level Modeling**: Generates text one character at a time
- **Configurable Architecture**: Easy to modify hyperparameters and model size
- **Training Loop**: Complete training pipeline with validation
- **Text Generation**: Sample text generation from trained model

## 📋 Requirements

```bash
torch>=1.9.0
numpy
```

## 🛠️ Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/gpt-from-scratch.git
cd gpt-from-scratch
```

2. Install dependencies:
```bash
pip install torch numpy
```

3. Download the dataset:
```bash
wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

## 🏗️ Model Architecture

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `batch_size` | 16 | Number of sequences processed in parallel |
| `block_size` | 32 | Maximum context length |
| `n_embd` | 64 | Embedding dimension |
| `n_head` | 4 | Number of attention heads |
| `n_layer` | 4 | Number of transformer blocks |
| `learning_rate` | 1e-3 | Adam optimizer learning rate |
| `dropout` | 0.0 | Dropout rate |

### Components

**1. Token and Position Embeddings**
- Character-level tokenization (65 unique characters)
- Learnable position embeddings up to `block_size`

**2. Multi-Head Self-Attention**
```python
class Head(nn.Module):
    """ one head of self-attention """
    
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
```

**3. Transformer Block**
- Multi-head attention
- Feed-forward network
- Residual connections
- Layer normalization

**4. Language Model Head**
- Linear projection to vocabulary size
- Cross-entropy loss for next-token prediction

## 📚 Usage

### Training the Model

```python
# Load and prepare data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Initialize model
model = BigramLanguageModel()
model = model.to(device)

# Train
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
for iter in range(max_iters):
    xb, yb = get_batch('train')
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
```

### Generating Text

```python
# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated_text = decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated_text)
```

### Sample Output

```
ROMEO:
But you fret, hell, where Volivius:
They now I usless like, to to see this grand; I'll must sprinks dray-wretor upwon alond, liege wan thy lace blaniess to hanger and nour gruest:
Now to by betwers tot, tow, go:
Eve nor to his ghards wifill my join not.
```

## 🔍 Key Concepts Explained

### Self-Attention Mechanism

The core innovation of transformers - allows each position to attend to all previous positions:

```python
# Compute attention weights
wei = q @ k.transpose(-2,-1) * C**-0.5  # Scale by sqrt(d_k)
wei = wei.masked_fill(tril == 0, float('-inf'))  # Causal masking
wei = F.softmax(wei, dim=-1)  # Normalize
out = wei @ v  # Weighted aggregation
```

### Mathematical Trick

Efficient computation using matrix multiplication instead of loops:
- **Version 1**: Explicit loops (slow)
- **Version 2**: Matrix multiplication with triangular weights
- **Version 3**: Softmax with masked attention
- **Version 4**: Full self-attention with queries, keys, values

## 📊 Training Results

- **Parameters**: ~0.21M parameters
- **Final Training Loss**: ~1.66
- **Final Validation Loss**: ~1.82
- **Training Time**: ~5000 steps

## 🧪 Experiments and Extensions

### Possible Improvements

1. **Larger Model**: Increase `n_embd`, `n_head`, `n_layer`
2. **Better Tokenization**: Use BPE instead of character-level
3. **Regularization**: Add dropout, weight decay
4. **Learning Rate Scheduling**: Cosine annealing, warmup
5. **Data Augmentation**: More diverse training data

### Ablation Studies

- Effect of attention heads (1, 2, 4, 8)
- Impact of layer depth (2, 4, 6, 8 layers)
- Context window size (16, 32, 64, 128)

## 📖 Educational Value

This implementation is designed for learning and includes:

- **Step-by-step progression** from bigram to full transformer
- **Detailed comments** explaining each component
- **Mathematical intuitions** behind self-attention
- **Visualization** of attention patterns
- **Performance monitoring** during training

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add attention visualization
- [ ] Implement different positional encodings
- [ ] Add model checkpointing
- [ ] Create evaluation metrics
- [ ] Add configuration files
- [ ] Implement beam search generation

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- **Andrej Karpathy** for the "Zero To Hero" tutorial series
- **Attention Is All You Need** paper by Vaswani et al.
- **Shakespeare dataset** for training data

## 📚 References

1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
3. [Zero To Hero Video Series](https://karpathy.ai/zero-to-hero.html)

---

**Note**: This is an educational implementation focused on understanding transformer architecture. For production use, consider using established libraries like HuggingFace Transformers.
