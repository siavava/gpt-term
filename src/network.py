#!/usr/bin/env python3
import torch; import torch.nn as nn; import torch.nn.functional as F
device = 'cuda' if torch.cuda.is_available() else 'cpu'
import yaml; import datasets
with open('config.yml', 'r') as f: globals().update(yaml.safe_load(f))


dataset = datasets.load_dataset('siavava/ai-tech-articles', split='train')
df = dataset.to_pandas()
df = df[df["year"] == 2023]

# concat all "text" column values into one string
text = " ".join(df["text"].tolist())
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
_encode = lambda s: [stoi.get(c, 0) for c in s]
_decode = lambda l: ''.join([itos.get(i, " ") for i in l])

class TextProcessor:  
  @classmethod
  def encode(cls, s) -> list[int]: return _encode(s)
  
  @classmethod
  def decode(cls, l):
    res = _decode(l).split()
    res = [res[i:i+10] for i in range(0, len(res), 10)]
    return "\n".join([" ".join(line) for line in res])

class Head(nn.Module):
  """One self-attention head."""
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(embeddings_size, head_size, bias=False)
    self.query = nn.Linear(embeddings_size, head_size, bias=False)
    self.value = nn.Linear(embeddings_size, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape; k = self.key(x)
    weights = self.query(x) @ k.transpose(-2, -1) * k.shape[-1] ** (-0.5)
    weights = F.softmax(weights.masked_fill(self.tril[:T, :T] == 0, float('-inf')), dim=-1)
    return self.dropout(weights) @ self.value(x)

class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel"""
  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, embeddings_size)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    return self.dropout(self.proj(torch.cat([h(x) for h in self.heads], dim=-1)))

class FeedFoward(nn.Module):
  """Feed-forward network with ReLU activation and dropout."""
  def __init__(self, embeddings_size):
    super().__init__()
    self.network = nn.Sequential(
      nn.Linear(embeddings_size, 4 * embeddings_size), nn.ReLU(),
      nn.Linear(4 * embeddings_size, embeddings_size), nn.Dropout(dropout),
    )

  def forward(self, x): return self.network(x)

class Block(nn.Module):
  """One transformer block. (self-attention, layer normalization, feed-forward, and residual connections)"""
  def __init__(self, embeddings_size, head_count):
    super().__init__()
    head_size = embeddings_size // head_count
    self.sa = MultiHeadAttention(head_count, head_size);  self.ffwd = FeedFoward(embeddings_size)
    self.ln1 = nn.LayerNorm(embeddings_size);             self.ln2 = nn.LayerNorm(embeddings_size)

  def forward(self, x):
    x += self.sa(self.ln1(x)); x += self.ffwd(self.ln2(x)); return x

class GPTLanguageModel(nn.Module):
  """GPT language model with a single embedding layer, n layers, and a linear output layer."""
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, embeddings_size)
    self.position_embedding_table = nn.Embedding(block_size, embeddings_size)
    self.blocks = nn.Sequential(*[Block(embeddings_size, head_count=head_count) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(embeddings_size)
    self.lm_head = nn.Linear(embeddings_size, vocab_size)
    self.apply(self._init_weights)

  def _init_weights(self, module):
    if isinstance(module, nn.Linear):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
      if module.bias is not None: torch.nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
      torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

  def forward(self, index, targets=None):
    """forward pass. returns logits for next token, and optionally the computed loss."""
    B, T = index.shape
    token_embeddings = self.token_embedding_table(index)
    positional_embeddings = self.position_embedding_table(torch.arange(T, device=device))
    logits = self.lm_head(self.ln_f(self.blocks(token_embeddings + positional_embeddings)))
    return logits

  def generate(self, index, count):
    """Generate up to *count* new tokens given a starting index."""
    for _ in range(count):
      index_next = torch.multinomial(F.softmax(self(index[:, -block_size:])[:, -1, :], dim=-1), num_samples=1)
      index = torch.cat((index, index_next), dim=1)
    return index

def load_model(path="checkpoints/transfusion.pth") -> GPTLanguageModel:
  model = GPTLanguageModel()
  model.load_state_dict(torch.load(path, map_location=device))
  return model.to(device)
