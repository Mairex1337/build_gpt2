import torch
import torch.nn as nn
from torch.nn import functional as F
import time

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# --------------


torch.manual_seed(1337)

#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt #tiny shakespeare
with open("/home/mairex/projects/test/src/input.txt", "r", encoding="utf-8") as f:
  text = f.read()

# characters that appear in the data
chars = sorted(list(set(text)))
vocab_size = len(chars)

# mapping from character to integers and vice versa
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[d] for d in l])

# split into train and val data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

# create data loeader
def get_batch(split):
  data = train_data if split == "train" else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,))
  x = torch.stack([data[i:i + block_size] for i in ix])
  y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
  x, y = x.to(device), y.to(device)
  return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "eval"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
  """Single head of attention"""

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias = False) # (C, head size)
    self.query = nn.Linear(n_embd, head_size, bias = False) # (C, head size)
    self.value = nn.Linear(n_embd, head_size, bias = False) # (C, head_size)
    self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape
    k = self.key(x) # (B, T, C) @  (*B*, C, head_size) = (B, T, head_size)
    q = self.query(x) # (B, T, head_size)

    # These are the attention affinities ("How much each token is interested in any token")
    # divide by sqrt(head size) to make sure specific affinities do not become too big
    wei = q @ k.transpose(-2, -1)* C**-0.5 # (B, T, head_size) @ (B, head_size, T) ---- (B, T, T) 
    # Filling everything but lower triangular form with -inf
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
    # normalize into row wise affinities that sum to one
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)
    v = self.value(x)
    out = wei @ v
    return out


class MultiHeadAttention(nn.Module):
  """Multiple heads of attention running in parralell"""

  def __init__(self, num_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(head_size * num_heads, n_embd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) 
    out = self.dropout(self.proj(out))
    return out
  


class FeedForward(nn.Module):
  """simple linear layer follow by non linearity (ReLU)"""

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
    )

  def forward(self, x):
    return self.net(x)
  


class Block(nn.Module):
  """Transformer block"""

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward(n_embd)
    self.ln1 = nn.LayerNorm(n_embd)
    self.ln2 = nn.LayerNorm(n_embd)


  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x
    


class GPTLLM(nn.Module):

  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embd) # final layer norm
    self.lm_head = nn.Linear(n_embd, vocab_size)


  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B, T) tensors of ints
    tok_embd = self.token_embedding_table(idx) # (B, T, C)
    pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_embd + pos_embd # (B, T, C)
    x = self.blocks(x) # (B, T, C)
    x = self.ln_f(x) # (B, T, C)
    logits = self.lm_head(x) # (B, T, vocab_size)
    
    if targets == None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)
      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, max_new_tokens):
    # idx is the context (B, T)
    for _ in range(max_new_tokens):
      # crop idx to block size
      idx_cond = idx[:, -block_size:]
      logits, loss = self(idx_cond)
      # pluck out only the last element
      logits = logits[:, -1, :] # (B, C)
      # calculate probs over C dim
      probs = F.softmax(logits, dim=-1)
      # sample from the probabilities
      # multinomial returns a tensor of (B, num_samples)
      idx_next = torch.multinomial(probs, num_samples=1)
      # append generated tokens column wise
      idx = torch.cat((idx, idx_next), dim=1) # (B, T + 1)

    return idx
  

model = GPTLLM()
model = torch.compile(model)
m = model.to(device)

# print num of parameters
print(sum(p.numel() for p in m.parameters())/1e6, "M parameters")

# create Optimizer object 
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
t1 = time.time()
# training loop
for iter in range(max_iters):
  
    # every once in a while calculate and print the loss
    if iter % eval_interval == 0 or iter == max_iters - 1:
        t3 = time.time()
        losses = estimate_loss()
        print(f"Step: {iter}, Train loss: {losses['train']:.4f}, Eval loss: {losses['eval']:.4f}, Time per step: {(t3 - t1)/ eval_interval:.4f}")
        t1 = time.time()

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print()
print("Output:", decode(m.generate(context, max_new_tokens=5000)[0].tolist()))
