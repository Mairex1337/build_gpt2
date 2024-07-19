import torch
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters
batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
eval_iters = 200
learning_rate = 1e-2
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device}")
# --------------


torch.manual_seed(1337)

#!wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt #tiny shakespeare
with open("/home/mairex/projects/test/buildGPT2/tinyshakespeare.txt", "r", encoding="utf-8") as f:
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

# super simple bigram language model
class BigramLanguageModel(nn.Module):

  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx) # (B, T, C)
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
      logits, loss = self(idx)
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
  

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create Optimizer object
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
print(sum(p.numel() for p in m.parameters())/1e3, "K parameters")

# training loop
for iter in range(max_iters):

    # every once in a while calculate and print the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step: {iter}, Train loss: {losses['train']:.4f}, Eval loss: {losses['eval']:.4f}")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print("Output:", decode(m.generate(context, max_new_tokens=500)[0].tolist()))
