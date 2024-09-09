# Build GPT-2

All of the code in here was a learning experience based on [Karpathy's](https://github.com/karpathy) awesome [video Lectures](https://www.youtube.com/@AndrejKarpathy)!
Thank you!!

This Repository includes:

- A **Bigram Model** (bigram.py) trained on Tiny Shakespeare
- A basic **Transformer model** (microGPT), also trained on Tiny Shakespeare
- An advanced **Transformer model** (gpt2s.py), essentially a GPT2 clone, trained on Fineweb-EDU

The advanced Transformer model includes:

- Weight tying scheme of Input Embedding and Output Embedding
- Pytorch DDP utilization
- Gradient Accumulation to fit on essentially any GPU('s)
- Weight decay and learning rate scheduler 
- Mixed precision (bfloat16)
- Top-k Sampling
- etc...

