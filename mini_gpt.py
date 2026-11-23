
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. DEVICE SETUP
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

torch.manual_seed(42)

# ==========================================
# 2. DATASET AND CHARACTER TOKENIZER
# ==========================================

text = """
hello world
hello ai
hello machine learning
hello transformer
hello deep learning
machine learning is interesting
artificial intelligence is powerful
"""

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

def encode(text):
    return [stoi[ch] for ch in text]

def decode(ids):
    return "".join([itos[i] for i in ids])

data = torch.tensor(encode(text), dtype=torch.long)

# ==========================================
# 3. HYPERPARAMETERS
# ==========================================

batch_size = 4
block_size = 16
n_embd = 32
num_heads = 4
n_layers = 2
learning_rate = 0.001
max_iters = 1000

head_size = n_embd // num_heads

# ==========================================
# 4. TRAINING BATCH
# ==========================================

def get_batch():

    ix = torch.randint(
        len(data) - block_size,
        (batch_size,)
    )

    x = torch.stack([
        data[i:i + block_size] for i in ix
    ])

    y = torch.stack([
        data[i + 1:i + block_size + 1] for i in ix
    ])

    return x.to(device), y.to(device)


# ==========================================
# 5. SELF-ATTENTION HEAD
# ==========================================

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

    def forward(self, x):

        B, T, C = x.shape

        key = self.key(x)
        query = self.query(x)

        # Attention scores
        scores = query @ key.transpose(-2, -1)

        scores = scores * (head_size ** -0.5)

        # Causal masking
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float("-inf")
        )

        # Attention probabilities
        weights = F.softmax(scores, dim=-1)

        value = self.value(x)

        # Weighted values
        output = weights @ value

        return output


# ==========================================
# 6. MULTI-HEAD ATTENTION
# ==========================================

class MultiHeadAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.heads = nn.ModuleList([
            Head(head_size) for _ in range(num_heads)
        ])

        self.projection = nn.Linear(n_embd, n_embd)

    def forward(self, x):

        output = torch.cat(
            [head(x) for head in self.heads],
            dim=-1
        )

        output = self.projection(output)

        return output


# ==========================================
# 7. FEED-FORWARD NETWORK
# ==========================================

class FeedForward(nn.Module):

    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd)
        )

    def forward(self, x):
        return self.network(x)


# ==========================================
# 8. TRANSFORMER BLOCK
# ==========================================

class Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.attention = MultiHeadAttention()

        self.ffwd = FeedForward()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):

        # Attention + residual connection
        x = x + self.attention(self.ln1(x))

        # FFN + residual connection
        x = x + self.ffwd(self.ln2(x))

        return x


# ==========================================
# 9. COMPLETE MINI GPT MODEL
# ==========================================

class MiniGPT(nn.Module):

    def __init__(self):
        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size, n_embd
        )

        self.position_embedding = nn.Embedding(
            block_size, n_embd
        )

        self.blocks = nn.Sequential(
            *[Block() for _ in range(n_layers)]
        )

        self.final_norm = nn.LayerNorm(n_embd)

        self.output_layer = nn.Linear(
            n_embd, vocab_size
        )

    def forward(self, idx, targets=None):

        B, T = idx.shape

        # Token embeddings
        token_emb = self.token_embedding(idx)

        # Position embeddings
        positions = torch.arange(T, device=idx.device)

        position_emb = self.position_embedding(positions)

        # Combine token and position information
        x = token_emb + position_emb

        # Transformer blocks
        x = self.blocks(x)

        x = self.final_norm(x)

        # Vocabulary logits
        logits = self.output_layer(x)

        loss = None

        if targets is not None:

            B, T, C = logits.shape

            logits_flat = logits.reshape(B * T, C)
            targets_flat = targets.reshape(B * T)

            loss = F.cross_entropy(
                logits_flat,
                targets_flat
            )

        return logits, loss

    # ======================================
    # 10. TEXT GENERATION
    # ======================================

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):

        self.eval()

        for _ in range(max_new_tokens):

            # Keep latest context
            idx_cond = idx[:, -block_size:]

            # Predict
            logits, loss = self(idx_cond)

            # Last token predictions
            logits = logits[:, -1, :]

            # Probabilities
            probabilities = F.softmax(logits, dim=-1)

            # Sample next character
            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )

            # Append next character
            idx = torch.cat(
                (idx, next_token),
                dim=1
            )

        return idx


# ==========================================
# 11. MODEL INITIALIZATION
# ==========================================

model = MiniGPT().to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)

print("\nModel initialized successfully!")
print("Vocabulary size:", vocab_size)
print("Parameters:", sum(p.numel() for p in model.parameters()))


# ==========================================
# 12. MODEL TRAINING
# ==========================================

print("\nTraining started...\n")

for step in range(max_iters):

    model.train()

    x, y = get_batch()

    logits, loss = model(x, y)

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    if step % 100 == 0:

        print(
            f"Step: {step:4d} | Loss: {loss.item():.4f}"
        )

print("\nTraining completed!")


# ==========================================
# 13. GENERATE TEXT
# ==========================================

model.eval()

start_text = "hello"

context = torch.tensor(
    [encode(start_text)],
    dtype=torch.long,
    device=device
)

generated = model.generate(
    context,
    max_new_tokens=50
)

result = decode(generated[0].tolist())

print("\nGenerated text:")
print("-------------------------")
print(result)
print("-------------------------")
