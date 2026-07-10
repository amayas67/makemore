import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Load data
# ============================================================
words = open('data/name.txt', 'r').read().splitlines()
print(f"First 10 names: {words[:10]}")
print(f"Total number of names: {len(words)}")

chars = sorted(list(set(''.join(words))))
chars = ['.'] + chars  # '.' represents start/end token
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}


# ============================================================
# Build the dataset
# ============================================================
block_size = 3  # context length: how many characters we use to predict the next

X, Y = [], []
for word in words:
    context = [0] * block_size  # start with all '.'
    for ch in word + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        context = context[1:] + [ix]  # slide the context window

X = torch.tensor(X)
Y = torch.tensor(Y)


# ============================================================
# Initialize parameters
# ============================================================
C = torch.randn((27, 2), requires_grad=True)  # character embedding table

W1 = torch.randn((6, 100), requires_grad=True)  # block_size(3) * emb_dim(2) = 6
b1 = torch.randn(100, requires_grad=True)

W2 = torch.randn((100, 27), requires_grad=True)
b2 = torch.randn(27, requires_grad=True)

params = [C, W1, b1, W2, b2]
lrs = 0.1


# ============================================================
# Training loop
# ============================================================
for epoch in range(1000):
    # Minibatch
    ix = torch.randint(0, X.shape[0], (32,))

    # Forward pass
    embeddings = C[X[ix]]          # (32, 3, 2)
    embd = embeddings.view(-1, 6)  # (32, 6)
    h = torch.tanh(embd @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y[ix])

    # Backward pass
    for p in params:
        p.grad = None
    loss.backward()

    # Gradient descent
    for p in params:
        p.data -= lrs * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

print("Training complete.")


# ============================================================
# Generate new names
# ============================================================
print("Generating new names:")
for _ in range(10):
    context = [0] * block_size
    name = ''

    while True:
        embeddings = C[torch.tensor([context])]
        embd = embeddings.view(-1, 6)
        h = torch.tanh(embd @ W1 + b1)
        logits = h @ W2 + b2
        probs = F.softmax(logits, dim=1)
        ix = torch.multinomial(probs, num_samples=1).item()

        if ix == 0:  # '.' marks the end of the name
            break

        name += itos[ix]
        context = context[1:] + [ix]

    print(name)