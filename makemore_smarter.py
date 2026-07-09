import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

words = open('data/name.txt', 'r').read().splitlines()

print(f"First 10 names: {words[:10]}")
print(f"Total number of names: {len(words)}")
chars = sorted(list(set(''.join(words))))
chars = ['.'] + chars  # Add the '.' character to the list of characters
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}

#building the dataset 
block_size = 3
X, Y = [], []
for word in words:
    # print(f"Processing word: {word}")
    context = [0] * block_size  # Start with a context of all '.'
    for ch in word + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        # print(f"Context: {''.join(itos[i] for i in context)} Next char: {ch}")
        context = context[1:] + [ix]  # Update the context

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27, 2), requires_grad=True)

W1 = torch.randn((6, 100), requires_grad=True)
b1 = torch.randn(100, requires_grad=True)

W2 = torch.randn((100, 27), requires_grad=True)
b2 = torch.randn(27, requires_grad=True)

params = [C, W1, b1, W2, b2]

for epoch in range(100):

    # Embedding lookup
    embeddings = C[X]
    embd = embeddings.view(-1, 6)

    # Forward pass
    h = torch.tanh(embd @ W1 + b1)
    logits = h @ W2 + b2
    loss = F.cross_entropy(logits, Y)

    # Backward pass
    for p in params:
        p.grad = None

    loss.backward()

    # Gradient descent
    for p in params:
        p.data -= 0.1 * p.grad
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")


print("Training complete.")
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
        if ix == 0:  # If the sampled character is '.', we stop generating
            break
        name += itos[ix]
        context.pop(0)
        context.append(ix)
    print(name)