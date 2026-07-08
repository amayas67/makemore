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
    print(f"Processing word: {word}")
    context = [0] * block_size  # Start with a context of all '.'
    for ch in word + '.':
        ix = stoi[ch]
        X.append(context)
        Y.append(ix)
        print(f"Context: {''.join(itos[i] for i in context)} Next char: {ch}")
        context.pop(0)  # Remove the oldest character from the context
        context = context + [ix]  # Update the context

X = torch.tensor(X)
Y = torch.tensor(Y)

C = torch.randn((27, 2), requires_grad=True)  # Character embeddings
embeddings = C[X]  # Look up the embeddings for each character in the context

W1 = torch.randn((6, 100), requires_grad=True)  # First layer weights 6 because we have 3 characters in the context and each character is represented by a 2-dimensional embedding
b1 = torch.randn(100, requires_grad=True)  # First layer bias

embd = embeddings.view(-1, 6)  # Flatten the embeddings to shape (batch_size, 6)
h = torch.tanh(embd @ W1 + b1)  # Hidden layer

W2 = torch.randn((100, 27), requires_grad=True)  # Second layer weights
b2 = torch.randn(27, requires_grad=True)  # Second layer bias
