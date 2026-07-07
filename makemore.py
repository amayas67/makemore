# ------------------------------------------------------------------
# Load dataset
# ------------------------------------------------------------------

DATASET_PATH = "data/name.txt"

words = open(DATASET_PATH, "r").read().splitlines()

print(f"First 10 names: {words[:10]}")
print(f"Total number of names: {len(words)}")

shortest_name = min(words, key=len)
longest_name = max(words, key=len)

print(f"Shortest name: {shortest_name}")
print(f"Longest name: {longest_name}")

print(f"Minimum length: {min(len(word) for word in words)}")
print(f"Maximum length: {max(len(word) for word in words)}")

# ------------------------------------------------------------------
# Vocabulary
# ------------------------------------------------------------------

import torch

chars = sorted(list(set(''.join(words))))

stoi = {ch: i + 1 for i, ch in enumerate(chars)}
stoi['.'] = 0

itos = {i: ch for ch, i in stoi.items()}

print(stoi)
print(itos)

# ------------------------------------------------------------------
# Bigram count matrix (N) — the "counting" baseline model
# ------------------------------------------------------------------

N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)


def build_bigram_matrix(words):
    for word in words:
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1


build_bigram_matrix(words)

# ------------------------------------------------------------------
# Visualization
# ------------------------------------------------------------------

import matplotlib.pyplot as plt

plt.figure(figsize=(20, 20))
plt.imshow(N, cmap="Blues")

chars_with_tokens = [itos[i] for i in range(len(stoi))]
plt.xticks(range(len(stoi)), chars_with_tokens)
plt.yticks(range(len(stoi)), chars_with_tokens)

# plt.colorbar()
# plt.show()

# ------------------------------------------------------------------
# Convert counts to probabilities (WITH SMOOTHING)
# ------------------------------------------------------------------

P = (N + 1).float()   # +1 smoothing: guarantees no cell is ever exactly 0
P = P / P.sum(1, keepdim=True)  # normalize row by row -> each row sums to 1

# ------------------------------------------------------------------
# Generate names from the counting model (P)
# ------------------------------------------------------------------

g = torch.Generator().manual_seed(2147483647)


def generate_name():
    name = []
    ix = 0

    while True:
        p = P[ix]  # only the row for the current character - no cross-row competition

        ix = torch.multinomial(
            p,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()

        if ix == 0:
            break

        name.append(itos[ix])

    return "".join(name)


print("\nGenerated names (counting model):\n")

for _ in range(10):
    print(generate_name())

# ------------------------------------------------------------------
# Log-Likelihood of the counting model (quality check for P)
# ------------------------------------------------------------------

log_likelihood = 0.0
num_bigrams = 0

for word in words:
    chars = ['.'] + list(word) + ['.']

    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]

        probability = P[ix1, ix2]

        log_likelihood += torch.log(probability)
        num_bigrams += 1

negative_log_likelihood = -log_likelihood
average_nll = negative_log_likelihood / num_bigrams

print(f"\nLog-Likelihood: {log_likelihood:.4f}")
print(f"Negative Log-Likelihood: {negative_log_likelihood:.4f}")
print(f"Average NLL: {average_nll:.4f}")

# ------------------------------------------------------------------
# Build the training set (xs -> ys) for the neural network version
# ------------------------------------------------------------------

xs, ys = [], []
for word in words:
    chars = ['.'] + list(word) + ['.']

    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)

# lowercase t (torch.tensor) infers dtype from the data, capital T (torch.Tensor)
# always gives float32 - lowercase is the safer default here since xs/ys are indices
xs = torch.tensor(xs)
ys = torch.tensor(ys)

# ------------------------------------------------------------------
# Neural network setup — a single linear layer (27 -> 27)
# ------------------------------------------------------------------

import torch.nn.functional as F

xenc = F.one_hot(xs, num_classes=len(stoi)).float()  # one-hot encode every input character once
W = torch.randn((len(stoi), len(stoi)), generator=g, requires_grad=True)  # trainable weight matrix

# forward pass (before any training, just to see the starting loss)
logits = xenc @ W                                  # log-counts
counts = logits.exp()                              # counts, equivalent to N
probs = counts / counts.sum(1, keepdim=True)        # probabilities, row-normalized (equivalent to P)

loss = -probs[torch.arange(len(xs)), ys].log().mean()  # negative log-likelihood loss
print(f"\nInitial loss (before training): {loss.item():.4f}")

# ------------------------------------------------------------------
# Training loop — gradient descent on W
# ------------------------------------------------------------------

for epoch in range(1000):
    # forward pass
    logits = xenc @ W
    counts = logits.exp()
    probs = counts / counts.sum(1, keepdim=True)
    loss = -probs[torch.arange(len(xs)), ys].log().mean()

    # backward pass
    if W.grad is not None:
        W.grad = None  # reset gradients to zero before backward pass
    loss.backward()

    # update
    W.data += -10 * W.grad  # gradient descent, learning rate = 10

print(f"Loss after training: {loss.item():.4f}")

# ------------------------------------------------------------------
# Generate names from the trained neural network
# ------------------------------------------------------------------

def generate_names_2():
    name = []
    ix = 0

    while True:
        xenc = F.one_hot(torch.tensor([ix]), num_classes=len(stoi)).float()  # shape (1, 27): only this ix's row
        logits = xenc @ W
        counts = logits.exp()
        probs = counts / counts.sum(1, keepdim=True)

        ix = torch.multinomial(probs, num_samples=1).item()

        if ix == 0:
            break

        name.append(itos[ix])

    return "".join(name)


print("\nGenerated names after training:\n")
for _ in range(10):
    print(generate_names_2())