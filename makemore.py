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
# Bigram count matrix
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

P = (N + 1).float()   # +1 smoothing
P = P / P.sum(1, keepdim=True) 

# ------------------------------------------------------------------
# Generate names
# ------------------------------------------------------------------

g = torch.Generator().manual_seed(2147483647)


def generate_name():
    name = []
    ix = 0

    while True:
        p = P[ix]

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


print("\nGenerated names:\n")

for _ in range(10):
    print(generate_name())

# ------------------------------------------------------------------
# Log-Likelihood
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

#trainig set 
xs, ys = [], []
for word in words:
    chars = ['.'] + list(word) + ['.']

    for ch1, ch2 in zip(chars, chars[1:]):
        ix1 = stoi[ch1]
        ix2 = stoi[ch2]
        xs.append(ix1)
        ys.append(ix2)
xs = torch.tensor(xs) # you can use capital T for Tensor, but lowercase t is more common because it gives you a tensor of the same type as the input.
ys = torch.tensor(ys)

import torch.nn.functional as F
xenc = F.one_hot(xs, num_classes=len(stoi)).float() # one-hot encoding of the input characters
W = torch.randn((len(stoi), len(stoi)), generator=g, requires_grad=True) # weight matrix for the linear layer
logits = xenc @ W 
counts = logits.exp() # convert logits to counts , we exponentiate the logits so they are positive and can be interpreted as counts
