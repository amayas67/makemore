# Path to the dataset
DATASET_PATH = "data/name.txt"

# Load all names into a list
words = open(DATASET_PATH, "r").read().splitlines()

# Display basic information
print(f"First 10 names: {words[:10]}")
print(f"Total number of names: {len(words)}")

# Find the shortest and longest names
shortest_name = min(words, key=len)
longest_name = max(words, key=len)

print(f"Shortest name: {shortest_name}")
print(f"Longest name: {longest_name}")

# Display the minimum and maximum name lengths
print(f"Minimum length: {min(len(word) for word in words)}")
print(f"Maximum length: {max(len(word) for word in words)}")


import torch

chars = sorted(list(set(''.join(words))))  # Unique characters
stoi = {ch: i+1 for i, ch in enumerate(chars)}  # Char to index
# print(f"Unique characters: {chars}")
# print(f"Character to index mapping: {stoi}")
stoi['.'] = 0  # Start token

N = torch.zeros((len(stoi), len(stoi)), dtype=torch.int32)  # Bigram count matrix
# print(f"Updated character to index mapping: {stoi}")

def build_bigram_matrix(words):
    for word in words:
        # Add start and end tokens
        chars = ['.'] + list(word) + ['.']
        for ch1, ch2 in zip(chars, chars[1:]):
            ix1 = stoi[ch1]
            ix2 = stoi[ch2]
            N[ix1, ix2] += 1

build_bigram_matrix(words)

itos = {i: ch for ch, i in stoi.items()}  # Index to char
print(itos)

import matplotlib.pyplot as plt

# Visualization of the bigram matrix N
plt.figure(figsize=(20, 20))
plt.imshow(N, cmap='Blues')

# Add character labels on the axes
chars_with_tokens = [itos[i] for i in range(len(stoi))]
plt.xticks(ticks=range(len(stoi)), labels=chars_with_tokens, fontsize=10)
plt.yticks(ticks=range(len(stoi)), labels=chars_with_tokens, fontsize=10)

# Add values in each cell
# for i in range(len(stoi)):
#     for j in range(len(stoi)):
#         if N[i, j] > 0:
#             plt.text(j, i, N[i, j].item(), ha='center', va='center', fontsize=8, color='black')

# plt.xlabel('Second character (j)')
# plt.ylabel('First character (i)')
# plt.title('Bigram Count Matrix N')
# plt.colorbar(label='Count')
# plt.tight_layout()
# plt.show()

g = torch.Generator().manual_seed(2147483647)  # For reproducibility
print(stoi)
def generate_name():
    name = []
    ix = 0  # Start with the start/end token '.'

    while True:
        # Get the probability distribution for the current character
        p = N[ix].float()
        p = p / p.sum()

        # Sample the next character
        ix = torch.multinomial(
            p,
            num_samples=1,
            replacement=True,
            generator=g
        ).item()

        # Stop if we reach the end token
        if ix == 0:
            break

        name.append(itos[ix])

    return ''.join(name)


print("Generated names:")
for _ in range(10):
    print(generate_name())