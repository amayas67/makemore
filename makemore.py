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

def get_bigrams(words):
    bigrams = {}
    for word in words:
        # Add start and end tokens
        chars = ['<S>'] + list(word) + ['<E>']
        for ch1, ch2 in zip(chars, chars[1:]):
            bigram = (ch1, ch2)
            bigrams[bigram] = bigrams.get(bigram, 0) + 1
    return bigrams

sorted_bigrams = sorted(get_bigrams(words).items(), key=lambda x: x[1], reverse=True)

print(f"Most common bigrams: {sorted_bigrams[:10]}")