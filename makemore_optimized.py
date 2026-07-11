import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

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
vocab_size = len(chars)  # plus de "27" en dur
block_size = 3  # context length

# ============================================================
# Build the dataset
# ============================================================
def build_dataset(words, block_size):
    X, Y = [], []
    for word in words:
        context = [0] * block_size  # start with all '.'
        for ch in word + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix]  # slide the context window
    return torch.tensor(X), torch.tensor(Y)

random.seed(42)
random.shuffle(words)
n1 = int(0.8 * len(words))
n2 = int(0.9 * len(words))
Xtr,  Ytr  = build_dataset(words[:n1],  block_size)  # 80%
Xdev, Ydev = build_dataset(words[n1:n2], block_size)  # 10%
Xte,  Yte  = build_dataset(words[n2:],  block_size)  # 10%

# ============================================================
# Initialize parameters
# ============================================================
n_embd   = 10   # dimensionality of the character embedding vectors
n_hidden = 200  # number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647)  # for reproducibility

C  = torch.randn((vocab_size, n_embd),               generator=g)
W1 = torch.randn((n_embd * block_size, n_hidden),    generator=g) *(5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                            generator=g)
W2 = torch.randn((n_hidden, vocab_size),              generator=g) * 0.01
b2 = torch.randn(vocab_size,                          generator=g) *0

bngain = torch.ones((1, n_hidden))
bnbias = torch.zeros((1, n_hidden))

# NEW: running estimates of the batch-norm mean/std.
# Not trained via backprop (no requires_grad) — just updated with a moving
# average during training, then used as fixed stats at eval/generation time.
bnmean_running = torch.zeros((1, n_hidden))
bnstd_running  = torch.ones((1, n_hidden))

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
print(sum(p.nelement() for p in parameters))  # number of parameters in total
for p in parameters:
    p.requires_grad = True

# Hyperparamètres d'entraînement (plus de magic numbers non plus)
lr         = 0.1
max_steps  = 100000
batch_size = 32
print_every = 100

# ============================================================
# Training loop
# ============================================================
lossi = []
for step in range(max_steps):

    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)

    # forward pass
    emb    = C[Xtr[ix]]                       # (batch_size, block_size, n_embd)
    embcat = emb.view(emb.shape[0], -1)       # (batch_size, block_size * n_embd)
    hpreact = embcat @ W1 + b1

    # batch normalization
    bnmeani = hpreact.mean(0, keepdim=True)
    bnstdi  = hpreact.std(0, keepdim=True)
    hpreact = bngain * (hpreact - bnmeani) / bnstdi + bnbias

    # NEW: update the running stats with a moving average (no gradient here —
    # this isn't a learned parameter, just a running record of the batch stats)
    with torch.no_grad():
        bnmean_running = 0.999 * bnmean_running + 0.001 * bnmeani
        bnstd_running  = 0.999 * bnstd_running  + 0.001 * bnstdi

    h       = torch.tanh(hpreact)             # (batch_size, n_hidden)
    logits  = h @ W2 + b2                     # (batch_size, vocab_size)
    loss    = F.cross_entropy(logits, Ytr[ix])

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()
    lr =  0.1 if step <10000 else 0.01
    # update
    for p in parameters:
        p.data += -lr * p.grad

    lossi.append(loss.log10().item())
    if step % print_every == 0:
        print(f"Step {step:7d}/{max_steps}: Loss {loss.item():.4f}")

print("Training complete.")
plt.plot(lossi)

# ============================================================
# Evaluate loss on a split (factorisé, plus de duplication)
# ============================================================
@torch.no_grad()
def split_loss(X, Y):
    emb    = C[X]
    embcat = emb.view(emb.shape[0], -1)
    h      = torch.tanh(embcat @ W1 + b1)
    # CHANGED: use the running stats instead of h.mean(0)/h.std(0)
    h = bngain * (h - bnmean_running) / bnstd_running + bnbias
    logits = h @ W2 + b2
    return F.cross_entropy(logits, Y).item()

print(f"Train Loss: {split_loss(Xtr, Ytr):.4f}")
print(f"Dev Loss:   {split_loss(Xdev, Ydev):.4f}")
print(f"Test Loss:  {split_loss(Xte, Yte):.4f}")

# ============================================================
# Generate new names
# ============================================================
# Ensemble de tous les mots du dataset pour vérification O(1)
words_set = set(words)

print("Generating new names:")
num_samples = 20
for _ in range(num_samples):
    out = []
    context = [0] * block_size
    while True:
        emb     = C[torch.tensor([context])]
        embcat  = emb.view(emb.shape[0], -1)
        h       = torch.tanh(embcat @ W1 + b1)
        # CHANGED: use the running stats instead of h.mean(0)/h.std(0)
        # (batch size is 1 here — h.std(0) would be NaN, this is the fix)
        h = bngain * (h - bnmean_running) / bnstd_running + bnbias
        logits  = h @ W2 + b2
        probs   = F.softmax(logits, dim=1)
        ix      = torch.multinomial(probs, num_samples=1, generator=g).item()
        context = context[1:] + [ix]
        if ix == 0:  # '.' marks the end of the name
            break
        out.append(ix)

    generated_name = ''.join(itos[i] for i in out)
    in_dataset = "Oui" if generated_name in words_set else "Non"
    print(f"{generated_name:15s} | Dans le dataset : {in_dataset}")