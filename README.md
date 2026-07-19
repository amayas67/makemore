# makemore — Character-Level Language Model from Scratch

Ces notes ne suivent pas la série *makemore* de Karpathy, elles la **redérivent** : chaque choix d'ingénierie est reconstruit depuis ses fondations, pas accepté comme une recette.

---

## What is implemented

| Component | File | Description |
|---|---|---|
| Baseline counting model | `makemore.py` | Bigram count matrix + Laplace smoothing → generative baseline |
| Neural bigram | `makemore.py` | One-hot + linear layer (27 → 27), gradient descent on NLL |
| Smarter neural net | `makemore_smarter.py` | Embedding layer, hidden layer, tanh, dropout, `F.cross_entropy` |
| Optimized version | `makemore_optimized.py` | Batch matrix multiply, optimized generation, full training loop |
| Refactor (WIP) | `makemore_refactored.py` | Part 3 — modules `Linear` / `BatchNorm1d` réutilisables |

### HTML notes — interactive derivations

`notes/intro.html`
: Empirical distribution of characters, bigram probability matrix, generation from the counting baseline.

`notes/makemore_smarter.html`
: Pourquoi une couche linéaire sur des one-hot est juste un bigramme déguisé → embeddings → couche cachée → dynamique d'entraînement.

`notes/makemore_optimized.html`
: Batch-efficient forward/backward pass, learning-rate finder, train/val/test split, overfitting analysis.

`notes/probability_foundations_deep_learning.html`
: Probabilistic grounding: MLE, NLL, cross-entropy as a loss, and why the model really optimizes a softmax classifier.

**Ordre de lecture recommandé :**
`intro.html` → `makemore_smarter.html` → `makemore_optimized.html` → `probability_foundations_deep_learning.html`

---

## Key insight

The counting model and the single-layer neural network are mathematically equivalent:
both compute `softmax(W · onehot(x))`. The neural net just learns `W` via backprop
instead of counting co-occurrences. Once this equivalence is exposed, every upgrade
(embeddings, hidden layers, deep nets) becomes a deliberate architectural choice,
not a magic trick.

---

## Architecture

```
makemore/
├── README.md                          # ← you are here
├── requirements.txt                   # torch, matplotlib
├── makemore.py                        # Baseline counting model + linear bigram
├── makemore_smarter.py               # Embedding + hidden layer + training loop
├── makemore_optimized.py             # Batch ops, LR finder, split data
├── makemore_refactored.py            # (WIP) Reusable Linear, BatchNorm1d
├── data/
│   └── name.txt                       # Dataset of names
└── notes/
    ├── intro.html
    ├── makemore_smarter.html
    ├── makemore_optimized.html
    └── probability_foundations_deep_learning.html
```

---

## References

- Karpathy, A. — *Let's build makemore* (series)
- Karpathy, A. — *The spelled-out intro to neural networks and backpropagation: building micrograd*