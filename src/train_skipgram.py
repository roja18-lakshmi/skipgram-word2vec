# STEP 1: Load Wikipedia Text
# with open("../data/text8", "r", encoding="utf-8") as f:
#     words = f.read().split()

# print("Total words:", len(words))
# MAX_WORDS = 1_000_000   # 10 lakh words
# words = words[:MAX_WORDS]

# print("Using words:", len(words))

import re
def clean_token(word):
    """
    Cleans Wikipedia tokens:
    - lowercase
    - remove wiki markup
    - remove non-alphabetic tokens
    """
    word = word.lower()

    # Remove Wikipedia markup
    word = re.sub(r"\[\[|\]\]|'''|==", "", word)

    # Remove URLs
    if word.startswith("http") or word.startswith("www"):
        return None

    # Keep only alphabetic words
    if not word.isalpha():
        return None

    return word


with open("../data/text8", "r", encoding="utf-8") as f:
    raw_words = f.read().split()

print("Total raw words:", len(raw_words))

MAX_WORDS = 1_000_000   # 10 lakh words
raw_words = raw_words[:MAX_WORDS]

words = []
for w in raw_words:
    clean_w = clean_token(w)
    if clean_w:
        words.append(clean_w)

print("Cleaned words:", len(words))


# STEP 3: Build vocabulary
from collections import Counter

word_counts = Counter(words)

# Remove rare words
vocab = {word: count for word, count in word_counts.items() if count >= 5}

print("Vocabulary size:", len(vocab))

# STEP 4: Create word-index mappings #word2idx["king"] → 1234 convert each word into a number
word2idx = {word: idx for idx, word in enumerate(vocab)}
idx2word = {idx: word for word, idx in word2idx.items()}
# STEP 5: Convert corpus to indices
# Replace words with their numeric IDs.
# "the king loves queen"
# → [5, 1523, 6789, 3892]

corpus = [word2idx[word] for word in words if word in word2idx]

print("Corpus length:", len(corpus))
# STEP 6: Generate skip-gram pairs
window_size = 2 #Look 2 words left and 2 words right
pairs = []

for i, target in enumerate(corpus):
    start = max(0, i - window_size)
    end = min(len(corpus), i + window_size + 1)

    for j in range(start, end):
        if i != j:
            pairs.append((target, corpus[j]))

# Limit number of pairs
pairs = pairs[:500_000]
print("Training pairs:", len(pairs))
# STEP 7: Initialize embeddings
import numpy as np

vocab_size = len(word2idx)
embedding_dim = 100 #number of values used to represent one word

W_in = np.random.randn(vocab_size, embedding_dim)
W_out = np.random.randn(vocab_size, embedding_dim)
# STEP 8: Negative sampling
import random

def get_negative_samples(context_idx, k=5):
    negatives = []
    while len(negatives) < k:
        neg = random.randint(0, vocab_size - 1)
        if neg != context_idx:
            negatives.append(neg)
    return negatives
# STEP 9: Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
# STEP 10: Train Skip-gram with Negative Sampling
from tqdm import tqdm

learning_rate = 0.025
epochs = 15

for epoch in range(epochs):
    print(f"Epoch {epoch+1}")
    for target, context in tqdm(pairs):
        # Positive sample
        score = sigmoid(np.dot(W_in[target], W_out[context]))
        grad = learning_rate * (1 - score)

        W_out[context] += grad * W_in[target]
        W_in[target] += grad * W_out[context]

        # Negative samples
        for neg in get_negative_samples(context):
            score = sigmoid(-np.dot(W_in[target], W_out[neg]))
            grad = learning_rate * (1 - score)

            W_out[neg] -= grad * W_in[target]
            W_in[target] -= grad * W_out[neg]

print("Training complete")
# STEP 11: Save trained vectors
# np.save("../models/word_vectors.npy", W_in)
# print("Word vectors saved")
import pickle

with open("../models/word2idx.pkl", "wb") as f:
    pickle.dump(word2idx, f)

print("Vocabulary saved")

