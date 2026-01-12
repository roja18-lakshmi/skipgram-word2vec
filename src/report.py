import numpy as np
import pickle
import gensim.downloader as api
import random

# Load your trained Skip-gram embeddings

W_in = np.load("../models/word_vectors.npy")

with open("../models/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

idx2word = {i: w for w, i in word2idx.items()}

print(" trained Skip-gram model vocab size:", len(word2idx))
print("Embedding shape:", W_in.shape)

#
# Cosine similarity function

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def word_cosine(w1, w2, embeddings=W_in):
    if w1 not in word2idx or w2 not in word2idx:
        return None
    return cosine_similarity(embeddings[word2idx[w1]], embeddings[word2idx[w2]])


# Compare with Gensim pretrained Word2Vec

print("\nLoading Gensim pretrained Word2Vec (25-dim)...")
gensim_model = api.load("glove-twitter-25")  # ~100MB
print("Gensim pretrained loaded")

# Selected word pairs to compare
word_pairs = [("king","queen"), ("man","woman"), ("doctor","nurse"), ("cat","dog")]

print("\nWord Cosine Similarity Comparison")
print(f"{'Word Pair':<15} | {'trained Model':<10} | {'Gensim':<10}")
for w1, w2 in word_pairs:
    your_sim = word_cosine(w1, w2)
    gensim_sim = gensim_model.similarity(w1, w2) if w1 in gensim_model.key_to_index and w2 in gensim_model.key_to_index else None
    print(f"{w1}-{w2:<10} | {your_sim:.3f}     | {gensim_sim:.3f}")

# Word Analogy Task

def analogy(a, b, c, embeddings=W_in, top_n=5):
    if a not in word2idx or b not in word2idx or c not in word2idx:
        return []
    vec = embeddings[word2idx[a]] - embeddings[word2idx[b]] + embeddings[word2idx[c]]
    sims = {}
    for word, idx in word2idx.items():
        sims[word] = cosine_similarity(vec, embeddings[idx])
    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]

analogy_examples = [("king","man","woman"), ("paris","france","italy"), ("walking","walked","swimming")]

print("\nWord Analogy Results (trained Model):")
for a, b, c in analogy_examples:
    result = analogy(a, b, c)
    print(f"{a} - {b} + {c} ≈ {[word for word, score in result]}")

# Gender Bias Detection

def gender_bias(word, embeddings=W_in):
    if word not in word2idx or "he" not in word2idx or "she" not in word2idx:
        return None
    return cosine_similarity(embeddings[word2idx[word]], embeddings[word2idx["he"]]) - \
           cosine_similarity(embeddings[word2idx[word]], embeddings[word2idx["she"]])

bias_words = ["doctor","nurse","king","queen","programmer","homemaker"]

print("\nGender Bias Scores (trained Model):")
for w in bias_words:
    bias = gender_bias(w)
    print(f"{w:<12} bias: {bias:.3f}" if bias is not None else f"{w:<12} bias: N/A")
