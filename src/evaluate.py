import numpy as np
import collections
import random


# STEP 1: Load trained embeddings


W_in = np.load("../models/word_vectors.npy")

# # Rebuild vocabulary (same way as training)
# with open("../data/text8", "r", encoding="utf-8") as f:
#     words = f.read().split()[:500_000]

# word_counts = collections.Counter(words)
# vocab = {w: c for w, c in word_counts.items() if c >= 5}

# word2idx = {w: i for i, w in enumerate(vocab)}
# idx2word = {i: w for w, i in word2idx.items()}

# print("Vocabulary size:", len(word2idx))
# print("Embedding shape:", W_in.shape)
import pickle
import numpy as np

W_in = np.load("../models/word_vectors.npy")

with open("../models/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

idx2word = {i: w for w, i in word2idx.items()}

print("Vocabulary size:", len(word2idx))
print("Embedding shape:", W_in.shape)



# STEP 2: Cosine similarity function


def cosine_similarity(w1, w2):
    v1 = W_in[word2idx[w1]]
    v2 = W_in[word2idx[w2]]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


# STEP 3: Test similarity using VALID vocab words


valid_words = list(word2idx.keys())

# w1, w2 = random.sample(valid_words, 2)

# print("\n--- Our Skip-gram Cosine Similarity (random words in vocab) ---")
# print(f"{w1} – {w2}:", cosine_similarity(w1, w2))


# STEP 4: Word Analogy (only if all words exist)


def analogy(a, b, c, top_n=5):
    vec = (
        W_in[word2idx[a]]
        - W_in[word2idx[b]]
        + W_in[word2idx[c]]
    )

    sims = {}
    for word, idx in word2idx.items():
        sims[word] = np.dot(vec, W_in[idx]) / (
            np.linalg.norm(vec) * np.linalg.norm(W_in[idx])
        )

    return sorted(sims.items(), key=lambda x: x[1], reverse=True)[:top_n]

# pick random analogy words
a, b, c = random.sample(valid_words, 3)

print("\n--- Word Analogy ---")
print(f"{a} - {b} + {c} ≈", analogy(a, b, c))

# STEP 5: Gender Bias (only if pronouns exist)


def gender_bias(word):
    return cosine_similarity(word, "he") - cosine_similarity(word, "she")

if "he" in word2idx and "she" in word2idx:
    test_word = random.choice(valid_words)
    print("\n--- Gender Bias ---")
    print(f"{test_word} bias:", gender_bias(test_word))
else:
    print("\n--- Gender Bias ---")
    print("he/she not present in vocab → cannot compute bias")
