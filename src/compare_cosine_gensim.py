import numpy as np
import pickle
import gensim.downloader as api

# ---------------------------
# 1) Load your trained vectors
# ---------------------------
W = np.load("../models/word_vectors.npy")
with open("../models/word2idx.pkl", "rb") as f:
    word2idx = pickle.load(f)

# words in your model
words = list(word2idx.keys())

print("Your model vocab size:", len(word2idx))
print("Your embedding shape:", W.shape)

# ---------------------------
# 2) Load Gensim pretrained Word2Vec
# ---------------------------

print("Loading Gensim pretrained Word2Vec...")

# You can choose one of the following:
# 'word2vec-google-news-300' — 300-dim GoogleNews vectors (~1.6GB)
# 'glove-wiki-gigaword-100' — smaller 100-dim GloVe vectors
gensim_model = api.load("glove-twitter-25")
print("Gensim pretrained loaded")

# ---------------------------
# 3) Compare cosine similarities
# ---------------------------

# Define a helper that gets a vector given a word
# for your model
def get_our_vec(word):
    return W[word2idx[word]]

# for gensim model
def get_gensim_vec(word):
    return gensim_model[word]  # Gensim KeyedVectors

# Example list of word pairs
pairs = [
    ("king", "queen"),
    ("man", "woman"),
    ("doctor", "nurse"),
    ("cat", "dog")
]

print("\nWord pair | Your model | Gensim model")
for w1, w2 in pairs:
    if w1 in word2idx and w2 in word2idx and w1 in gensim_model.key_to_index and w2 in gensim_model.key_to_index:
        # your model similarity
        v1, v2 = get_our_vec(w1), get_our_vec(w2)
        cos_our = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

        # gensim pretrained similarity
        cos_gen = gensim_model.similarity(w1, w2)

        print(f"{w1}-{w2} | {cos_our:.3f} | {cos_gen:.3f}")
    else:
        print(f"{w1}-{w2} | missing in vocab")

