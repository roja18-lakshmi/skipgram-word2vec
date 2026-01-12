# Implementation and Evaluation of Skip-gram with Negative Sampling

This project implements **Skip-gram with Negative Sampling (SGNS)** from scratch and trains it on a Wikipedia corpus.  
The learned word embeddings are evaluated using **cosine similarity**, **word analogy tasks**, and **gender bias detection**, and are compared with **Gensim’s pre-trained Word2Vec vectors**.



## 1. Dataset

The model is trained using the **text8 dataset**, derived from Wikipedia and available at:

https://mattmahoney.net/dc/textdata.html

### Dataset Characteristics
- Lowercased Wikipedia text  
- Cleaned and tokenized  
- Widely used for benchmarking word embedding models  
- Only words with a minimum frequency threshold are retained to build the vocabulary  

---

## 2. Methodology

### 2.1 Skip-gram with Negative Sampling

The Skip-gram model learns word embeddings by predicting surrounding **context words** given a **target word**.

For a target word \( w_t \) and a context word \( w_c \), the objective is to maximize the probability of observing \( w_c \) given \( w_t \), while minimizing the probability of randomly sampled negative words.

Negative sampling improves training efficiency by avoiding the expensive full softmax computation.

---

### 2.2 Model Configuration

| Parameter | Value |
|---------|------|
| Vocabulary size | 10,796 |
| Embedding dimension | 100 |
| Training corpus | Wikipedia (text8 subset) |
| Training method | Skip-gram with Negative Sampling |

The learned embeddings are stored and later evaluated using cosine similarity.

---

## 3. Evaluation

The trained embeddings are compared with **Gensim pre-trained Word2Vec vectors**  
(https://radimrehurek.com/gensim/models/word2vec.html).

### 3.1 Cosine Similarity

Cosine similarity measures semantic similarity between two word vectors:

\[
\text{cosine}(w_1, w_2) = \frac{v_1 \cdot v_2}{\|v_1\| \|v_2\|}
\]

This metric is used to compare similarity scores from:
- Our trained Skip-gram model  
- Gensim’s pre-trained Word2Vec model  

---

### 3.2 Word Analogy Task

Word analogies are solved using vector arithmetic:

\[
\text{vector} = v(a) - v(b) + v(c)
\]

The nearest words to this resulting vector are retrieved to evaluate semantic relationships.

---

### 3.3 Gender Bias Detection

Gender bias is measured as:

\[
\text{bias(word)} = \cos(word, he) - \cos(word, she)
\]

- Positive values → male association  
- Negative values → female association  

---

## 4. Results (Sample Output)

### Cosine Similarity Comparison

| Word Pair | Our Model | Gensim |
|----------|-----------|--------|
| king – queen | 0.257 | 0.920 |
| man – woman | 0.351 | 0.765 |
| doctor – nurse | 0.162 | 0.659 |
| cat – dog | 0.167 | 0.959 |

### Word Analogy (Our Model)
- king − man + woman → *woman, king, engaged, knowledge, award*

### Gender Bias (Our Model)
- doctor bias: 0.089  
- nurse bias: 0.078  
- king bias: 0.053  
- queen bias: 0.006  

---

## 5. Conclusion

This project demonstrates a complete implementation of **Skip-gram with Negative Sampling** trained on Wikipedia text.  
Although the learned embeddings capture basic semantic relationships, their performance is lower than large-scale pre-trained Gensim models due to limited data size and training time.  
The experiments highlight the effectiveness of SGNS and provide insights into semantic similarity, analogy reasoning, and bias in word embeddings.

---

## References
- Mikolov et al., *Efficient Estimation of Word Representations in Vector Space*
- Gensim Word2Vec Documentation
- Wikipedia text8 dataset
