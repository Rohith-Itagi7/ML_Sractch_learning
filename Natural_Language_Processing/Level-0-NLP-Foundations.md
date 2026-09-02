# Level 0 — NLP Foundations

This section covers the fundamental concepts required to understand
Natural Language Processing (NLP) before moving into word embeddings,
deep learning, and Transformers.

---

## 📚 Topics

### 1. Introduction to NLP

Natural Language Processing (NLP) is a field of Artificial Intelligence
that focuses on enabling computers to understand, process, and generate
human language.

**Topics covered:**

- What is NLP?
- Applications of NLP
- NLP pipeline
- Challenges in NLP

---

### 2. Text Normalization

Text normalization converts raw text into a consistent format that
can be processed by NLP algorithms.

**Topics covered:**

- Lowercasing
- Removing unnecessary characters
- Handling punctuation
- Handling whitespace
- Expanding contractions
- Unicode normalization

---

### 3. Tokenization

Tokenization is the process of breaking text into smaller units called
tokens.

**Topics covered:**

- Word tokenization
- Sentence tokenization
- Character tokenization
- Subword tokenization

---

### 4. Stopwords

Stopwords are common words that may carry little useful information
for certain NLP tasks.

**Examples:**

```text
the
is
a
an
and
of
in
```

---

### 5. Stemming

Stemming reduces words to a root-like form by removing prefixes or
suffixes.

**Example:**

```text
playing  → play
played   → play
plays    → play
```

> Note: Stemming can sometimes produce words that are not valid
> dictionary words.

---

### 6. Lemmatization

Lemmatization converts a word into its meaningful dictionary form,
called a lemma.

**Example:**

```text
running → run
better  → good
mice    → mouse
```

---

### 7. Regular Expressions

Regular Expressions (Regex) are patterns used to search, extract,
replace, or manipulate text.

**Applications in NLP:**

- Extracting emails
- Extracting URLs
- Finding numbers
- Removing unwanted characters
- Pattern matching

---

## 🔄 NLP Preprocessing Pipeline

A basic NLP preprocessing pipeline looks like:

```text
Raw Text
   ↓
Text Normalization
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Stemming / Lemmatization
   ↓
Clean Text
   ↓
Feature Extraction
```

---

## 🧪 Practical Approach

For every concept, the goal is to understand:

1. What is it?
2. Why do we need it?
3. How does it work?
4. Python implementation
5. From-scratch implementation
6. Real-world example
7. Limitations
8. Practical experiment

---

## 🛠️ Tools

The following Python libraries will be used throughout this section:

- Python
- NLTK
- spaCy
- NumPy
- Pandas
- Regular Expressions

---
---

## 🚀 Next Level

After completing Level 0, we move to:

**Level 1 — Word Embeddings**

```text
BoW / TF-IDF
      ↓
Limitations of Sparse Representations
      ↓
Dense Vectors
      ↓
Word Embeddings
      ↓
Word2Vec
      ↓
CBOW / Skip-gram
      ↓
Negative Sampling
      ↓
Word Similarity
      ↓
Word Analogies
```
