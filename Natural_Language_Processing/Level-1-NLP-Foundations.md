
# 🧠 NLP — Level 1: Word Representations & Statistical Language Models

> **Goal:** Understand how NLP evolved from simple word-count methods (BoW) to dense word embeddings, contextual representations, and statistical language models.

---

# 1. Bag of Words (BoW)

## What is Bag of Words?

**Bag of Words (BoW)** is one of the simplest text representation techniques in NLP.

> **Definition:** It represents a document using the occurrence (or count) of words while **ignoring their original order**.

### Why is it called a "Bag"?

Imagine putting every word into a bag. The bag remembers **which words exist**, but **not the sequence** they appeared in.

### Example Corpus

```text
D1 = "I love NLP"
D2 = "I love Python"
D3 = "Python is powerful"
```

### Step 1: Build the Vocabulary

```text
["I", "love", "NLP", "Python", "is", "powerful"]
```

Each vocabulary word becomes one feature (column).

### Step 2: Convert Documents into Vectors

| Word | I | love | NLP | Python | is | powerful |
|------|---|------|-----|--------|----|----------|
| D1 | 1 | 1 | 1 | 0 | 0 | 0 |
| D2 | 1 | 1 | 0 | 1 | 0 | 0 |
| D3 | 0 | 0 | 0 | 1 | 1 | 1 |

### Mental Model

```text
Raw Text
    │
    ▼
Build Vocabulary
    │
    ▼
Count Words
    │
    ▼
Numerical Vector
```

### Where is BoW used?

- Spam detection
- Basic sentiment analysis
- Text classification
- Keyword-based retrieval

---

# 2. Binary Bag of Words

## What is Binary BoW?

Binary BoW records **only whether a word is present or absent**.

> **Question answered:** *Does this word occur?*

### Rule

| Count | Binary Value |
|--------|-------------|
| 0 | 0 |
| 1 | 1 |
| 2 | 1 |
| 10 | 1 |

### Example

```text
Document:
"I love love NLP"
```

Vocabulary:

```text
[I, love, NLP]
```

Binary Representation:

```text
[1, 1, 1]
```

Even though **love** appears twice, it becomes **1**.

### Use Case

Useful when **presence matters more than frequency**.

---

# 3. Count Bag of Words

## What is Count BoW?

Instead of presence, Count BoW stores **how many times each word appears**.

### Example

```text
D1 = "cat eats fish"
D2 = "cat eats fish fish"
```

Vocabulary:

```text
["cat", "eats", "fish"]
```

| Document | cat | eats | fish |
|----------|-----|------|------|
| D1 | 1 | 1 | 1 |
| D2 | 1 | 1 | 2 |

### Difference from Binary BoW

| Binary | Count |
|---------|------|
| Presence | Frequency |
| 0 / 1 only | Any count |
| Simpler | More informative |

---

# 4. Manual BoW Implementation (Python)

Instead of using scikit-learn, let's build BoW manually.

```python
documents = [
    "I love NLP",
    "I love Python",
    "Python is powerful"
]

vocab = ["I", "love", "NLP", "Python", "is", "powerful"]

for document in documents:
    words = document.split()
    vector = []

    for word in vocab:
        vector.append(words.count(word))

    print(vector)
```

### Output

```text
[1, 1, 1, 0, 0, 0]
[1, 1, 0, 1, 0, 0]
[0, 0, 0, 1, 1, 1]
```

### What happens internally?

```text
Document
   │
Split into Words
   │
Compare with Vocabulary
   │
Count Each Word
   │
Create Vector
```

---

# 5. TF–IDF (Term Frequency–Inverse Document Frequency)

## Why do we need TF-IDF?

BoW treats every word equally.

But words like:

```text
the
is
and
```

appear in almost every document.

> **TF-IDF gives higher importance to words that are frequent in one document but rare across the corpus.**

### Formula

<math block value="TF\\text{-}IDF = TF \\times IDF"/>

---

# 6. Term Frequency (TF)

## Definition

Term Frequency measures **how often a word appears inside a document**.

### Formula

<math block value="TF(t,d)=\\frac{\\text{Count of term }t}{\\text{Total words in document}}"/>

### Example

Document:

```text
cat cat dog
```

Total words = **3**

| Word | Count | TF |
|------|------|----|
| cat | 2 | 2/3 |
| dog | 1 | 1/3 |

### Intuition

Higher frequency → Higher TF.

---

# 7. Document Frequency (DF)

## Definition

Document Frequency counts **how many documents contain a word**.

> **Important:** DF is **not** the total number of occurrences.

### Example

```text
D1 = cat cat dog
D2 = cat dog fish
D3 = dog fish
```

| Word | DF |
|------|----|
| cat | 2 |
| dog | 3 |
| fish | 2 |

Dog appears in **all documents**, so DF = 3.

---

# 8. Inverse Document Frequency (IDF)

## Why IDF?

Words appearing everywhere are less informative.

### Formula

<math block value="IDF(t)=\\log\\left(\\frac{N}{DF(t)}\\right)"/>

Where:

- **N** = Total documents
- **DF** = Documents containing the word

### Example

```text
N = 3
```

| Word | DF | IDF |
|------|----|------|
| cat | 2 | log(3/2) |
| dog | 3 | 0 |
| fish | 2 | log(3/2) |

### Important Relationship

```text
Higher DF
    │
    ▼
Lower IDF

Lower DF
    │
    ▼
Higher IDF
```

---

# 9. TF–IDF Example

Corpus:

```text
D1 = cat cat dog
D2 = cat dog fish
D3 = dog fish
```

### Step 1: TF

For D1:

| Word | TF |
|------|----|
| cat | 2/3 |
| dog | 1/3 |

### Step 2: IDF

```text
cat  → log(3/2)
dog  → 0
```

### Step 3: Multiply

<math block value="TF\\text{-}IDF = TF \\times IDF"/>

Since dog has IDF = 0:

```text
TF-IDF(dog) = 0
```

Meaning dog carries **little distinguishing information**.

---

# 10. Limitations of BoW & TF-IDF

Although useful, these methods have major weaknesses.

## 1. Sparse Vectors

Large vocabulary → Huge vectors.

```text
[0,0,0,0,1,0,0,0,...]
```

Most values are zero.

## 2. No Semantic Understanding

BoW cannot understand:

```text
cat ≈ dog
```

## 3. Synonyms

```text
car
automobile
vehicle
```

They become completely different dimensions.

## 4. Word Order is Lost

Both produce identical counts:

```text
The dog bit the man.
The man bit the dog.
```

> **Key Limitation:** Same vector, different meaning.

---

# 11. Word Embeddings

## Why do we need Embeddings?

Sparse vectors are inefficient.

Word embeddings represent words using **dense numerical vectors**.

### Example

```text
king
  │
  ▼
[0.72, -0.31, 0.48, 0.91]
```

### Incorrect Understanding

```text
k → 0.72
i → -0.31
```

❌ Wrong.

### Correct Understanding

The **entire word** receives one vector.

```text
king
   │
   ▼
Dense Vector
```

---

# 12. Dense vs Sparse Representation

## Sparse

```text
[0,0,0,1,0,0,0,1]
```

Characteristics:

- Mostly zeros
- Large memory
- Weak semantics

## Dense

```text
[0.42,-0.17,0.83,0.21,-0.55]
```

Characteristics:

- Mostly non-zero
- Compact
- Semantic meaning

| Sparse | Dense |
|---------|------|
| BoW | Embeddings |
| Large vectors | Small vectors |
| No semantics | Rich semantics |

---

# 13. Distributional Hypothesis

## The Most Important NLP Idea

> **Words appearing in similar contexts tend to have similar meanings.**

### Example

```text
The cat eats fish.
The dog eats fish.
```

Both occur in:

```text
The ___ eats fish
```

Therefore:

```text
cat ≈ dog
```

The model learns similar representations.

### Mental Model

```text
Similar Context
      │
      ▼
Similar Meaning
      │
      ▼
Similar Vector
```

---

# 14. Vector Space

Embeddings create a **vector space**.

Conceptually:

```text
          dog
         /
        /
      cat


                    banana
```

Nearby vectors → Similar meaning.

Far apart → Different meaning.

> **Important:** Individual numbers are less meaningful than relationships between vectors.

---

# 15. Cosine Similarity

## Measuring Similarity

Cosine similarity compares the **direction** of vectors.

### Formula

<math block value="\\cos(\\theta)=\\frac{A\\cdot B}{||A||||B||}"/>

### Interpretation

```text
Same Direction
      │
      ▼
High Similarity

Different Direction
      │
      ▼
Low Similarity
```

Example:

| Pair | Similarity |
|------|------------|
| cat – dog | High |
| cat – banana | Lower |
| cat – car | Much lower |

---

# 16. Embedding Matrix

Suppose:

```text
Vocabulary Size = 5
Embedding Dimension = 4
```

### Matrix Shape

```text
5 × 4
```

| Word | Embedding |
|------|-----------|
| the | [0.2,0.7,-0.1,0.4] |
| cat | [0.4,0.8,-0.2,0.1] |
| dog | [0.3,0.7,-0.1,0.2] |
| fish | [0.1,0.2,0.7,0.5] |
| milk | [0.5,0.3,0.2,0.8] |

### Shape Rule

```text
Rows = Vocabulary Size
Columns = Embedding Dimension
```

---

# 17. How Embeddings Are Learned

The numbers are **not manually assigned**.

### Training Pipeline

```text
Random Initialization
        │
        ▼
Training Data
        │
        ▼
Prediction
        │
        ▼
Loss Calculation
        │
        ▼
Backpropagation
        │
        ▼
Parameter Update
        │
        ▼
Learned Embeddings
```

> Embeddings are **learned parameters**.

---

# 18. Word2Vec

## What is Word2Vec?

Word2Vec learns word representations using **surrounding context**.

### Two Architectures

```text
Word2Vec
   │
   ├── CBOW
   │
   └── Skip-gram
```

---

# 19. CBOW (Continuous Bag of Words)

## Idea

Predict the **target word** from surrounding words.

```text
Context
   │
   ▼
Target Word
```

### Example

Sentence:

```text
The cat eats fish
```

Context:

```text
the
eats
```

Target:

```text
cat
```

Pipeline:

```text
the + eats
      │
      ▼
    Model
      │
      ▼
     cat
```

---

# 20. Skip-gram

Skip-gram is the opposite of CBOW.

> **Predict surrounding words from the target word.**

### Example

Target:

```text
cat
```

Context:

```text
the
eats
```

Training samples:

```text
cat → the
cat → eats
```

### Difference

| CBOW | Skip-gram |
|------|-----------|
| Context → Target | Target → Context |
| Faster | Better for rare words |

---

# 21. Word2Vec Training Flow

A simplified Skip-gram pipeline:

```text
Sentence
   │
   ▼
Tokenization
   │
   ▼
Token IDs
   │
   ▼
Choose Target
   │
   ▼
Choose Context
   │
   ▼
Embedding Lookup
   │
   ▼
Hidden Layer
   │
   ▼
Softmax
   │
   ▼
Predicted Context
   │
   ▼
Loss
   │
   ▼
Backpropagation
```

---

# 22. One-Hot Encoding

Vocabulary:

```text
["the","cat","eats","fish"]
```

| Word | One-Hot |
|------|---------|
| the | [1,0,0,0] |
| cat | [0,1,0,0] |
| eats | [0,0,1,0] |
| fish | [0,0,0,1] |

### Conversion

```text
Word
  │
  ▼
One-Hot
  │
  ▼
Embedding Matrix
  │
  ▼
Dense Vector
```

In practice, embedding lookup is used instead of explicit matrix multiplication.

---

# 23. N-grams

## What is an N-gram?

An **N-gram** is a sequence of **N consecutive tokens**.

Sentence:

```text
I love natural language processing
```

---

# 24. Types of N-grams

### Unigram (1)

```text
I
love
natural
language
processing
```

### Bigram (2)

```text
I love
love natural
natural language
language processing
```

### Trigram (3)

```text
I love natural
love natural language
natural language processing
```

### 4-gram

```text
I love natural language
love natural language processing
```

---

# 25. Number of N-grams

### Formula

<math block value="\\text{Number of N-grams}=n-N+1"/>

Where:

- **n** = Total tokens
- **N** = Gram size

### Example

```text
n = 5
N = 2
```

Result:

```text
5 - 2 + 1 = 4
```

So there are **4 bigrams**.

---

# 26. N-gram Language Models

N-grams can also estimate **probabilities of word sequences**.

### Unigram

<math block value="P(w)"/>

### Bigram

<math block value="P(w_n\\mid w_{n-1})"/>

### Trigram

<math block value="P(w_n\\mid w_{n-2},w_{n-1})"/>

### Mental Model

```text
Previous Words
      │
      ▼
Predict Next Word
```

---

# 27. Bigram Probability

### Formula

<math block value="P(w_2\\mid w_1)=\\frac{Count(w_1,w_2)}{Count(w_1)}"/>

### Example

```text
Count(I love) = 3
Count(I) = 5
```

Therefore:

<math block value="P(love\\mid I)=\\frac{3}{5}=0.6"/>

Meaning there is a **60% probability** of seeing *love* after *I* in the training corpus.

---

# 28. Trigram Probability

### Formula

<math block value="P(w_3\\mid w_1,w_2)=\\frac{Count(w_1,w_2,w_3)}{Count(w_1,w_2)}"/>

Instead of one previous word, trigram uses **two previous words**.

Example:

```text
I love NLP
```

Prediction:

```text
P(NLP | I love)
```

---

# 29. Markov Assumption

Ideally, language depends on **all previous words**.

```text
P(word | every previous word)
```

But this is computationally expensive.

### Approximation

| Model | Context Used |
|--------|--------------|
| Bigram | Previous 1 word |
| Trigram | Previous 2 words |

### Mental Model

```text
Full History
      │
      ▼
Recent N-1 Words
      │
      ▼
Predict Next Word
```

This approximation is known as the **Markov Assumption**.

---

# 30. Zero-Frequency Problem

Suppose training data contains:

```text
I love NLP
I love Python
```

But never contains:

```text
love machine
```

Then:

```text
Count(love,machine)=0
```

Therefore:

```text
P(machine | love)=0
```

### Why is this bad?

> An unseen sequence is **not necessarily impossible**.

This motivates smoothing.

---

# 31. Smoothing

## What is Smoothing?

Smoothing assigns a **small non-zero probability** to unseen sequences.

### Common Methods

- Laplace (Add-One)
- Add-k
- Good-Turing
- Backoff
- Interpolation
- Kneser-Ney

The first method to learn is **Laplace Smoothing**.

---

# 32. Laplace (Add-One) Smoothing

### Formula

<math block value="P(w_2\\mid w_1)=\\frac{Count(w_1,w_2)+1}{Count(w_1)+V}"/>

Where:

- **V** = Vocabulary size

### Example

```text
Count(study,Python)=0
Count(study)=1
V=5
```

Without smoothing:

```text
0
```

With smoothing:

<math block value="\\frac{0+1}{1+5}=\\frac{1}{6}\\approx0.167"/>

Now the unseen sequence has a valid probability.

---

# 33. Limitations of N-gram Models

## Major Problems

### 1. Limited Context

A trigram remembers only two previous words.

### 2. Data Sparsity

Most possible word combinations never appear.

### 3. Weak Semantics

Statistical counts do not truly understand meaning.

### 4. Computational Cost

Larger N creates exponentially more combinations.

### 5. Poor Generalization

Unseen sequences remain difficult to model.

> These limitations motivated **neural language models**.

---

# 34. GloVe (Global Vectors)

## What is GloVe?

GloVe learns embeddings from **global co-occurrence statistics**.

### Word2Vec vs GloVe

| Word2Vec | GloVe |
|----------|--------|
| Predict context | Use co-occurrence matrix |
| Local context | Global statistics |

### Co-occurrence Example

```text
The cat drinks milk.
The cat eats fish.
The dog drinks milk.
The dog eats meat.
```

Conceptual Matrix:

| | cat | dog | milk | fish |
|---|---|---|---|---|
| cat | 0 | 0 | 5 | 3 |
| dog | 0 | 0 | 6 | 1 |
| milk | 5 | 6 | 0 | 0 |
| fish | 3 | 1 | 0 | 0 |

### Learning Idea

```text
Entire Corpus
      │
      ▼
Co-occurrence Matrix
      │
      ▼
Training
      │
      ▼
Word Vectors
```

---

# 35. FastText

## What is FastText?

FastText improves embeddings using **character n-grams**.

### Traditional Embedding

```text
playing
    │
    ▼
One Word Vector
```

### FastText

```text
playing
    │
    ▼
Character N-grams
    │
    ▼
Subword Vectors
    │
    ▼
Combined Embedding
```

### Why is it Useful?

Words sharing subwords:

```text
play
played
playing
player
playful
```

can share useful information.

### Benefits

- Rare words
- Unseen words
- Morphology
- Word variations
- Some spelling errors

---

# 36. ELMo

## What is ELMo?

**ELMo (Embeddings from Language Models)** introduced **contextual embeddings**.

### Static vs Contextual

Static:

```text
bank
  │
  ▼
One Fixed Vector
```

Contextual:

```text
Money Context
      │
      ▼
Financial Vector

River Context
      │
      ▼
River Vector
```

The same word receives **different vectors depending on context**.

---

# 37. How ELMo Works

ELMo uses **Bidirectional LSTMs**.

```text
               Sentence
                  │
      ┌───────────┴───────────┐
      ▼                       ▼
Forward LSTM            Backward LSTM
      │                       │
      └───────────┬───────────┘
                  ▼
      Contextual Representation
```

Example:

Forward:

```text
The → cat → drinks → milk
```

Backward:

```text
milk → drinks → cat → The
```

Both directions contribute to the final embedding.

---

# 38. FastText vs BPE / WordPiece / SentencePiece

These are **not the same thing**.

| FastText | BPE / WordPiece / SentencePiece |
|----------|-------------------------------|
| Embedding method | Tokenization method |
| Uses character n-grams | Splits text into subwords |
| Produces embeddings | Produces tokens |

### Pipeline

```text
Text
  │
  ▼
Tokenizer
(BPE / WordPiece)
  │
  ▼
Token IDs
  │
  ▼
Embeddings
(FastText / Neural Models)
```

---

# 39. Evolution of NLP

The history of NLP is the evolution of **better language representations**.

```text
RAW TEXT
    │
    ▼
TOKENIZATION
    │
    ▼
BAG OF WORDS
    │
    ▼
TF-IDF
    │
    ▼
WORD2VEC
    │
    ▼
GLOVE
    │
    ▼
FASTTEXT
    │
    ▼
ELMO
    │
    ▼
RNN
    │
    ▼
LSTM / GRU
    │
    ▼
SEQ2SEQ
    │
    ▼
ATTENTION
    │
    ▼
TRANSFORMERS
    │
    ▼
BERT / GPT
    │
    ▼
MODERN LLMs
```

---

# 40. One-Sentence Summary of Each Technique

| Technique | Core Idea |
|-----------|-----------|
| BoW | Does this word occur? |
| TF-IDF | How important is this word? |
| Word2Vec | Learn from surrounding words |
| GloVe | Learn from global co-occurrence |
| FastText | Learn using subword information |
| ELMo | Context changes word meaning |
| RNN | Model sequential information |
| Attention | Focus on important tokens |
| Transformer | Self-attention models relationships |
| LLM | Large pretrained language understanding |

---

# 📚 Final Mental Model

```text
Raw Text
    │
    ▼
Tokenization
    │
    ▼
BoW
    │
    ▼
TF-IDF
    │
    ▼
Word2Vec
    │
    ▼
GloVe
    │
    ▼
FastText
    │
    ▼
ELMo
    │
    ▼
RNN / LSTM
    │
    ▼
Attention
    │
    ▼
Transformer
    │
    ▼
Large Language Models
```

---

# ✅ What You Should Understand

- [ ] Explain Bag of Words and build a vocabulary.
- [ ] Differentiate Binary BoW and Count BoW.
- [ ] Calculate TF, DF, IDF, and TF-IDF manually.
- [ ] Describe why sparse vectors are inefficient.
- [ ] Explain dense word embeddings.
- [ ] Understand the Distributional Hypothesis.
- [ ] Interpret cosine similarity conceptually.
- [ ] Explain the embedding matrix shape.
- [ ] Compare CBOW and Skip-gram.
- [ ] Build unigram, bigram, and trigram examples.
- [ ] Calculate bigram probability.
- [ ] Explain the Markov Assumption.
- [ ] Solve the zero-frequency problem using Laplace smoothing.
- [ ] Compare Word2Vec, GloVe, FastText, and ELMo.
- [ ] Distinguish static and contextual embeddings.
- [ ] Explain why modern NLP evolved toward Transformers.

---

# 🚀 What Comes Next?

**Level 2 — Sequence Models**

```text
Word Embeddings
       │
       ▼
Recurrent Neural Networks (RNN)
       │
       ▼
Hidden State
       │
       ▼
Backpropagation Through Time
       │
       ▼
Vanishing Gradient
       │
       ▼
LSTM
       │
       ▼
GRU
       │
       ▼
Seq2Seq
```

After mastering sequence models, you'll move to:

```text
Attention
     │
     ▼
Self-Attention
     │
     ▼
Transformers
     │
     ▼
BERT & GPT
```

---

# 🎯 Key Takeaway

> **The evolution of NLP is the evolution of language representation.**

```text
Words
  │
  ▼
Counts (BoW)
  │
  ▼
Weighted Counts (TF-IDF)
  │
  ▼
Dense Vectors (Word2Vec, GloVe)
  │
  ▼
Subword-Aware Vectors (FastText)
  │
  ▼
Contextual Vectors (ELMo)
  │
  ▼
Attention-Based Representations
  │
  ▼
Modern Large Language Models
```
