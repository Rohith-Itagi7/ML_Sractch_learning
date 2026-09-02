# NLP Foundations — Level 0

Practical NLP learning notes from scratch, focused on understanding what happens under the hood before using libraries.

---

## 1. What is NLP?

NLP (Natural Language Processing) is a field of AI that combines:

- Artificial Intelligence
- Machine Learning
- Linguistics

to enable computers to process, analyze, understand, and generate human language.

### Examples

- Chatbots
- Sentiment analysis
- Machine translation
- Search engines
- Spam detection
- Text classification
- Question answering
- Large Language Models (LLMs)

### Basic NLP Pipeline

```text
Human Language
      ↓
Text / Speech
      ↓
Representation
      ↓
Algorithm / Model
      ↓
Prediction / Generation
```

The fundamental challenge is that human language is not simple.

Language can be:

- Ambiguous
- Context-dependent
- Flexible
- Non-literal
- Incomplete
- Noisy

### Example

> "I went to the bank."

"Bank" could mean:

- A financial institution
- The side of a river

The surrounding context is required to determine the meaning.

---

## 2. Basic Language Concepts

Before working with NLP algorithms, it is important to understand the basic units of language.

### Character

A single symbol.

```text
c
a
t
```

The word `cat` contains three characters.

### Word

A linguistic unit such as:

```text
cat
Python
running
```

### Token

A token is a unit selected by a tokenizer.

A token does not necessarily have to be a complete word.

Depending on the tokenizer, a token can be:

- A word
- A subword
- A character
- A punctuation mark
- A byte-level unit

### Example

```text
playing
```

could potentially be represented as:

```text
["playing"]
```

or using a subword tokenizer:

```text
["play", "ing"]
```

### Important Distinction

```text
TOKEN
  ↓
The actual unit selected by the tokenizer

TOKEN ID
  ↓
An integer identifier assigned to that token

EMBEDDING
  ↓
A numerical vector representing a token/context
```

> A token ID is not a representation of meaning. It is simply an identifier in a vocabulary.

---

## 3. Sentence

A sentence is a sequence of words/tokens expressing a complete thought.

### Example

```text
I love NLP.
```

---

## 4. Document

A document is a larger text unit.

Examples:

- An article
- A book
- An email
- A research paper

---

## 5. Corpus

A corpus is a collection of documents used for NLP analysis or model training.

### Example

```text
D1 = "I love NLP"
D2 = "I love Python"
D3 = "Python is powerful"
```

Together, these documents form a small corpus.

---

## 6. Vocabulary

The vocabulary is the collection of unique tokens/words being considered.

For:

```text
D1 = "I love NLP"
D2 = "I love Python"
D3 = "Python is powerful"
```

One possible vocabulary is:

```text
["I", "love", "NLP", "Python", "is", "powerful"]
```

---

## 7. Context

Context is the surrounding information that helps determine meaning.

### Example

```text
I deposited money at the bank.
```

Here, "bank" most likely refers to a financial institution.

But:

```text
I sat near the bank of the river.
```

Here, "bank" refers to the side of a river.

NLP systems increasingly rely on context to understand language.

---

## 8. Linguistics and NLP

Linguistics is the scientific study of language.

Important linguistic areas for NLP include:

### Morphology

Study of the internal structure and formation of words.

Example:

```text
un + happy
```

or:

```text
play + ing
```

### Syntax

Study of sentence structure and how words are arranged.

Example:

```text
The cat eats fish.
```

Changing the arrangement can change the grammatical structure and potentially the meaning.

### Semantics

Study of meaning.

Example:

```text
The cat is sleeping.
```

Semantics concerns what the sentence means.

### Pragmatics

Study of meaning in context, including speaker intention.

Example:

> "Can you open the window?"

Literally, this is a question about ability.

In context, it is usually a request to open the window.

---

## 9. Tokenization

Tokenization converts text into smaller units called tokens.

```text
Text
 ↓
Tokenizer
 ↓
Tokens
```

### Example

```text
"I love NLP"
```

could become:

```text
["I", "love", "NLP"]
```

### Why Tokenization Matters

Machine learning models cannot directly process raw human language.

We need to convert:

```text
Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Numerical Representations
 ↓
Model
```

---

## 10. Types of Tokenization

### Character-Level Tokenization

```text
"cat"
  ↓
["c", "a", "t"]
```

#### Advantage

- Small vocabulary

#### Disadvantage

- Very long sequences
- Limited direct word-level information

---

### Word-Level Tokenization

```text
"I love NLP"
  ↓
["I", "love", "NLP"]
```

#### Advantage

- Simple and intuitive

#### Disadvantage

- Large vocabulary
- Out-of-Vocabulary (OOV) problems

For example:

```text
play
playing
played
player
playful
```

A word-level tokenizer may need to treat every form as a separate vocabulary item.

---

### Subword-Level Tokenization

A word can be split into meaningful or reusable pieces.

For example:

```text
playing
   ↓
play + ing
```

The exact split depends on the tokenizer.

Subword tokenization helps reduce vocabulary size and handle previously unseen word forms.

---

### Byte-Level Tokenization

Some modern tokenizers operate around byte representations, allowing them to handle a broad range of text.

Modern Transformer/LLM systems commonly use subword- or byte-oriented tokenization strategies rather than simple whitespace word tokenization.

---

## 11. Simple Tokenizer From Scratch

A basic tokenizer can start with whitespace splitting:

```python
text = "I love NLP"

tokens = text.split()

print(tokens)
```

### Output

```text
['I', 'love', 'NLP']
```

But this is simplistic.

Consider:

```text
"Hello, world!"
```

Using:

```python
text.split()
```

we get:

```text
["Hello,", "world!"]
```

Punctuation is attached to the words.

A basic regex tokenizer can separate words and punctuation:

```python
import re

text = "Hello, world!"

tokens = re.findall(r"\w+|[^\w\s]", text)

print(tokens)
```

### Possible Output

```text
['Hello', ',', 'world', '!']
```

This is useful for understanding the idea, but production tokenizers use much more sophisticated algorithms.

---

## 12. Token → Token ID → Embedding

A simplified NLP pipeline looks like:

```text
"I love NLP"
      ↓
Tokenization
      ↓
["I", "love", "NLP"]
      ↓
Token IDs
      ↓
[12, 45, 91]
      ↓
Embeddings
      ↓
Dense Numerical Vectors
      ↓
Model
```

### Important

The IDs:

```text
12, 45, 91
```

do not mean:

```text
12 = less meaningful
45 = more meaningful
91 = most meaningful
```

They are simply vocabulary identifiers.

---

## 13. Text Preprocessing

After tokenization, text may be normalized or cleaned depending on the task.

Common preprocessing operations include:

- Normalization
- Lowercasing
- Punctuation handling
- Stopword handling
- Stemming
- Lemmatization
- Regular expressions

---

## 14. Normalization

Normalization converts different textual forms into a more consistent representation.

### Example

```text
NLP
nlp
Nlp
```

A simple normalization could be:

```python
text = text.lower()
```

### Result

```text
nlp
```

> **Important warning:** Normalization can remove useful information.

For example:

```text
US
us
```

can have different meanings.

Therefore:

> Preprocessing should depend on the task.

Do not blindly clean everything.

---

## 15. Punctuation

Punctuation can sometimes be removed:

```text
"Hello!"
     ↓
"Hello"
```

But punctuation can contain useful information.

### Examples

```text
Really?
Really!
WHAT?!
```

The punctuation may carry information about:

- Questions
- Emotion
- Emphasis
- Tone

Therefore, punctuation removal is also task-dependent.

---

## 16. Stopwords

Stopwords are very common words that traditionally may be removed in some NLP pipelines.

### Examples

```text
the
is
a
an
of
to
```

### Example

```text
"The cat is on the table."
```

A traditional preprocessing pipeline might remove:

```text
the
is
on
the
```

### Important Warning

Stopword removal can destroy meaning.

Example:

```text
I like this.
I don't like this.
```

Removing `don't` carelessly can dramatically change the meaning.

Therefore:

> Never automatically remove stopwords without considering the task.

---

## 17. Stemming

Stemming tries to reduce words to a root-like form by applying simple rules.

### Example

```text
studies
studying
studied
```

A stemmer may produce something like:

```text
studi
```

The result does not necessarily have to be a real English word.

### Key Idea

```text
Stemming
    ↓
Mechanical word chopping
```

It is generally faster but less linguistically precise.

---

## 18. Lemmatization

Lemmatization attempts to convert words into their proper dictionary/base form.

### Examples

```text
studies → study
running → run
ran → run
```

### Key Idea

```text
Lemmatization
      ↓
Linguistically informed normalization
```

It can require information about:

- Vocabulary
- Morphology
- Part of speech
- Context

---

## 19. Stemming vs Lemmatization

| Feature | Stemming | Lemmatization |
|---|---|---|
| Approach | Mechanical rules | Linguistic analysis |
| Speed | Usually faster | Usually slower |
| Output | May not be a real word | Usually a valid lemma |
| Example | `studies → studi` | `studies → study` |
| Precision | Lower | Higher |

---

## 20. Regular Expressions (Regex)

Regex is a pattern-matching technique.

It is useful for detecting structured patterns such as:

- Emails
- Phone numbers
- URLs
- Dates
- Numbers
- Specific text patterns

### Example

```python
import re

text = "Contact me at test@example.com"

pattern = r"\w+@\w+\.\w+"

result = re.findall(pattern, text)

print(result)
```

### Possible Output

```text
['test@example.com']
```

### Important

Regex performs pattern matching.

It does not inherently understand the semantic meaning of language.

---

# Level 0 — Key Takeaways

By completing this level, you should understand:

- What NLP is
- Why human language is difficult for computers
- Character, word, token, sentence, document, and corpus
- Vocabulary and context
- Basic linguistic concepts
- Tokenization
- Character, word, subword, and byte-level tokenization
- Token IDs vs embeddings
- Text preprocessing
- Normalization
- Punctuation handling
- Stopwords
- Stemming
- Lemmatization
- Regex

---

# Next Level

## Level 1 — Word Embeddings

After understanding these foundations, the next step is to understand how words and tokens can be represented as numerical vectors.

```text
BoW / TF-IDF
      ↓
Why Sparse Representations Are Limited
      ↓
Dense Vectors
      ↓
Word Embeddings
      ↓
Word2Vec
      ↓
CBOW
      ↓
Skip-gram
      ↓
Negative Sampling
      ↓
Training Word2Vec From Scratch
      ↓
Word Similarity
      ↓
Word Analogies
```
