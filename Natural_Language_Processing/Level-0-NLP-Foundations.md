NLP Foundations — Level 0

Practical NLP learning notes from scratch, focused on understanding what happens under the hood before using libraries.

1. What is NLP?

NLP (Natural Language Processing) is a field of AI that combines:

Artificial Intelligence

Machine Learning

Linguistics

to enable computers to process, analyze, understand, and generate human language.

Examples:

Chatbots

Sentiment analysis

Machine translation

Search engines

Spam detection

Text classification

Question answering

Large Language Models (LLMs)

Basic NLP pipeline

Human Language
      ↓
Text / Speech
      ↓
Representation
      ↓
Algorithm / Model
      ↓
Prediction / Generation

The fundamental challenge is that human language is not simple.

Language can be:

Ambiguous

Context-dependent

Flexible

Non-literal

Incomplete

Noisy

Example

"I went to the bank."

"Bank" could mean:

A financial institution

The side of a river

The surrounding context is required to determine the meaning.

2. Basic Language Concepts

Before working with NLP algorithms, it is important to understand the basic units of language.

Character

A single symbol.

c
a
t

The word cat contains three characters.

Word

A linguistic unit such as:

cat
Python
running

Token

A token is a unit selected by a tokenizer.

A token does not necessarily have to be a complete word.

Depending on the tokenizer, a token can be:

A word

A subword

A character

A punctuation mark

A byte-level unit

Example:

"playing"

could potentially be represented as:

["playing"]

or using a subword tokenizer:

["play", "ing"]

Important distinction

TOKEN
  ↓
The actual unit selected by the tokenizer

TOKEN ID
  ↓
An integer identifier assigned to that token

EMBEDDING
  ↓
A numerical vector representing a token/context

A token ID is not a representation of meaning. It is simply an identifier in a vocabulary.

3. Sentence

A sentence is a sequence of words/tokens expressing a complete thought.

Example:

I love NLP.

4. Document

A document is a larger text unit.

Example:

An article
A book
An email
A research paper

5. Corpus

A corpus is a collection of documents used for NLP analysis or model training.

Example:

D1 = "I love NLP"
D2 = "I love Python"
D3 = "Python is powerful"

Together, these documents form a small corpus.

6. Vocabulary

The vocabulary is the collection of unique tokens/words being considered.

For:

D1 = "I love NLP"
D2 = "I love Python"
D3 = "Python is powerful"

One possible vocabulary is:

["I", "love", "NLP", "Python", "is", "powerful"]

7. Context

Context is the surrounding information that helps determine meaning.

Example:

I deposited money at the bank.

Here, "bank" most likely refers to a financial institution.

But:

I sat near the bank of the river.

Here, "bank" refers to the side of a river.

NLP systems increasingly rely on context to understand language.

8. Linguistics and NLP

Linguistics is the scientific study of language.

Important linguistic areas for NLP include:

Morphology

Study of the internal structure and formation of words.

Example:

un + happy

or:

play + ing

Syntax

Study of sentence structure and how words are arranged.

Example:

The cat eats fish.

Changing the arrangement can change the grammatical structure and potentially the meaning.

Semantics

Study of meaning.

Example:

The cat is sleeping.

Semantics concerns what the sentence means.

Pragmatics

Study of meaning in context, including speaker intention.

Example:

"Can you open the window?"

Literally, this is a question about ability.

In context, it is usually a request to open the window.

9. Tokenization

Tokenization converts text into smaller units called tokens.

Text
 ↓
Tokenizer
 ↓
Tokens

Example:

"I love NLP"

could become:

["I", "love", "NLP"]

Why tokenization matters

Machine learning models cannot directly process raw human language.

We need to convert:

Text
 ↓
Tokens
 ↓
Token IDs
 ↓
Numerical representations
 ↓
Model

10. Types of Tokenization

Character-level

"cat"
↓
["c", "a", "t"]

Advantage

Small vocabulary.

Disadvantage

Very long sequences and limited direct word-level information.

Word-level

"I love NLP"
↓
["I", "love", "NLP"]

Advantage

Simple and intuitive.

Disadvantage

Large vocabulary and Out-of-Vocabulary (OOV) problems.

For example:

play
playing
played
player
playful

A word-level tokenizer may need to treat every form as a separate vocabulary item.

Subword-level

A word can be split into meaningful/reusable pieces.

For example:

playing
↓
play + ing

The exact split depends on the tokenizer.

Subword tokenization helps reduce vocabulary size and handle previously unseen word forms.

Byte-level tokenization

Some modern tokenizers operate around byte representations, allowing them to handle a broad range of text.

Modern Transformer/LLM systems commonly use subword- or byte-oriented tokenization strategies rather than simple whitespace word tokenization.

11. Simple Tokenizer From Scratch

A basic tokenizer can start with whitespace splitting:

text = "I love NLP"

tokens = text.split()

print(tokens)

Output:

['I', 'love', 'NLP']

But this is simplistic.

Consider:

"Hello, world!"

Using:

text.split()

we get:

["Hello,", "world!"]

Punctuation is attached to the words.

A basic regex tokenizer can separate words and punctuation:

import re

text = "Hello, world!"

tokens = re.findall(r"\w+|[^\w\s]", text)

print(tokens)

Possible output:

['Hello', ',', 'world', '!']

This is useful for understanding the idea, but production tokenizers use much more sophisticated algorithms.

12. Token → Token ID → Embedding

A simplified NLP pipeline looks like:

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
Dense numerical vectors
      ↓
Model

Important

The IDs:

12, 45, 91

do not mean:

12 = less meaningful
45 = more meaningful
91 = most meaningful

They are simply vocabulary identifiers.

13. Text Preprocessing

After tokenization, text may be normalized or cleaned depending on the task.

Common preprocessing operations include:

Normalization

Lowercasing

Punctuation handling

Stopword handling

Stemming

Lemmatization

Regular expressions

14. Normalization

Normalization converts different textual forms into a more consistent representation.

Example:

NLP
nlp
Nlp

A simple normalization could be:

text = text.lower()

Result:

nlp

Important warning

Normalization can remove useful information.

For example:

US
us

are different meanings.

Therefore:

Preprocessing should depend on the task.

Do not blindly clean everything.

15. Punctuation

Punctuation can sometimes be removed:

"Hello!"
↓
"Hello"

But punctuation can contain useful information.

Examples:

Really?
Really!
WHAT?!

The punctuation may carry information about:

Questions

Emotion

Emphasis

Tone

Therefore, punctuation removal is also task-dependent.

16. Stopwords

Stopwords are very common words that traditionally may be removed in some NLP pipelines.

Examples:

the
is
a
an
of
to

Example:

"The cat is on the table."

A traditional preprocessing pipeline might remove:

the
is
on
the

Important warning

Stopword removal can destroy meaning.

Example:

I like this.
I don't like this.

Removing don't carelessly can dramatically change the meaning.

Therefore:

Never automatically remove stopwords without considering the task.

17. Stemming

Stemming tries to reduce words to a root-like form by applying simple rules.

Example:

studies
studying
studied

A stemmer may produce something like:

studi

The result does not necessarily have to be a real English word.

Key idea

Stemming
= mechanical word chopping

It is generally faster but less linguistically precise.

18. Lemmatization

Lemmatization attempts to convert words into their proper dictionary/base form.

Examples:

studies → study
running → run
ran → run

Key idea

Lemmatization
= linguistically informed normalization

It can require information about:

Vocabulary

Morphology

Part of speech

Context

19. Stemming vs Lemmatization

Feature

Stemming

Lemmatization

Approach

Mechanical rules

Linguistic analysis

Speed

Usually faster

Usually slower

Output

May not be a real word

Usually a valid lemma

Example

studies → studi

studies → study

Precision

Lower

Higher

20. Regular Expressions (Regex)

Regex is a pattern-matching technique.

It is useful for detecting structured patterns such as:

Emails

Phone numbers

URLs

Dates

Numbers

Specific text patterns

Example:

import re

text = "Contact me at test@example.com"

pattern = r"\w+@\w+\.\w+"

result = re.findall(pattern, text)

print(result)

Possible output:

['test@example.com']

Important

Regex performs pattern matching.

It does not inherently understand the semantic meaning of language.

21. Bag of Words (BoW)

After tokenization, we need a way to convert text into numbers.

One of the simplest methods is:

Bag of Words

BoW represents a document based on the words it contains.

It ignores word order and focuses mainly on word presence/frequency.

22. Example of BoW

Consider:

D1 = "I love NLP"
D2 = "I love Python"
D3 = "Python is powerful"

Vocabulary:

["I", "love", "NLP", "Python", "is", "powerful"]

Then:

D1 → [1, 1, 1, 0, 0, 0]

D2 → [1, 1, 0, 1, 0, 0]

D3 → [0, 0, 0, 1, 1, 1]

Each position corresponds to a vocabulary word.

23. Binary BoW

Binary BoW only records whether a word exists.

0 = absent
1 = present

Suppose:

D1 = "cat eats fish"
D2 = "cat eats fish fish"

Vocabulary:

["cat", "eats", "fish"]

Binary representation:

D1 → [1, 1, 1]
D2 → [1, 1, 1]

Even though fish occurs twice in D2, the binary representation remains 1.

Rule

count = 0  → 0
count = 1  → 1
count = 2  → 1
count = 10 → 1

24. Count BoW

Count BoW records the actual frequency.

Using:

D1 = "cat eats fish"
D2 = "cat eats fish fish"

Vocabulary:

["cat", "eats", "fish"]

We get:

D1 → [1, 1, 1]

D2 → [1, 1, 2]

Difference

Binary BoW
"What words are present?"

Count BoW
"How many times does each word appear?"

25. BoW From Scratch

Example:

documents = [
    "I love NLP",
    "I love Python",
    "Python is powerful"
]

vocabulary = ["I", "love", "NLP", "Python", "is", "powerful"]

for document in documents:
    words = document.split()

    vector = []

    for word in vocabulary:
        vector.append(words.count(word))

    print(vector)

This produces count-based BoW vectors.

26. Limitations of BoW

BoW is simple and useful, but it has major limitations.

1. No word order

Consider:

The dog bit the man.

and:

The man bit the dog.

The words are the same, but the meanings are different.

BoW can produce the same/similar representation because it largely ignores order.

2. No real semantic understanding

BoW does not inherently know that:

king
queen

are semantically related.

3. High dimensionality

If a corpus contains 100,000 unique words, each document can potentially become a vector with 100,000 dimensions.

4. Sparse vectors

Most documents use only a small fraction of the entire vocabulary.

Therefore, many values are zero.

Example:

[0, 0, 0, 1, 0, 0, 0, 0, 1, 0, ...]

This is called a sparse vector.

27. TF-IDF

BoW treats word counts without considering how common a word is across the entire corpus.

TF-IDF improves this by assigning weights based on:

How frequently a word occurs in a document

How common the word is across documents

TF-IDF means:

Term Frequency × Inverse Document Frequency

TF-IDF = TF × IDF

28. Term Frequency (TF)

TF measures how frequently a term appears in a document.

A simple formula is:

[
TF(t,d)=
\frac{\text{number of occurrences of }t\text{ in }d}
{\text{total number of terms in }d}
]

Example:

D1 = "cat cat dog"

Total words = 3.

For cat:

[
TF(cat,D1)=\frac{2}{3}
]

For dog:

[
TF(dog,D1)=\frac{1}{3}
]

29. Document Frequency (DF)

DF measures how many documents contain a particular word.

Example:

D1 = "cat cat dog"
D2 = "cat dog fish"
D3 = "dog fish"

Then:

cat  → D1, D2 → DF = 2

dog  → D1, D2, D3 → DF = 3

fish → D2, D3 → DF = 2

Important:

DF counts documents, not total word occurrences.

30. Inverse Document Frequency (IDF)

A basic IDF formula is:

[
IDF(t)=\log\left(\frac{N}{DF(t)}\right)
]

where:

N = total number of documents

DF(t) = number of documents containing the term

For:

N = 3

and:

DF(cat) = 2

we get:

[
IDF(cat)=\log(3/2)
]

approximately:

0.405

For dog:

DF(dog) = 3

so:

[
IDF(dog)=\log(3/3)=0
]

Important intuition

If a word appears in every document:

IDF = 0

It provides little information for distinguishing documents.

31. TF-IDF Calculation

The basic formula is:

[
TFIDF(t,d)=TF(t,d)\times IDF(t)
]

Example:

D1 = "cat cat dog"

For cat:

TF = 2/3
IDF = log(3/2)

Therefore:

\frac{2}{3}\times\log(3/2)
]

Using natural logarithm:

≈ 0.2703

For dog:

TF = 1/3
IDF = 0

Therefore:

TF-IDF = 0

32. TF-IDF Intuition

TF-IDF asks:

"How important is this word to this document compared with the whole collection?"

Generally:

Frequent in this document
        +
Rare across the corpus
        ↓
Higher TF-IDF

While:

Common across many documents
        ↓
Lower IDF
        ↓
Lower TF-IDF

Example:

Word

Frequency in document

Common across documents

General importance

the

High

Very high

Low

cat

High

Medium

Medium/High

quantum

Low

Very low

High

33. TF-IDF From Scratch

Example corpus:

import math

documents = [
    "cat cat dog",
    "cat dog fish",
    "dog fish"
]

vocabulary = ["cat", "dog", "fish"]

N = len(documents)

# TF
tf_values = []

for document in documents:
    words = document.split()
    total_words = len(words)

    tf = {}

    for word in vocabulary:
        count = words.count(word)
        tf[word] = count / total_words

    tf_values.append(tf)

# DF
df = {}

for word in vocabulary:
    count = 0

    for document in documents:
        words = document.split()

        if word in words:
            count += 1

    df[word] = count

# IDF
idf = {}

for word in vocabulary:
    idf[word] = math.log(N / df[word])

# TF-IDF
tfidf = []

for tf in tf_values:
    document_tfidf = {}

    for word in vocabulary:
        document_tfidf[word] = tf[word] * idf[word]

    tfidf.append(document_tfidf)

for i, document_tfidf in enumerate(tfidf):
    print(f"D{i+1}:", document_tfidf)

This implementation demonstrates the complete conceptual pipeline:

Documents
   ↓
TF
   ↓
DF
   ↓
IDF
   ↓
TF × IDF
   ↓
TF-IDF

34. Why TF-IDF Is Better Than Basic BoW

BoW:

Word → count/presence

TF-IDF:

Word → weighted importance

TF-IDF reduces the importance of words that appear throughout many documents.

However, TF-IDF still has important limitations.

35. Limitations of TF-IDF

TF-IDF still does not truly understand semantic meaning.

For example:

king
queen

are semantically related, but TF-IDF treats them as independent vocabulary dimensions.

It also has weak understanding of:

Word relationships

Context

Synonyms

Polysemy

Word order

Therefore, NLP representation methods evolved further.

36. The NLP Representation Evolution

The concepts learned so far form this progression:

Raw Text
   ↓
Tokenization
   ↓
Preprocessing
   ↓
Bag of Words
   ↓
TF-IDF
   ↓
Word Embeddings
   ↓
Contextual Embeddings
   ↓
Attention
   ↓
Transformers
   ↓
LLMs

37. Level 0 Completion Checklist

NLP Fundamentals

What is NLP?

NLP pipeline

Why language is difficult

Character

Word

Token

Token ID

Embedding

Sentence

Document

Corpus

Vocabulary

Context

Linguistics

Morphology

Syntax

Semantics

Pragmatics

Tokenization

What is tokenization?

Character-level tokenization

Word-level tokenization

Subword tokenization

Byte-level tokenization

Token vs Token ID vs Embedding

Basic tokenizer from scratch

Regex tokenizer

Preprocessing

Normalization

Lowercasing

Punctuation handling

Stopwords

Stemming

Lemmatization

Stemming vs Lemmatization

Regex

Task-dependent preprocessing

Text Representation

Bag of Words

Binary BoW

Count BoW

Vocabulary construction

Sparse vectors

BoW limitations

TF-IDF

TF

DF

IDF

TF-IDF

TF-IDF intuition

TF-IDF from scratch

TF-IDF limitations

38. What Comes After Level 0?

The next level starts with Word Embeddings.

The progression is:

Level 0 — NLP Foundations
        ↓
Level 1 — Word Embeddings
        ↓
Word2Vec
        ↓
CBOW
        ↓
Skip-gram
        ↓
Negative Sampling
        ↓
GloVe
        ↓
FastText
        ↓
Contextual Embeddings
        ↓
RNN / LSTM / GRU
        ↓
Attention
        ↓
Transformers
        ↓
BERT / GPT
        ↓
Modern LLMs
        ↓
RAG
        ↓
Agents
        ↓
MCP
        ↓
NLP/LLM Research

Next Topic

Word Embeddings

The central question:

How can we represent the meaning and relationships of words using dense numerical vectors?

This is where NLP moves beyond simple word counts and starts representing semantic relationships.
