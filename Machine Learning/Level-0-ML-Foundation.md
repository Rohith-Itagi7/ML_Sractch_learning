# Machine Learning — Level 0: Foundations

> **Goal:** Build a strong mental model of Machine Learning before learning individual algorithms.

---

## 1. What is Machine Learning?

**Machine Learning (ML)** is a branch of Artificial Intelligence where computers learn patterns from data and use those patterns to make predictions or decisions.

### Traditional Programming

In traditional programming, the programmer explicitly defines the rules.

```text
Rules + Data → Output
```

Example:

```python
if marks >= 40:
    result = "Pass"
else:
    result = "Fail"
```

Here, the programmer has manually written the rule:

```text
marks >= 40 → Pass
marks < 40  → Fail
```

### Machine Learning

In Machine Learning, instead of manually writing every rule, we provide examples and allow an algorithm to learn the relationship.

```text
Data + Expected Output
          ↓
    Learning Algorithm
          ↓
        Model
```

Example:

```text
Past student data
       ↓
   ML Algorithm
       ↓
   Trained Model
       ↓
New student's marks
       ↓
Prediction: Pass / Fail
```

### Core Idea

> **Instead of explicitly programming every rule, we allow the model to learn patterns from examples.**

---

# 2. AI vs ML vs Deep Learning

These terms are related but they are **not identical**.

```text
Artificial Intelligence
│
├── Machine Learning
│   │
│   ├── Traditional ML
│   │   ├── Linear Regression
│   │   ├── Logistic Regression
│   │   ├── Decision Trees
│   │   ├── Random Forest
│   │   └── SVM
│   │
│   └── Deep Learning
│       ├── Neural Networks
│       ├── CNN
│       ├── RNN
│       └── Transformers
│
└── Other AI approaches
    ├── Search
    ├── Planning
    ├── Rules
    └── Knowledge-based systems
```

### Artificial Intelligence

**Artificial Intelligence (AI)** is the broad field of building systems capable of performing tasks that normally require intelligent behavior.

AI can involve:

- Reasoning
- Search
- Planning
- Decision-making
- Learning
- Perception
- Problem solving

### Machine Learning

**Machine Learning** is a way of building AI systems by allowing systems to learn patterns from data.

```text
Data
 ↓
Learning
 ↓
Patterns
 ↓
Model
 ↓
Prediction / Decision
```

### Deep Learning

**Deep Learning** is a subfield of Machine Learning that uses neural networks with multiple layers.

Examples:

- Neural Networks
- CNNs
- RNNs
- Transformers

### Simple Mental Model

```text
AI
└── ML
    └── Deep Learning
```

But remember:

> **Not every AI system uses Machine Learning.**

Classical AI approaches such as search, planning, rules, and knowledge-based systems can work without ML.

---

# 3. Why Do We Need Machine Learning?

Some problems are difficult to solve using manually written rules.

Consider **Spam Detection**.

### Traditional Approach

We could write rules such as:

```python
if "free money" in email:
    spam = True
```

But real spam emails can contain millions of variations.

For example:

```text
"Congratulations! You won money."

"You have been selected for a reward."

"Claim your prize now."

"Get rich quickly."
```

It is difficult to manually write rules for every possible variation.

### Machine Learning Approach

Instead:

```text
Thousands of Emails
        ↓
    ML Algorithm
        ↓
   Learn Patterns
        ↓
   Trained Model
        ↓
New Email
        ↓
Spam / Not Spam
```

The model can learn patterns from previous examples.

### When is ML Useful?

Machine Learning is especially useful when:

- Rules are difficult to write manually
- There are large amounts of data
- Patterns exist in the data
- The system needs to make predictions
- The problem changes over time
- The relationship between inputs and outputs is complex

---

# 4. What is Data?

**Data** is the information from which a Machine Learning model learns.

Example:

| Age | Experience | Salary |
|---:|---:|---:|
| 22 | 1 | 30000 |
| 25 | 3 | 45000 |
| 28 | 5 | 60000 |
| 32 | 8 | 90000 |

In this dataset:

```text
Age         → Feature
Experience  → Feature
Salary      → Target
```

The model can learn a relationship between:

```text
Age + Experience → Salary
```

---

# 5. Features and Target

These are two of the most important concepts in Machine Learning.

## Feature

A **feature** is an input variable used by the model to make a prediction.

Examples:

```text
Age
Experience
Education
Location
Bedrooms
Area
Temperature
```

For example, when predicting salary:

```text
Age
Experience
Education
Location
```

could be features.

---

## Target

The **target** is what we want the model to predict.

Example:

```text
Features                  Target

Age                       Salary
Experience       →        ↑
Education                 Prediction
Location
```

In Python:

```python
X = data[["age", "experience"]]
y = data["salary"]
```

Conventionally:

```text
X → Features / Inputs
y → Target / Output
```

### Important Mental Model

```text
X → What the model receives
y → What the model should learn to predict
```

---

# 6. Sample

A **sample** is one individual observation or row in a dataset.

Example:

| Age | Experience | Salary |
|---:|---:|---:|
| 22 | 1 | 30000 |

This entire row represents **one sample**.

If the dataset contains:

```text
1000 rows
```

then we generally have:

```text
1000 samples
```

Another way to think about it:

```text
One row = One sample
```

---

# 7. Dataset

A **dataset** is a collection of samples.

Example:

```text
Dataset
│
├── Sample 1
├── Sample 2
├── Sample 3
├── Sample 4
├── ...
└── Sample 1000
```

A dataset normally contains:

```text
Rows    → Samples
Columns → Features / Target
```

Example:

| Age | Experience | Salary |
|---:|---:|---:|
| 22 | 1 | 30000 |
| 25 | 3 | 45000 |
| 28 | 5 | 60000 |

Here:

```text
Rows:
3 samples

Columns:
2 features + 1 target
```

---

# 8. Types of Machine Learning

The three fundamental categories are:

```text
Machine Learning
│
├── Supervised Learning
├── Unsupervised Learning
└── Reinforcement Learning
```

Each type learns differently.

---

# 9. Supervised Learning

In **Supervised Learning**, the model learns from data where the correct answer is already known.

```text
Input + Correct Output
          ↓
      ML Algorithm
          ↓
         Model
```

Example:

```text
House Size → House Price
```

Training data:

| Size | Price |
|---:|---:|
| 1000 sq ft | ₹50L |
| 1500 sq ft | ₹75L |
| 2000 sq ft | ₹1Cr |

The model learns the relationship:

```text
House Size → House Price
```

After training, we can provide:

```text
2500 sq ft
```

and ask the model to predict:

```text
Predicted Price
```

### Two Major Types

```text
Supervised Learning
│
├── Regression
└── Classification
```

---

# 10. Regression

**Regression** predicts a continuous numerical value.

Examples:

```text
House Price
Salary
Temperature
Demand
Sales
Weight
Age
```

Example:

```text
House Size → Predicted Price

1000 sq ft → ₹50L
1500 sq ft → ₹75L
2000 sq ft → ₹1Cr
```

The output is a number.

### Common Regression Algorithms

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting

### Simple Rule

> **If the target is a continuous numerical value, it is usually a regression problem.**

---

# 11. Classification

**Classification** predicts a category or class.

Examples:

```text
Spam / Not Spam
Pass / Fail
Disease / No Disease
Cat / Dog
Fraud / Not Fraud
```

Example:

```text
Email
  ↓
Model
  ↓
Spam
```

The output belongs to a class.

### Common Classification Algorithms

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- SVM
- Naive Bayes

### Simple Rule

> **If the target represents a category/class, it is a classification problem.**

---

# 12. Regression vs Classification

| Regression | Classification |
|---|---|
| Predicts numerical value | Predicts category |
| Salary | Spam / Not Spam |
| House Price | Cat / Dog |
| Temperature | Pass / Fail |
| Demand | Fraud / Not Fraud |
| Sales | Disease / No Disease |

### Easy Mental Model

```text
Number → Regression

Category → Classification
```

---

# 13. Unsupervised Learning

In **Unsupervised Learning**, the dataset does **not have labeled answers**.

```text
Input Data
    ↓
ML Algorithm
    ↓
Find Hidden Patterns
```

The algorithm tries to discover structure in the data.

### Example

Suppose a company has customer data:

```text
Age
Income
Purchases
Frequency
```

But the company does not know the customer groups.

Machine Learning can discover groups:

```text
Customers
    ↓
 Clustering
    ↓
 ┌───────┬───────┬───────┐
 ↓       ↓       ↓
Group 1 Group 2 Group 3
```

### Common Unsupervised Learning Techniques

- Clustering
- Dimensionality Reduction
- Association Rule Learning

### Common Algorithms

- K-Means
- DBSCAN
- Hierarchical Clustering
- PCA

---

# 14. Reinforcement Learning

**Reinforcement Learning (RL)** is based on learning through interaction with an environment.

The basic idea is:

```text
Environment
     ↑
     │
   Action
     │
     ↓
   Agent
     ↑
   Reward
```

The agent:

1. Observes the environment
2. Takes an action
3. Receives a reward or penalty
4. Learns which actions are better

### Example — Game

```text
Game
 ↓
Agent chooses action
 ↓
Wins → Positive Reward
Loses → Negative Reward
 ↓
Agent learns better actions
```

### Examples

- Game playing
- Robotics
- Control systems
- Recommendation systems
- Autonomous systems

### Important RL Terms

- Agent
- Environment
- State
- Action
- Reward
- Policy

---
