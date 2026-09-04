# 33. Data Leakage

**Data leakage** occurs when information that should not be available to the model during prediction or training enters the training process.

This can make evaluation look unrealistically good.

Conceptually:

```text
Future Information
       ↓
Training Data
       ↓
Model
       ↓
Artificially High Performance
```

### Common Causes

- Splitting data after preprocessing incorrectly
- Using future information
- Target-derived features
- Contamination between train and test sets
- Applying transformations using information from the entire dataset before splitting

### Important Rule

> **Information from the test set should not influence model training or model selection.**

### Example

Suppose we want to predict whether a customer will cancel a subscription.

If we create a feature:

```text
Cancellation Date
```

then we have accidentally given the model future information.

That is leakage.

---

# 34. Feature Engineering

**Feature engineering** means creating or transforming useful input features from raw data.

Example:

Raw data:

```text
Date of Birth
Current Date
```

Create:

```text
Age
```

Another example:

```text
Height
Weight
```

Create:

```text
BMI
```

Another example:

```text
Total Amount
Number of Purchases
```

Create:

```text
Average Purchase Amount
```

Good features can significantly improve model performance.

### Mental Model

```text
Raw Data
   ↓
Transform / Create Features
   ↓
Better Representation
   ↓
Model
```

---

# 35. Feature Scaling

Some algorithms work better when numerical features are on comparable scales.

Example:

```text
Age        → 20–60
Salary     → 20,000–2,00,000
```

If features have very different scales, some algorithms can be affected by those differences.

Scaling transforms features into comparable ranges.

---

## Standardization

Standardization transforms values using:

```text
z = (x - mean) / standard deviation
```

The resulting feature has approximately:

```text
Mean = 0
Standard Deviation = 1
```

---

## Min-Max Scaling

Min-Max scaling transforms values using:

```text
x' = (x - min) / (max - min)
```

This commonly maps values into:

```text
0 → 1
```

---

## Algorithms Where Scaling is Often Important

Scaling is especially important for algorithms such as:

- KNN
- K-Means
- SVM
- Gradient-based models

Tree-based models generally do not require feature scaling.

Examples:

```text
Decision Tree
Random Forest
Gradient-Boosted Trees
```

---

# 36. Categorical Features

Machine Learning models generally require numerical representations of categorical values.

Example:

```text
City

Bangalore
Mumbai
Delhi
```

One possible transformation is **One-Hot Encoding**.

```text
Bangalore → [1, 0, 0]
Mumbai    → [0, 1, 0]
Delhi     → [0, 0, 1]
```

### Common Techniques

- One-Hot Encoding
- Ordinal Encoding
- Target Encoding

> Target Encoding must be performed carefully to avoid data leakage.

---

# 37. Missing Values

Real-world datasets often contain missing values.

Example:

| Age | Salary |
|---:|---:|
| 22 | 30000 |
| — | 40000 |
| 28 | — |

Possible approaches include:

- Remove rows
- Remove columns
- Mean imputation
- Median imputation
- Most-frequent imputation
- Model-based imputation

### Important

There is no single best method for every dataset.

The correct approach depends on:

- Amount of missing data
- Why the data is missing
- Feature type
- Model
- Business/problem requirements

---

# 38. Outliers

An **outlier** is an observation that is unusually far from the rest of the data.

Example:

```text
Salaries:

30k
35k
40k
42k
45k
500k ← Possible outlier
```

An outlier can represent:

```text
Real unusual observation
```

or:

```text
Data-entry error
```

or:

```text
Measurement problem
```

### Possible Approaches

- Investigate the data
- Correct data-entry errors
- Transform the feature
- Use robust methods
- Remove observations only when justified

> **Never automatically remove every outlier.**

---

# 39. Data Preprocessing

Before training a model, raw data often needs to be transformed.

A typical preprocessing pipeline can look like:

```text
Raw Data
   ↓
Clean Data
   ↓
Handle Missing Values
   ↓
Handle Categorical Data
   ↓
Feature Engineering
   ↓
Feature Scaling
   ↓
Train Model
```

However, preprocessing should be designed carefully to avoid leakage.

For example:

```text
Split Data
   ↓
Fit preprocessing on Training Data
   ↓
Apply same transformation to Validation/Test Data
```

### Important Principle

> **Learn preprocessing parameters from the training data, then apply the learned transformation to unseen data.**

---

# 40. The Basic Machine Learning Workflow

A typical ML workflow is:

```text
1. Define Problem
       ↓
2. Collect Data
       ↓
3. Explore Data
       ↓
4. Clean Data
       ↓
5. Split Data
       ↓
6. Preprocess Data
       ↓
7. Select / Engineer Features
       ↓
8. Choose Model
       ↓
9. Train Model
       ↓
10. Validate / Tune
       ↓
11. Evaluate on Test Data
       ↓
12. Deploy
       ↓
13. Monitor
```

This is not always strictly linear.

In real projects, you may repeatedly move between:

```text
Data
  ↕
Features
  ↕
Model
  ↕
Evaluation
```

---

# 41. Train → Predict → Evaluate

One of the simplest mental models for Machine Learning is:

```text
             TRAINING
                ↓
Data → Algorithm → Model
                     │
                     ↓
                  New Data
                     ↓
                  Predict
                     ↓
                 Evaluate
```

Remember:

> **Training teaches the model. Testing checks whether it generalizes.**

---

# 42. What is an ML Algorithm?

An **algorithm** is a mathematical procedure used to learn patterns from data.

Examples:

```text
Linear Regression
Decision Tree
Random Forest
K-Means
SVM
Neural Network
```

Different algorithms make different assumptions about the structure of the problem.

For example:

```text
Linear Regression
→ Assumes a linear relationship

Decision Tree
→ Learns decision rules from features

K-Means
→ Finds groups based on similarity
```

The goal is not to find the "best algorithm" universally.

The goal is to find an appropriate algorithm for the problem and data.

---

# 43. Parametric vs Non-Parametric Models

Another useful way to categorize models is:

```text
Parametric
vs
Non-Parametric
```

---

## Parametric Models

Parametric models assume a specific functional form and learn a fixed number of parameters.

Example:

```text
Linear Regression
```

```text
y = wx + b
```

The model learns:

```text
w
b
```

The functional form is specified beforehand.

---

## Non-Parametric Models

Non-parametric models do not assume a fixed functional form in the same way and can adapt their effective complexity based on the data.

Examples:

```text
KNN
Decision Trees
```

### Simple Mental Model

```text
Parametric
→ Fixed-form assumption + parameters

Non-Parametric
→ More flexible structure
```

---

# 44. Baseline Model

Before building a sophisticated model, establish a **simple baseline**.

Example:

```text
Baseline:
Always predict the majority class

Advanced Model:
Random Forest
```

Suppose:

```text
Baseline Accuracy = 85%
Random Forest     = 86%
```

The improvement may be small.

But if:

```text
Baseline Accuracy = 85%
Random Forest     = 95%
```

the advanced model provides a much stronger improvement.

### Why Baselines Matter

A baseline helps answer:

> **Is my sophisticated model actually better than a simple approach?**

If an advanced model cannot beat a sensible baseline, something may need investigation.

---

# 45. Cross-Validation

**Cross-validation** helps estimate how well a model generalizes and is useful for model and hyperparameter selection.

One common approach is **K-Fold Cross-Validation**.

Example:

### 5-Fold Cross-Validation

```text
Dataset
────────────────────────────────

Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5
```

First iteration:

```text
Validation → Fold 1
Training   → Fold 2 + Fold 3 + Fold 4 + Fold 5
```

Second iteration:

```text
Validation → Fold 2
Training   → Fold 1 + Fold 3 + Fold 4 + Fold 5
```

Continue until every fold has been used for validation.

```text
Repeat 5 times
      ↓
Calculate validation scores
      ↓
Average performance
```

### Why Use Cross-Validation?

It gives a more reliable estimate than relying on a single validation split, especially when data is limited.

---

# 46. Regularization

**Regularization** helps control model complexity and can reduce overfitting.

Common forms include:

```text
L1 Regularization
L2 Regularization
```

---

## L1 Regularization

L1 regularization is associated with **Lasso Regression**.

It adds a penalty related to the absolute values of coefficients.

Conceptually:

```text
Loss + λ × Σ|w|
```

L1 can encourage some coefficients to become exactly zero.

This can make it useful for feature selection in some settings.

---

## L2 Regularization

L2 regularization is associated with **Ridge Regression**.

It adds a penalty related to the squared values of coefficients.

Conceptually:

```text
Loss + λ × Σw²
```

L2 penalizes large coefficients and can help reduce model complexity.

---

## Simple Mental Model

```text
Regularization
      ↓
Control Complexity
      ↓
Reduce Overfitting
      ↓
Improve Generalization
```

---

# 47. Correlation vs Causation

This is extremely important in Machine Learning.

## Correlation

Two variables change together.

Example:

```text
Ice Cream Sales ↑
Swimming Incidents ↑
```

This does **not** mean:

```text
Ice Cream
    ↓
Swimming Incidents
```

A third variable may influence both:

```text
Hot Weather
   ↙      ↘
Ice Cream  Swimming
Sales      Incidents
```

Therefore:

> **Correlation does not automatically imply causation.**

---

## ML and Causality

A Machine Learning model can learn a predictive relationship without proving that one variable causes another.

For example:

```text
Feature X
   ↓
ML Model
   ↓
Prediction Y
```

If X is useful for predicting Y, that does not automatically mean:

```text
X causes Y
```

This distinction is important when interpreting ML models.

---

# 48. A Simple End-to-End Example

Suppose we want to predict **house prices**.

---

## Step 1 — Data

We collect:

```text
Size
Bedrooms
Location
Age
Price
```

Example:

| Size | Bedrooms | Location | Age | Price |
|---:|---:|---|---:|---:|
| 1000 | 2 | Bangalore | 10 | ₹50L |
| 1500 | 3 | Bangalore | 5 | ₹75L |
| 2000 | 4 | Mumbai | 3 | ₹1Cr |

---

## Step 2 — Features

```text
X = Size, Bedrooms, Location, Age
```

These are the inputs.

---

## Step 3 — Target

```text
y = Price
```

This is what we want to predict.

---

## Step 4 — Split

Separate the data into appropriate subsets:

```text
Training Data
Validation Data
Test Data
```

---

## Step 5 — Preprocess

Possible preprocessing:

```text
Handle missing values
        ↓
Encode location
        ↓
Scale features if appropriate
```

---

## Step 6 — Train

```text
X_train + y_train
        ↓
   ML Algorithm
        ↓
      Model
```

The model learns patterns from the training data.

---

## Step 7 — Predict

Give the model a new house:

```text
Size      = 1800 sq ft
Bedrooms  = 3
Location  = Bangalore
Age       = 4
```

The model may produce:

```text
Predicted Price
      ↓
₹80,00,000
```

---

## Step 8 — Evaluate

Use appropriate regression metrics:

```text
MAE
RMSE
R²
```

Then investigate whether the model generalizes well.

---

# 49. The Most Important ML Mental Model

Always think in this sequence:

```text
PROBLEM
   ↓
DATA
   ↓
FEATURES (X)
   ↓
TARGET (y)
   ↓
SPLIT
   ↓
PREPROCESS
   ↓
MODEL
   ↓
TRAIN
   ↓
PREDICT
   ↓
EVALUATE
   ↓
IMPROVE
   ↓
DEPLOY
   ↓
MONITOR
```

This is one of the most important workflows to remember.

### In Simple Words

```text
What problem am I solving?
        ↓
What data do I have?
        ↓
What are my inputs?
        ↓
What do I want to predict?
        ↓
How should I prepare the data?
        ↓
Which model should I use?
        ↓
Train it
        ↓
Make predictions
        ↓
Evaluate performance
        ↓
Improve it
        ↓
Deploy it
        ↓
Monitor it
```

---
