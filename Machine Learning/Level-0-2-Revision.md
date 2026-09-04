# 50. Level 0-2 — What You Should Understand

Before moving to individual ML algorithms, make sure you can explain the following concepts in your own words.

---

## Core Concepts

- What is Machine Learning?
- AI vs ML vs Deep Learning
- What is data?
- What is a dataset?
- What is a sample?
- What is a feature?
- What is a target?
- What is a model?
- What is an algorithm?

---

## Learning Types

- Supervised Learning
- Unsupervised Learning
- Reinforcement Learning
- Regression
- Classification
- Clustering

---

## Model Concepts

- Parameters
- Hyperparameters
- Training
- Prediction / Inference
- Generalization
- Overfitting
- Underfitting
- Bias
- Variance
- Parametric Models
- Non-Parametric Models
- Baseline Models
- Regularization

---

## Data Concepts

- Training Set
- Validation Set
- Test Set
- Missing Values
- Outliers
- Categorical Variables
- Feature Engineering
- Feature Scaling
- Data Leakage

---

## Evaluation

- Loss Function
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- MAE
- MSE
- RMSE
- R²
- Confusion Matrix
- Cross-Validation

---

# 51. Quick Concept Map

Use this map to connect the concepts:

```text
                         MACHINE LEARNING
                                │
             ┌──────────────────┼──────────────────┐
             ↓                  ↓                  ↓
       SUPERVISED         UNSUPERVISED       REINFORCEMENT
             │                  │                  │
       ┌─────┴─────┐       ┌────┴────┐             │
       ↓           ↓       ↓         ↓             ↓
  Regression  Classification  Clustering   Dimensionality   Agent
                                             Reduction        ↓
                                                              Action
                                                              ↓
                                                            Reward
```

---

# 52. Complete ML Pipeline

A more detailed view of a Machine Learning project:

```text
                    PROBLEM
                       ↓
                  Collect Data
                       ↓
                  Explore Data
                       ↓
                  Clean Data
                       ↓
                Split the Data
                       ↓
            ┌──────────┴──────────┐
            ↓                     ↓
       Training Data        Test Data
            ↓
       Preprocessing
            ↓
    Feature Engineering
            ↓
       Model Selection
            ↓
          Training
            ↓
       Validation
            ↓
    Hyperparameter Tuning
            ↓
       Final Model
            ↓
       Test Evaluation
            ↓
          Deploy
            ↓
         Monitor
            ↓
       Retrain / Improve
```
### Important

The test set should remain isolated until final evaluation.

---

# 53. Training vs Validation vs Test

A simple way to remember the three:

```text
TRAINING
"Teach the model."

VALIDATION
"Help me choose and tune the model."

TEST
"Tell me how the final model performs on unseen data."
```

Or:

```text
Training → Learn
Validation → Choose / Tune
Test → Final Check
```

---

# 54. Parameters vs Hyperparameters

Another important distinction:

```text
              MODEL
                │
        ┌───────┴───────┐
        ↓               ↓
   Parameters      Hyperparameters
        ↓               ↓
 Learned from       Chosen / Tuned
     Data             by Us
```

Example:

```text
Linear Regression

Parameters:
w
b

Neural Network

Parameters:
Weights
Biases

Hyperparameters:
Learning Rate
Number of Layers
Batch Size
```

---

# 55. Loss vs Evaluation Metric

These concepts are related but not identical.

### Loss

Used to guide model training.

```text
Prediction
    ↓
Loss
    ↓
Optimization
    ↓
Update Model
```

### Evaluation Metric

Used to measure model performance.

```text
Model
  ↓
Predictions
  ↓
Evaluation Metric
  ↓
Performance
```

Example:

```text
Training:
Cross-Entropy Loss

Evaluation:
Precision
Recall
F1
```

---

# 56. Overfitting vs Underfitting

Quick comparison:

| Concept | Training Performance | Test Performance | Main Problem |
|---|---|---|---|
| Underfitting | Poor | Poor | Model too simple |
| Good Fit | Good | Good | Good generalization |
| Overfitting | Very Good | Poor | Model memorizes training data |

Mental model:

```text
Too Simple
   ↓
Underfitting

Balanced
   ↓
Good Generalization

Too Complex
   ↓
Overfitting
```

---

# 57. Supervised vs Unsupervised vs Reinforcement Learning

| Type | Has Target Labels? | Main Goal | Example |
|---|---|---|---|
| Supervised | Yes | Learn input → output relationship | House price prediction |
| Unsupervised | No | Discover hidden structure | Customer clustering |
| Reinforcement | No fixed target label | Learn actions through rewards | Game-playing agent |

---

# 58. Regression vs Classification vs Clustering

```text
Regression
    ↓
Predict a number

Example:
Salary = ₹70,000
```

```text
Classification
    ↓
Predict a category

Example:
Spam = Yes
```

```text
Clustering
    ↓
Discover groups

Example:
Customer Group = 2
```

---
# 60. What Comes Next?

After completing **Level 0 to 2 — ML Foundations**, the next step is to understand the mathematics and data concepts behind Machine Learning.

Recommended learning path:

```text
Level 0 — ML Foundations
        ↓
Level 1 — Mathematics for Machine Learning
        ↓
Level 2 — Data Preprocessing
        ↓
Level 3 — Regression
        ↓
Level 4 — Classification
        ↓
Level 5 — Tree-Based Models
        ↓
Level 6 — Unsupervised Learning
        ↓
Level 7 — Model Evaluation & Optimization
        ↓
Level 8 — Ensemble Learning
        ↓
Level 9 — Real-World ML Projects
        ↓
Level 10 — Deployment & MLOps
```

---
