# 15. What is a Model?

A **model** is the learned representation of patterns or relationships from training data that can be used to make predictions.

Basic process:

```text
Training Data
     ↓
Learning Algorithm
     ↓
   Model
     ↓
New Data
     ↓
Prediction
```

For example, Linear Regression uses:

```text
y = wx + b
```

The model learns:

```text
w → Weight
b → Bias
```

These learned values allow the model to make predictions.

---

# 16. Parameters

**Parameters** are values that the model learns from the training data.

For:

```text
y = wx + b
```

the parameters are:

```text
w
b
```

The model adjusts these values during training.

Example:

```text
Before Training

w = 0.2
b = 5

        ↓
      Training
        ↓

After Training

w = 2.5
b = 10
```

The exact values depend on:

- Training data
- Learning algorithm
- Objective/loss
- Optimization process

### Important

```text
Parameters
    ↓
Learned by the model
```

---

# 17. Hyperparameters

**Hyperparameters** are settings chosen before or outside the model's learning process.

Examples:

```text
Learning rate
Number of trees
Maximum tree depth
Number of neighbors (K)
Regularization strength
```

### Parameters vs Hyperparameters

| Parameters | Hyperparameters |
|---|---|
| Learned by the model | Set/tuned by us |
| Learned during training | Usually chosen before/during model development |
| Example: weights | Example: learning rate |
| Example: bias | Example: tree depth |

### Simple Rule

```text
Parameters
→ Model learns them

Hyperparameters
→ We choose/tune them
```

---

# 18. Training

**Training** is the process of allowing a model to learn patterns from data.

A simplified training loop:

```text
Training Data
     ↓
   Model
     ↓
 Prediction
     ↓
Compare with Actual Answer
     ↓
Calculate Error
     ↓
Update Model
     ↓
Repeat
```

The objective is generally to reduce prediction error according to a chosen training objective.

### Simple Example

Suppose:

```text
Actual Price     = ₹50L
Predicted Price  = ₹40L
```

The model has made an error.

The training process attempts to adjust the model so that future predictions become better.

---

# 19. Inference / Prediction

After training, we can give the model new data.

```text
New Input
   ↓
Trained Model
   ↓
Prediction
```

Example:

```text
Experience = 5 years
        ↓
   Trained Model
        ↓
Predicted Salary = ₹65,000
```

This process of using a trained model to produce outputs for new inputs is often called **inference**.

### Training vs Inference

```text
Training
Data → Learn Parameters → Model

Inference
New Data → Trained Model → Prediction
```

---

# 20. Training Data, Validation Data and Test Data

A dataset is commonly divided into:

```text
Complete Dataset
       │
       ├── Training Set
       ├── Validation Set
       └── Test Set
```

The exact split depends on the problem.

---

## Training Set

The **training set** is used to train the model.

```text
Training Data
      ↓
Learn Parameters
```

---

## Validation Set

The **validation set** is used during model development.

It can be used to:

- Tune hyperparameters
- Compare models
- Select configurations
- Make development decisions

Example:

```text
Model A → Validation Score: 85%
Model B → Validation Score: 90%

Choose Model B
```

---

## Test Set

The **test set** is used for final evaluation on data that was not used to train or tune the model.

```text
Final Model
     ↓
Test Data
     ↓
Final Evaluation
```

### Important Rule

> **Do not use the test set to repeatedly make model-selection decisions.**

---

# 21. Generalization

A good Machine Learning model should not simply memorize the training data.

It should perform well on **unseen data**.

This ability is called:

> **Generalization**

Example:

```text
Training Data
     ↓
Model learns useful patterns
     ↓
New / Unseen Data
     ↓
Good Predictions
```

The real goal of Machine Learning is not:

```text
Memorize training data
```

The goal is:

```text
Learn useful patterns
        ↓
Perform well on unseen data
```

---

# 22. Overfitting

**Overfitting** happens when a model learns the training data too closely, including noise or accidental patterns.

Typical behavior:

```text
Training Performance → Very Good
Test Performance     → Poor
```

The model has effectively memorized too much of the training data.

### Example

```text
Model memorizes:
"These exact examples"

Instead of learning:
"The underlying pattern"
```

### Causes of Overfitting

- Model too complex
- Too little training data
- Noise in the dataset
- Too many features
- Insufficient regularization

### Possible Solutions

- More training data
- Simpler model
- Regularization
- Feature selection
- Cross-validation
- Early stopping for applicable models

---

# 23. Underfitting

**Underfitting** happens when the model is too simple to capture the important patterns in the data.

Typical behavior:

```text
Training Performance → Poor
Test Performance     → Poor
```

### Possible Causes

- Model too simple
- Insufficient features
- Excessive regularization
- Insufficient training
- Poor feature representation

### Mental Model

```text
Too Simple
    ↓
Underfitting
```

---

# 24. Bias and Variance

Bias and variance provide a useful mental model for understanding model complexity and generalization.

## High Bias

A high-bias model is often too simple.

```text
High Bias
    ↓
Underfitting
```

The model makes strong simplifying assumptions and fails to capture important patterns.

---

## High Variance

A high-variance model is very sensitive to the training data.

```text
High Variance
     ↓
Overfitting
```

It may perform extremely well on training data but poorly on unseen data.

---

## Goal

```text
High Bias              High Variance
    ↓                       ↓
Underfitting             Overfitting

             ↓
       Find a Balance
             ↓
     Better Generalization
```

### Simple Mental Model

```text
Bias → Model too simple

Variance → Model too sensitive
```

---

# 25. Loss Function

A **loss function** measures how wrong a model's prediction is for a training example or batch.

Example:

```text
Actual      = 100
Predicted   = 80
```

The prediction error is:

```text
20
```

A loss function converts prediction errors into a numerical value that can be optimized.

---

## Mean Squared Error

MSE is commonly used for regression.

```text
MSE = average((actual - predicted)²)
```

Example:

```text
Actual      = 100
Predicted   = 80

Error = 20

Squared Error = 20² = 400
```

---

## Cross-Entropy Loss

Cross-entropy is commonly used for classification.

It measures the difference between the true class distribution and the model's predicted probabilities.

---

## Important

During training, the model generally attempts to:

```text
Minimize Loss
```

The loss function therefore provides a signal that helps the learning process improve the model.

---

# 26. Evaluation Metrics

A model needs to be evaluated using appropriate metrics.

Different problems require different evaluation metrics.

---

## Regression Metrics

Common metrics include:

```text
MAE
MSE
RMSE
R²
```

### MAE

Mean Absolute Error.

```text
MAE = average(|actual - predicted|)
```

It measures the average absolute prediction error.

---

### MSE

Mean Squared Error.

```text
MSE = average((actual - predicted)²)
```

It gives larger errors more weight because errors are squared.

---

### RMSE

Root Mean Squared Error.

```text
RMSE = √MSE
```

It is expressed in the same units as the target.

---

### R²

R² measures how much of the variation in the target is explained by the model relative to a baseline.

---

## Classification Metrics

Common metrics include:

```text
Accuracy
Precision
Recall
F1 Score
ROC-AUC
```

---

## Important Distinction

> **Loss function and evaluation metric do not necessarily have to be the same.**

For example:

```text
Training:
Cross-Entropy Loss

Evaluation:
Accuracy + Precision + Recall + F1
```

---

# 27. Accuracy

**Accuracy** measures the fraction of predictions that are correct.

```text
Accuracy =
Correct Predictions / Total Predictions
```

Example:

```text
100 predictions
90 correct

Accuracy = 90%
```

Accuracy is simple and useful when classes are reasonably balanced.

However:

> **Accuracy can be misleading when classes are highly imbalanced.**

### Example

Suppose:

```text
1000 emails
950 → Not Spam
50  → Spam
```

A model that predicts:

```text
Every email → Not Spam
```

gets:

```text
950 / 1000 = 95% accuracy
```

But it detects:

```text
0 spam emails
```

So accuracy alone is not enough for many classification problems.

---

# 28. Confusion Matrix

For binary classification:

```text
                  Predicted
                Positive Negative

Actual Positive    TP       FN
Actual Negative    FP       TN
```

Where:

```text
TP → True Positive
TN → True Negative
FP → False Positive
FN → False Negative
```

---

## True Positive

The model predicts positive and the actual class is positive.

```text
Actual:    Positive
Predicted: Positive

→ TP
```

---

## True Negative

The model predicts negative and the actual class is negative.

```text
Actual:    Negative
Predicted: Negative

→ TN
```

---

## False Positive

The model predicts positive but the actual class is negative.

```text
Actual:    Negative
Predicted: Positive

→ FP
```

This is also called a **Type I error**.

---

## False Negative

The model predicts negative but the actual class is positive.

```text
Actual:    Positive
Predicted: Negative

→ FN
```

This is also called a **Type II error**.

---

## Why is the Confusion Matrix Important?

These four values are the foundation for:

```text
Precision
Recall
F1 Score
```

---

# 29. Precision

Precision answers:

> **Of everything the model predicted as positive, how many were actually positive?**

Formula:

```text
Precision = TP / (TP + FP)
```

Example:

```text
Model predicted 100 emails as Spam.

80 were actually Spam.
20 were Not Spam.

Precision = 80 / 100
          = 80%
```

High precision means:

```text
Few False Positives
```

---

# 30. Recall

Recall answers:

> **Of all the actual positive examples, how many did the model correctly identify?**

Formula:

```text
Recall = TP / (TP + FN)
```

Example:

```text
There are 100 actual Spam emails.

Model detects 90.

Recall = 90 / 100
       = 90%
```

High recall means:

```text
Few False Negatives
```

---

# 31. F1 Score

F1 Score combines Precision and Recall.

```text
F1 = 2 × (Precision × Recall)
         --------------------
         Precision + Recall
```

F1 is useful when we want a balance between:

```text
Precision
+
Recall
```

---

# 32. ROC-AUC

**ROC-AUC** is a classification evaluation metric that summarizes how well a model ranks positive examples above negative examples across different classification thresholds.

The ROC curve considers:

```text
True Positive Rate
vs
False Positive Rate
```

A higher AUC generally indicates better ranking ability.

---

