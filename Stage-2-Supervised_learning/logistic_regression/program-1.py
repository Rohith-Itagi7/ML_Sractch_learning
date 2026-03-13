import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    log_loss,
    roc_curve,
    roc_auc_score
)

# ------------------------------------------------
# 1️⃣ SIGMOID FUNCTION
# ------------------------------------------------

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10,10,100)
sig = sigmoid(z)

plt.figure()
plt.plot(z,sig)
plt.title("Sigmoid Function")
plt.xlabel("z")
plt.ylabel("sigmoid(z)")
plt.show()

# ------------------------------------------------
# 2️⃣ PROBABILITY INTERPRETATION
# ------------------------------------------------

x_test = np.array([-2,-1,0,1,2])
prob = sigmoid(x_test)

print("\nProbability Interpretation")
for x,p in zip(x_test,prob):
    print(f"Input {x} -> Probability {p:.2f}")

# ------------------------------------------------
# 3️⃣ BINARY CLASSIFICATION DATASET
# ------------------------------------------------

X,y = make_classification(
    n_samples=300,
    n_features=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=42
)

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

# ------------------------------------------------
# 4️⃣ TRAIN LOGISTIC REGRESSION
# ------------------------------------------------

model = LogisticRegression()

model.fit(X_train,y_train)

pred = model.predict(X_test)
prob = model.predict_proba(X_test)[:,1]

# ------------------------------------------------
# 5️⃣ DECISION BOUNDARY VISUALIZATION
# ------------------------------------------------

plt.figure()

plt.scatter(X[:,0],X[:,1],c=y)

x_min,x_max = X[:,0].min(),X[:,0].max()

w = model.coef_[0]
b = model.intercept_

x_vals = np.linspace(x_min,x_max,100)
y_vals = -(w[0]*x_vals + b)/w[1]

plt.plot(x_vals,y_vals)

plt.title("Decision Boundary")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# ------------------------------------------------
# 6️⃣ LOG LOSS
# ------------------------------------------------

loss = log_loss(y_test,prob)

print("\nLog Loss:",loss)

# ------------------------------------------------
# 7️⃣ EVALUATION METRICS
# ------------------------------------------------

acc = accuracy_score(y_test,pred)
prec = precision_score(y_test,pred)
rec = recall_score(y_test,pred)
f1 = f1_score(y_test,pred)

print("\nEvaluation Metrics")
print("Accuracy:",acc)
print("Precision:",prec)
print("Recall:",rec)
print("F1 Score:",f1)

# ------------------------------------------------
# 8️⃣ ROC CURVE
# ------------------------------------------------

fpr,tpr,_ = roc_curve(y_test,prob)
auc = roc_auc_score(y_test,prob)

plt.figure()

plt.plot(fpr,tpr,label="AUC="+str(round(auc,3)))
plt.plot([0,1],[0,1],'--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

iris = load_iris()

X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.3,random_state=42
)

multi_model = LogisticRegression(
    multi_class="multinomial",
    max_iter=200
)

multi_model.fit(X_train,y_train)

pred = multi_model.predict(X_test)

print("\nMulticlass Accuracy (Iris):",accuracy_score(y_test,pred))
