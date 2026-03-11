import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Create dataset
np.random.seed(0)
X = np.linspace(0, 10, 20)
y = X**2 + np.random.randn(20)*10   # quadratic relation + noise

X = X.reshape(-1,1)


X_plot = np.linspace(0,10,100).reshape(-1,1)

# ---------- Underfitting (Linear model) ----------
lin_model = LinearRegression()
lin_model.fit(X,y)
y_lin = lin_model.predict(X_plot)

# ---------- Good fit (Polynomial degree 2) ----------
poly2 = PolynomialFeatures(degree=2)
X_poly2 = poly2.fit_transform(X)

model2 = LinearRegression()
model2.fit(X_poly2,y)

y_poly2 = model2.predict(poly2.transform(X_plot))

poly10 = PolynomialFeatures(degree=10)
X_poly10 = poly10.fit_transform(X)

model10 = LinearRegression()
model10.fit(X_poly10,y)

y_poly10 = model10.predict(poly10.transform(X_plot))

plt.figure(figsize=(15,4))

# Underfitting
plt.subplot(1,3,1)
plt.scatter(X,y,color="blue")
plt.plot(X_plot,y_lin,color="red")
plt.title("Underfitting (Linear Model)")

# Good fit
plt.subplot(1,3,2)
plt.scatter(X,y,color="blue")
plt.plot(X_plot,y_poly2,color="green")
plt.title("Good Fit (Polynomial Degree 2)")

# Overfitting
plt.subplot(1,3,3)
plt.scatter(X,y,color="blue")
plt.plot(X_plot,y_poly10,color="red")
plt.title("Overfitting (Polynomial Degree 10)")

plt.show()
