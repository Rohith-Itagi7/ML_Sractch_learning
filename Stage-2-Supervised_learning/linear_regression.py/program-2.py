import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
df = pd.read_excel(r"C:\Users\user\OneDrive\Desktop\complete ML\Stage-2-Supervised_learning\linear_regression.py\Book1.xlsx")

print("Original Dataset")
print(df)

print("\nMissing values:")
print(df.isnull().sum())


# METHOD 1: Mean
df_mean = df.fillna(df.mean())

# METHOD 2: Median
df_median = df.fillna(df.median())

# METHOD 3: Forward fill
df_ffill = df.ffill()

# We will use median dataset
df = df_median

print("\nDataset after filling missing values")
print(df)


X = df[["size_sqt","bedrooms","Age","Distance"]]
y = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)


mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("R2:", r2)


ridge = Ridge(alpha=1)

ridge.fit(X_train, y_train)

ridge_pred = ridge.predict(X_test)

print("\nRidge R2:", r2_score(y_test, ridge_pred))


lasso = Lasso(alpha=1)

lasso.fit(X_train, y_train)

lasso_pred = lasso.predict(X_test)

print("Lasso R2:", r2_score(y_test, lasso_pred))

poly = PolynomialFeatures(degree=2)

X_poly = poly.fit_transform(X)

poly_model = LinearRegression()

poly_model.fit(X_poly, y)

print("\nPolynomial model trained")

new_house = pd.DataFrame([[2800,4,5,6]],
columns=["size_sqt","bedrooms","Age","Distance"])

new_house_scaled = scaler.transform(new_house)

prediction = model.predict(new_house_scaled)

print("\nPredicted Price:", prediction)

plt.figure(figsize=(6,4))

plt.scatter(df["size_sqt"], df["Price"], color="green")

plt.xlabel("Size (sqft)")
plt.ylabel("Price")

plt.title("House Size vs Price")

plt.show()