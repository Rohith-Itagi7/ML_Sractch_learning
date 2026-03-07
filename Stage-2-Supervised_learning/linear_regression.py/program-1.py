# #        Task-1
# import numpy as np

# # Data
# X = np.array([1, 2, 3, 4])
# y = np.array([2, 4, 6, 8])

# w = 0.0
# lr = 0.01
# n = len(X)

# for i in range(200):

#     # 1️⃣ Prediction
#     y_pred =w*X

#     # 2️⃣ Residuals
#     error = y_pred- y

#     # 3️⃣ MSE Loss
#     loss = np.mean(error**2)

#     # 4️⃣ Gradient
#     gradient =2* np.mean(error*X)

#     # 5️⃣ Update
#     w = w-lr*gradient


# print("Final weight:", w)

# #       Task-2

# import numpy as np

# y = np.array([3, -0.5, 2, 7])
# y_pred = np.array([2.5, 0.0, 2, 8])

# # MAE
# mae = np.mean(np.abs(error))

# # MSE
# mse = np.mean(error**2)

# # RMSE
# rmse = np.sqrt(mse)

# # R²
# ss_total = np.sum((y - np.mean(y))**2)
# ss_res =np.sum((y - y_pred)**2)
# r2 = 1 - (ss_res / ss_total)

# print("MAE:", mae)
# print("RMSE:", rmse)
# print("R2:", r2)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LinearRegression,Ridge,Lasso
# from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# data=pd.DataFrame({
#     "Size":[1000,1500,2000,2500,3000],
#     "Rooms":[1,2,3,np.nan,5],
#     "Price":[2000,3000,4000,5000,6000]
# })

# print("Dataset:\n", data)
# data = data.dropna()
# print(data)

# X=data[["Size","Rooms"]]
# y=data["Price"]

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

# model=LinearRegression()

# model.fit(X_train,y_train)

# y_pred=model.predict(X_test)

# mae=mean_absolute_error(y_test,y_pred)
# mse = mean_squared_error(y_test, y_pred)
# rmse = np.sqrt(mse)
# r2 = r2_score(y_test, y_pred)
# print("\nModel Evaluation:")
# print("MAE:", mae)
# print("MSE:", mse)
# print("RMSE:", rmse)
# print("R2 Score:", r2)

# print("\nModel Coefficients:")
# print("Weights:", model.coef_)
# print("Bias:", model.intercept_)

# ridge_model = Ridge(alpha=1.0)
# ridge_model.fit(X_train, y_train)

# ridge_pred = ridge_model.predict(X_test)

# print("\nRidge Coefficients:", ridge_model.coef_)
# lasso_model = Lasso(alpha=1.0)
# lasso_model.fit(X_train, y_train)

# lasso_pred = lasso_model.predict(X_test)

# print("\nLasso Coefficients:", lasso_model.coef_)

# new_house = np.array([[5000, 3]])
# prediction = model.predict(new_house)

# print("\nPredicted Price for new house:", prediction)

# plt.scatter(X["Size"], y, color="blue", label="Actual Data")

# plt.plot(X["Size"],
#          model.predict(X),
#          color="red",
#          label="Regression Line")

# plt.xlabel("Size")
# plt.ylabel("Price")
# plt.title("Linear Regression Visualization")
# plt.legend()

# plt.show()


