#        Task-1
import numpy as np

# Data
X = np.array([1, 2, 3, 4])
y = np.array([2, 4, 6, 8])

w = 0.0
lr = 0.01
n = len(X)

for i in range(200):

    # 1️⃣ Prediction
    y_pred =w*X

    # 2️⃣ Residuals
    error = y_pred- y

    # 3️⃣ MSE Loss
    loss = np.mean(error**2)

    # 4️⃣ Gradient
    gradient =2* np.mean(error*X)

    # 5️⃣ Update
    w = w-lr*gradient


print("Final weight:", w)

#       Task-2

import numpy as np

y = np.array([3, -0.5, 2, 7])
y_pred = np.array([2.5, 0.0, 2, 8])

# MAE
mae = np.mean(error)

# MSE
mse = np.mean(error**2)

# RMSE
rmse = np.sqrt(mse)

# R²
ss_total = np.sum((y - np.mean(y))**2)
ss_res =np.sum(y-y_pred**2)
r2 = 1-ss_total/ss_res

print("MAE:", mae)
print("RMSE:", rmse)
print("R2:", r2)