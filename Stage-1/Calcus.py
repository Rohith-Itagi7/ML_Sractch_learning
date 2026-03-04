import numpy as np

# Data
X = np.array([1, 2, 3 ])
y = np.array([40, 50, 60])

# Initialize weight
w = 0.0

# Learning rate
lr = 0.01

# Number of samples
n = len(X)

# Training loop
for i in range(200):

    # 1️⃣ Prediction
    y_pred = w * X

    # 2️⃣ Raw Error (IMPORTANT: NOT squared)
    error = y_pred - y

    # 3️⃣ Loss (Mean Squared Error)
    loss = np.mean(error ** 2)

    # 4️⃣ Gradient
    gradient = 2 * np.mean(error * X)

    # 5️⃣ Update weight
    w = w - lr * gradient

    # Print progress
    if i % 20 == 0:
        print(f"Iteration {i}")
        print("Weight:", w)
        print("Loss:", loss)
        print("------")

print("Final Weight:", w)