
Dataset
   ↓
Initialize weights
   ↓
Linear Equation (wX + b)
   ↓
Sigmoid Function
   ↓
Probability
   ↓
Log Loss
   ↓
Gradient Descent
   ↓
Update weights
   ↓
Repeat

Linear equation
↓
Sigmoid
↓
Log loss
↓
Gradient descent

Attendance
   |
   |    Pass
   |      *
   |    *
   |  *
---|----------------
   | *   Fail
   | *
   | *
      Study Hours
import numpy as np

y = 1
p = 0.8

loss = -(y*np.log(p) + (1-y)*np.log(1-p))
print(loss)
