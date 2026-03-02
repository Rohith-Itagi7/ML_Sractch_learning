import numpy as np
data=np.array([[1,2],     #Vector
              [3,4]])
data_1=np.array([[5,5],
                 [3,4]])

#Shape is used to know the dimenstionality
print(data.shape)      

# Element-wise operations
print(data*3)          
print(data+data_1)

#dot product
x=np.array([[1,2,3,4]])
w=np.array([[3,4,5,6]])
multiplication_dot=np.dot(x,w.T)
print(multiplication_dot)

# Manual dot product
data=0
for i in range(len(x)):
    data+=x[i]*w[i]
print(data)

#Matrix Multiplication
X_1 = np.array([[1, 2],
              [3, 4]])

W_1 = np.array([[5],
                [6]])
data_mul=np.dot(X_1,W_1)
print(data_mul)      #Here for dot the row frist coloumn element should match second row frist element
print(data_mul.shape)

#Norm
v=np.array([[3,4],
            [5,6]])
print(np.sqrt(np.sum(v**2)))
print(np.linalg.norm(v))

#Eigenvectors-Eigenvalues

A = np.array([[5, 0],
              [0, 3]])

values, vectors = np.linalg.eig(A)
print(values)
print(vectors)


