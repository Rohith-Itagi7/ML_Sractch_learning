import numpy as np
# Manual mean
data = np.array([10, 20, 30, 40, 50])

total=0
for i in data:
    total+=i
print(total)
data_total=total/len(data)
print(data_total)

#Using Numpy
v1=np.mean(data)
print(v1)

squared=0
for value in data:
    squared+=(value-v1)**2
    var_1=squared/len(data)
print(var_1)

v2=np.var(data)
print(v2)

#Standard deviation
variance = 200
std_manual = np.sqrt(variance)

print("Manual Standard Deviation:", std_manual)

import matplotlib.pyplot as plt
import numpy as np

data_1 = np.random.normal(0, 1, 1000)
print(data_1)