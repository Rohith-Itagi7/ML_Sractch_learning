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

# Manual varience
squared=0
for value in data:
    squared+=(value-v1)**2
    var_1=squared/len(data)
print(var_1)

#Using Numpy
v2=np.var(data)
print(v2)

#Standard deviation
variance = 200
std_manual = np.sqrt(variance)

print("Manual Standard Deviation:", std_manual)

#Normal Distribution
import matplotlib.pyplot as plt
import numpy as np

data_1 = np.random.normal(0, 1, 1000)
print(data_1)

#Conditional Probalility
study = 60
pass_total = 50
study_and_pass = 50

p_pass_given_study = study_and_pass / study
p_study_given_pass = study_and_pass / pass_total

print(p_pass_given_study)
print(p_study_given_pass)