import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

data = pd.read_csv('iris.data.txt', header = None)
data=data.replace(['Iris-setosa'], 1)
data=data.replace(['Iris-versicolor'], 2)
data=data.replace(['Iris-virginica'], 3)
#print (data)
data = data.values
np.random.shuffle(data)
df =pd.DataFrame(data,dtype=float)

#print(A)
Y = df.iloc[:,4:5]
e = []

for k in range(1,11):
    centroids = []
    A = df.iloc[:,0:4]
    for i in range(k):
        centroids.append([random.uniform(A.iloc[:,i].min(),A.iloc[:,i].max()) for i in range(4)])
    A = A.values    
    total_error = []
    for _ in range(100):
        distances = {}    
        for i in range(len(A)):
            distances[i] = []
            for j in range(k):
                dist = np.sqrt(np.sum((A[i]-centroids[j])**2))
                distances[i].append(dist)
            ind = distances[i].index(min(distances[i]))
            distances[i] = min(distances[i]),ind    
        
        c = {}
        for i in range(k):
            c[i] = 0    
        centroids = np.zeros((len(centroids),4))
        for i in range(len(A)):
            centroids[distances[i][1]] += A[i] 
            c[distances[i][1]]+=1
        for i in range(k):
            if c[i]>0:
                centroids[i] = centroids[i]/c[i] 
        error = sum([i**2 for i,j in distances.values()])
        total_error.append(error)
    #print(centroids)
    e.append(error)

plt.plot(e)
plt.show()
