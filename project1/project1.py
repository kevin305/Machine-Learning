import pandas as pd
import numpy as np
from numpy.linalg import inv

data = pd.read_csv('iris.data.txt', header = None)
data=data.replace(['Iris-setosa'], 1)
data=data.replace(['Iris-versicolor'], 2)
data=data.replace(['Iris-virginica'], 3)
#print (data)
data = data.values
np.random.shuffle(data)
df =pd.DataFrame(data,dtype=float)
print(df)
A = df.iloc[:,0:4]
#print(A)
Y = df.iloc[:,4:5]
#print(Y)

#cross validation
k =5
set_size = int(150/k)
accuracies = []
for i in range(k):
    test_A = A.iloc[i*set_size:i*set_size+set_size,0:4]  
    test_Y = Y.iloc[i*set_size:i*set_size+set_size,:] 
    if i==0:
        train_A =A.iloc[i*set_size+set_size:,0:4]
        train_Y =Y.iloc[i*set_size+set_size:,:]   
    elif i==k-1:    
        train_A =A.iloc[:i*set_size,0:4]
        train_Y =Y.iloc[:i*set_size,:]
    else:    
        train_A = A.iloc[i*set_size-set_size:i*set_size,0:4].append(A.iloc[i*set_size+set_size:,0:4])
        train_Y =Y.iloc[i*set_size-set_size:i*set_size,:].append(Y.iloc[i*set_size+set_size:,:])
    train_A_T = np.transpose(train_A)
    C =np.dot(train_A_T,train_A)
    #print(C)
    D =inv(C)
    #print(D)
    E = np.dot(train_A_T.values,train_Y.values)
    #print(E)
    B =np.dot(D,E)
    #print(B)

    F=np.dot(test_A,B)
    #print(F)

    F=np.round(F)
    #print(F)

    accuracy = sum(np.equal(F,test_Y.values))*100/30
    accuracies.append(accuracy)
print(np.mean(accuracies))



