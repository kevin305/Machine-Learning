# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:19:50 2018

@author: kevin
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 19:27:13 2018
@author: lenovo
"""
# importing the libraries
import pandas as pd
import numpy as np
#import sys

# reading data from input file
data = pd.read_csv("iris.data.txt",header=None)

# adding the bias term
b = pd.DataFrame(np.ones(len(data)))
data = pd.concat([b,data],axis=1)

# shuffling the data to train on different classes properly
data = data.sample(frac=1)


# separating the dependent and independent variables and splitting data into training and test data
X = data.iloc[:,:5]
y = data.iloc[:,5]

# function to encode the output labels
def encodeLabels(a):
    j = 0
    d = {}
    for i in set(a):
        d[i] = j
        j+=1        
    return d

# function to compute the weights (Linear Regression)
def findWeights(A,Y):
    return np.dot(np.dot(np.linalg.inv(np.dot(np.transpose(A),A)),np.transpose(A)),Y)

# function to compute the classification accuracy
def accuracy(y_pred,y_test):
    return sum(np.equal(y_pred,y_test))*100/len(y_pred)

# shuffle function to pick training and test set
def shuffleData(X,y,psize):
    X = X.iloc[psize:,:].append(X.iloc[:psize,:])
    y = y.iloc[psize:].append(y.iloc[:psize])
    return X,y

# call to label encoding method
d = encodeLabels(y)
y = y.map(d)

# cross validation computation    
acc = []
k = 5
psize = int(len(data)/k)
variance = []
for i in range(k):
    
    # splitting data into training and test data
    X_train = X.iloc[:(k-1)*psize,:].values
    y_train = y.iloc[:(k-1)*psize].values
    X_test = X.iloc[(k-1)*psize:,:].values
    y_test = y.iloc[(k-1)*psize:].values
    
    # shuffle data function to place new block on top
    X,y = shuffleData(X,y,psize)
    
    # calling the method to find weights
    beta = findWeights(X_train,y_train)
    
    # predicted values
    y_pred = np.abs(np.round(np.dot(X_test,beta)))
    
    # decoding the output labels
    #y_pred = pd.DataFrame(y_pred).iloc[::,0]
    #d = {v:k for k,v in d.items()}
    #y_pred = y_pred.map(d)
    
    # call to compute accuracy
    acc.append(accuracy(y_pred,y_test))
    variance.append(sum(np.power(y_pred-y_test,2))/len(y_test))
    
print(acc)
# priting the cross validation accuracy
print("Accuracy:",np.mean(acc))

# printing the error in the output
print("Variance:",np.mean(variance))
