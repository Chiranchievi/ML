#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data = pd.read_csv("C:\\D files\\datasets\\titanic_processed.csv")
x = data.drop("Survived",axis=1).values
y = data["Survived"].values


# In[ ]:


hidden_layers = [5,3]
lr = 30
n_epochs = 20
batch_size = 25
n = len(hidden_layers)+1


# In[ ]:


s = int(0.8*x.shape[0])
x_train = x[:s].T
y_train = y[:s]
outputs = [None]*4
delta = [None]*3
x_test = x[s:]
y_test = y[s:]
n_input = x_train.shape[0]
z = y_train.reshape(-1,1)
classes = np.unique(y_train)
target = (z==classes).T
n_output = target.shape[0]
m = x_train.shape[1]


# In[ ]:


np.random.seed(0)
weights = [np.random.standard_normal((o,i+1))*0.01
for i,o in zip([n_input]+hidden_layers,hidden_layers+[n_output])]


# In[ ]:


def act(x):
    return 1/(1+np.exp(-x))


# In[ ]:


def dact(x):
    return x*(1-x)


# In[ ]:


def pad(x):
    pad_w = [(1,0),(0,0)]
    return np.pad(x,pad_w,constant_values=1)


# In[ ]:


def forward_propagate(n,input):
    outputs[-1] = input
    for i in range(n):
        outputs[i] = act(weights[i]@pad(outputs[i-1]))
    return outputs[n-1]
        


# In[ ]:


def error(actual,predicted):
    error = actual - predicted
    return np.sum(error*error)


# In[ ]:


def back_propagate(n,target):
    for i in range(n-1,-1,-1):
        if i == n-1:
            errors = outputs[i]-target
        else:
            errors = weights[i+1][:,1:].T@delta[i+1]
        delta[i] = errors*dact(outputs[i])


# In[ ]:


def update_weight(n,lr,batch_size):
    for i in range(n):
        weights[i] -= lr*delta[i]@pad(outputs[i-1]).T/batch_size


# In[ ]:


def predict(n,inputs):
    pred = forward_propagate(n,inputs.T).T.argmax(axis=-1)
    return pred


# In[ ]:


def accuracy(x_test,y_test,n):
    return (predict(n,x_test)==y_test).mean()


# In[ ]:


verbose = True
if verbose:
    print("Initital Weights:")
    print(*weights,sep="\n")
    print("Training:")

for epoch in range(n_epochs):
    sum_error = 0
    fs = range(m+batch_size)
    for f,t in zip(fs,fs[1:]):
        forwd_out = forward_propagate(n,x_train[:,f:t])
        sum_error += error(target[:,f:t],forwd_out)
        back_propagate(n,target[:,f:t])
        update_weight(n,lr,batch_size)
    if verbose:
        print(f'> epoch={epoch+1}, lrate={lr:.1f}, error={sum_error/m:.5f}')

if verbose:
    print("Trained Weights:")
    print(*weights,sep="\n")
    print("Evaluation:")
    print(f"Accuracy of the classifer:{accuracy(x_test,y_test,n)*100:.2f}%")

