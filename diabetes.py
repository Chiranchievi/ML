#!/usr/bin/env python
# coding: utf-8

# <h1>Logistic Regression for predicting diabetes using Sklearn</h1>

# In[ ]:


from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd


# <h1>Loading and Processing the DataSet</h1>

# In[ ]:


data = pd.read_csv("C:\\D files\\datasets\\diabetes.csv")

x = minmax_scale(data.iloc[:,:-1].values)
y = data.iloc[:,-1:].values.reshape(-1)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 1/3, random_state=6
)


# <h3>Implementing Logistic Regression</h3>

# In[ ]:


model = LogisticRegression().fit(x_train,y_train)


# In[ ]:


def compute_accuracy(model,x_test,y_test):
    y_hat = model.predict(x_test)
    return (y_hat == y_test).mean() * 100


# In[ ]:


print(f"Accuracy of the test set using sklearn module: {compute_accuracy(model,x_test,y_test)}")


# In[ ]:




