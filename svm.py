#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cvxopt import solvers
from cvxopt import matrix
from functools import partial


# In[2]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist


# In[3]:


data = pd.read_csv("C:\\D files\\datasets\\titanic_processed.csv")


# In[4]:


X = data.drop('Survived', axis=1).values
y = data['Survived'].values
y[y==0] = -1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[5]:


sc = StandardScaler()
X_scale = sc.fit_transform(X_train)
y_scale = y_train.reshape(-1,1)


# In[7]:


sigma = 2.5
c = .35

def rbf_kernel(x1, x2, sigma):
    return np.exp(-(cdist(x1, x2, 'sqeuclidean'))/(2*sigma**2))
kernel = partial(rbf_kernel, sigma = sigma)


# In[9]:


n = X_scale.shape[0]
I_n = np.eye(n)
P = (y_scale@y_scale.T)*kernel(X_scale, X_scale)
q = np.full(n, -1)
G = np.vstack((-1*I_n, I_n))
h = np.hstack((np.zeros(n), np.full(n, c)))
A = y_train.reshape(1,-1)
b = np.zeros(1)

P,q,G,h,A,b = map(lambda x:matrix(x, tc="d"),(P,q,G,h,A,b))


# In[10]:


solutions = solvers.qp(P,q,G,h,A,b)


# In[12]:


a = np.asarray(solutions['x']).squeeze()

support_index = np.logical_and(a>=1e-10, a<c)
x_s = X_scale[support_index]
b = np.mean(y_scale-a*y_scale.T@kernel(X_scale, x_s))

X_test = sc.fit_transform(X_test)
predict = np.sign(a*y_scale.T@kernel(X_scale, X_test)+b)
print(f"Accuracy of the classifier:{(y_test==predict).mean() * 100:.2f}%")


# In[ ]:




