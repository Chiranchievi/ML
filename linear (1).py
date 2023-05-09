#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


# In[ ]:


data = pd.read_csv("C:\\D files\\datasets\\headbrain.csv")
print(f"size:{data.size} ; shape:{data.shape}")
data.head()


# In[ ]:


x = data['Head Size(cm^3)'].values
y = data['Brain Weight(grams)'].values


# In[ ]:


def regression_plot(x,y,model,title=""):
    plt.figure(figsize=(14,7))
    plt.title(title)
    plt.xlabel("Head Size(cm^3)")
    plt.ylabel("Brain Weights(grams)")
    x_line = np.array([np.min(x) - 100,np.max(x) + 100]).reshape(-1,1)
    y_line = model.predict(x_line)
    plt.scatter(x, y,c='orange', label='Original Data Points')
    plt.plot(x_line, y_line,linewidth=4, label='Regression Line')
    plt.legend()
    


# In[ ]:


x1 = x.reshape(-1,1)

linear_reg_model = LinearRegression().fit(x1,y)

regression_plot(x1, y, linear_reg_model, title="Linear Regression using Scikit Learn")
plt.scatter(x1, y, color="orange", label = "Data points")

y_hat = linear_reg_model.predict(x1)
rmse = np.sqrt(mean_squared_error(y, y_hat))
r2_score = linear_reg_model.score(x1,y)
print("Root Mean Squared Error:", rmse)
print("R^2 value:", r2_score)

