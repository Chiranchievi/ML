#!/usr/bin/env python
# coding: utf-8

# <h1>Importing Modules</h1>

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


# <h2>Loading The Dataset</h2>

# In[ ]:


data = pd.read_csv(
    "C:\\D files\\datasets\\iris.csv",
    names = ["sepal length", "sepal width", "petal length", "petal width", "target"]               
)
display(data)


# In[ ]:


X_iris = data.iloc[:,:-1].values
y_iris = data.iloc[:,-1].values


# <h3>Preprocessing the Data</h3>

# In[ ]:


sc = StandardScaler()
x_scaled = sc.fit_transform(X_iris)


# <h3>Implementing PCA in Dataset</h3>

# In[ ]:


pca = PCA(n_components=2)
reduced_iris_x= pca.fit_transform(x_scaled)


# In[ ]:


reduced_data = pd.DataFrame(reduced_iris_x,columns=["PC1","PC2"])
reduced_data["target"] = data["target"]
display(reduced_data)


# <h3>Visualizing The Priciple Components of The Reduced Iris Dataset</h3>

# In[ ]:


plt.title('2 component PCA in Iris Dataset')
plt.xlabel('pc1')
plt.ylabel('pc2')
targets = np.unique(y_iris)
for target in targets:
    idxs = y_iris == target
    plt.scatter(
        reduced_iris_x[idxs,0],
        reduced_iris_x[idxs,1],
        s=25,label = target
    )
plt.legend()
plt.grid()
plt.show()


# <h1>PCA for Breast Cancer Dataset</h1>

# In[ ]:


from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[ ]:


cancer = load_breast_cancer()
cancer_x = cancer.data
cancer_y = cancer.target_names[cancer.target]
cancer_df = pd.DataFrame(
    cancer_x, columns = cancer.feature_names
)
cancer_df['target'] = cancer_y
cancer_df


# In[ ]:


sc = StandardScaler()
scaled_x = sc.fit_transform(cancer_x)


# In[ ]:


pc = PCA(n_components=2)
reduced_data_x = pc.fit_transform(scaled_x)


# In[ ]:


reduced_data = pd.DataFrame(reduced_data_x,columns=['PC1','PC2'])
# reduced_data['target'] = reduced_data_x['target']
display(reduced_data)


# In[ ]:


plt.title('2 component PCA in Iris Dataset')
plt.xlabel('pc1')
plt.ylabel('pc2')

for target in cancer.target_names:
    idxs = cancer_y == target
    plt.scatter(
        reduced_data_x[idxs,0],
        reduced_data_x[idxs,1],
        s=10, label=target
    )
plt.legend()
plt.grid()
plt.show()


# In[ ]:




