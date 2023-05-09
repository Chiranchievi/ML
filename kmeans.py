#!/usr/bin/env python
# coding: utf-8

# <h1>Importing Modules</h1>

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# <h2>Loading DataSets</h2>

# In[ ]:


names = ['s_len', 's_wid', 'p_len', 'p_wid', 'species']
data = pd.read_csv("C:\\D files\\datasets\\iris.csv",header=None,names=names)
display(data.head())


# <h2>Implementing K-means Clustering</h1>

# In[ ]:


train_x = data.drop("species", axis=1).values
k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(train_x)
all_data = data.copy(deep = True)
centroids = pd.DataFrame(kmeans.cluster_centers_, columns=names[:-1])
centroids["cluster"] = "centroid"
all_data['cluster'] = kmeans.labels_.astype("str")
all_data = pd.concat([all_data, centroids])
all_data


# <h2>Comparing The Clustering Results with Original Classes</h2>

# In[ ]:


plt.figure(figsize=(15,6))
plt.suptitle("classes vs clusters")
plt.subplot(121)
plt.title("Original Classes")
plt.title("Original Classes")
sns.scatterplot(data=all_data,x="s_len", y="s_wid", hue="species", s=80)
plt.legend()

plt.subplot(122)
plt.title("Predicted Clusters and Centroids")
sns.scatterplot(
    data = all_data, x="s_len", y="s_wid",
    hue = "cluster", style = "cluster",
    markers = "osPD", s=80
)
plt.legend(loc="upper right")
plt.show()


# <h3>Finding Optimal K using Elbow Method</h3>

# In[ ]:


distortion = []
K = range(2, 6)
for k in K:
    kmeans = KMeans(n_clusters=k)
    model = kmeans.fit(train_x)
    distortion.append(kmeans.inertia_/100)
    ss = silhouette_score(train_x, kmeans.labels_)
    print(f"For k={k:<4} Avg.Sil.Coef: {ss:<10.5f} Distortion: {distortion[-1]:.5f}")


# In[ ]:


plt.plot(K, distortion, 's-', markersize=8)
plt.xlabel('k')
plt.xticks(K)
plt.ylabel('Distortion')
plt.title("Elbow method for finding optimal k")
plt.show()

