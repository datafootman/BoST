# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 17:50:45 2019

@author: 
"""


#!/usr/bin/env python
# -*- coding: utf-8 -*-
 
import numpy as np
from matplotlib import pyplot as plt
import BoST_ as BoST 
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# [doc_num, topics_num]
doc_topic = np.load("BoST_doc_topic_distributions.npy",allow_pickle=True)

# show them
print("visualizing document vectors of BoST")
tsne = TSNE(n_components=2, init='pca')
w1_y1 = tsne.fit_transform(doc_topic)
plt.figure(figsize=(4,4))
plt.xticks([])
plt.yticks([])
plt.scatter(w1_y1[:,0], w1_y1[:,1],s= 1, c='black', alpha=0.6)
plt.show()

# [doc_num, topics_num]
doc_topic = np.load("LDAdoc_topic_distributions.npy",allow_pickle=True)

# show them
print("visualizing document vectors of LDA")
tsne = TSNE(n_components=2, init='pca')
w1_y1 = tsne.fit_transform(doc_topic)
plt.figure(figsize=(4,4))
plt.xticks([])
plt.yticks([])
plt.scatter(w1_y1[:,0], w1_y1[:,1],s= 1, c='black', alpha=0.6)
plt.show()
