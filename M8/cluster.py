import numpy as np
from sklearn.cluster import KMeans
# from matplotlib.pyplot import plt


data = np.loadtxt("M8/seeds_dataset.txt")
data = data[:, :-1]

# print(data)

k_means = KMeans(n_clusters=3).fit(data)
