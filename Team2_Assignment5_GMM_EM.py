#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import itertools
from scipy import linalg
import matplotlib as mpl
X = loadmat('Assignment5_D1_2.mat')
X1 = X['samples']
Y1 = X['labels']
scaler = StandardScaler()
#z-score normalisation
scaled_data = scaler.fit_transform(X1)
sklearn_pca = PCA(n_components=2)
#PCA implementation
reduced_data = sklearn_pca.fit_transform(scaled_data)
#PCA plot
plt.scatter(reduced_data[:, 0], reduced_data[:, 1], label='Data Points')
plt.title("PCA")
plt.xlim(-5., 5.)
plt.ylim(-4., 6.)
plt.legend()
plt.show()
gmm = GaussianMixture(n_components=3)
gmm.fit(reduced_data)
print('mean')
print(gmm.means_)
print('\n')
print('covariance')
print(gmm.covariances_)
#For colours in plot
color_iter = itertools.cycle(['red', 'gold', 'cornflowerblue', 'c','darkorange','navy','green'])
#function for plotting
def plot_results(X, Y_, means, covariances, index, title):
    plt.figure(figsize=(10,10))
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-5., 5.)
    plt.ylim(-4., 6.)
    plt.title(title)
#GMM with 3 gaussian
gmm = GaussianMixture(n_components=3, covariance_type='full').fit(reduced_data)
plot_results(reduced_data, gmm.predict(reduced_data), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture with 3 Gaussians')
#GMM with 5 gaussian
gmm = GaussianMixture(n_components=5, covariance_type='full').fit(reduced_data)
plot_results(reduced_data, gmm.predict(reduced_data), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture with 5 Gaussians')
#GMM with 7 gaussian
gmm = GaussianMixture(n_components=7, covariance_type='full').fit(reduced_data)
plot_results(reduced_data, gmm.predict(reduced_data), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture with 7 Gaussians')


# In[ ]:




