# %%
import numpy as np
import pandas
from matplotlib import pyplot as plt


# find the average vector of X and create an m.n matrix x_tilda
def x_tilde(x):
    return x - np.average(x, axis=1).reshape(x.shape[0], 1)


# svd
# decompose the covariance  and product first l columns and x_tilda as the matrix of data with lower dimension
def decomposition(x_tilde, l):
    cov = x_tilde @ np.transpose(x_tilde)
    u, s, vh = np.linalg.svd(cov)
    y = np.transpose(u[:, :l]) @ x_tilde
    return y


# read the Excel
x = pandas.read_csv('iris.csv').to_numpy()
x = np.transpose(x)
x = np.array(x, dtype=float)
l = 2
y = decomposition(x_tilde(x), l)
# plot the final diagram
plt.plot(y[0, :50], y[1, :50], 'o', color='pink')
plt.plot(y[0, 50:100], y[1, 50:100], 'o', color='green')
plt.plot(y[0, 100:150], y[1, 100:150], 'o', color='yellow')
plt.show()
