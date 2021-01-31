import numpy as np
import matplotlib.pyplot as plt

# Make some random data
x, y, z = np.random.random((3, 25))

# Bin the data onto a 10x10 grid
# Have to reverse x & y due to row-first indexing
zi, yi, xi = np.histogram2d(y, x, bins=(20, 20), weights=z, normed=False)

print(xi)
zi = zi
zi = np.nan_to_num(zi, 0)
print(zi)