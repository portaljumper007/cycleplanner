import numpy as np
import scipy
from scipy import spatial
from scipy.spatial import distance
import time
import random

#Code to calculate 'differences' between n dimensional scatter graphs.
#There will always be a 'base', defined as A, and then 'others' which are compared with A. There are 5 others - consisting of 10 coordinates and 3 dimensions.

#base = np.array([[random.uniform(0,10) for i in range(2)] for i in range(5)])
base = np.array([[1,2,3,4], [1,2,3,4]])
#others = np.array([[[random.uniform(0,10) for a in range(2)] for b in range(5)] for c in range(4)])
others = np.array([[[1,2,3.5,5], [1,2,3.5,5]], [[1,2,3,5], [1,2,3,5]]])

def ndSimilarity(base, others):
    base = np.transpose(base)
    others = [np.transpose(others[i]) for i in range(len(others))]
    finale = []
    for which in range(np.size(others, axis=0)):
        toAverage = []
        for point in range(np.size(base, axis=0)):
            temp = []
            for otherPoint in range(np.size(others[which], axis=0)):
                temp.append(distance.euclidean(base[point], others[which][otherPoint]))
            temp.sort()
            toAverage.append(temp)
        finale.append(1/(np.std(np.transpose(toAverage).mean(axis=1)) + 1))
    return finale

    #return np.array([np.mean([np.std([distance.euclidean(base[point], others[which][otherPoint]) for otherPoint in range(np.size(others[which], axis=0))]) for point in range(np.size(base, axis=0))]) for which in range(np.size(others, axis=0))])

print(ndSimilarity(base, others))