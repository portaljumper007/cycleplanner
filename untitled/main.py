#import scipy
#from scipy import ndimage
import numpy as np
import math
import numpy as np
import itertools
import time
from scipy import optimize
from scipy.spatial import distance

np.set_printoptions(precision=10)
np.set_printoptions(suppress=True)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d, Axes3D
figure = plt.figure() #gcf()
figureTwo = plt.figure()
figureThree = plt.figure()
ax1 = Axes3D(figure) #figure.add_subplot(111, projection='3d')
ax2 = Axes3D(figureTwo)
ax3 = Axes3D(figureThree)

def ndEvenDist(ndBounds, count):
    z = np.array([[np.linspace(ndBounds[dim][param][0].astype(np.float), ndBounds[dim][param][1].astype(np.float), num=math.ceil(count ** (1 / np.size(ndBounds[dim], axis=0))), endpoint=True, retstep=False, dtype=None, axis=0) for param in range(np.size(ndBounds[dim], axis=0))] for dim in range(np.size(ndBounds, axis=0))])
    z = z.reshape(-1, z.shape[-1])
    meshgrid = np.meshgrid(*z)
    meshgrid = [meshgrid[i].flatten() for i in range(np.size(meshgrid, axis=0))]
    print(meshgrid)
    return [meshgrid[a][list(set(np.floor(np.linspace(0, len(meshgrid[a]), num=count + 2, endpoint=True, retstep=False, dtype=None, axis=0)[1:-1]).astype(int)))] for a in range(len(meshgrid))]

def ndSimilarity(base, others):
    return np.array([np.mean([np.mean([distance.euclidean(base[point], others[which][otherPoint]) for otherPoint in range(np.size(others[which], axis=0))]) for point in range(np.size(base, axis=0))]) for which in range(np.size(others, axis=0))])



def phase(origin, dimPhase):
    # origin must be a numpy array
    result = origin.astype(float) + dimPhase.astype(float)
    return result

def scale(origin, factor, center):
    # origin must be a numpy array
    result = (((origin - center ) * factor) + center)
    return result

def doubleRefl(points, AA, AB, BA, BB):
    A_vector = AB - AA
    B_vector = BA - BB
    A_hat = A_vector / (A_vector**2).sum()**0.5
    B_hat = B_vector / (B_vector**2).sum()**0.5
    result = np.array([])
    for i in range(np.size(points,0)):
        point_refl = -points[i] + 2 * AA + 2 * A_hat * np.dot((points[i] - AA), A_hat)
        point_doubleRefl = -point_refl + 2 * BA + 2 * B_hat * np.dot((point_refl - BA), B_hat)
        result = np.vstack((*[result[i] for i in range(len(result))], point_doubleRefl))
    return result



DATA = np.array([[i for i in range(10)], [i for i in range(10)], [i for i in range(10)]])

# Scaling # Computational plausibility put on the line.
# Displacement
# Rotation

perf = [[], []]

for i in range(10, 11, 1): #Nearing ##math.ceil(count ** (1 / np.size(ndBounds[dim], axis=0)))## may cause weird distribution

    ax1.clear()
    ax3.clear()
    ax3.scatter(*DATA, color='blue', s=10 ** 2)
    count = i#((np.sin(i / 5) + 2) * 10).astype(int)
    perf[0].append(count)
    startTime = time.time_ns() / (10 ** 9)

    ndBounds = np.zeros(shape=(np.size(DATA, axis=0),3,2))
    for dim in range(np.size(DATA, axis=0)):
        dimRange = np.amax(DATA[dim]) - np.amin(DATA[dim])
        lowerBound = np.amin(DATA[dim]) - (1 * dimRange)
        upperBound = np.amax(DATA[dim]) + (1 * dimRange)
        ndBounds[dim] = np.array([[0.25, 4], [lowerBound, upperBound], [-dimRange, dimRange]]) #scale factor, scale center, phase             #doubleRefl center???, ["rotate", 0, 360 - ((360 - 0) / (count + 1))]

    result = ndEvenDist(ndBounds, count)
    print(ndEvenDist(ndBounds, count))

    params = 3
    for task in range(np.size(result[0], axis=0)):
        startTimeTwo = time.perf_counter()
        tempDATA = np.array([[i for i in range(10)], [i for i in range(10)], [i for i in range(10)]])
        for dim in range(np.size(result, axis=0) // params): # amount of parameters (dimensions) passed into ndEvenDist
            offset = params * dim
            temp = scale(DATA[dim], result[offset][task], result[offset + 1][task])
            temp = phase(temp, result[offset + 2][task])
            tempDATA[dim] = temp
        similarity = ndSimilarity(np.array([[i for i in range(10)], [i for i in range(10)], [i for i in range(10)]]), np.array([tempDATA]))
        print(1 / (time.perf_counter() - startTimeTwo))
        colormap = plt.cm.get_cmap('binary')
        ax3.scatter(*tempDATA, [similarity for a in range(np.size(tempDATA[0], axis=0))], vmin=0, vmax=2000000, s=12, cmap=colormap)
        plt.draw()
        plt.pause(0.0000000001)







    #sampleOutput = np.array([])
    #for inputDim in range(len(result) // 4):
        #for transf in range(len(result[0])):
            #for part in range(len(result)):
                #if part == 0:
                    #temp = scale(sampleInput[inputDim], result[part][transf], result[part + 1][transf])
                #if part == 2:
                    #temp = phase(temp, result[part][transf])
            #if sampleOutput.size < 1:
                #sampleOutput = np.vstack([temp])
            #else:
                #sampleOutput = np.vstack(([sampleOutput[i] for i in range(len(sampleOutput))], [temp]))
        #print(sampleOutput)

    perf[1].append((time.time_ns() / (10 ** 9)) - startTime)
    ax1.scatter(result[0], result[1], result[2])
    ax2.clear()
    ax2.plot(np.array(perf[0]), perf[1])

    if len(perf[0]) > 2:
        def test_func(x, a, b, c):
            return (a * (x ** 2)) + (b * x) + c
        params = optimize.curve_fit(test_func, perf[0], perf[1])[0]
        x = np.linspace(min(perf[0]), max(perf[0]) * 2,100)
        y = test_func(x, params[0], params[1], params[2])
        ax2.plot(x, y, label='{}'.format([i, params]))
        ax2.legend()
plt.pause(100)