from scipy import signal
import numpy as np
from scipy.interpolate import PchipInterpolator

def weightedAvgConvolve(first, second):
    convOne = signal.convolve(first, second)
    convTwo = signal.convolve([1 for i in range(len(first))], second)
    return convOne, convTwo #np.divide(convOne, convTwo) #

def autocorr(x):
    diffs = np.array([])
    for lag in range(x.size - 1):
        diffs = np.append(diffs, (1 / (np.mean(np.absolute(x[:x.size - lag - 1] - x[lag + 1:])) + 1)**1))
    diffs = np.insert(diffs, 0, 1)
    diffs = np.insert(diffs, 0, diffs[1:][::-1])
    return diffs


DATA = np.array([[1,2,3,4,5], [1,2,3,4,5]])
samples = 5

def predictor(DATA, samples):
    tempDATA = np.zeros((2, samples))
    autocorrs = np.zeros((2, (samples*2)-1))

    for dim in range(2):
        pchip = PchipInterpolator(np.arange(1, DATA[1 - dim].size + 1), DATA[dim])
        tempDATA[dim] = pchip(np.linspace(1, DATA[dim].size, samples))
        autocorrs[dim] = autocorr(tempDATA[dim])

    rowCount = autocorrs[1].size
    rows = []
    divisors = []
    for row in range(rowCount):
        result = weightedAvgConvolve(tempDATA[1] + row - ((rowCount - 1) / 2), autocorrs[0] * autocorrs[1][row])
        rows.append(result[0])
        divisors.append(result[1])
    return np.arange(-samples + 2, tempDATA[1].size + samples), np.sum(rows, axis=0) / np.sum(divisors, axis=0)

def predictorTwo(DATA, samples):
    tempDATA = np.zeros((2, samples))
    autocorrs = np.zeros((2, (samples * 2) - 1))

    for dim in range(2):
        pchip = PchipInterpolator(np.arange(1, DATA[1 - dim].size + 1), DATA[dim])
        tempDATA[dim] = pchip(np.linspace(1, DATA[dim].size, samples))
        autocorrs[dim] = autocorr(tempDATA[dim])

    rowCount = autocorrs[1].size
    rows = []
    divisors = []
    for row in range(rowCount):
        result = weightedAvgConvolve(tempDATA[1] + row - ((rowCount - 1) / 2), autocorrs[0] * autocorrs[1][row])
        rows.append(result[0])
        divisors.append(result[1])
    x = np.arange(-samples + 2, tempDATA[1].size + samples)
    y = np.sum(rows, axis=0) / np.sum(divisors, axis=0)
    z = [0 for x in range((samples * 3) - 2)]
    return [[x[i], y[i], z[i]] for i in range(x.size)]

print(predictorTwo(np.array([[1,2,3], [1,2,3]]), 3))