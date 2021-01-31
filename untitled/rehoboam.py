import numpy as np
from scipy.interpolate import PchipInterpolator
import math
from scipy.spatial import distance
import sys

### VISPY ###
import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'
scatter = visuals.Markers()
view.add(scatter)
axis = visuals.XYZAxis(parent=view.scene)
### VISPY ###

def ndEvenDist(ndBounds, count):
    count += 1 #adding 1 because the center is removed
    param = 0
    z = [np.linspace(ndBounds[dim][param][0], ndBounds[dim][param][1], num=math.ceil(count ** (1 / np.size(ndBounds, axis=0)))) for dim in range(np.size(ndBounds, axis=0))]
    coords = np.transpose(np.meshgrid(*z))
    coords = coords.reshape(-1, coords.shape[-1])
    return coords[np.linspace(0, np.size(coords, axis=0) - 1, num=count, dtype=int).astype(int)]

dataTimes = [[1,2,3]]
dataValues = [[1,2,3]]

comb_times = np.unique(np.concatenate(dataTimes))
overallTimeSpan = np.amax(comb_times) - np.amin(comb_times)
ndBounds = [[[-overallTimeSpan, overallTimeSpan]]]
for set in range(len(dataTimes)):
    pchip = PchipInterpolator(dataTimes[set], dataValues[set], extrapolate=False)
    dataValues[set] = pchip(comb_times)
    dimSpan = np.amax(dataValues[set]) - np.amin(dataValues[set])
    ndBounds.append([[-dimSpan, dimSpan]])

COUNT = 5
disps = ndEvenDist(ndBounds, COUNT)

def ndSimilarity(base, others):
    base = np.transpose(base)
    others = [np.transpose(others[i]) for i in range(len(others))]
    finale = []
    for which in range(np.size(others, axis=0)):
        toAverage = []
        for point in range(np.size(base, axis=0)):
            temp = []
            for otherPoint in range(np.size(others[which], axis=0)):
                temp.append(1/(distance.euclidean(base[point], others[which][otherPoint]) + 1)**20)
            temp.sort()
            toAverage.append(temp)
        finale.append(np.mean(np.transpose(toAverage).mean(axis=1)))
    finale = np.array(finale)
    finale = finale / np.linalg.norm(finale)
    return finale

def ndAutocorr(dataTimes, dataValues, disps):
    dispData = []
    generic = [comb_times, *dataValues]
    for disp in range(len(disps)):
        dispData.append(np.array(generic))
        for dim in range(len(disps[disp])):
            dispData[-1][dim] += disps[disp][dim]
    return dispData, ndSimilarity(generic, dispData)

result = ndAutocorr(dataTimes, dataValues, disps)
def update(ev):
    global scatter
    global result
    for disp in range(COUNT):
        data = np.transpose([result[0][disp][0], result[0][disp][1], np.zeros(len(result[0][disp][1]))])
        print(disp)
    scatter.set_data(data, edge_color=None, face_color=(1, 1, 1, .5), size=5)

timer = app.Timer()
timer.connect(update)
timer.start(0)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
