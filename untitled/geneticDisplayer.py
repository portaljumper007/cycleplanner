import numpy as np
import time
import sys
import geopy.distance
import matplotlib.pyplot as plt
from statistics import mean
from itertools import chain
figure = plt.figure() #gcf()

### P L O T T I N G #############################################################
#################################################################################
from vispy import app, visuals, scene

import random

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'fly'
view.camera.fov = 45
view.camera.distance = 6
#################################################################################
#################################################################################

properties = np.load("properties.npy")
boundBoxCenter = [np.mean([properties[0][0][0], properties[0][1][0]]), np.mean([properties[0][0][1], properties[0][1][1]]), 0]

def roadDraw(roads, dims, dimScal, dimDisp):
    out = []
    totalN = 0
    heights = []
    for road in range(len(roads)):
        if dims < 3:
            heights.extend(np.ones(len(roads[road][0])))
        try:
            N = np.size(roads[road][0])
            temp = np.empty((N - 1, 2), np.int32)
            temp[:, 0] = np.arange(totalN, totalN + N - 1)
            temp[:, 1] = np.arange(totalN, totalN + N - 1) + 1
            totalN += N
            out.extend(temp)
        except:
            print(roads[road][0])
    connect = np.array(out)
    roadsConc = np.concatenate(roads,axis=1)
    if dims < 3:
        roadsConc = np.append(roadsConc, [heights], axis=0)
    for dim in range(len(roadsConc)):
        roadsConc[dim] = ((roadsConc[dim] - boundBoxCenter[dim]) * dimScal[dim]) + dimDisp[dim]
    roadsTranspos = np.transpose(roadsConc)
    return roadsTranspos, connect

def normalize(array):
    return (array - np.min(array)) / np.ptp(array)

def addToOne(array):
    return array / np.sum(array)

joints = np.load("joints.npy", allow_pickle=True).tolist()

weights = []
for seg in range(len(joints)):
    weights.append([[], []])
    for sOE in range(-1, 1):
        for option in range(len(joints[seg][sOE][0])):
            weights[seg][sOE].append([])

splitRoads = np.load("splitRoadsWithHeights.npy", allow_pickle=True)
length = len(splitRoads)
splitRoadsWithHeights = splitRoads

metric = False
if metric:
    print("Gathering segment data for routes to be judged by...")
    segMetric_dists = []
    segMetric_elChange = []
    for segment in range(length):
        if segment % np.floor(length / 50) == 0:
            perc = ((segment) / (length - 1))
            print(perc * 100, "%")
        temp1 = []
        temp2 = []
        for coord in range(len(splitRoadsWithHeights[segment][0]) - 1):
            coord1 = (splitRoadsWithHeights[segment][0][coord], splitRoadsWithHeights[segment][1][coord])
            coord2 = (splitRoadsWithHeights[segment][0][coord+1], splitRoadsWithHeights[segment][1][coord+1])
            heightDiff = splitRoadsWithHeights[segment][2][coord+1] - splitRoadsWithHeights[segment][2][coord]
            temp1.append(np.sqrt((geopy.distance.distance(coord1, coord2).km)**2 + (heightDiff**2)))
            temp2.append(heightDiff)
        segMetric_dists.append(temp1)
        segMetric_elChange.append(temp2)
    segMetric_dists = np.array(segMetric_dists)
    segMetric_elChange = np.array(segMetric_elChange)
    np.save("segMetric_dists", segMetric_dists)
    np.save("segMetric_elChange", segMetric_elChange)
    print("Done")
else:
    segMetric_dists = np.load("segMetric_dists.npy", allow_pickle=True)
    segMetric_elChange = np.load("segMetric_elChange.npy", allow_pickle=True)

out = roadDraw(splitRoads.tolist(), len(splitRoads[0]), [75000, 75000, 2], [0,0,0])
p1 = Plot3D(out[0], marker_size=1, width=0.1,
            edge_color='w', symbol='-',
            parent=view.scene, connect=out[1], color=(1, 1, 0, 0.5), face_color=(0, 1, 1, 0.1))  # face_color=(0.2, 0.2, 1, 0.8)

#for i in range(length):
    #print(100 * (i / (length-1)))
    #plt.plot(splitRoads[i][0], splitRoads[i][1], color="blue", linewidth=1)
#plt.draw()
#plt.pause(0.000001)

routeNum = 500
countNum = 1000000
r = routeNum
count = countNum
routeMetric_q_log = []
routeMetric_qBest_log = []

def routeFinder(ev):
    global joints
    global splitRoads
    global r
    global count
    global routeNum
    global countNum
    global seg
    global sOE
    global segLog
    global sOELog
    global choLog
    global disLog
    global elcLog
    global p2
    global p3
    global p4
    global startTime
    global weights
    global routeMetric_q_log
    global routeMetric_qBest_log
    if r >= routeNum:
        r = -1
    r += 1
    ###
    ### route statistics
    ###
    if r != 0:
        routeMetric_dist = np.sum(disLog)
        routeMetric_grad = np.mean(np.array(elcLog) / np.array(disLog))
        if routeMetric_dist > 25*100 and count != 0:
            routeMetric_q = (1 / np.sum(np.abs(np.array(elcLog)))) ** 1
            routeMetric_q_log.append(np.sum(np.abs(np.array(elcLog))))
            routeMetric_qBest_log.append(np.amin(routeMetric_q_log))
            for seg in range(len(segLog)):
                #print(segLog[seg], sOELog[seg], choLog[seg])
                #print("data for this segment end", weights[segLog[seg]][sOELog[seg]])
                #print(weights[segLog[seg]][sOELog[seg]])
                weights[segLog[seg]][sOELog[seg]][choLog[seg]].append(routeMetric_q)
                #print(segLog[seg], sOELog[seg], choLog[seg], "/", len(weights[segLog[seg]][sOELog[seg]]))
            if len(routeMetric_q_log) > 1:
                p2.parent = None
                #p3.parent = None
                #p4.parent = None
            #print(countNum / ((time.time_ns() / (10 ** 9)) - startTime))
            out = roadDraw(splitRoads[segLog].tolist(), len(splitRoads[0]), [75000, 75000, 2], [0,0,200])
            p2 = Plot3D(out[0], marker_size=10, width=10,
                        edge_color='w', symbol='-',
                        parent=view.scene, connect=out[1],
                        color=(0.5, 0.5, 0, 1), face_color=(1, 0.5, 0, 1))  # face_color=(0.2, 0.2, 1, 0.8)
            if len(routeMetric_qBest_log) > 1 and routeMetric_qBest_log[-1] != routeMetric_qBest_log[-2]:
                print(routeMetric_qBest_log[-1])
                out = roadDraw(splitRoads[segLog].tolist(), len(splitRoads[0]), [75000, 75000, 2], [0, 0, 200])
                try:
                    p3.parent = None
                except:
                    pass
                p3 = Plot3D(out[0], marker_size=1, width=10,
                            edge_color='w', symbol='-',
                            parent=view.scene, connect=out[1],
                            color=(0.5, 0, 1, 1), face_color=(0.5, 0, 1, 1))  # face_color=(0.2, 0.2, 1, 0.8)
            #p3 = Plot3D(np.transpose([((np.cumsum(disLog)*1)).tolist(), (((np.cumsum(elcLog))*0.5) - 10000).tolist(), np.full( shape=len(elcLog), fill_value=500, dtype=np.int).tolist()]), marker_size=1, width=10,
                        #edge_color='w', symbol='-',
                        #parent=view.scene,
                        #color=(0.2, 1, 1, 0.5))  # face_color=(0.2, 0.2, 1, 0.8)
            #p4 = Plot3D(np.transpose([np.linspace(0, 10000, np.size(routeMetric_qBest_log)).tolist(), ((normalize(routeMetric_qBest_log)*500) - 10000).tolist(), np.full( shape=np.size(routeMetric_qBest_log), fill_value=500, dtype=np.int).tolist()]).tolist(), marker_size=1, width=10,
                        #edge_color='w', symbol='-',
                        #parent=view.scene,
                        #color=(0.2, 1, 1, 0.5))  # face_color=(0.2, 0.2, 1, 0.8)
            count = countNum
    if count >= countNum:
        count = -1
        segLog = []
        sOELog = []
        choLog = []
        disLog = []
        elcLog = []
        startTime = time.time_ns() / (10 ** 9)
        seg = 4500
        sOE = 0
        sOE = (-1) - sOE
    count += 1
    if np.size(joints[seg][sOE][0]) == 0:
        sOE = (-1) - sOE

    optionWeights = []
    for choice in range(len(weights[seg][sOE])):
        if weights[seg][sOE][choice] == []:
            optionWeights.append(1 + ((1/(10**10))*choice))
        else:
            optionWeights.append((1/(np.std(weights[seg][sOE][choice])+1)) * np.mean(weights[seg][sOE][choice]))
    optionWeights = addToOne(optionWeights)
    if len(optionWeights) > 1:
        choice = np.random.choice(np.size(joints[seg][sOE][0]), p=optionWeights)
    else:
        choice = 0
    segLog.append(seg)
    sOELog.append(sOE)
    choLog.append(choice)
    disLog.extend(segMetric_dists[seg])
    elcLog.extend(segMetric_elChange[seg])
    sOE, seg = (-1) - joints[seg][sOE][1][choice], joints[seg][sOE][0][choice]

timer = app.Timer()
timer.connect(routeFinder)
timer.start(0.00000000001)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()

