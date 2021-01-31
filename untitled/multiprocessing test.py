import time
import multiprocessing
import numpy as np
from igraph import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

joints = np.load("joints.npy", allow_pickle=True).tolist()

segMetric = np.load("segMetric.npy", allow_pickle=True)
segMetricNormal = np.load("segMetricNormal.npy", allow_pickle=True)
roadProperties = np.load("roadProperties.npy", allow_pickle=True)
properties = np.load("properties.npy", allow_pickle=True)

### GRAPH ###
g = Graph(directed=True)

try:
    graphFile = np.load("graph.npy", allow_pickle=True)
except:
    graphFile = [0,0,0,"#"]
if properties[5] == graphFile[3]:
    edgesToAdd, vertexToRoad, vertexCount = graphFile[0], graphFile[1], graphFile[2]
    for i in range(vertexCount + 1):
        g.add_vertices(1)
else:
    np.save("cacheWeight.npy", [[], [], []])
    np.save("cacheRoute.npy", [[], [], [], []])

    edgesToAdd = []

    log = []
    vertexLog = []
    segmentVertices = [[False, False] for i in range(len(splitRoadsWithHeights))]
    vertexCount = -1
    length = len(splitRoadsWithHeights)
    vertexToRoad = []
    for segment in range(length):
        if segment % np.floor(length / 20) == 0:
            print((segment / length)*100)
        for startOrEnd in range(0,2):
            connections = joints[segment][startOrEnd][0]
            startOrEnds = joints[segment][startOrEnd][1]
            for c in range(len(connections)):
                if (connections[c], startOrEnds[c]) not in log:
                    #print(connections[c], startOrEnds[c])
                    log.append((connections[c], startOrEnds[c]))
                    vertexCount += 1
                    g.add_vertices(1)
                    vertexToRoad.append([connections[c], startOrEnds[c]])
                    segmentVertices[connections[c]][startOrEnds[c]] = vertexCount

    for segment in range(length):
        if segment % np.floor(length / 20) == 0:
            print((segment / length)*100)
        for startOrEnd in range(-1, 1):
            connections = joints[segment][startOrEnd][0]
            startOrEnds = joints[segment][startOrEnd][1]
            for c in range(len(connections)):
                #if (connections[c], startOrEnds[c]) in log:
                edgesToAdd.append((segmentVertices[segment][startOrEnd], segmentVertices[connections[c]][startOrEnds[c]]))
    tally = 0
    for i in range(len(segmentVertices)):
        if segmentVertices[i][0] != False and segmentVertices[i][1] != False:
            if int(roadProperties[i][0]) == 0:
                edgesToAdd.append((segmentVertices[i][1], segmentVertices[i][0]))
            edgesToAdd.append((segmentVertices[i][0], segmentVertices[i][1]))
        else:
            tally += 1
    edgesToAdd = [i for i in edgesToAdd if i[0] != False and i[1] != False]
    np.save("graph.npy", np.array([edgesToAdd, vertexToRoad, vertexCount, properties[5]]))

g.add_edges(edgesToAdd)


def multiprocessing_func(id, Global, stuff, waypoints, goals):
    Global.startedIDs.append(id)
    ranges = properties[4]
    rang = stuff[0]
    steps = stuff[1]
    waypointsPoss = stuff[2]
    howManyVariable = stuff[3]
    length = stuff[4]

    min = False
    for a in range(waypointsPoss):
        vertexes = []
        notStaticCount = -1
        for i in range(length):
            if waypoints[i] == None:
                notStaticCount += 1
                vertexes.append(rang[int((a // (steps ** notStaticCount)) % steps)])
            else:
                vertexes.append(waypoints[i])
        possibleRoute = True
        route = []
        for i in range(length - 1):
            directions = g.get_shortest_paths(vertexes[i], to=vertexes[i+1], weights=[1 for i in range(len(edgesToAdd))], mode=OUT)[0]
            if len(directions) > 1:
                route.extend(directions)
            else:
                possibleRoute = False
                break
        if possibleRoute:
            arr = np.take(np.transpose(vertexToRoad)[0], route)
            arrDir = np.take(np.transpose(vertexToRoad)[1], route)
            # u = [x for x in set(arr) if arr.count(x) > 0]

            ###finding the route metrics...
            routeMetric = [0, 0, 0]  # distance, elevation up, elevation down
            for seg in range(len(arr)):
                if seg > 0 and arr[seg] != arr[seg - 1]:
                    routeMetric[0] += segMetric[arr[seg]][0][arrDir[seg]]
                    routeMetric[1] += segMetric[arr[seg]][1][arrDir[seg]]
                    routeMetric[2] += segMetric[arr[seg]][1][-1 - arrDir[seg]]
            score = []
            for o in range(len(goals)):
                score.append((ranges[o] + np.abs(goals[o] - routeMetric[o])) / ranges[o])
            score = np.mean(score)
            if min == False or score < min:
                min = score
                minRouteMetric = routeMetric
                minDirections = directions
                # print(i / length, score)
                optBatch = Global.optBatch
                optBatch[0][id] = (min)
                optBatch[1][id] = (minRouteMetric)
                optBatch[2][id] = (minDirections)
                Global.optBatch = optBatch
        optPerc = Global.optPerc
        optPerc[id] = (a + 1) / waypointsPoss
        Global.optPerc = optPerc


class OptProgThread(QThread):
    def run(self):
        global lastUpdateProg
        global Global
        while True:
            #print(Global.optBatch[0])
            if Global.optInterrupt == False and time.clock() - lastUpdateProg >= 0.25:
                lastUpdateProg = time.clock()
                percs = np.take(Global.optBatch[0], Global.startedIDs)
                if len(Global.startedIDs) > 0:
                    print(np.mean(percs))

if __name__ == '__main__':
    coreCount = multiprocessing.cpu_count() - 2

    global Global
    manager = multiprocessing.Manager()
    Global = manager.Namespace()
    Global.optBatch = [[None for i in range(coreCount)], [None for i in range(coreCount)], [None for i in range(coreCount)]] #scores, routeMetrics, directions
    Global.optPerc = [0 for i in range(coreCount)]
    Global.optInterrupt = False
    Global.startedIDs = []

    lastUpdateProg = time.clock()

    waypoints = [None, None, 1376, 1567]
    howManyVariable = sum([1 for i in waypoints if i == None])
    length = len(waypoints)

    optProgThread = OptProgThread()
    optProgThread.start()
    quality = ((vertexCount * 0.025) ** (1 / howManyVariable) * coreCount) / vertexCount
    goals = [30000, 0, 0]

    steps = int((vertexCount * quality) / coreCount)
    waypointsPoss = int(steps ** howManyVariable)
    distribution = np.linspace(0, vertexCount-1, num=steps, endpoint=True)
    spread = distribution[1] - distribution[0]
    spreadPerCore = spread / (coreCount)
    rangs = []
    for core in range(coreCount):
        rangs.append((distribution + (spreadPerCore * (core))).astype(int).tolist())

    processes = []

    for core in range(coreCount):
        stuff = [rangs[core], steps, waypointsPoss, howManyVariable, length]
        p = multiprocessing.Process(target=multiprocessing_func, args=(core, Global, stuff, waypoints, goals))
        p.daemon = True
        processes.append(p)
        processes[-1].start()

    shallIContinue = input()