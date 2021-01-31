import numpy as np
from numpy import int64
import sys
#import PyQt5
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt
from PyQt5.QtCore import QThread
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from igraph import *

import gpxpy
import gpxpy.gpx

import time

import inspect

import multiprocessing
from multiprocessing import Process

# Creating a new file:
# --------------------

gpx = gpxpy.gpx.GPX()


def makeGpx(y, x):
    # Create first track in our GPX:
    gpx_track = gpxpy.gpx.GPXTrack()
    gpx.tracks.append(gpx_track)

    # Create first segment in our GPX track:
    gpx_segment = gpxpy.gpx.GPXTrackSegment()
    gpx_track.segments.append(gpx_segment)

    # Create points:
    for i in range(len(x)):
        gpx_segment.points.append(gpxpy.gpx.GPXTrackPoint(x[i], y[i]))

    # You can add routes and waypoints, too...

    with open("output.gpx", "w") as f:
        f.write(gpx.to_xml())

### P L O T T I N G #############################################################
#################################################################################

import vispy
from vispy import app, visuals, scene
from vispy.util.quaternion import Quaternion
from vispy import color
from scipy.spatial.transform import Rotation as R
from vispy.scene.visuals import Text
from vispy.io import read_png
from functools import partial

import random

properties = np.load("properties.npy", allow_pickle=True)
boundBoxCenter = [np.mean([properties[0][0][0], properties[0][1][0]]),
                  np.mean([properties[0][0][1], properties[0][1][1]])]

boundBoxX = np.subtract(properties[0][1][0], properties[0][0][0])
boundBoxY = np.subtract(properties[0][1][1], properties[0][0][1])

RES = properties[1]

### DRAW ###

latitudeUnsquash = (1 + ((2 * np.sin(np.deg2rad((90 - properties[0][1][1]) / 2))) / (2 ** 0.5)))
def roadDraw(roads, dims, dimScal, dimDisp):
    # print("Drawing roads...")
    out = []
    totalN = 0
    heights = []
    x = []
    y = []
    for road in range(len(roads)):
        log = []
        if (x,y) not in log:
            log.append((x,y))
            x.extend(roads[road][0])
            y.extend(roads[road][1])
        if dims < 3:
            heights.extend(np.ones(len(roads[road][0])))
        # else:
        # heights.extend(roads[road][2])
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
    roadsConc = np.concatenate(roads, axis=1)
    if dims < 3:
        roadsConc = np.append(roadsConc, [heights], axis=0)  # heights.reshape((-1, len(heights)))
    for dim in range(len(roadsConc) - 1):
        if dim == 1:
            roadsConc[dim] = (roadsConc[dim] - boundBoxCenter[dim]) * latitudeUnsquash * dimScal[dim] + dimDisp[dim] #properties[0][0][1] + ((roadsConc[dim] - properties[0][0][1]) * latitudeUnsquash)
        else:
            roadsConc[dim] = (roadsConc[dim] - boundBoxCenter[dim]) * dimScal[dim] + dimDisp[dim]
    roadsConc[2] = roadsConc[2] * dimScal[2] + dimDisp[2]
    roadsTranspos = np.transpose(roadsConc)
    return roadsTranspos, connect, x, y
    # Plot3D(roadsTranspos, width=20, color='red',
    # edge_color='w', symbol='disc',
    # parent=view.scene,connect=connect, face_color=(0.2, 0.2, 1, 0.8)) #face_color=(0.2, 0.2, 1, 0.8)
    # print(roadsTranspos)

splitRoadsWithHeights = np.load("splitRoadsWithHeights.npy", allow_pickle=True)
splitRoadsWithHeights = splitRoadsWithHeights.tolist()
joints = np.load("joints.npy", allow_pickle=True).tolist()

segMetric = np.load("segMetric.npy", allow_pickle=True)
segMetricNormal = np.load("segMetricNormal.npy", allow_pickle=True)
roadProperties = np.load("roadProperties.npy", allow_pickle=True)

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

#print(edgesToAdd)
g.add_edges(edgesToAdd)

lastUpdateProg = time.clock()
progTask = False
lastPerc = 999999999
def updateProg(perc):
    global lastUpdateProg
    global prog2
    global prog1
    global progTask
    global lastPerc
    return None
    if perc < lastPerc:
        stack = inspect.stack()
        try:
            the_class = stack[1][0].f_locals["self"].__class__.__name__
            the_method = stack[1][0].f_code.co_name
            progTask = "{}.{}".format(the_class, the_method)
        except:
            progTask = ""
        if Global.onMainMenu:
            prog2.reset()
            prog2.setFormat(progTask + "%p%")
        else:
            prog1.reset()
            prog1.setFormat(progTask + "%p%")

    lastPerc = perc
    if perc == 1:
        if Global.onMainMenu:
            prog2.reset()
        else:
            prog1.reset()
    elif time.clock() - lastUpdateProg >= 0.5:
        if Global.onMainMenu:
            prog2.setValue(perc*100)
        else:
            prog1.setValue(perc*100)
        lastUpdateProg = time.clock()

#Weights:
#Distance, higher weight to prioritise shorter distance
#Height increase, higher wait to prioritise a route with less increase

class UpdateRouteThread(QRunnable):
    global r1a
    global r1b
    def __init__(self, out):
        self.out = out
        self.routeLatLon = []
        super(UpdateRouteThread, self).__init__()
    def run(self):
        r1a.parent = view.scene
        r1b.parent = view.scene
        if self.out != "False":
            updateProg(0)
            route, routeMetric = self.out
            self.out = roadDraw(np.take(splitRoadsWithHeights, route, axis=0).tolist(), len(roads[0]), [111000, 111000, 3], [0, 0, 10])
            self.routeLatLon = [self.out[2], self.out[3]]
            if len(self.routeLatLon) == 0:
                l1.text = "graph traversal error (missing edges?)"
                posLog = posLog[:-1]
                waypoints[0], waypoints[1] = waypoints[0][:-1], waypoints[1][:-1]
                colors = np.array([[1, 0.4, 0]] * len(posLog))
                m1.set_data(np.array(posLog), face_color=colors, symbol='o', size=7.5, edge_width=7.5,
                            edge_color='green')
            else:
                l1.text = str(np.round(routeMetric[0] / 1000, 2)) + "km   " + str(
                    np.round(routeMetric[1], 2)) + "m up   " + str(np.round(routeMetric[2], 2)) + "m down   "
                r1a.set_data(pos=self.out[0], connect=self.out[1], width=1000, color=(0.5, 0.5, 1, 1))#, marker_size=7.5, width=1000, edge_color='w',
                            #symbol='x',
                            #color=(0.5, 0.5, 1, 1), face_color=(0.5, 0.5, 1, 1))
                r1b.set_data(self.out[0], connect=self.out[1], width=1000, marker_size=7.5, edge_color='w',
                            symbol='x',
                            color=(0.5, 0.5, 1, 1), face_color=(0.5, 0.5, 1, 1))
        else:
            r1a.parent = None
            r1b.parent = None
        updateCanvas()
        updateProg(1)

# def highwayTypeToWeight(htype, Global):
#     if htype in ["motorway", "motorway_link"]:
#         return Global.UITypeWeights[0]
#     elif htype in ["trunk", "trunk_link"]:
#         return Global.UITypeWeights[1]
#     elif htype in ["primary", "primary_link"]:
#         return Global.UITypeWeights[2]
#     elif htype in ["secondary", "secondary_link"]:
#         return Global.UITypeWeights[3]
#     elif htype in ["tertiary", "tertiary_link"]:
#         return Global.UITypeWeights[4]
#     elif htype in ["unclassified", "road"]:
#         return Global.UITypeWeights[5]
#     elif htype in ["residential", "living_street"]:
#         return Global.UITypeWeights[6]
#     elif htype in ["cycleway"]:
#         return Global.UITypeWeights[7]
#     elif htype in ["footway", "bridleway", "steps", "corridor", "path", "sidewalk", "crossing", "pedestrian"]:
#         return Global.UITypeWeights[8]
#     elif htype in ["track"]:
#         return Global.UITypeWeights[9]
#     else:
#         return Global.UITypeWeights[5]

class WeightBuilderThread():
    def __init__(self, edgesToAdd, weights, typeWeights, drawOrNot=False):
        self.edgesToAdd = edgesToAdd
        self.weights = weights
        self.typeWeights = typeWeights
        self.drawOrNot = drawOrNot
        self.currentWeights = []
        super(WeightBuilderThread, self).__init__()
    def run(self):
        print("WeightBuilderThread: run")
        if len(waypoints) > 1:
            length = len(self.edgesToAdd)
            result = [0 for i in range(length)]
            i = -1
            weights = np.array(self.weights)
            typeWeights = np.array(self.typeWeights)
            minPossible = np.sum((weights < 0) * weights) / 3
            # maxPossible = np.sum((weights>0)*weights)
            cacheWeight = np.load("cacheWeight.npy", allow_pickle=True).tolist()
            indexes = [x for x in range(len(cacheWeight[0])) if np.array_equal(cacheWeight[0][x], [weights, typeWeights])]
            if len(indexes) > 0:
                self.currentWeights = cacheWeight[1][indexes[0]]
            else:
                x = minPossible
                for edge in self.edgesToAdd:
                    i += 1
                    updateProg(i / (length - 1))
                    for startOrEnd in range(-1, 1):
                        segNum = vertexToRoad[edge[startOrEnd]]
                        d = highwayTypeToWeight(roadProperties[segNum[0]][3])
                        print(d)
                        if segNum[1] == -1 and segNum[1] == 0:  ##in reverse
                            a = (segMetricNormal[segNum[0]][0][1] * weights[0])
                            b = (segMetricNormal[segNum[0]][1][1] * weights[1])
                            c = (segMetricNormal[segNum[0]][1][0] * weights[2])
                            result[i] = (x + (a + b + c) / 3) / d
                        else:  ##forwards
                            a = (segMetricNormal[segNum[0]][0][0] * weights[0])
                            b = (segMetricNormal[segNum[0]][1][0] * weights[1])
                            c = (segMetricNormal[segNum[0]][1][1] * weights[2])
                            result[i] = (x + a + b + c) / d
                        # if result[i] < 0:
                        # print(result[i], segMetricNormal[start[0]].tolist(), weights)
                cacheWeight[0].append([weights, typeWeights])
                cacheWeight[1].append(result)
                np.save("cacheWeight.npy", cacheWeight)
                self.currentWeights = result
        if self.drawOrNot:
            routeBuilderThread = RouteBuilderThread(waypoints, self.currentWeights, True)
            QThreadPool.globalInstance().start(routeBuilderThread)

def WeightBuilder(edgesToAdd, Global, waypoints, drawOrNot=False):
    currentWeights = []
    weights = Global.UIMetricWeights
    typeWeights = Global.UITypeWeights
    print("WeightBuilderFunction: run")
    Global.perc = 0
    if len(waypoints) > 1:
        length = len(edgesToAdd)
        result = [0 for i in range(length)]
        i = -1
        weights = np.array(weights)
        typeWeights = np.array(typeWeights)
        minPossible = (np.sum((weights < 0) * weights)) / 3
        # maxPossible = np.sum((weights>0)*weights)
        cacheWeight = Global.cacheWeight
        indexes = [x for x in range(len(cacheWeight[0])) if np.array_equal(cacheWeight[0][x], weights) and np.array_equal(cacheWeight[1][x], typeWeights)]
        if len(indexes) > 0:
            currentWeights = cacheWeight[2][indexes[0]]
        else:
            for edge in edgesToAdd:
                i += 1
                #updateProgThread = UpdateProgThread(i / (length - 1))
                #QThreadPool.globalInstance().start(updateProgThread)
                #startTime = time.clock()
                for startOrEnd in range(-1, 1):
                    segNum = vertexToRoad[edge[startOrEnd]]
                    x = minPossible
                    d = typeWeights[int(roadProperties[segNum[0]][3])]
                    if segNum[1] == -1 and segNum[1] == 0:  ##in reverse
                        a = (segMetricNormal[segNum[0]][0][1] * weights[0])
                        b = (segMetricNormal[segNum[0]][1][1] * weights[1])
                        c = (segMetricNormal[segNum[0]][1][0] * weights[2])
                        result[i] = (x + ((a + b + c) / 3)) / (d+0.0000000001)
                    else:  ##forwards
                        a = (segMetricNormal[segNum[0]][0][0] * weights[0])
                        b = (segMetricNormal[segNum[0]][1][0] * weights[1])
                        c = (segMetricNormal[segNum[0]][1][1] * weights[2])
                        result[i] = (x + ((a + b + c) / 3)) / (d+0.0000000001)
                Global.perc = i / (length - 1)
                    # if result[i] < 0:
                    # print(result[i], segMetricNormal[start[0]].tolist(), weights)
                #print(time.clock() - startTime)
            cacheWeight[0].append(weights)
            cacheWeight[1].append(typeWeights)
            cacheWeight[2].append(result)
            Global.cacheWeight = cacheWeight
            currentWeights = result
    if drawOrNot:
        Global.perc = 0
        print("RouteBuilderFunction: run")
        #Global.perc = 0
        # global Global
        # updateProg(0, Global)
        ###finding the directions to take through each section...
        route = []
        cacheRoute = Global.cacheRoute
        for section in range(len(waypoints[0]) - 1):
            indexes = [x for x in range(len(cacheRoute[0])) if
                       cacheRoute[0][x] == [waypoints[0][section], waypoints[0][section + 1]]]
            done = False
            for i in indexes:
                if Global.UIMetricWeights == cacheRoute[2][i] and Global.UITypeWeights == cacheRoute[3][i]:
                    done = True
                    route.append(cacheRoute[1][i])
            if not done:
                directions = g.get_shortest_paths(waypoints[0][section], to=waypoints[0][section + 1],
                                                  weights=currentWeights, mode=OUT)[0]
                print(waypoints[0][section], waypoints[0][section + 1])
                if len(directions) > 0:
                    route.append(directions)
                    cacheRoute[0].append([waypoints[0][section], waypoints[0][section + 1]])
                    cacheRoute[1].append(directions)
                    cacheRoute[2].append(Global.UIMetricWeights)
                    cacheRoute[3].append(Global.UITypeWeights)
        Global.cacheRoute = cacheRoute
        print(len(route), "route part count")
        # route = []
        # for section in range(len(waypoints[0]) - 1):
        #     #if section > 0:
        #         #if metricWeights[section] != metricWeights[section-1]:
        #             #weights = WeightBuilder(edgesToAdd, vertexToRoad, segMetric, metricWeights[section])
        #     #else:
        #     sectionRoute = g.get_shortest_paths(waypoints[0][section], to=waypoints[0][section+1], weights=weights, mode=OUT)
        #     if len(sectionRoute[0]) > 1:
        #         route.append(sectionRoute[0])
        #     else:
        #         return False

        ###converting the node sections into a road route...
        combinedRoute = []
        for i in range(len(route)):
            combinedRoute.extend(route[i])
        #print("combinedRoute", combinedRoute)
        if len(combinedRoute) > 1:
            # combinedRoute = route
            arr = np.take(np.transpose(vertexToRoad)[0], combinedRoute)
            arrDir = np.take(np.transpose(vertexToRoad)[1], combinedRoute)
            # u = [x for x in set(arr) if arr.count(x) > 0]

            ###finding the route metrics...
            routeMetric = [0, 0, 0]  # distance, elevation up, elevation down
            for seg in range(len(arr)):
                if seg > 0 and arr[seg] != arr[seg - 1]:
                    routeMetric[0] += segMetric[arr[seg]][0][arrDir[seg]]
                    routeMetric[1] += segMetric[arr[seg]][1][arrDir[seg]]
                    routeMetric[2] += segMetric[arr[seg]][1][-1 - arrDir[seg]]
            Global.out = arr.tolist()
            Global.routeMetric = routeMetric
        Global.perc = 1


def normalize(values, bounds):
    a = bounds[1][1] - bounds[1][0]
    b = (bounds[0][1] - bounds[0][0])
    return [bounds[1][0] + (x - bounds[0][0]) * a / b for x in values]

class GlideCameraThread(QThread):
    def __init__(self, steps, location, rotation):
        self.steps = steps
        self.location = location
        self.rotation = rotation
        super(GlideCameraThread, self).__init__()
    def run(self):
        rang = np.linspace(0, np.pi, self.steps)
        rang = (np.sin(rang - (np.pi/2)) + 1) / 2

        locSteps = []
        for i in range(3):
            locSteps.append(normalize(rang, [[0, 1], [view.camera.center[i], self.location[i]]]))
        locSteps = np.transpose(locSteps)

        r = R.from_quat(
            [view.camera.rotation.w, view.camera.rotation.x, view.camera.rotation.y, view.camera.rotation.z])
        startRot = r.as_euler('zyx', degrees=True)
        rotSteps = []
        for i in range(3):
            rotSteps.append(normalize(rang, [[0, 1], [startRot[i], self.rotation[i]]]))
        rotSteps = np.transpose(rotSteps)
        for i in range(self.steps):
            view.camera.center = locSteps[i]
            view.camera.rotation = Quaternion.create_from_euler_angles(*rotSteps[i], degrees=True)
    def __del__(self):
        self.wait()


def UISliderChange():
    global progThread
    global Global
    UIMetricWeights = Global.UIMetricWeights
    UITypeWeights = Global.UITypeWeights
    UIMetricWeights[0] = (s1.value()/100)
    UIMetricWeights[1] = (s2.value() / 100)
    UIMetricWeights[2] = (s3.value() / 100)
    UITypeWeights[0] = (s5.value() / 100)
    UITypeWeights[1] = (s6.value() / 100)
    UITypeWeights[2] = (s7.value() / 100)
    UITypeWeights[3] = (s8.value() / 100)
    UITypeWeights[4] = (s9.value() / 100)
    UITypeWeights[5] = (s10.value() / 100)
    UITypeWeights[6] = (s11.value() / 100)
    UITypeWeights[7] = (s12.value() / 100)
    UITypeWeights[8] = (s13.value() / 100)
    UITypeWeights[9] = (s14.value() / 100)
    Global.UIMetricWeights = UIMetricWeights
    Global.UITypeWeights = UITypeWeights

    drawOrNot = False
    if len(waypoints[0]) > 1:
        drawOrNot = True

    ### threading jam time ###
    p = Process(target=WeightBuilder, args=(edgesToAdd, Global, waypoints, drawOrNot))
    p.daemon = True
    weightThreads.append(p)
    for i in range(len(weightThreads[:-1])):
        if weightThreads[i].is_alive():
            print("I just killed a process!")
            Global.interrupt = True
            weightThreads[i].terminate()
    weightThreads[-1].start()
    progThread = ProgThread(drawOrNot)
    progThread.start()
    progThread.progress_update.connect(updateProgressBar)
    progThread.route_update.connect(updateRoute)
    ### ################## ###

def UISliderChange2():
    global cacheRoute
    print(cacheRoute[2])
    print(s2.value())

def OptSliderChange():
    print(opts1.value())

routeLatLon = []
def UIButtonPress():
    global posLog
    global m1
    global r1a
    global waypoints
    r1a.parent = None
    r1b.parent = None
    posLog = []
    m1.parent = None
    l1.text = ''
    waypoints = [[], []]
    for i in range(len(waypointWidgetHandler.waypoints)):
        waypointWidgetHandler.remove(0)

def UIButtonPress2():
    routeLatLon = updateRouteThread.routeLatLon
    try:
        if len(routeLatLon[0]) > 0:
            makeGpx(routeLatLon[0], routeLatLon[1])
            print("GPX Saved.")
    except:
        pass

### OPTIMISER ###
def multiprocessing_func(id, Global, stuff, waypoints, goals):
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
            try:
                directions = g.get_shortest_paths(vertexes[i], to=vertexes[i + 1], weights=[1 for i in range(len(edgesToAdd))],
                                    mode=OUT)[0]
            except:
                break
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
                minDirections = arr
                # print(i / length, score)
                optBatch = Global.optBatch
                optBatch[0][id] = (min)
                optBatch[1][id] = (minRouteMetric)
                optBatch[2][id] = (minDirections)
                Global.optBatch = optBatch
        optPerc = Global.optPerc
        optPerc[id] = (a + 1) / waypointsPoss
        Global.optPerc = optPerc

def UIButtonPress3():
    global optimiserThreads
    global waypoints
    global processes
    if optb1.isChecked():
        if len(waypoints[0]) > 1:
            optWaypoints = waypoints[0]
            count = 0
            for i in range(len(waypointWidgetHandler.waypoints)):
                if waypointWidgetHandler.waypoints[i][0] == True:
                    optWaypoints[i] = None
                    count += 1
            if count > 0:
                goals = [float(opte1.text()), float(opte2.text()), float(opte3.text())]
                quality = opts1.value() / 100
                optimiserThread = OptimiserThread(optWaypoints, quality, goals)
                optimiserThreads.append(optimiserThread)
                optimiserThreads[-1].start()
    else:
        Global.optInterrupt = True
        for i in range(len(processes)):
            processes[i].terminate()
        processes = []


def UIButtonPress4():
    s1.setValue(0)
    s2.setValue(0)
    s3.setValue(0)
    UISliderChange()

def clearLayout(layout):
  while layout.count():
    child = layout.takeAt(0)
    if child.widget():
      child.widget().deleteLater()

r = 0
tau = 0
toggle1 = False
length = len(splitRoadsWithHeights)
roads = []
indexes = []
roads = splitRoadsWithHeights
def update(ev):
    if view.camera.center[2] < 0:
        previousPos = view.camera.center
        view.camera.center = (previousPos[0], previousPos[1], 0)
    global tau
    global r
    global roads
    global p1
    global toggle1
    global toggle1x
    global toggle1y
    global toggle1height
    global locSteps
    global rotSteps
    global steps
    global menuP1
    global menuL1
    global menuL2
    global menuL3
    length = len(roads)

    ### GLIDE CAMERA CODE, ported
    if tau == 0:
        view.camera.scale_factor = 90490.23177079092
        toggle1height = ((boundBoxX)) * 111000 * 1.225  # fov?
        steps, location, rotation = 100, (0, 0, toggle1height*1.25), (0, 0, 0)

        rang = np.linspace(0, np.pi, steps)
        rang = (np.sin(rang - (np.pi / 2)) + 1) / 2
        rang2 = np.linspace(0, np.pi, steps)
        rang2 = np.arctan(rang2*10)

        locSteps = []
        for i in range(3):
            locSteps.append(normalize(rang, [[0, 1], [view.camera.center[i], location[i]]]))
        locSteps = np.transpose(locSteps)

        rot = R.from_quat(
            [view.camera.rotation.w, view.camera.rotation.x, view.camera.rotation.y, view.camera.rotation.z])
        startRot = rot.as_euler('zyx', degrees=True)
        rotSteps = []
        for i in range(3):
            rotSteps.append(normalize(rang2, [[0, rang2[-1]], [startRot[i], rotation[i]]]))
        rotSteps = np.transpose(rotSteps)
    if tau < steps:
        view.camera.center = locSteps[tau]
        view.camera.rotation = Quaternion.create_from_euler_angles(*rotSteps[tau], degrees=True)
    if tau == steps:
        menuP1.parent = None
        menuL1.parent = None
        menuL2.parent = None
        menuL3.parent = None

    increment = int(np.ceil(length / 100))
    if (tau+1) % 5000000000000000 == 0:
        if r >= length:
            r = 0
        # print(t, joints[t])
        out = roadDraw(roads[r:r + increment], len(roads[0]), [111000, 111000, 3], [0,0,10])
        print("---")
        print(joints[r:r+increment])
        try:
            print(vertexToRoad.index([r,0]), vertexToRoad.index([r,-1]))
        except:
            print("i can't find those nodes")
        print(edgesToAdd)
        print(len(edgesToAdd))
        if r != 0:
            p1.parent = None
        p1 = Plot3D(out[0], marker_size=10, width=20, color='blue',
                    edge_color='w', symbol='-',
                    parent=view.scene, connect=out[1], face_color=(1, 0.2, 0.2, 0.8))  # face_color=(0.2, 0.2, 1, 0.8)
        r += increment
    tau += 1
    if toggle1 == True and (view.camera.center != (toggle1x, toggle1y, toggle1height) or view.camera.scale_factor != 90490.23177079092):
        view.camera.scale_factor = 90490.23177079092
        view.camera.center = (toggle1x, toggle1y, toggle1height)
    #r = R.from_quat([view.camera.rotation.w, view.camera.rotation.x, view.camera.rotation.y, view.camera.rotation.z])
    #print(r.as_euler('zyx', degrees=True))

### DRAW ###

class WaypointWidgetHandler:
    global waypointWidget

    def __init__(self):
        self.waypoints = []  # For each waypoint: [ToRandomize,]
        self.backstore = []

    def remove(self, index):
        del self.waypoints[index]
        self.update()

    def add(self):
        ###index isn't useful at the moment, we only ever add a new point at the end of the current waypoints
        self.waypoints.append([False])
        self.update()

    def update(self):
        for i in reversed(range(waypointWidget.layout().count())):
            waypointWidget.layout().itemAt(i).widget().setParent(None)
        self.backstore = []
        for i in range(len(self.waypoints)):
            self.backstore.append([QLabel(), QPushButton()])
            self.backstore[-1][0].setStyleSheet("QLabel {background-color:white}")
            self.backstore[-1][0].setText(str(i))

            self.backstore[-1][1].toggle()
            self.backstore[-1][1].setCheckable(True)
            self.backstore[-1][1].setStyleSheet("QPushButton"
                                                "{"
                                                "background-color : white;"
                                                "}"
                                                "QPushButton::pressed"
                                                "{"
                                                "background-color : green;"
                                                "}")
            self.backstore[-1][1].setText('')
            self.backstore[-1][1].clicked.connect(partial(self.interactionHandler, i))
            waypointWidget.layout().addWidget(self.backstore[-1][0], i, 0)
            waypointWidget.layout().addWidget(self.backstore[-1][1], i, 1)

    def interactionHandler(self, i):
        self.waypoints[i][0] = self.backstore[i][1].isChecked()
waypointWidgetHandler = WaypointWidgetHandler()

def PlannerUI():
    global waypointWidgetHandler
    global update
    global timer
    global view
    global m1
    global r1a
    global r1b
    global r1s
    global prog1
    global l1
    global s1
    global s2
    global s3
    global s4
    global s5
    global s6
    global s7
    global s8
    global s9
    global s10
    global s11
    global s12
    global s13
    global s14
    global optb1
    global waypointWidget
    global opte1
    global opte2
    global opte3
    global opts1
    Global.onMainMenu = False

    w.setWindowOpacity(1)

    widget = QWidget()
    w.setCentralWidget(widget)
    widget.setLayout(QGridLayout())

    #vispyQT = canvas.native
    #vispyQTsp = vispyQT.sizePolicy()
    vispyQTsp.setVerticalStretch(255)
    # vispyQTsp.setHorizontalStretch(255)
    vispyQT.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    widget.layout().addWidget(vispyQT, 1, 0, 17, 5)

    widget.layout().setColumnStretch(0, 2)
    widget.layout().setColumnStretch(1, 2)
    widget.layout().setColumnStretch(2, 2)
    widget.layout().setColumnStretch(3, 0)
    widget.layout().setColumnStretch(4, 1)
    widget.layout().setRowStretch(0, 0)
    widget.layout().setRowStretch(1, 0)
    widget.layout().setRowStretch(2, 0)
    widget.layout().setRowStretch(3, 0)
    widget.layout().setRowStretch(4, 0)
    widget.layout().setRowStretch(5, 0)
    widget.layout().setRowStretch(5, 255)
    widget.layout().setRowStretch(6, 0)

    widget.layout().addWidget(prog1, 0, 0, 1, 3)

    waypointWidget = QWidget()
    waypointWidget.setLayout(QGridLayout())
    widget.layout().addWidget(waypointWidget, 4, 4)

    optimiserWidget = QWidget()
    optimiserWidget.setLayout(QGridLayout())
    widget.layout().addWidget(optimiserWidget, 5, 4, 2, 2)

    optl1 = QLabel()
    optl1.setStyleSheet("QLabel {background-color:white}")
    optl1.setText("Distance")
    optimiserWidget.layout().addWidget(optl1, 0,0)

    opte1 = QLineEdit()
    opte1.setStyleSheet("QSlider {background-color:black}")
    onlyDouble = QDoubleValidator()
    opte1.setValidator(onlyDouble)
    opte1.setText('10')
    optimiserWidget.layout().addWidget(opte1, 0,1)

    optd1 = QDial()
    optd1.setStyleSheet("QDial {background-color:black}")
    optd1.setMinimum(0)
    optd1.setMaximum(100)
    optd1.setValue(100)
    optd1.sliderReleased.connect(OptSliderChange)
    optimiserWidget.layout().addWidget(optd1, 0,2)

    optl2 = QLabel()
    optl2.setStyleSheet("QLabel {background-color:white}")
    optl2.setText("Elevation gain")
    optimiserWidget.layout().addWidget(optl2, 1,0)

    opte2 = QLineEdit()
    opte2.setStyleSheet("QSlider {background-color:black}")
    opte2.setValidator(onlyDouble)
    opte2.setText('10')
    optimiserWidget.layout().addWidget(opte2, 1,1)

    optd2 = QDial()
    optd2.setStyleSheet("QDial {background-color:black}")
    optd2.setMinimum(0)
    optd2.setMaximum(100)
    optd2.setValue(100)
    optd2.sliderReleased.connect(OptSliderChange)
    optimiserWidget.layout().addWidget(optd2, 1,2)

    optl3 = QLabel()
    optl3.setStyleSheet("QLabel {background-color:white}")
    optl3.setText("Elevation loss")
    optimiserWidget.layout().addWidget(optl3, 2,0)

    opte3 = QLineEdit()
    opte3.setStyleSheet("QSlider {background-color:black}")
    opte3.setValidator(onlyDouble)
    opte3.setText('10')
    optimiserWidget.layout().addWidget(opte3, 2,1)

    optd3 = QDial()
    optd3.setStyleSheet("QDial {background-color:black}")
    optd3.setMinimum(0)
    optd3.setMaximum(100)
    optd3.setValue(100)
    optd3.sliderReleased.connect(OptSliderChange)
    optimiserWidget.layout().addWidget(optd3, 2,2)

    optl4 = QLabel()
    optl4.setStyleSheet("QLabel {background-color:white}")
    optl4.setText("Quality")
    optimiserWidget.layout().addWidget(optl4, 3,0)

    opts1 = QSlider(Qt.Horizontal)
    opts1.setStyleSheet("QSlider {background-color:white}")
    opts1.setMinimum(1)
    opts1.setMaximum(100)
    opts1.setTickInterval(1)
    optimiserWidget.layout().addWidget(opts1, 3, 1, 1, 2)

    optb1 = QPushButton()
    optb1.setText('Optimise')
    optb1.clicked.connect(UIButtonPress3)
    optb1.toggle()
    optb1.setCheckable(True)
    #b3.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    optimiserWidget.layout().addWidget(optb1, 4, 0, 1, 3)

    ###
    s5 = QSlider(Qt.Horizontal)
    s5.setStyleSheet("QSlider {background-color:white}")
    s5.setMinimum(0)
    s5.setMaximum(100)
    s5.setTickInterval(1)
    s5.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s5, 7, 4)


    l5 = QLabel()
    l5.setStyleSheet("QLabel {background-color:white}")
    l5.setText("Motorway")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l5, 7, 3)

    s6 = QSlider(Qt.Horizontal)
    s6.setStyleSheet("QSlider {background-color:white}")
    s6.setMinimum(0)
    s6.setMaximum(100)
    s6.setTickInterval(1)
    s6.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s6, 8, 4)

    l6 = QLabel()
    l6.setStyleSheet("QLabel {background-color:white}")
    l6.setText("Trunk")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l6, 8, 3)

    s7 = QSlider(Qt.Horizontal)
    s7.setStyleSheet("QSlider {background-color:white}")
    s7.setMinimum(0)
    s7.setMaximum(100)
    s7.setTickInterval(1)
    s7.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s7, 9, 4)

    l7 = QLabel()
    l7.setStyleSheet("QLabel {background-color:white}")
    l7.setText("Primary")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l7, 9, 3)

    s8 = QSlider(Qt.Horizontal)
    s8.setStyleSheet("QSlider {background-color:white}")
    s8.setMinimum(0)
    s8.setMaximum(100)
    s8.setTickInterval(1)
    s8.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s8, 10, 4)

    l8 = QLabel()
    l8.setStyleSheet("QLabel {background-color:white}")
    l8.setText("Secondary")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l8, 10, 3)

    s9 = QSlider(Qt.Horizontal)
    s9.setStyleSheet("QSlider {background-color:white}")
    s9.setMinimum(0)
    s9.setMaximum(100)
    s9.setTickInterval(1)
    s9.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s9, 11, 4)

    l9 = QLabel()
    l9.setStyleSheet("QLabel {background-color:white}")
    l9.setText("Tertiary")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l9, 11, 3)

    s10 = QSlider(Qt.Horizontal)
    s10.setStyleSheet("QSlider {background-color:white}")
    s10.setMinimum(0)
    s10.setMaximum(100)
    s10.setTickInterval(1)
    s10.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s10, 12, 4)

    l10 = QLabel()
    l10.setStyleSheet("QLabel {background-color:white}")
    l10.setText("Unclassified")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l10, 12, 3)

    s11 = QSlider(Qt.Horizontal)
    s11.setStyleSheet("QSlider {background-color:white}")
    s11.setMinimum(0)
    s11.setMaximum(100)
    s11.setTickInterval(1)
    s11.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s11, 13, 4)

    l11 = QLabel()
    l11.setStyleSheet("QLabel {background-color:white}")
    l11.setText("Residential")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l11, 13, 3)

    s12 = QSlider(Qt.Horizontal)
    s12.setStyleSheet("QSlider {background-color:white}")
    s12.setMinimum(0)
    s12.setMaximum(100)
    s12.setTickInterval(1)
    s12.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s12, 14, 4)

    l12 = QLabel()
    l12.setStyleSheet("QLabel {background-color:white}")
    l12.setText("Cycleway")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l12, 14, 3)

    s13 = QSlider(Qt.Horizontal)
    s13.setStyleSheet("QSlider {background-color:white}")
    s13.setMinimum(0)
    s13.setMaximum(100)
    s13.setTickInterval(1)
    s13.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s13, 15, 4)

    l13 = QLabel()
    l13.setStyleSheet("QLabel {background-color:white}")
    l13.setText("Footway")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l13, 15, 3)

    s14 = QSlider(Qt.Horizontal)
    s14.setStyleSheet("QSlider {background-color:white}")
    s14.setMinimum(0)
    s14.setMaximum(100)
    s14.setTickInterval(1)
    s14.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s14, 16, 4)

    l14 = QLabel()
    l14.setStyleSheet("QLabel {background-color:white}")
    l14.setText("Tracks")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l14, 16, 3)
    ###

    s5.setValue(50)
    s6.setValue(50)
    s7.setValue(50)
    s8.setValue(50)
    s9.setValue(50)
    s10.setValue(50)
    s11.setValue(50)
    s12.setValue(50)
    s13.setValue(50)
    s14.setValue(50)

    b1 = QPushButton()
    b1.setText('Start afresh')
    b1.clicked.connect(UIButtonPress)
    #b1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(b1, 17, 0)

    b2 = QPushButton()
    b2.setText('Export as GPX')
    b2.clicked.connect(UIButtonPress2)
    #b2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(b2, 17, 1)

    l4 = QLabel()
    l4.setStyleSheet("QLabel {background-color:white}")
    l4.setText("                                                   Time of day")
    #l1.move(0,0)
    #l1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l4, 17, 3)

    s4 = QSlider(Qt.Horizontal)
    s4.setStyleSheet("QSlider {background-color:white}")
    s4.setMinimum(-100)
    s4.setMaximum(100)
    s4.setTickInterval(1)
    s4.sliderReleased.connect(UISliderChange2)
    widget.layout().addWidget(s4, 17, 4)

    l1 = QLabel()
    l1.setStyleSheet("QLabel {background-color:white}")
    l1.setText("Maximise Distance -> Minimise Distance")
    #l1.move(0,0)
    #l1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l1, 0, 3)

    s1 = QSlider(Qt.Horizontal)
    s1.setStyleSheet("QSlider {background-color:white}")
    s1.setMinimum(-100)
    s1.setMaximum(100)
    s1.setTickInterval(1)
    s1.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s1, 0, 4)

    l2 = QLabel()
    l2.setStyleSheet("QLabel {background-color:white}")
    l2.setText("Maximise height gain -> Minimise height gain")
    #l1.move(0,0)
    #l2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l2, 1, 3)

    s2 = QSlider(Qt.Horizontal)
    s2.setStyleSheet("QSlider {background-color:white}")
    s2.setMinimum(-100)
    s2.setMaximum(100)
    s2.setTickInterval(1)
    s2.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s2, 1, 4)

    l3 = QLabel()
    l3.setStyleSheet("QLabel {background-color:white}")
    l3.setText("Maximise height loss -> Minimise height loss")
    #l1.move(0,0)
    #l3.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(l3, 2, 3)

    s3 = QSlider(Qt.Horizontal)
    s3.setStyleSheet("QSlider {background-color:white}")
    s3.setMinimum(-100)
    s3.setMaximum(100)
    s3.setTickInterval(1)
    s3.sliderReleased.connect(UISliderChange)
    widget.layout().addWidget(s3, 2, 4)

    b4 = QPushButton()
    b4.setText('Reset Weights')
    b4.clicked.connect(UIButtonPress4)
    #b3.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    widget.layout().addWidget(b4, 3, 4)

    r = 0
    length = len(splitRoadsWithHeights)
    roads = []
    indexes = []
    roads = splitRoadsWithHeights

    out = roadDraw(roads, len(roads[0]), [111000, 111000, 3], [0, 0, 10])
    a1 = Plot3D(out[0], marker_size=1, width=20, color=(1, 0.125, 0.125, 1),
                edge_color='w', symbol='-',
                parent=view.scene, connect=out[1], face_color=(1, 0.125, 0.125, 0.1))  # face_color=(0.2, 0.2, 1, 0.8)
    Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)
    m1 = Scatter3D(parent=view.scene)
    r1a = Arrow3D(pos=np.array([[0, 0, 0]]), parent=view.scene)
    r1b = Plot3D([[0, 0, 0]], parent=view.scene)
    r1s = Plot3D([[0, 0, 0]], parent=view.scene)

    ############################################################
    ############################################################
    ############################################################

    l1 = Text('', bold=True, parent=view, color='white')
    l1.font_size = 12
    l1.pos = canvas.size[0] // 3, canvas.size[1] // 4

    timer.disconnect()
    timer.connect(update)
    timer.start(0.0000001)



def MenuUIButtonPress():
    #app.quit()
    #clearLayout(menuWidget.layout())
    PlannerUI()

def MenuUIButtonPress2():
    print("exit?")

class BaseMapperThread(QThread):
    def run(self):
        basemaps = np.load("basemaps.npy", allow_pickle=True)
        length = len(basemaps) - 1
        x = []
        y = []
        z = []
        xMin = None
        xMax = None
        yMin = None
        yMax = None
        for b in range(length):
            basemap = basemaps[b]
            print("basemapper", b, "/", length, "   ", str(100 * ((b + 1) / length)) + "%")
            zGrid = basemap[1]
            zGridBottomLeft = basemap[0][0]
            zGridTopRight = basemap[0][1]
            zGridMiddle = np.mean(np.transpose([zGridBottomLeft, zGridTopRight]), axis=1)
            zGridXRange = zGridTopRight[0] - zGridBottomLeft[0]
            zGridYRange = zGridTopRight[1] - zGridBottomLeft[1]
            zGrid = np.transpose(zGrid)

            if "output_srtm.asc" in basemap[2]:
                quality = 0.1
                zGridComp = []
                step = int(1 / quality)
                row = 0

                rows = 0
                columns = len(zGrid[0][0::step])
                while row < len(zGrid):
                    rows += 1
                    zGridComp.append(zGrid[row][0::step])
                    row += step
                zGrid = np.array(zGridComp)
                p1 = scene.visuals.SurfacePlot(z=zGrid, color=(1, 1, 1, 1))
                zGridMax = abs(np.amax(zGrid))
                cnorm = zGrid / zGridMax
                cnorm = (((1 - cnorm) ** 1.5) / 2)
                c = np.transpose(color.get_colormap("greens").map(cnorm))
                c[3] = np.full(np.shape(c[3]), 1)
                c[0] = c[0] * 0.15
                c[1] = c[1] * 0.15
                c[2] = c[2] * 0.15
                c = np.transpose(c)
                c = c.reshape(zGrid.shape + (-1,))
                c = c.flatten().tolist()
                c = list(map(lambda x, y, z, w: (x, y, z, w), c[0::4], c[1::4], c[2::4], c[3::4]))
                p1.mesh_data.set_vertex_colors(c) # but explicitly setting vertex colors does work?
                p1.shininess = 0
                p1.ambient_light_color = (0.75, 0.75, 0.75, 1)
                p1.shading = "smooth"
                p1.transform = scene.transforms.MatrixTransform()
                texture = np.flipud(read_png("CMYK_color_swatches.png"))
                texRes = np.shape(texture)[:2][::-1]  # width, height
                # print(texRes[0])
                xPerCol = (zGridXRange / rows)
                yPerRow = (zGridYRange / columns)
                # y, x = np.meshgrid((np.linspace(properties[0][0][0], properties[0][1][0], texRes[0]) - boundBoxCenter[0]) * 111000, (np.linspace(properties[0][0][1], properties[0][1][1], texRes[1]) - boundBoxCenter[1]) * 111000)
                # y, x = np.meshgrid(np.linspace(0, 1, texRes[0]), np.linspace(0, 1, texRes[1]))
                # y = np.zeros(texRes[0])
                # x = np.linspace(0, 1, texRes[0])
                # texcoords = np.transpose((x, y))
                # texcoords = texcoords.reshape(texRes[0], 2)
                #z = texture
                #coords = np.meshgrid(np.arange(z.shape[0]) / z.shape[0], np.arange(z.shape[1]) / z.shape[1])
                #texcoords = np.empty((z.shape[0] * z.shape[1], 2), dtype=np.float32)
                #texcoords[:, 0] = coords[0].ravel()
                #texcoords[:, 1] = coords[1].ravel()
                # p1.attach(TextureFilter(texture, texcoords, enabled=True))
                p1.transform.translate([(-rows / 2) + ((zGridMiddle[0] - boundBoxCenter[0]) / xPerCol),
                                        (-columns / 2) + ((zGridMiddle[1] - boundBoxCenter[1]) / yPerRow)])
                p1.transform.scale([xPerCol * 111000, yPerRow * 111000 * latitudeUnsquash, 3])
                view.add(p1)
            else:
                zGrid = np.rot90(zGrid)
                lengthZGrid = len(zGrid)
                lengthZGridColumns = len(zGrid[0])
                for row in range(0, lengthZGrid, 5):
                    for column in range(0, len(zGrid[row]), 5):
                        z.append(zGrid[row][column])
                        x.append(basemap[3][0][(row * lengthZGridColumns) + column])
                        if xMin == None or x[-1] < xMin:
                            xMin = x[-1]
                        if xMax == None or x[-1] > xMax:
                            xMax = x[-1]
                        y.append(basemap[3][1][(row * lengthZGridColumns) + column])
                        if yMin == None or y[-1] < yMin:
                            yMin = y[-1]
                        if yMax == None or y[-1] > yMax:
                            yMax = y[-1]


        print(np.shape(x), np.shape(y), np.shape(z), basemaps[-1][0], basemaps[-1][1])
        print(xMax, xMin)
        width = basemaps[-1][0]
        height = basemaps[-1][1]
        zi, yi, xi = np.histogram2d(y, x, bins=(width/10, height/10), weights=z, normed=False)
        zi2, yi2, xi2 = np.histogram2d(y, x, bins=(width/10, height/10), normed=False)
        zi = zi/zi2
        xi = xi[:-1]
        yi = yi[:-1]
        zi = np.nan_to_num(zi, 0)
        zGrid = zi
        print(np.shape(xi), np.shape(yi), np.shape(zi))
        print("xi", xi[0], "yi", yi[0], "zi", zGrid[0])

        p2 = scene.visuals.SurfacePlot(x=xi, y=yi, z=zGrid, color=(1, 1, 1, 1))
        zGridMax = abs(np.amax(zGrid))
        cnorm = zGrid / zGridMax
        cnorm = (((1 - cnorm) ** 1.5) / 2)
        c = np.transpose(color.get_colormap("greens").map(cnorm))
        c[3] = np.full(np.shape(c[3]), 1)
        c[0] = c[0] * 0.15
        c[1] = c[1] * 0.15
        c[2] = c[2] * 0.15
        c = np.transpose(c)
        c = c.reshape(zGrid.shape + (-1,))
        c = c.flatten().tolist()
        c = list(map(lambda x, y, z, w: (x, y, z, w), c[0::4], c[1::4], c[2::4], c[3::4]))
        p2.mesh_data.set_vertex_colors(c)  # but explicitly setting vertex colors does work?
        p2.shininess = 0
        p2.ambient_light_color = (0.75, 0.75, 0.75, 1)
        p2.shading = "smooth"
        p2.transform = scene.transforms.MatrixTransform()
        texture = np.flipud(read_png("CMYK_color_swatches.png"))
        texRes = np.shape(texture)[:2][::-1]  # width, height
        # print(texRes[0])
        xPerCol = ((xMax - xMin) / width)
        yPerRow = ((yMax - yMin) / height)
        # y, x = np.meshgrid((np.linspace(properties[0][0][0], properties[0][1][0], texRes[0]) - boundBoxCenter[0]) * 111000, (np.linspace(properties[0][0][1], properties[0][1][1], texRes[1]) - boundBoxCenter[1]) * 111000)
        # y, x = np.meshgrid(np.linspace(0, 1, texRes[0]), np.linspace(0, 1, texRes[1]))
        # y = np.zeros(texRes[0])
        # x = np.linspace(0, 1, texRes[0])
        # texcoords = np.transpose((x, y))
        # texcoords = texcoords.reshape(texRes[0], 2)
        # z = texture
        # coords = np.meshgrid(np.arange(z.shape[0]) / z.shape[0], np.arange(z.shape[1]) / z.shape[1])
        # texcoords = np.empty((z.shape[0] * z.shape[1], 2), dtype=np.float32)
        # texcoords[:, 0] = coords[0].ravel()
        # texcoords[:, 1] = coords[1].ravel()
        # p1.attach(TextureFilter(texture, texcoords, enabled=True))
        p2.transform.translate([-(xMax + xMin)/2, -(yMax + yMin)/2])
        p2.transform.scale([111000, 111000 * latitudeUnsquash, 3])

        view.add(p2)
        print("lidar added")

        initial_camera_dir = (0, -1, 0)  # for a default initialised camera
        p2.light_dir = initial_camera_dir
        initial_light_dir = view.camera.transform.imap(initial_camera_dir)[:3]

def MenuUI():
    global prog2
    global view
    global canvas
    global t
    global menuP1
    global menuL1
    global menuL2
    global menuL3
    global vispyQT
    global vispyQTsp
    global timer
    global manager
    global Global
    global waypoints

    w.setWindowTitle("Cycle Planner")

    menuWidget = QWidget()
    w.setWindowOpacity(0.85)
    w.setAttribute(Qt.WA_NoSystemBackground)
    w.setAttribute(Qt.WA_TranslucentBackground)
    w.setCentralWidget(menuWidget)
    menuWidget.setLayout(QGridLayout())

    ### Initialise Vispy
    vispyQT = canvas.native
    vispyQTsp = vispyQT.sizePolicy()
    vispyQTsp.setVerticalStretch(255)
    # vispyQTsp.setHorizontalStretch(255)
    vispyQT.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
    menuWidget.layout().addWidget(vispyQT, 0, 0, 7, 3)

    ### Main menu buttons
    menuWidget.layout().setColumnStretch(0, 1)
    menuWidget.layout().setColumnStretch(1, 0.5)
    menuWidget.layout().setColumnStretch(2, 1)
    menuWidget.layout().setRowStretch(0, 2)
    menuWidget.layout().setRowStretch(1, 0)
    menuWidget.layout().setRowStretch(2, 2)
    menuWidget.layout().setRowStretch(3, 0)
    menuWidget.layout().setRowStretch(4, 0)
    menuWidget.layout().setRowStretch(5, 0)
    menuWidget.layout().setRowStretch(6, 2)
    menuWidget.layout().setRowStretch(7, 0)

    menuWidget.layout().addWidget(prog2, 7, 0, 1, 3)
    #updateProg(0)

    menuL1 = Text('Cycle Planner', bold=True, parent=view, color='black')
    menuL1.font_size = 52
    menuL1.pos = canvas.size[0] // 2, canvas.size[1] // 4

    menuL2 = Text('Cycle Planner', bold=True, parent=view, color='grey')
    menuL2.font_size = 50
    menuL2.pos = canvas.size[0] // 2, canvas.size[1] // 4

    menuL3 = Text('Cycle Planner', bold=True, parent=view, color='white')
    menuL3.font_size = 48
    menuL3.pos = canvas.size[0] // 2, canvas.size[1] // 4

    menub1 = QPushButton()
    menub1.setText('Plan a route')
    menub1.clicked.connect(MenuUIButtonPress)
    menub1.setFont(QFont('Arial', 15))
    # menub1.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    menuWidget.layout().addWidget(menub1, 3, 1)

    menub2 = QPushButton()
    menub2.setText('Exit')
    menub2.clicked.connect(MenuUIButtonPress2)
    menub2.setFont(QFont('Arial', 15))
    # menub2.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Minimum)
    menuWidget.layout().addWidget(menub2, 4, 1)

    menuP1 = Plot3D([[0, 0, 0]], parent=view.scene)

    view.camera.center = (0, 0, 0.5)
    view.camera.rotation = Quaternion.create_from_euler_angles(0, 0, 0, degrees=True)
    # view.camera.interactive = False
    view.camera.scale_factor = 1

    t = 0
    def menuUpdate(ev):
        global menuP1
        global t
        rang = np.linspace(0 + t, 360 + t, 1000)
        x = np.cos(rang)
        y = np.absolute(np.sin(rang))
        # canvasSize = [canvas.size[0], canvas.size[1]]
        # menuMousePos = np.array(menuMousePos) / np.array(canvasSize)
        z = (np.tan(rang * (3)) - 3)  # + (np.sqrt((x - menuMousePos[0])**2 + (y - menuMousePos[1])**2) / 1)
        xyz = np.transpose([x, y, z])
        squashedPi = 0.5 + ((np.sin(t*200) + 1) / 4)
        squashedPi2 = 0.5 + ((np.sin(t * 300) + 1) / 4)
        menuL1.color = (0, 0, 0, squashedPi)
        menuL2.color = (1-squashedPi, squashedPi, 1-squashedPi, squashedPi2)
        menuL3.color=(0.25, 1, 0.25, 1)
        menuP1.set_data(xyz, connect="strip", width=1000, marker_size=0, edge_color='black',
                        color=(0.5, 1, 0, 0.25), face_color=(0.5, 0.5, 0, 0))
        t += 0.00001

    w.show()

    def updateProgressBar(onMainMenu, perc):
        print("updateProgressBar")
        if onMainMenu:
            if perc == 1.0:
                prog2.reset()
            else:
                prog2.setValue(perc)
        else:
            if perc == 1.0:
                prog1.reset()
            else:
                prog1.setValue(perc)

    #baseMapperThread = BaseMapperThread()
    #baseMapperThread.start()

    p = Process(target=WeightBuilder, args=(edgesToAdd, Global, waypoints))
    p.daemon = True
    weightThreads.append(p)
    for i in range(len(weightThreads[:-1])):
        weightThreads[i].terminate()
    weightThreads[-1].start()
    progThread = ProgThread(False)
    progThread.start()
    progThread.progress_update.connect(updateProgressBar)
    progThread.route_update.connect(updateRoute)

    #weightBuilderThread = WeightBuilderThread(edgesToAdd, UIMetricWeights, UITypeWeights)
    #QThreadPool.globalInstance().start(weightBuilderThread)

    timer = vispy.app.Timer()
    timer.connect(menuUpdate)
    timer.start(0.0000001)

    if __name__ == '__main__':
        canvas.show()

        if sys.flags.interactive == 0:
            vispy.app.run()

#cacheRoute = [[], [], [], []]  # waypoint pairs, route sections, metric weights
#np.save("cacheRoute.npy", cacheRoute)

class UpdateProgThread(QRunnable):
    def __init__(self, perc):
        self.perc = perc
        super(UpdateProgThread, self).__init__()
    def run(self):
        perc = self.perc
        global lastUpdateProg
        global prog2
        global prog1
        global progTask
        global lastPerc
        if perc < lastPerc:
            stack = inspect.stack()
            try:
                the_class = stack[1][0].f_locals["self"].__class__.__name__
                the_method = stack[1][0].f_code.co_name
                progTask = "{}.{}".format(the_class, the_method)
            except:
                progTask = ""
            if Global.onMainMenu:
                prog2.reset()
                prog2.setFormat(progTask + "%p%")
            else:
                prog1.reset()
                prog1.setFormat(progTask + "%p%")

        lastPerc = perc
        if perc == 1:
            if Global.onMainMenu:
                prog2.reset()
            else:
                prog1.reset()
        elif time.clock() - lastUpdateProg >= 0.25:
            print(perc)
            if Global.onMainMenu:
                prog2.setValue(perc * 100)
            else:
                prog1.setValue(perc * 100)
            lastUpdateProg = time.clock()



class RouteBuilderThread(QRunnable):
    def __init__(self, waypoints, weights, drawOrNot=False):
        self.waypoints = waypoints
        self.weights = weights
        self.drawOrNot = drawOrNot
        self.out = "False"
        super(RouteBuilderThread, self).__init__()
    def run(self):
        global Global
        print("RouteBuilderThread: run")
        Global.perc = 0
        print("a")
        #global Global
        #updateProg(0, Global)
        ###finding the directions to take through each section...
        route = []
        cacheRoute = np.load("cacheRoute.npy", allow_pickle=True).tolist()
        for section in range(len(self.waypoints[0]) - 1):
            indexes = [x for x in range(len(cacheRoute[0])) if
                       cacheRoute[0][x] == [self.waypoints[0][section], self.waypoints[0][section + 1]]]
            done = False
            for i in indexes:
                if Global.UIMetricWeights == cacheRoute[2][i]:
                    done = True
                    route.append(cacheRoute[1][i])
            if not done:
                directions = g.get_shortest_paths(self.waypoints[0][section], to=self.waypoints[0][section + 1], weights=self.weights, mode=OUT)[0]
                if len(directions) > 0:
                    route.append(directions)
                    cacheRoute[0].append([self.waypoints[0][section], self.waypoints[0][section + 1]])
                    cacheRoute[1].append(directions)
                    cacheRoute[2].append(Global.UIMetricWeights)
        np.save("cacheRoute.npy", cacheRoute)
        print(route)
        # route = []
        # for section in range(len(waypoints[0]) - 1):
        #     #if section > 0:
        #         #if metricWeights[section] != metricWeights[section-1]:
        #             #weights = WeightBuilder(edgesToAdd, vertexToRoad, segMetric, metricWeights[section])
        #     #else:
        #     sectionRoute = g.get_shortest_paths(waypoints[0][section], to=waypoints[0][section+1], weights=weights, mode=OUT)
        #     if len(sectionRoute[0]) > 1:
        #         route.append(sectionRoute[0])
        #     else:
        #         return False

        ###converting the node sections into a road route...
        if len(route) > 0:
            combinedRoute = []
            for i in range(len(route)):
                combinedRoute.extend(route[i])
            # combinedRoute = route
            arr = np.take(np.transpose(vertexToRoad)[0], combinedRoute)
            arrDir = np.take(np.transpose(vertexToRoad)[1], combinedRoute)
            # u = [x for x in set(arr) if arr.count(x) > 0]

            ###finding the route metrics...
            routeMetric = [0, 0, 0]  # distance, elevation up, elevation down
            for seg in range(len(arr)):
                if seg > 0 and arr[seg] != arr[seg - 1]:
                    routeMetric[0] += segMetric[arr[seg]][0][arrDir[seg]]
                    routeMetric[1] += segMetric[arr[seg]][1][arrDir[seg]]
                    routeMetric[2] += segMetric[arr[seg]][1][-1 - arrDir[seg]]
            self.out = (arr.tolist(), routeMetric)
        print(self.out)
        if self.drawOrNot:
            updateRouteThread = UpdateRouteThread(self.out)
            QThreadPool.globalInstance().start(updateRouteThread)
        updateProg(1)

def RouteBuilder(waypoints, weights, Global, drawOrNot=False):
    waypoints = waypoints
    weights = weights
    drawOrNot = drawOrNot
    out = "False"
    updateProg(0, Global)
    ###finding the directions to take through each section...
    route = []
    cacheRoute = np.load("cacheRoute.npy", allow_pickle=True).tolist()
    for section in range(len(waypoints[0]) - 1):
        indexes = [x for x in range(len(cacheRoute[0])) if
                   cacheRoute[0][x] == [waypoints[0][section], waypoints[0][section + 1]]]
        done = False
        for i in indexes:
            if Global.UIMetricWeights == cacheRoute[2][i]:
                done = True
                route.append(cacheRoute[1][i])
        if not done:
            directions = g.get_shortest_paths(waypoints[0][section], to=waypoints[0][section + 1], weights=weights, mode=OUT)[0]
            if len(directions) > 0:
                route.append(directions)
                cacheRoute[0].append([waypoints[0][section], waypoints[0][section + 1]])
                cacheRoute[1].append(directions)
                cacheRoute[2].append(Global.UIMetricWeights)
    np.save("cacheRoute.npy", cacheRoute)
    # route = []
    # for section in range(len(waypoints[0]) - 1):
    #     #if section > 0:
    #         #if metricWeights[section] != metricWeights[section-1]:
    #             #weights = WeightBuilder(edgesToAdd, vertexToRoad, segMetric, metricWeights[section])
    #     #else:
    #     sectionRoute = g.get_shortest_paths(waypoints[0][section], to=waypoints[0][section+1], weights=weights, mode=OUT)
    #     if len(sectionRoute[0]) > 1:
    #         route.append(sectionRoute[0])
    #     else:
    #         return False

    ###converting the node sections into a road route...
    if len(route) > 0:
        combinedRoute = []
        for i in range(len(route)):
            combinedRoute.extend(route[i])
        # combinedRoute = route
        arr = np.take(np.transpose(vertexToRoad)[0], combinedRoute)
        arrDir = np.take(np.transpose(vertexToRoad)[1], combinedRoute)
        # u = [x for x in set(arr) if arr.count(x) > 0]

        ###finding the route metrics...
        routeMetric = [0, 0, 0]  # distance, elevation up, elevation down
        for seg in range(len(arr)):
            if seg > 0 and arr[seg] != arr[seg - 1]:
                routeMetric[0] += segMetric[arr[seg]][0][arrDir[seg]]
                routeMetric[1] += segMetric[arr[seg]][1][arrDir[seg]]
                routeMetric[2] += segMetric[arr[seg]][1][-1 - arrDir[seg]]
        out = (arr.tolist(), routeMetric)
    if drawOrNot:
        print(out)
        updateRouteThread = UpdateRouteThread(out)
        QThreadPool.globalInstance().start(updateRouteThread)
    updateProg(1)

roadChunks = np.load("roadChunks.npy", allow_pickle=True)

def whatClicked(eventPos, maxDistance):
    global r1s
    viewBox = view.size
    zoom = toggle1initialHeight / view.camera.center[2]
    mapHeight = view.size[1]
    mapWidth = view.size[1] / ((((properties[0][1][1] - boundBoxCenter[1]) * (latitudeUnsquash)) - (
                (properties[0][0][1] - boundBoxCenter[1]) * (latitudeUnsquash))) / (
                                           properties[0][1][0] - properties[0][0][0]))
    xPad = (viewBox[0] - mapWidth) / 2
    xMiddle = viewBox[0] / 2
    yMiddle = viewBox[1] / 2
    # xOffset =
    # yOffset =
    mapBottomLeft = [xMiddle - ((xMiddle - xPad) * zoom), yMiddle - (yMiddle * zoom)]
    mapTopRight = [xMiddle + (((xPad + mapWidth) - xMiddle) * zoom), yMiddle + ((viewBox[1] - yMiddle) * zoom)]

    latlonPos = [
        normalize([eventPos[0]], [[mapBottomLeft[0], mapTopRight[0]], [properties[0][0][0], properties[0][1][0]]])[0],
        normalize([eventPos[1]], [[mapBottomLeft[1], mapTopRight[1]], [properties[0][1][1], properties[0][0][1]]])[0]]

    gridWidth = np.int((properties[0][1][0] - properties[0][0][0]) / RES[0]) + 1
    whichBox = [(latlonPos[0] - properties[0][0][0]) // RES[0], (latlonPos[1] - properties[0][0][1]) // RES[1]]
    roadsToCheck = []
    for i in range(-2, 3):
        for j in range(-2, 3):
            if gridWidth - 1 > whichBox[1] + i >= 0 and gridWidth - 1 > whichBox[0] + j >= 0:
                index = int(((whichBox[1] + i) * gridWidth) + (whichBox[0] + j) - (whichBox[1] + i))
                roadsToCheck.extend(roadChunks[index])
    roadsToCheck = list(set(roadsToCheck))
    #print(roadsToCheck)
    if len(roadsToCheck) > 0:
        r1s.parent=view.scene
        out = roadDraw(np.take(splitRoadsWithHeights, roadsToCheck, axis=0).tolist(), len(roads[0]), [111000, 111000, 3], [0, 0, 10])
        r1s.set_data(out[0], connect=out[1], width=1000, marker_size=7.5, edge_color='w',
                            symbol='x',
                            color=(0.5, 0.5, 1, 1), face_color=(0.5, 0.5, 1, 1))

    ###
    minDistance = False
    whichNode = False
    for road in range(len(roadsToCheck)):
        for coord in range(len(splitRoadsWithHeights[roadsToCheck[road]][0])):
            currentDistance = [np.abs(splitRoadsWithHeights[roadsToCheck[road]][0][coord] - latlonPos[0]),
                               np.abs(splitRoadsWithHeights[roadsToCheck[road]][1][coord] - latlonPos[1])]
            if currentDistance[0] < (maxDistance) and currentDistance[1] < (maxDistance) and (
                    (not minDistance) or minDistance > currentDistance):
                minDistance = currentDistance
                whichNode = [roadsToCheck[road], coord]
    nodePos = "False"
    if minDistance:
        nodePos = [splitRoadsWithHeights[whichNode[0]][0][whichNode[1]], splitRoadsWithHeights[whichNode[0]][1][whichNode[1]]]
    return whichNode, minDistance, nodePos

def whatClickedWaypoint(eventPos, maxDistance, waypoints):
    whichWaypoint = "False"
    if len(waypoints) > 0:
        viewBox = view.size
        zoom = toggle1initialHeight / view.camera.center[2]
        mapHeight = view.size[1]
        mapWidth = view.size[1] / ((((properties[0][1][1] - boundBoxCenter[1]) * (latitudeUnsquash)) - (
                    (properties[0][0][1] - boundBoxCenter[1]) * (latitudeUnsquash))) / (
                                               properties[0][1][0] - properties[0][0][0]))
        xPad = (viewBox[0] - mapWidth) / 2
        xMiddle = viewBox[0] / 2
        yMiddle = viewBox[1] / 2
        # xOffset =
        # yOffset =
        mapBottomLeft = [xMiddle - ((xMiddle - xPad) * zoom), yMiddle - (yMiddle * zoom)]
        mapTopRight = [xMiddle + (((xPad + mapWidth) - xMiddle) * zoom), yMiddle + ((viewBox[1] - yMiddle) * zoom)]

        latlonPos = [
            normalize([eventPos[0]], [[mapBottomLeft[0], mapTopRight[0]], [properties[0][0][0], properties[0][1][0]]])[0],
            normalize([eventPos[1]], [[mapBottomLeft[1], mapTopRight[1]], [properties[0][1][1], properties[0][0][1]]])[0]]

        minDistance = False
        for w in range(len(waypoints[1])):
            currentDistance = [np.abs(waypoints[1][w][0] - latlonPos[0]), np.abs(waypoints[1][w][1] - latlonPos[1])]
            if currentDistance[0] < (maxDistance[0]) and currentDistance[1] < (maxDistance[1]) and ((not minDistance) or minDistance > currentDistance):
                minDistance = currentDistance
                whichWaypoint = w
    return whichWaypoint

def updateWaypoints(waypoints):
    posLog = []
    if len(waypoints[0]) > 0:
        for i in range(len(waypoints[0])):
            whichNode = vertexToRoad[waypoints[0][i]]
            startOrEnd = whichNode[1]
            posLog.append([(splitRoadsWithHeights[whichNode[0]][0][startOrEnd] - boundBoxCenter[0]) * 111000, (
                splitRoadsWithHeights[whichNode[0]][1][startOrEnd] - boundBoxCenter[
            1]) * 111000 * latitudeUnsquash, (splitRoadsWithHeights[whichNode[0]][2][startOrEnd] * 3) + 10])
        m1.parent = view.scene
        m1.set_data(np.array(posLog), face_color=np.array([[1, 0, 0]] * len(posLog)), symbol='o', size=7.5, edge_width=7.5,
                    edge_color=(1, 0, 1, 1))
    else:
        m1.parent = None
    updateCanvas()
    return posLog

def updateCanvas():
    pass
    #global canvas
    #r1a.parent = None
    #r1b.parent = None
    #r1a.parent = view.scene
    #r1b.parent = view.scene
    #canvas.update()

### DEBUG FOR BOUND BOX CENTER
#minX = 100
#maxX = -100
#for road in range(len(splitRoads)):
    #print(road/len(splitRoads))
    #temp = np.amin(splitRoads[road][0])
    #if temp < minX:
        #minX = temp
    #temp = np.amax(splitRoads[road][0])
    #if temp > maxX:
        #maxX = temp
#print(minX, maxX)
#print(boundBoxCenter)

# p1.set_gl_state('translucent', blend=True, depth_test=True)

############################################################
############################################################
############################################################

    #@canvas.connect
    #def on_mouse_move(event):
        #global menuMousePos
        #menuMousePos = event.pos

#deltas = [0, 0]
#
# @canvas.connect
# def on_mouse_wheel(event):
#     if not onMainMenu:
#         global toggle1
#         global deltas
#         if toggle1 == True:
#             view.camera.scale_factor = 90490.23177079092
#             deltas[0] = deltas[1]
#             deltas[1] = event.delta[1]

if __name__ == '__main__':
    global Global
    global manager
    manager = multiprocessing.Manager()
    Global = manager.Namespace()
    Global.onMainMenu = True
    Global.perc = 0
    Global.UIMetricWeights = [0,0,0]
    Global.UITypeWeights = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    Global.cacheWeight = [[], [], []] #np.load("cacheWeight.npy", allow_pickle=True).tolist()
    Global.cacheRoute = [[], [], [], []] #np.load("cacheRoute.npy", allow_pickle=True).tolist()
    Global.out = []
    Global.routeMetric = []
    Global.interrupt = False

    # build visuals
    Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)
    Arrow3D = scene.visuals.create_visual_node(visuals.ArrowVisual)

    # build canvas
    canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)
    #vispy.use(gl="gl2")

    # Add a ViewBox to let the user zoom/rotate
    view = canvas.central_widget.add_view()
    view.camera = 'fly'
    view.camera.fov = 45
    view.camera.distance = 6
    #################################################################################
    #################################################################################

    ### Cycling UI
    prog1 = QProgressBar()
    prog1.setStyleSheet("background-color : rgba(255, 255, 255, 0);")
    prog1.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    prog1.setMinimum(0)
    prog1.setMaximum(100)

    ### Main Menu
    global weightThreads
    weightThreads = []

    global optimiserThreads
    optimiserThreads = []

    global processes
    processes = []

    w = QMainWindow()

    prog2 = QProgressBar()
    prog2.setStyleSheet("background-color : rgba(0, 0, 0, 255);")
    prog2.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
    prog2.setMaximum(100)

    class OptProgThread(QThread):
        def __init__(self):
            super(OptProgThread, self).__init__()
        progress_update = pyqtSignal(bool, float)
        def run(self):
            global Global
            global processes
            runningCoreCount = 1
            lastUpdateProg = 0
            hasStarted = False
            while Global.optInterrupt == False and (runningCoreCount > 0 or not hasStarted):
                # print(Global.optBatch[0])
                if time.clock() - lastUpdateProg >= 0.25:
                    lastUpdateProg = time.clock()
                    runningIDs = [x for x in range(len(processes)) if processes[x].is_alive()]
                    percs = [x for x in np.take(Global.optPerc, runningIDs) if x != None]
                    runningCoreCount = len(runningIDs)
                    if runningCoreCount > 0:
                        hasStarted = True
                        self.progress_update.emit(Global.onMainMenu, float(np.mean(percs) * 100))
                        print(str(np.round(np.mean(percs) * 100, 2)) + "%     " + str(runningCoreCount) + " threads")

                    optBatch = Global.optBatch
                    if len([x for x in optBatch[0] if x != None]) > 0:
                        optBatch[0] = [np.nan if v is None else v for v in optBatch[0]]
                        bestID = np.nanargmin(optBatch[0])
                        try:
                            updateRoute(optBatch[2][bestID], optBatch[1][bestID])
                        except:
                            pass

    class OptimiserThread(QThread):
        def __init__(self, waypoints, quality, goals):
            self.quality = quality
            self.waypoints = waypoints
            self.goals = goals
            super(OptimiserThread, self).__init__()

        def run(self):
            global processes
            global Global

            coreCount = int(multiprocessing.cpu_count() * 0.5)

            Global.optBatch = [[None for i in range(coreCount)], [None for i in range(coreCount)],
                               [None for i in range(coreCount)]]  # scores, routeMetrics, directions
            Global.optPerc = [None for i in range(coreCount)]
            Global.optInterrupt = False

            # waypoints = [None, 1376, 1567]
            howManyVariable = sum([1 for i in self.waypoints if i == None])
            length = len(self.waypoints)

            quality = ((vertexCount * self.quality) ** (1 / howManyVariable) * coreCount) / vertexCount
            goals = self.goals

            steps = int((vertexCount * quality) / coreCount)
            waypointsPoss = int(steps ** howManyVariable)
            distribution = np.linspace(0, vertexCount - 1, num=steps, endpoint=True)
            spread = distribution[1] - distribution[0]
            spreadPerCore = spread / (coreCount)
            rangs = []
            for core in range(coreCount):
                rangs.append((distribution + (spreadPerCore * (core))).astype(int).tolist())
            # rangs.append(None)

            # stuff = [rangs[0], steps, waypointsPoss, howManyVariable, length]
            # multiprocessing_func(core, Global, stuff, waypoints, goals)
            # (shallIContinue = input()

            for core in range(coreCount):
                stuff = [rangs[core], steps, waypointsPoss, howManyVariable, length]
                p = multiprocessing.Process(target=multiprocessing_func,
                                            args=(core, Global, stuff, self.waypoints, goals))
                p.daemon = True
                processes.append(p)
            optProgThread = OptProgThread()
            optProgThread.start()
            optProgThread.progress_update.connect(updateProgressBar2)

            for core in range(coreCount):
                processes[core].start()

            processes[-1].join()
    ### OPTIMISER ###

    def updateProgressBar(onMainMenu, perc):
        if onMainMenu:
            if perc == 1.0:
                prog2.reset()
            else:
                prog2.setValue(perc)
        else:
            if perc == 1.0:
                prog1.reset()
            else:
                prog1.setValue(perc)

    def updateProgressBar2(onMainMenu, perc):
        print("BAM")

    def updateRoute(out, routeMetric):
        r1a.parent = view.scene
        r1b.parent = view.scene
        route = out
        out = roadDraw(np.take(splitRoadsWithHeights, route, axis=0).tolist(), len(roads[0]),
                            [111000, 111000, 3], [0, 0, 11])
        #routeLatLon = [self.out[2], self.out[3]]
        #if len(self.routeLatLon) == 0:
            #l1.text = "graph traversal error (missing edges?)"
            #posLog = posLog[:-1]
            #waypoints[0], waypoints[1] = waypoints[0][:-1], waypoints[1][:-1]
            #colors = np.array([[1, 0.4, 0]] * len(posLog))
            #m1.set_data(np.array(posLog), face_color=colors, symbol='o', size=7.5, edge_width=7.5, edge_color='green')
        #else:
        l1.text = str(np.round(routeMetric[0] / 1000, 2)) + "km   " + str(
            np.round(routeMetric[1], 2)) + "m up   " + str(np.round(routeMetric[2], 2)) + "m down   "
        r1a.set_data(pos=out[0], connect=out[1], width=1000,
                     color=(0.5, 0.5, 1, 1))  # , marker_size=7.5, width=1000, edge_color='w',
        # symbol='x',
        # color=(0.5, 0.5, 1, 1), face_color=(0.5, 0.5, 1, 1))
        r1b.set_data(out[0], connect=out[1], width=1000, marker_size=7.5, edge_color='w',
                     symbol='x',
                     color=(0.5, 0.5, 1, 1), face_color=(0.5, 0.5, 1, 1))
        #else:
            #r1a.parent = None
            #r1b.parent = None
        #updateCanvas()
        #updateProg(1)

    class ProgThread(QThread):
        def __init__(self, drawOrNot):
            self.drawOrNot = drawOrNot
            super(ProgThread, self).__init__()
        progress_update = pyqtSignal(bool, float)
        route_update = pyqtSignal(list, list)
        def run(self):
            print("ProgThread: run")
            global lastUpdateProg
            if self.drawOrNot:
                while Global.out == [] and not Global.interrupt:
                    if time.clock() - lastUpdateProg >= 0.25:
                        lastUpdateProg = time.clock()
                        print(str(Global.perc * 100) + "%")
                        self.progress_update.emit(Global.onMainMenu, Global.perc * 100)
                if not Global.interrupt:
                    self.route_update.emit(Global.out, Global.routeMetric)
                    Global.out = []
                    Global.routeMetric = []
            else:
                while Global.perc < 1 and not Global.interrupt:
                    if time.clock() - lastUpdateProg >= 0.25:
                        lastUpdateProg = time.clock()
                        print(str(Global.perc*100) + "%")
                        self.progress_update.emit(Global.onMainMenu, Global.perc * 100)
            if Global.interrupt:
                print("Progthread caught an interrupt!")
                Global.interrupt = False
            else:
                Global.perc = 0
                self.progress_update.emit(Global.onMainMenu, 0)
                np.save("cacheWeight.npy", Global.cacheWeight)
                np.save("cacheRoute.npy", Global.cacheRoute)
            return

    @canvas.connect
    def on_key_press(event):
        if not Global.onMainMenu:
            global toggle1
            global toggle1loc
            global toggle1rot
            global toggle1sca
            global toggle1height
            global toggle1x
            global toggle1y
            global toggle1wideOrTall
            global toggle1initialHeight
            global view
            if toggle1 == True:
                try:
                    if event.key.name == "Q":
                        toggle1height += 500
                    if event.key.name == "E":
                        toggle1height -= 500
                    if event.key.name == "W":
                        toggle1y += 500
                    if event.key.name == "S":
                        toggle1y -= 500
                    if event.key.name == "D":
                        toggle1x += 500
                    if event.key.name == "A":
                        toggle1x -= 500
                except:
                    pass
            if event.key.name == "1":
                if toggle1 == False:
                    toggle1loc = view.camera.center
                    toggle1rot = view.camera.rotation
                    toggle1sca = view.camera.scale_factor
                    view.camera.interactive = False
                    # if boundBoxX/(view.size[0] / view.size[1]) > boundBoxY:
                    # toggle1height = boundBoxX * 136000
                    toggle1height = ((boundBoxX)) * 111000 * 1.225  # fov?
                    toggle1initialHeight = toggle1height
                    toggle1x = 0
                    toggle1y = 0
                    view.camera.center = (0, 0, toggle1height)
                    view.camera.scale_factor = 1
                    # view.camera.center = (0, 0, -179.97778545)
                    view.camera.rotation = Quaternion.create_from_euler_angles(0, 0, 0, degrees=True)
                    toggle1 = True
                else:
                    view.camera.interactive = True
                    view.camera.center = toggle1loc
                    view.camera.rotation = toggle1rot
                    view.camera.scale_factor = toggle1sca
                    toggle1 = False

    @canvas.connect
    def on_resize(event):
        if Global.onMainMenu:
            menuL1.pos = canvas.size[0] // 2, canvas.size[1] // 4
            menuL2.pos = canvas.size[0] // 2, canvas.size[1] // 4
            menuL3.pos = canvas.size[0] // 2, canvas.size[1] // 4
        else:
            global viewBox
            viewBox = view.size

    posLog = []
    route = []
    global waypoints
    waypoints = [[], []]

    @canvas.connect
    def on_mouse_release(event):
        global r1s
        if not Global.onMainMenu:
            r1s.parent = None

    @canvas.connect
    def on_mouse_press(event):
        if not Global.onMainMenu:
            global viewBox
            global posLog
            global latitudeUnsquash
            global route
            global toggle1initialHeight
            global waypoints
            global routeLatLon
            global updateRouteThread
            global initial_light_dir
            global waypointWidgetHandler
            global weightThreads
            global progThread
            if event.button == 2:
                if toggle1:
                    maxDistance = [boundBoxX / 5, boundBoxY / 5]
                    nodePos = whatClickedWaypoint(event.pos, maxDistance, waypoints)
                    if nodePos != "False":
                        l1.text = "Waypoint removed."
                        waypointWidgetHandler.remove(nodePos)
                        del waypoints[0][nodePos]
                        del waypoints[1][nodePos]
                        posLog = updateWaypoints(waypoints)

                        if len(waypoints[0]) > 1:
                            ### threading jam time ###
                            p = Process(target=WeightBuilder, args=(edgesToAdd, Global, waypoints, True))
                            p.daemon = True
                            weightThreads.append(p)
                            for i in range(len(weightThreads[:-1])):
                                if weightThreads[i].is_alive():
                                    print("I just killed a process!")
                                    Global.interrupt = True
                                    weightThreads[i].terminate()
                            weightThreads[-1].start()
                            progThread = ProgThread(True)
                            progThread.start()
                            progThread.progress_update.connect(updateProgressBar)
                            progThread.route_update.connect(updateRoute)
                            ### ################## ###
                        else:
                            r1a.parent = None
                            r1b.parent = None
                    else:
                        l1.text = "I can't find that waypoint."
                if not toggle1:
                    # print(p1.light_dir)
                    transform = view.camera.transform
                    dir = np.concatenate((initial_light_dir, [0]))
                    p1.light_dir = transform.map(dir)[:3]
                    # print(view.camera.scale_factor)
                    # print(view.camera.fov)
                    # print(view.camera.center)
                    # r = R.from_quat([view.camera.rotation.w, view.camera.rotation.x, view.camera.rotation.y, view.camera.rotation.z])
                    # print(r.as_euler('zyx', degrees=True))
                    # print(event.pos)
            if toggle1 and event.button == 1:
                maxDistance = RES[0] * 2
                whichNode, distanceFromClick, nodePos = whatClicked(event.pos, maxDistance)
                if nodePos != "False":
                    startOrEnd = int(np.round(whichNode[1] / (len(splitRoadsWithHeights[whichNode[0]][0]) - 1)) - 1)
                    pos = [(splitRoadsWithHeights[whichNode[0]][0][startOrEnd] - boundBoxCenter[0]) * 111000, (
                            splitRoadsWithHeights[whichNode[0]][1][startOrEnd] - boundBoxCenter[
                        1]) * 111000 * latitudeUnsquash, splitRoadsWithHeights[whichNode[0]][2][startOrEnd]]
                    if [whichNode[0], startOrEnd] in vertexToRoad:
                        l1.text = "Waypoint added."
                        waypointWidgetHandler.add()
                        waypoints[0].append(vertexToRoad.index([whichNode[0], startOrEnd]))
                        waypoints[1].append(nodePos)
                        # posLog.append(pos)
                        posLog = updateWaypoints(waypoints)
                        canvas.update()
                        if len(waypoints[0]) > 1:
                            ### threading jam time ###
                            p = Process(target=WeightBuilder, args=(edgesToAdd, Global, waypoints, True))
                            p.daemon = True
                            weightThreads.append(p)
                            for i in range(len(weightThreads[:-1])):
                                if weightThreads[i].is_alive():
                                    print("I just killed a process!")
                                    Global.interrupt = True
                                    weightThreads[i].terminate()
                            weightThreads[-1].start()
                            progThread = ProgThread(True)
                            progThread.start()
                            progThread.progress_update.connect(updateProgressBar)
                            progThread.route_update.connect(updateRoute)
                            ### ################## ###
                    else:
                        l1.text = "I can't find that waypoint."
    MenuUI()