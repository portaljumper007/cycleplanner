import requests
import json
from altitudo import altitudo
import numpy as np
import sys

import matplotlib.pyplot as plt
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


overpass_url = "http://overpass-api.de/api/interpreter"
#area["ISO3166-1"="GB"][admin_level=2];
overpass_query = """
[out:json]
;
area(3601402218)->.searchArea;
(
  way["highway"](area.searchArea);
  way["cycleway"](area.searchArea);
  way["footway"](area.searchArea);
);
out geom;
"""
response = requests.get(overpass_url,
                        params={'data': overpass_query})

data = response.json()

roads = []
roadBounds = []

for element in data['elements']:
    if element['type'] == 'way':
        road = element['geometry']
        roads.append(road)
        #print(element)
        roadBounds.append([element['bounds']['minlat'], element['bounds']['minlon'], element['bounds']['maxlat'], element['bounds']['maxlon']])
    #elif 'center' in element:

roadCoords = []
allLons = []
allLats = []
totalCoords = 0

import os
filenames = []
for root, dirs, files in os.walk(os.getcwd()):
    for file in files:
        if file.endswith('.npy'):
            filenames.append(file)

splitRoads = [[], []]
tempSplits = []
for r in range(len(roads)):
    roadCoords.append([[], []])
    for coord in range(len(roads[r])):
        totalCoords += 1
        roadCoords[r][0].append(roads[r][coord]['lon'])
        allLons.append(roads[r][coord]['lon'])
        roadCoords[r][1].append(roads[r][coord]['lat'])
        allLats.append(roads[r][coord]['lat'])
        # print(altitudo(lat=roadCoords[r][1][coord], lon=roadCoords[r][0][coord]))
    # plt.plot(*roadCoords[r])
    # plt.draw()
    # plt.pause(0.000001)

def splitter(array, indices):
    indices = sorted(list(set(indices)))
    indices.insert(0, 0)
    indices.append(len(array) - 1)
    result = []
    for i in range(len(indices)-1):
        if indices[i] != indices[i+1]:
            result.append(array[indices[i]:indices[i+1]+1])
    return result

if "splitRoads.npy" not in filenames:

    # plt.show()
    length = len(roadCoords)
    print("-" * 20)
    print("Scanning", length, "roads to split them at their intersections")

    for r in range(length):
        if r % np.floor(length / 10) == 0 :
            print((r/len(roadCoords)) * 100, "%")
        for coord in range(len(roadCoords[r][0])):
            for otherR in range(len(roadCoords)): #for each road, iterate through every other road
                if otherR != r: #for each road, iterate through every other road
                    for otherCoord in range(len(roadCoords[otherR])):
                        if roadCoords[otherR][0][otherCoord] == roadCoords[r][0][coord] and roadCoords[otherR][1][otherCoord] == roadCoords[r][1][coord]:
                            if 0 < coord < len(roadCoords[r][0]): #if coord isn't at the end of the road already, hence needs splitting
                                diff = r - (len(tempSplits) - 1)
                                for i in range(diff):
                                    tempSplits.append([])
                                tempSplits[r].append(coord)
                                #plt.scatter(roadCoords[otherR][0][otherCoord], roadCoords[otherR][1][otherCoord])
                            if 0 < otherCoord < len(roadCoords[otherR][0]):  # if coord isn't at the end of the road already, hence needs splitting
                                diff = otherR - (len(tempSplits) - 1)
                                for i in range(diff):
                                    tempSplits.append([])
                                tempSplits[otherR].append(otherCoord)
                                #plt.scatter(roadCoords[otherR][0][otherCoord], roadCoords[otherR][1][otherCoord])
                            #plt.draw()
                            #plt.pause(0.000001)
    length = len(tempSplits)
    for r in range(length):
        if r % np.floor(length / 10) == 0 :
            print((r/length) * 100, "%")
        splitsX = splitter(roadCoords[r][0], tempSplits[r])
        splitsY = splitter(roadCoords[r][1], tempSplits[r])
        for s in range(len(splitsX)):
            if splitsX[s] == []:
                print("splitsX", splitsX, "roadCoords", roadCoords[r][0], "tempSplits", tempSplits[r])
                shallIContinue = input()
            splitRoads[0].append([splitsX[s], splitsY[s]])
            splitRoads[1].append(roadBounds[r])
    np.save("splitRoads", splitRoads)
else:
    splitRoads = np.load("splitRoads.npy", allow_pickle=True)

if "joints.npy" not in filenames:
    joints = []
    length = len(splitRoads[0])
    print("-" * 20)
    print("Finding segment joints from", length, "segments")

    for segment in range(length):
        if segment % np.floor(length / 10) == 0 :
            print((segment/length) * 100, "%")
        joints.append([]) #which it connects to #whether it's this ones start or end #whether it's that one's start or end
        for startOrEnd in range(-1,0):
            for otherSegment in range(length):
                if otherSegment != segment:
                    for otherStartOrEnd in range(-1,0):
                        try:
                            if splitRoads[0][segment][0][startOrEnd] == splitRoads[0][otherSegment][0][otherStartOrEnd] and splitRoads[0][segment][1][startOrEnd] == splitRoads[0][otherSegment][1][otherStartOrEnd]:
                                joints[-1].extend([otherSegment, startOrEnd, otherStartOrEnd])
                        except:
                            print(segment, otherSegment, splitRoads[0][segment], splitRoads[0][otherSegment])
                            shallIContinue = input()
    np.save("joints", joints)
else:
    joints = np.load("joints.npy", allow_pickle=True)

### DRAW ###
def roadDraw(roads):
    print("Drawing roads...")
    out = []
    totalN = 0
    heights = []
    for road in range(len(roads)):
        heights.extend(roads[road][2])
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
    #roadsConc = np.append(roadsConc, splitRoadsWithHeights, axis=0) #heights.reshape((-1, len(heights)))
    for dim in range(len(roadsConc) - 1):
        roadsConc[dim] = (roadsConc[dim] - np.mean(roadsConc[dim])) * 50000
    roadsTranspos = np.transpose(roadsConc)
    Plot3D(roadsTranspos, width=20, color='red',
       edge_color='w', symbol='disc',
       parent=view.scene,connect=connect, face_color=(0.2, 0.2, 1, 0.8)) #face_color=(0.2, 0.2, 1, 0.8)

splitRoadsWithHeights = np.load("splitRoadsWithHeights.npy", allow_pickle=True)
roadDraw(splitRoadsWithHeights.tolist())


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()
### DRAW ###

#def roadTime(lats, lons, alts):
