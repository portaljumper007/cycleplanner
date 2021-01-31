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

properties = np.load("properties.npy")
boundBoxCenter = [np.mean([properties[0][0][0], properties[0][1][0]]), np.mean([properties[0][0][1], properties[0][1][1]])]
print(boundBoxCenter)

### DRAW ###
def roadDraw(roads, dims):
    #print("Drawing roads...")
    out = []
    totalN = 0
    heights = []
    for road in range(len(roads)):
        if dims < 3:
            heights.extend(np.ones(len(roads[road][0])))
        #else:
            #heights.extend(roads[road][2])
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
        roadsConc = np.append(roadsConc, [heights], axis=0) #heights.reshape((-1, len(heights)))
    for dim in range(len(roadsConc) - 1):
        roadsConc[dim] = (roadsConc[dim] - boundBoxCenter[dim]) * 75000
    roadsTranspos = np.transpose(roadsConc)
    return roadsTranspos, connect
    #Plot3D(roadsTranspos, width=20, color='red',
       #edge_color='w', symbol='disc',
       #parent=view.scene,connect=connect, face_color=(0.2, 0.2, 1, 0.8)) #face_color=(0.2, 0.2, 1, 0.8)
    #print(roadsTranspos)

#splitRoadsWithHeights = np.load("splitRoadsWithHeights.npy", allow_pickle=True)
#splitRoadsWithHeights = splitRoadsWithHeights.tolist()
splitRoads = np.load("splitRoadsWithHeights.npy", allow_pickle=True)
splitRoads = splitRoads.tolist()
joints = np.load("joints.npy", allow_pickle=True).tolist()

r = 0
print(splitRoads)
length = len(splitRoads)
roads = []
indexes = []
#print(splitRoads[564])
#while r < length:
    #tally = 0
    #for coord in range(len(splitRoads[r][0])):
        #if tally == 0 and (properties[0][0][0] < splitRoads[r][0][coord] < properties[0][1][0] and properties[0][0][1] < splitRoads[r][1][coord] < properties[0][1][1]):
            #roads.append(splitRoads[r])
            #indexes.append(r)
    #r += 1
roads = splitRoads
t = 0
#p1.set_gl_state('translucent', blend=True, depth_test=True)

out = roadDraw(roads, len(roads[0]))
p1 = Plot3D(out[0], marker_size=2, width=20, color='red',
            edge_color='w', symbol='-',
            parent=view.scene, connect=out[1], face_color=(0.2, 0.2, 1, 0.8))  # face_color=(0.2, 0.2, 1, 0.8)

def update(ev):
    global t
    global roads
    global p2
    length = len(roads)
    increment = int(np.ceil(length / 1000))
    if t >= length:
        t = 0
    print(t, joints[t])
    out = roadDraw(roads[t:t+increment], len(roads[0]))
    if t != 0:
        p2.parent = None
    p2 = Plot3D(out[0], marker_size=10, width=20, color='blue',
       edge_color='w', symbol='-',
       parent=view.scene,connect=out[1], face_color=(0.2, 0.2, 1, 0.8)) #face_color=(0.2, 0.2, 1, 0.8)
    t += increment

timer = app.Timer()
timer.connect(update)
timer.start(2)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()
### DRAW ###
