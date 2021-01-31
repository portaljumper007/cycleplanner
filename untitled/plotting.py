# pyline: disable=no-member
""" plot3d using existing visuals : LinePlotVisual """

import numpy as np
import sys

from vispy import app, visuals, scene

import random

# build visuals
Plot3D = scene.visuals.create_visual_node(visuals.LinePlotVisual)

# build canvas
canvas = scene.SceneCanvas(keys='interactive', title='plot3d', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 6

def roadDraw(roads, heights):
    out = []
    totalN = 0
    for road in range(len(roads)):
        N = np.size(roads[road][0])
        temp = np.empty((N - 1, 2), np.int32)
        temp[:, 0] = np.arange(totalN, totalN + N - 1)
        temp[:, 1] = np.arange(totalN, totalN + N - 1) + 1
        totalN += N
        out.extend(temp)
    connect = np.array(out)
    roadsConc = np.concatenate(roads,axis=1)
    roadsConc = np.append(roadsConc, heights.reshape((-1, len(heights))), axis=0)
    roadsTranspos = np.transpose(roadsConc)
    Plot3D(roadsTranspos, width=2.0, color='red',
       edge_color='w', symbol='vbar', face_color=(0.2, 0.2, 1, 0.8),
       parent=view.scene,connect=connect)

print(roadDraw([[[10,20,30,40], [10,20,30,40]], [[20,30,40,50], [10,20,30,40]]], np.zeros(8)))
if __name__ == '__main__':
    if sys.flags.interactive != 1:
        app.run()