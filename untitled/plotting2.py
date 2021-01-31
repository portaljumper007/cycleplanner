import sys

# pylint: disable=no-member
""" scatter using MarkersVisual """

import numpy as np
from vispy import app, visuals, scene

# build your visuals
Scatter3D = scene.visuals.create_visual_node(visuals.MarkersVisual)

# The real-things : plot using scene
# build canvas
canvas = scene.SceneCanvas(keys='interactive', show=True)

# Add a ViewBox to let the user zoom/rotate
view = canvas.central_widget.add_view()
view.camera = 'turntable'
view.camera.fov = 45
view.camera.distance = 500

# data
def func(t):
    n = 5000
    pos = np.zeros((n, 3))
    colors = np.ones((n, 4), dtype=np.float32)
    radius, theta, dtheta = 1.0, 0.0, 5.5 / 180.0 * np.pi
    for i in range(n):
        theta += dtheta
        x = 0.0 + radius * np.cos(theta)
        y = 0.0 + radius * np.sin(theta) + np.sin(t * i)
        z = 1.0 * radius
        r = 10.1 - i * 0.02
        radius -= 0.1
        pos[i] = x, y, z
        colors[i] = (i/n, 1.0-i/n, 0, 0.8)
    return pos, colors

t = 0
p1 = Scatter3D(parent=view.scene)
p1.set_gl_state('translucent', blend=True, depth_test=True)
def update(ev):
    global t
    global p1
    t += 0.1
    out = func(t)
    p1.set_data(out[0], face_color=out[1])
    p1.symbol = visuals.marker_types[10]

timer = app.Timer()
timer.connect(update)
timer.start(0)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()