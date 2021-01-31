import numpy as np
import vispy
import vispy.scene
from vispy.scene import visuals
from vispy import app
import sys

canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()
view.camera = 'turntable'
# generate data
def solver(t):
    pos = np.array([[np.sinh(t * i ** 0.01) * np.tan(i * t * 10) * 2, np.sin(t * i + t) * np.sin(i) * 1, np.tanh(i * t) * np.cos(i ** 2) * 0.25] for i in range(1, 5000, 1)])
    return pos
# These are the data that need to be updated each frame --^

scatter = visuals.Markers()
view.add(scatter)


#view.camera = scene.TurntableCamera(up='z')

# just makes the axes
axis = visuals.XYZAxis(parent=view.scene)


t = 0.0
def update(ev):
    global scatter
    global t
    t += 0.000001
    scatter.set_data(solver(t), edge_color=None, face_color=(1, 1, 1, .5), size=5)

timer = app.Timer()
timer.connect(update)
timer.start(0)
if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()