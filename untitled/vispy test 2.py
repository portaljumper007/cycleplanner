import sys
import numpy as np
from scipy.interpolate import griddata
import vispy
from vispy import app, scene
from vispy import color
from vispy.util.filter import gaussian_filter

canvas = scene.SceneCanvas(keys='interactive', bgcolor='w')
view = canvas.central_widget.add_view()
view.camera = 'fly'
view.camera.fov = 45
view.camera.distance = 6

heights = np.load("heights.npy", allow_pickle=True)
properties = np.load("properties.npy", allow_pickle=True)

RES = properties[1]

yBounds = (properties[0][0][1], properties[0][1][1])
xBounds = (properties[0][0][0], properties[0][1][0])

gridWidth = np.int((xBounds[1] - xBounds[0]) / RES[0]) + 1
gridHeight = np.int((yBounds[1] - yBounds[0]) / RES[1]) + 1

heightChunks = [[[], [], []] for i in range(gridWidth * gridHeight)]

tally = 0
try:
    zGrid = np.load("regularZGrid.npy", allow_pickle=True)
except:
    print("Gridding the heightmap...")
    lenHeights = len(heights[0])
    for coord in range(lenHeights):
        if coord % 1000 == 0:
            print(coord / lenHeights)
        whichBox = [(heights[0][coord] - xBounds[0]) // RES[0], (heights[1][coord] - yBounds[0]) // RES[1]] #time to do some upside down deinterlacing magic
        if 0 <= whichBox[0] < gridWidth and 0 <= whichBox[1] < gridHeight:
            index = int((whichBox[1] * gridWidth) + whichBox[0])
            heightChunks[index][0].append(heights[0][coord])
            heightChunks[index][1].append(heights[1][coord])
            heightChunks[index][2].append(heights[2][coord])

    lon = np.linspace(xBounds[0], xBounds[1], 100+2)[1:-1]
    lat = np.linspace(yBounds[0], yBounds[1], 100+2)[1:-1]
    X, Y = np.meshgrid(lon, lat)
    X = X.flatten()
    Y = Y.flatten()
    #result = griddata((heights[0][:9000000], heights[1][:9000000]), heights[2][:9000000], (X, Y), method='cubic')
    #print(result)
    coordsToInterpolate = [[], [], []]
    regularHeights = [[], [], []]
    zGrid = [[] for i in range(len(lat))]
    lenX = len(X)
    for coord in range(lenX):
        coordsToInterpolate = [[], [], []]
        if coord % 100 == 0:
            print(coord / lenX)
        for i in range(-1, 2):
            for j in range(-1, 2):
                whichBox = [(X[coord] - xBounds[0] + (i*RES[0])) // RES[0], (Y[coord] - yBounds[0] + (j*RES[1])) // RES[1]] #time to do some upside down deinterlacing magic
                if 0 <= whichBox[0] < gridWidth and 0 <= whichBox[1] < gridHeight:
                    index = int((whichBox[1] * gridWidth) + whichBox[0])
                    coordsToInterpolate[0].extend(heightChunks[index][0])
                    coordsToInterpolate[1].extend(heightChunks[index][1])
                    coordsToInterpolate[2].extend(heightChunks[index][2])
        result = griddata((coordsToInterpolate[0], coordsToInterpolate[1]), coordsToInterpolate[2], (X[coord], Y[coord]), method='cubic')
        #regularHeights[0].append(X[coord])
        #regularHeights[1].append(Y[coord])
        #regularHeights[2].append(result)
        zGrid[(coord)//len(lon)].append(result)
    np.save("regularZGrid.npy", zGrid)

#print("a")
#z = np.random.normal(size=(101, 101), scale=200)
#print("b")
#z[100, 100] += 50000
#print("a")
#z = gaussian_filter(z, (10, 10))
#print("b")
#print(z)
#zGrid = np.array(zGrid)
zGrid = np.load("basemaps.npy", allow_pickle=True)[0]
quality = 0.1

zGridComp = []
step = int(1/quality)
row = 0
while row < len(zGrid):
    zGridComp.append(zGrid[row][0::step])
    row += step

zGrid = np.array(zGridComp)

p1 = scene.visuals.SurfacePlot(z=zGrid, color=(0.3, 1, 0.1, 1))
cnorm = zGrid / abs(np.amax(zGrid))
c = color.get_colormap("greens").map(cnorm).reshape(zGrid.shape + (-1,))
c = c.flatten().tolist()
c=list(map(lambda x,y,z,w:(x,y,z,w), c[0::4],c[1::4],c[2::4],c[3::4]))
p1.mesh_data.set_vertex_colors(c) # but explicitly setting vertex colors does work?
p1.transform = scene.transforms.MatrixTransform()
p1.transform.scale([1110000, 1110000., 1])
view.add(p1)
print("ok")

xax = scene.Axis(pos=[[0, 0], [1, 0]], tick_direction=(0, -1), axis_color='r', tick_color='r', text_color='r', font_size=16, parent=view.scene)
yax = scene.Axis(pos=[[0, 0], [0, 1]], tick_direction=(-1, 0), axis_color='g', tick_color='g', text_color='g', font_size=16, parent=view.scene)

zax = scene.Axis(pos=[[0, 0], [-1, 0]], tick_direction=(0, -1), axis_color='b', tick_color='b', text_color='b', font_size=16, parent=view.scene)
zax.transform = scene.transforms.MatrixTransform()  # its acutally an inverted xaxis
zax.transform.rotate(90, (0, 1, 0))  # rotate cw around yaxis
zax.transform.rotate(-45, (0, 0, 1))  # tick direction towards (-1,-1)

initial_camera_dir = (0, -1, 0) # for a default initialised camera
p1.light_dir = initial_camera_dir
initial_light_dir = view.camera.transform.imap(initial_camera_dir)[:3]

#@canvas.connect
#def on_mouse_move(event):
    #print(p1.light_dir)
    #transform = view.camera.transform
    #dir = np.concatenate((initial_light_dir, [0]))
    #p1.light_dir = transform.map(dir)[:3]

if __name__ == '__main__':
    canvas.show()
    if sys.flags.interactive == 0:
        app.run()