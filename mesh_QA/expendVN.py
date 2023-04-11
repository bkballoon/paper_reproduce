# from utils import loadobj, getAllNormVector
import csv
import utils
import numpy as np

path = "/home/simula/Dataset/liris/Armadillo_models/arma.obj"
vertices, faces = utils.loadobj(path)
v_to_mesh = utils.buildMapVerMesh(vertices, faces)
vn = np.zeros(len(vertices)*3).reshape(len(vertices), 3)
for i in v_to_mesh.keys():
    v_index = int(i)
    linked_mesh = v_to_mesh[i]
    v = np.array([0, 0, 0])
    for i in linked_mesh:
        threeP = faces[i]
        vec1 = np.array(vertices[threeP[0]-1])
        vec2 = np.array(vertices[threeP[1]-1])
        vec3 = np.array(vertices[threeP[2]-1])
        v = v + np.cross(vec1 - vec2, vec1 - vec3)
    v = v / len(linked_mesh)
    vn[v_index-1] = v

with open(path, "a") as file:
    for i in vn:
        string = 'vn '
        string += ' '.join([str(j) for j in i]) + '\n'
        file.write(string)

file.close()

# with open(path, "a") as file:
#     csv_writer = csv.writer(file)
#     csv_writer.writerow()