import numpy as np
import csv
import os

def readAllObj(data_uppper_path):
    '''
        batch read .obj file
    '''
    file_list = []
    files = os.listdir(data_uppper_path)
    # print(files)
    for file_name in files:
        if file_name != 'rockerArm.obj':
            new_path = data_uppper_path + str(file_name)
            file_list.append(new_path)
    return file_list

def computeNormVectorOnPlane(point_index, faces, vertices):
    point1 = faces[point_index - 1][0] - 1 
    point2 = faces[point_index - 1][1] - 1
    point3 = faces[point_index - 1][2] - 1
    vec1 = np.array(vertices[point1]) - np.array(vertices[point2])
    vec2 = np.array(vertices[point1]) - np.array(vertices[point3])
    # print(vec1, vec2)
    vector = np.cross(vec1, vec2)
    return vector

def getAllNormVector(vertices, faces):
    normals = []
    for i in range(len(faces)):
        normals.append(computeNormVectorOnPlane(i, faces, vertices))
    return normals

def loadobj(filename):
    # there just process two elements f and v
    vertices = []
    faces = []
    for line in open(filename):
        if line.startswith('#'):continue
        values = line.split()
        if values[0] == 'v':
            v = [float(x) for x in values[1:4]]
            v = [v[0], v[2], v[1]]
            vertices.append(v)
        elif values[0] == 'f':
            f = [int(values[1].split('/')[0]),
                 int(values[2].split('/')[0]),
                 int(values[3].split('/')[0])]
            faces.append(f)
    return vertices, faces

def computeLinesInSphere(v_to_mesh, vertex_index, vertices, faces):

    face_list = v_to_mesh[vertex_index]
    line_list = dict()
    # 这里想个点子，在记录线的同时记录下对应的两个面
    # 用字典来记录，
    for i in face_list:
        face_ = faces[i]
        # 这里的face_即三个顶点的列表
        # keys 的形式类似“199200”其中199 200 分别是点在vertices的index
        for k in range(3):
            s1 = k%3
            s2 = (k + 1)%3
            line = sorted([face_[s1], face_[s2]])
            line = '+'.join([str(i) for i in line])
            if line_list.get(line):
                line_list[line] += [i]
            else:
                line_list[line] = [i]
    # 这里得到的line_list就是线和相关的面
    return line_list

def getOneRingField(vertex_index,v_to_mesh, faces):

    # 获取点相关mesh
    # v_to_mesh要求的点的是从1开始的
    face_list = v_to_mesh[vertex_index]
    # 一环邻域的点的容器
    point_set = set()
    for i in face_list:
        i -= 1
        face_with_3_point = faces[i]
        for j in range(3):
            point_set.add(face_with_3_point[j])
    return point_set

def buildMapVerMesh(vertices, faces):
    # 这里的keys是从0开始的faces的index
    v_to_mesh = dict()
    for index in range(len(faces)):
        face = faces[index]
        for vertex in face:
            if vertex not in v_to_mesh.keys():
                v_to_mesh[vertex] = [index]
            else:
                v_to_mesh[vertex].append(index)
    return v_to_mesh

def prepareLinesLists(vertices, faces):
    v_to_mesh = buildMapVerMesh(vertices, faces) # 对所有的点及其face
    line_lists = []
    l = 40
    for vertex_index in range(len(vertices)):
        line_list = computeLinesInSphere(v_to_mesh, vertex_index+1, vertices, faces)
        # print(line_list)
        tmp = []        
        for k, v in line_list.items():
            points = k.split('+')
            # print(points)
            if len(v) >= 2:
                tmp += [int(points[0])] + [int(points[1])] + v
        tmp += list(np.zeros(int(l - len(tmp)/4)*4))
        line_lists.append(tmp)
    return line_lists


def preparePointSets(vertices, faces):
    v_to_mesh = buildMapVerMesh(vertices, faces)
    point_sets = []
    l = 40
    for i in range(len(vertices)):
        point_set = getOneRingField(i + 1, v_to_mesh, faces)
        t = list(point_set)
        t2 = list(np.zeros(l - len(t)))
        point_sets.append(t + t2)
    return point_sets

def prepareVtoFace(v_to_mesh):
    l = 40
    v_to_face = [list(np.zeros(l))]
    for i in range(len(v_to_mesh)):
        v_to_face.append([])
    for i in range(len(v_to_mesh)):
        # print(l - len(v_to_mesh[i+1]))
        t = list(np.zeros(l - len(v_to_mesh[i+1])))
        v_to_face[int(i+1)] = v_to_mesh[i+1] + t
    return v_to_face

def loadRelatedFace(path):
    # path = 'E:\\zy_QA\\SVR-mesh-v2\\data\\related_face.csv'
    relate_face = []
    file = open(path, 'r')
    rows = csv.reader(file)
    for i in rows:
        if len(i) > 0:
            relate_face.append(int(float(i[0])))
    return relate_face

def loadValueAndVector(path):
    
    eigen_value = []
    eigen_vector = []
    file = open(path, 'r')
    rows = csv.reader(file)
    for i in rows:
        if len(i) == 3:
            eigen_value.append(np.array(i, dtype=np.float))
        elif len(i) == 9:
            eigen_vector.append(np.array(i, dtype=np.float).reshape(3,3))        
        else:
            continue
    return eigen_value, eigen_vector

def writeValueAndVector(upper_dir):
    """这个函数的功能是读取两个模型的曲率矩阵的列表，然后分别计算特征值和特征向量
    @param upper_dir: 上层目录所在位置
    """
    load_path = upper_dir + '\\eigen_1.csv'
    curvature_list = loadCurvatureEigen(load_path)
    write_path = upper_dir + '\\eigen_1_vv.csv'
    file = open(write_path, 'w')
    csv_writer = csv.writer(file)
    for i in curvature_list:
        tmp = []
        value, vector = np.linalg.eig(i)
        for j in vector:
            tmp += list(j)
        csv_writer.writerow(value)
        csv_writer.writerow(tmp)
    file.close()
    
    path = upper_dir + '\\eigen_0.csv'
    curvature_list = loadCurvatureEigen(path)
    write_path = upper_dir + '\\eigen_0_vv.csv'
    file = open(write_path, 'w')
    csv_writer = csv.writer(file)
    for i in curvature_list:
        tmp = []
        value, vector = np.linalg.eig(i)
        for j in vector:
            tmp += list(j)
        csv_writer.writerow(value)
        csv_writer.writerow(tmp)
    file.close()

def loadCurvatureEigen(path):
    # count = 0
    curvature_lists = []
    with open(path, 'r') as file:
        rows = csv.reader(file)
        for i in rows:
            if len(i) == 9:
                tmp = np.zeros(9).reshape(3,3)
                for j in range(3):
                    tmp[j][0] = i[j*3+0]
                    tmp[j][1] = i[j*3+1]
                    tmp[j][2] = i[j*3+2]
                curvature_lists.append(tmp)
    return curvature_lists
