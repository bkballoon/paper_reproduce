import numpy as np
from matplotlib import pyplot as plt 
import time
import os
from math import sqrt
import numba
from numba import cuda
from numba import njit, types, typed

import csv

# use igl AABB tree to match the related point in the other mesh
# why the closest point is not itself >:<

def main2():
    filename = '/home/zy/Desktop/avr/LIRIS_EPFL_GenPurpose/RockerArm_models/rockerArm-Taubin20.obj'
    # set a draw plane
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    # load(.obj file)
    vertices, faces = loadobj(filename)
    point_index = 10
    # vertices start from 0 the problem is there
    query_point = vertices[point_index-1]
    # v_to_mesh start from 1
    v_to_mesh = buildMapVerMesh(vertices, faces)
    face_contain_query = computePointInPlane(faces, query_point, vertices)
    print("contain the point {}".format(face_contain_query))
    print("point related mesh = {}".format(v_to_mesh[point_index]))
    axes.scatter3D(
        vertices[point_index][0], vertices[point_index][1], vertices[point_index][2], 
        color='yellow'
    )
    for i in face_contain_query:
        displayTriangle(vertices, axes, faces, i, 'red')
    
    plt.show()

def norm(v):
    length = sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
    # print("length {}".format(length))

    return length

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
            f = [int(values[1]), int(values[3]), int(values[2])]
            faces.append(f)
    return vertices, faces

def computePointToTriangle(vertices, query_point, i, triangle):
    try:
        # minus one cause face order begins from 1 not zero
        point_A = np.array(vertices[int(triangle[0]) - 1])
        point_B = np.array(vertices[int(triangle[1]) - 1])
        point_C = np.array(vertices[int(triangle[2]) - 1])
    except:
        print(i)
    try:
        AB = point_A - point_B
        AC = point_A - point_C
    except:
        print(point_A, "-----", point_B)
    N  = np.cross(AB, AC)
    Ax = N[0]
    By = N[1]
    Cz = N[2]
    D  = -(Ax * point_A[0] + By * point_A[1] + Cz * point_A[2])
    # plane equation => Ax + By + Cz + D = 0
    mod_d = Ax * query_point[0] + By * query_point[1] + Cz * query_point[2] + D
    mod_area = np.sqrt(np.sum(np.square([Ax, By, Cz])))
    if mod_d == 0.0:
        print("face_index = {} fenzi is zero".format(i))
    if mod_area == 0.0:
        print("face_index = {} fenmu is zero".format(i))

    try:
        d = abs(mod_d) / mod_area
    except:
        print(abs(mod_area), mod_d)
    return d

def computePointToPlane(faces, query_point, vertices):
    global f
    d_min = 100000
    d_min_index = 0
    for i in range(len(faces)):
        triangle = faces[i]
        d = computePointToTriangle(vertices, query_point, i, triangle)
        if d < d_min:
            if d == 0.0:
                f.append(i)
            d_min = d
            d_min_index = i
    return d_min, d_min_index

def computePointInPlane(faces, query_point, vertices):
    """
    @function:判断点是否在平面内部
    @thinking:对于每个点，判断给定模型的每个平面内是否含有该点
    """
    face_contain_query = []
    for i in range(len(faces)):
        triangle = faces[i]
        d = np.array(query_point)
        A = np.array(vertices[int(triangle[0]) - 1])
        B = np.array(vertices[int(triangle[1]) - 1])
        C = np.array(vertices[int(triangle[2]) - 1])
        S_ABd = norm(np.cross(A - d, B - d))
        S_ACd = norm(np.cross(A - d, C - d))
        S_BCd = norm(np.cross(B - d, C - d))
        S     = norm(np.cross(A - B, A - C))
        if S == (S_ABd + S_ACd + S_BCd):
            face_contain_query.append(i)
    return face_contain_query

def displayTriangle(vertices, axes, faces, face_index, color):
    
    threepoint = faces[face_index]
    p1, p2, p3 = threepoint[0], threepoint[1], threepoint[2]

    axes.plot3D(
        [vertices[p1][0], vertices[p2][0]],
        [vertices[p1][1], vertices[p2][1]],
        [vertices[p1][2], vertices[p2][2]], color = color
    )
    axes.plot3D(
        [vertices[p1][0], vertices[p3][0]],
        [vertices[p1][1], vertices[p3][1]],
        [vertices[p1][2], vertices[p3][2]], color = color
    )
    axes.plot3D(
        [vertices[p3][0], vertices[p2][0]],
        [vertices[p3][1], vertices[p2][1]],
        [vertices[p3][2], vertices[p2][2]], color = color
    )

def readAllObj():
    '''
        batch read .obj file
    '''
    file_list = []
    mother_path = '/home/zy/Desktop/avr/LIRIS_EPFL_GenPurpose/RockerArm_models/'
    files = os.listdir(mother_path)
    for file_name in files:
        new_path = mother_path + str(file_name)
        vertices, faces = loadobj(new_path)
        print(" file_name = {}, faces_len = {}, vertices_len = {}".format(file_name, len(faces), len(vertices)))
        file_list.append(new_path)
    return file_list

def writeobj(filename, tag, tag_list):
    with open(filename, 'w') as file:
        for i in tag_list:
            tag_line = ' '.join(i)
            file.readline(str(tag) + ' ' + tag_line)
    file.close()
    return 

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

def displayComparison(axes, vertices, faces):
    # faces1是使得平面为方程为0的mesh列表
    faces1 = [39533, 39664, 39767, 39920, 39984, 40172]
    # faces2是v-to-mesh求出的点相关的mesh列表
    faces2 = [37234, 37312, 37478, 37586, 37705, 37786]
    for i in faces1:
        displayTriangle(vertices, axes, faces, i, 'pink')
    for i in faces2:
        displayTriangle(vertices, axes, faces, i, 'green')
    return 

def main1():
    global f
    f = []
    # set a draw plane
    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')

    # this is reference mesh
    filename = '/home/zy/Desktop/avr/LIRIS_EPFL_GenPurpose/RockerArm_models/rockerArm.obj'
    vertices, faces = loadobj(filename)
    point_index = 10
    query_point = vertices[point_index]
    # 例：第10个点在obj文件中表示为11，在list中表现为9

    # 显示参考网格中的点
    point_index += 1
    axes.scatter3D(
        vertices[point_index][0], vertices[point_index][1], vertices[point_index][2], 
        color='red'
    )
    point_index -= 1

    filename = '/home/zy/Desktop/avr/LIRIS_EPFL_GenPurpose/RockerArm_models/rockerArm-Taubin20.obj'
    vertices, faces = loadobj(filename)
    # v_to_mesh = buildMapVerMesh(vertices, faces)
    min_dis, face_index = computePointToPlane(faces, query_point, vertices)
    print("min_dis = {}, face_index = {}".format(min_dis, face_index))
    # print("point related mesh = {}".format(v_to_mesh[point_index]))

    # 显示畸变网格中的距离点最近的平面
    displayTriangle(vertices, axes, faces, face_index, 'red')
    # 显示畸变网格中对应序号的点
    # for face_index in v_to_mesh[point_index]:
    #     displayTriangle(vertices, axes, faces, face_index, 'blue')
    plt.show()

@cuda.jit
def computeRelatedTriangle(reference_vertices, distorted_faces, distorted_vertices, point_related_face, v_length):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    if idx < v_length:
        count = 0
        d_min = 1000
        AB_0, AB_1, AB_2 = 0.0, 0.0, 0.0
        AC_0, AC_1, AC_2 = 0.0, 0.0, 0.0
        Ax = types.float32(0.0)
        By = types.float32(0.0)
        Cz = types.float32(0.0)
        D  = types.float32(0.0)
        query_point_0, query_point_1, query_point_2 = reference_vertices[idx][0], reference_vertices[idx][1], reference_vertices[idx][2]

        for i in range(v_length):
            count += 1
            triangle_0, triangle_1, triangle_2 = distorted_faces[i][0], distorted_faces[i][1], distorted_faces[i][2]
            pointA_0, pointA_1, pointA_2 = distorted_vertices[triangle_0 - 1][0], distorted_vertices[triangle_0 - 1][1], distorted_vertices[triangle_0 - 1][2]
            pointB_0, pointB_1, pointB_2 = distorted_vertices[triangle_1 - 1][0], distorted_vertices[triangle_1 - 1][1], distorted_vertices[triangle_1 - 1][2]
            pointC_0, pointC_1, pointC_2 = distorted_vertices[triangle_2 - 1][0], distorted_vertices[triangle_2 - 1][1], distorted_vertices[triangle_2 - 1][2]

            AB_0 = pointA_0 - pointB_0
            AC_0 = pointA_0 - pointC_0
            AB_1 = pointA_1 - pointB_1
            AC_1 = pointA_1 - pointC_1
            AB_2 = pointA_2 - pointB_2
            AC_2 = pointA_2 - pointC_2

            Ax = AB_1*AC_2 - AB_2*AC_1
            By = AB_2*AC_0 - AB_0*AC_2
            Cz = AB_0*AC_1 - AB_1*AC_0

            D  = -(Ax * pointA_0 + By * pointA_1 + Cz * pointA_2)
            mod_d = Ax * query_point_0 + By * query_point_1 + Cz * query_point_2 + D
            mod_area = Ax*Ax + By*By + Cz*Cz 
            d = abs(mod_d)/mod_area

            if d < d_min:
                d_min = d
                point_related_face[idx] = i

def AccRelateFace(ref_file_name, dis_file_name, related_face_path):
    """指定Ref_file_name和Dis_file_name，然后计算Ref点相关面，然后写入指定文件
    
    @param ref_file_name: this filename is stable, future version could be placed with
    its .obj file
    @param dis_file_name: this filename is distorted file's name
    @param related_face_path: this file path stores the data computed by this function
    """
    res = []
    ref_vertices, ref_faces = loadobj(ref_file_name)
    
    # file_list = readAllObj()[:1]

    dis_vertices, dis_faces = loadobj(dis_file_name)
    v_length = len(ref_vertices)

    N = 1000
    M = int(v_length/1000) + 1
    related_face = np.zeros(v_length)

    ref_vertices = np.array(ref_vertices)
    dis_vertices = np.array(dis_vertices)
    dis_faces = np.array(dis_faces)
    
    # 把几个数组传到GPU中去
    reference_vertices = cuda.to_device(ref_vertices)
    distorted_vertices = cuda.to_device(dis_vertices)
    distorted_faces = cuda.to_device(dis_faces)
    point_related_face = cuda.to_device(np.zeros(v_length))
    # call kernal function
    computeRelatedTriangle[M, N](reference_vertices, distorted_faces, distorted_vertices, point_related_face, v_length)
    # 传回来
    point_related_face.copy_to_host(related_face)
    cuda.synchronize()
    res.append(related_face)
    f = open(related_face_path + '\\related_face.csv', 'w', encoding='utf-8')
    writer = csv.writer(f)
    for line in related_face:
        writer.writerow([str(line)])
    f.close()

# 从结果上来看，完全可以在笔记本上，完成这个部分的内容
# 一组模型大概耗时不到3秒，这个时间完全是可以承受的
# 然后将得到的所有的数据，写成csv的文件，然后将所有的文件移植到本机上，这样就可以完成程序。

# 这部分的函数暂时不进行测试

# 我真是吐了，