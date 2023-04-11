import sys 
sys.path.append("..") 

import math
import numpy as np
import csv

from math import acos
from numba import typed, types, jit
from numba import cuda

from utils import loadobj, getAllNormVector, buildMapVerMesh, \
                        prepareLinesLists, preparePointSets, prepareVtoFace

'''
并行化思路：
    1. 首先并行的计算所有的点的曲率矩阵。
        1. 可以先将所有的预准备数据处理好，然后全部送到GPU中
        2. 计算所有点的曲率矩阵，然后返回这个曲率矩阵的集合
        3. 如果GPU不支持高维的列表，可以先将所有的矩阵转化成一维的，然后再转回来
        4. 
    2. 计算出曲率矩阵的列表之后
        1. 并行的计算
    3. python的这个代码好像不能并行化
    4. 总的思路就是并行化any可并行的程序
'''

'''
并行化曲率矩阵
    1. 每个线程做如下的操作
        1. 计算当前点的radius
        2. 计算曲率矩阵
    2. v_to_mesh, line_list 提前准备好
'''

@cuda.jit(device=True)
def a_device_function(a,b):
    return a + b, a+b+1

@cuda.jit(device=True)
def computeNormVectorOnPlane(point_index, faces, vertices):
    # point1, point2, point3 = types.int32(0), types.int32(0), types.int32(0)
    point1 = faces[point_index - 1][0] - 1 
    point2 = faces[point_index - 1][1] - 1
    point3 = faces[point_index - 1][2] - 1

    point1_x, point1_y, point1_z = vertices[point1][0], vertices[point1][1], vertices[point1][2]
    point2_x, point2_y, point2_z = vertices[point2][0], vertices[point2][1], vertices[point2][2]
    point3_x, point3_y, point3_z = vertices[point3][0], vertices[point3][1], vertices[point3][2]

    vec1_x, vec1_y, vec1_z = point1_x - point2_x, point1_y - point2_y, point1_z - point2_z
    vec2_x, vec2_y, vec2_z = point1_x - point3_x, point1_y - point3_y, point1_z - point3_z

    #  a.y*b.z-b.y*a.z , b.x*a.z-a.x*b.z , a.x*b.y-b.x*a.y
    return vec1_y*vec2_z-vec2_y*vec1_z, vec2_x*vec1_z-vec1_x*vec2_z, vec1_x*vec2_y-vec2_x*vec1_y

@cuda.jit(device=True)
def norm(a, b, c):
    return math.sqrt(a*a + b*b + c*c)
@cuda.jit(device=True)
def dot(a, b, c, d, e, f):
    return a*d + b*e + c*f
@cuda.jit
def testFunction(f):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x 
    acos(0.5)
    a, b = types.int32(1), types.int32(2)
    c = 2
    d, e = a_device_function(a, b)
    print(idx + d, c)

    return
    # 这个函数可以得知
    # cuda支持高维数组，cuda支持高维数组的取值

'''
1. 提前给定的参数有：v_to_mesh, line_list, vertices, faces, line_list的字典的keys转换成2个列表
2. 线程计算的参数有：radius
3. 本身主要的计算流程为：计算曲率矩阵
4. 返回曲率矩阵的列表
'''
'''
1. v_to_mesh:
2. points_sets[idx] vertex对应的相关点
3. line_sets[idx] vertex对应的面的相关线
4. 返回曲率矩阵的列表
'''
# v_to_facee, point_setss, line_listss,
        # reff_faces, reff_vetices, curvature_listss
@cuda.jit
def gpuCurvature(s, v_to_face, point_sets, line_lists, faces, vertices, curvature_lists):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    idx += s
    # print(idx)
    if idx < len(vertices):
        # print(idx)
        v_index = types.int32(0)
        x, y, z = types.float64(0.), types.float64(0.), types.float64(0.)
        radius = types.float64(0.)
        x_min, x_max = types.float64(9999), types.float64(-9999)
        y_min, y_max = types.float64(9999), types.float64(-9999)
        z_min, z_max = types.float64(9999), types.float64(-9999)
        # for i in range(len(point_sets[idx])):
        #     print(idx + int(point_sets[idx][1]))
        for i in range(len(point_sets[idx])):
            # 这里需要判断值是否为0，为0跳出循环
            v_index = int(point_sets[idx][i])
            if v_index > 0:
                x = vertices[v_index - 1][0]
                if x > x_max: x_max = x
                if x < x_min: x_min = x
                y = vertices[v_index - 1][1]
                if y > y_max: y_max = y
                if y < y_min: y_min = y
                z = vertices[v_index - 1][2]
                if z > z_max: z_max = z
                if z < z_min: z_min = z
        radius = math.sqrt(
            abs(x_max*x_max - x_min*x_min) + 
            abs(y_max*y_max - y_min*y_min) + 
            abs(z_max*z_max - z_min*z_min)
        )/2

        # print(radius)
        # 接下来就是将计算curvature矩阵
        B_modulus = types.float64(0.0)
        pi = types.float64(3.141592653589793)
        B_modulus = pi*radius*radius*4/3
        length = types.int32(0)
        length = int(len(line_lists[idx])/4)
        # for i in range(4):
        #     print(line_lists[idx][i])

        for i in range(length):
            v_index = int(line_lists[idx][i*4])
            # print(v_index, "----------")
            if v_index > 0:
                # 然后计算两个面的法向量
                # 这里调用上面的那个GPU函数即可
                v2 = int(line_lists[idx][4*i+2])
                v3 = int(line_lists[idx][4*i+3])
                vec1_x, vec1_y, vec1_z = computeNormVectorOnPlane(v2,faces, vertices)
                vec2_x, vec2_y, vec2_z = computeNormVectorOnPlane(v3,faces, vertices)
                beta_E = math.acos(dot(vec1_x, vec1_y, vec1_z, vec2_x, vec2_y, vec2_z)/norm(vec1_x, vec1_y, vec1_z)/norm(vec2_x, vec2_y, vec2_z))
                # print(vec1_x, vec1_y, vec1_z, vec2_x, vec2_y, vec2_z)
                v0 = int(line_lists[idx][4*i+0]) - 1
                v1 = int(line_lists[idx][4*i+1]) - 1
                # print(v0, v1, v2, v3)
                E_x, E_y, E_z = vertices[v0][0]-vertices[v1][0], \
                                vertices[v0][1]-vertices[v1][1], \
                                vertices[v0][2]-vertices[v1][2]
                l = norm(E_x, E_y, E_z)
                E_x = E_x/l
                E_y = E_y/l
                E_z = E_z/l
                # print(E_x, E_y, E_z, beta_E, l)
                # print(type(E_x*E_x*l*beta_E))
                # print(curvature_lists[idx][0][0])
                # print(curvature_lists[0][0])
                curvature_lists[idx][0][0] += E_x*E_x*l*beta_E
                curvature_lists[idx][0][1] += E_x*E_y*l*beta_E
                curvature_lists[idx][0][2] += E_x*E_z*l*beta_E
                
                curvature_lists[idx][1][0] += E_y*E_x*l*beta_E
                curvature_lists[idx][1][1] += E_y*E_y*l*beta_E
                curvature_lists[idx][1][2] += E_y*E_z*l*beta_E

                curvature_lists[idx][2][0] += E_z*E_x*l*beta_E
                curvature_lists[idx][2][1] += E_z*E_y*l*beta_E
                curvature_lists[idx][2][2] += E_z*E_z*l*beta_E
        for i in range(3):
            curvature_lists[idx][i][0] /= B_modulus
            curvature_lists[idx][i][1] /= B_modulus
            curvature_lists[idx][i][2] /= B_modulus


def callFunction(v_to_facee, point_setss, line_listss, \
    reff_faces, reff_vetices, curvature_listss, \
    upper_dir, tag):
    length = len(reff_vetices)
    l = int(length/2000)+1
    basic_mat = [
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ]
    curvature_lists = []
    for i in range(length):
        curvature_lists.append(basic_mat)
    curvature_lists = np.array(curvature_lists)
    for i in range(l):
        # print('进入第 {} 个2000'.format(i))
        gpuCurvature[20, 100](i*2000, v_to_facee, point_setss, line_listss,
            reff_faces, reff_vetices, curvature_listss)
        curvature_listss.copy_to_host(curvature_lists)
        start = i*2000
        for j in range(2000):
            index = start+j
            if index < length:
                writeEigen(curvature_lists[index], upper_dir, tag)
            else:
                break
    # testFunction[2,2](1.0)

def writeEigen(mat, upper_dir, tag):
    # with open(upper_dir + '\\eigen_' + str(tag) + '.csv', 'a') as file:
    # with open("/home/simula/Pro/textured_mesh_saliency/yang/geometry/arma.csv", 'a') as file:
    with open("/home/simula/Dataset/textured_model/bunny/saliency.csv", 'a') as file:
        csv_writer = csv.writer(file)
        li = []
        for i in range(3):
            for k in range(3):
                li.append(mat[i][k])
        csv_writer.writerow(li)

def AccEigen(mesh_path, upper_dir, tag):

    """这个函数的功能是计算某个mesh的所有点的曲率矩阵
    @param mesh_path: the mesh which needs to be computed 
    @param eigen_path: the path stores all the eigen
    @param tag: 0 means ref mesh while 1 means dis mesh
    """
    vertices, faces = loadobj(mesh_path)
    normals = getAllNormVector(vertices, faces)
    normals = np.array(normals)
    v_to_mesh = buildMapVerMesh(vertices, faces)
    v_to_face = prepareVtoFace(v_to_mesh)
    line_lists = prepareLinesLists(vertices, faces)
    point_sets = preparePointSets(vertices, faces)

    basic_mat = [
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.]
    ]
    curvature_lists = []
    l = len(vertices)
    for i in range(l):
        curvature_lists.append(basic_mat)

    curvature_lists = np.array(curvature_lists)
    # translate to the GPU
    veticess = cuda.to_device(vertices)
    facess = cuda.to_device(faces)
    # length = len(vertices)
    line_listss = cuda.to_device(line_lists)
    point_setss = cuda.to_device(point_sets)
    # normalss = cuda.to_device(normals)
    v_to_facee = cuda.to_device(v_to_face)
    curvature_listss = cuda.to_device(curvature_lists)
    
    # gpuRW[2,2](normalss, reff_faces, reff_vetices, length)
    # v_to_mesh, point_sets, line_lists, vertices, faces, curvature_lists
    # 输入的各种列表也好，矩阵也罢，必须有一个前提，那就是维数一定
    callFunction(v_to_facee, point_setss, line_listss, \
        facess, veticess, curvature_listss, \
        upper_dir, tag)
    # writeEigen(curvature_lists[0])
    


'''
1. 为什么出现keyerror这个bug, 因为两个build_vtomesh的函数写法不同
2. 接下来的程序该如何并行化呢？
3. 如果不能并行化的话，该如何最大限度的减少程序运行的时间？
4. 将所有可并行化的部分全部并行化
'''

'''
# 根据之前写的代码来看，出现角度nan值是代码的问题
1. 为什么第2000个点的二面角得出的结果是无限的
2. 为什么
'''
