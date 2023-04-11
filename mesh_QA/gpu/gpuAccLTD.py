import numpy as np 
import math
import csv

import numba
from numba import jit
from numba import cuda
from numba import types

from utils import \
    loadobj, getAllNormVector, buildMapVerMesh, \
    prepareLinesLists, preparePointSets, prepareVtoFace, \
    loadCurvatureEigen, loadValueAndVector, loadRelatedFace

pi = 3.141592653589793

@cuda.jit(device=True)
def getMinMaxValueIndex(eigen_value):
    max_value_index = types.int32(0)
    min_value_index = types.int32(0)
    max_value = types.float64(-9999.0)
    min_value = types.float64(9999.0)

    for i in range(3):
        if eigen_value[i] > max_value:
            max_value = eigen_value[i]
            max_value_index = i
        if eigen_value[i] < min_value:
            min_value = eigen_value[i]
            min_value_index = i
    return min_value_index, max_value_index


@cuda.jit(device=True)
def norm(a, b, c):
    # print(a, '=', b, '=', c, 'length = ', math.sqrt(a*a + b*b + c*c))
    return math.sqrt(a*a + b*b + c*c)

@cuda.jit(device=True)
def dot(a, b, c, d, e, f):
    return a*d + b*e + c*f

@cuda.jit(device=True)
def getAngula(a, b, c, d, e, f):
    return math.acos(dot(a,b,c,d,e,f)/norm(a,b,c)/norm(d,e,f))

@cuda.jit(device=True)
def mean(li):
    tmp = types.float64(0.0)
    length = len(li)
    for i in range(length):
        tmp += li[i]
    return tmp/length
# TIP：
# 1代表REF，2代表DIS
@cuda.jit
def gpuLTD(s, relate_face, dis_faces, \
    eigen_values1, eigen_vectors1, \
    eigen_values2, eigen_vectors2, \
    res_list):

    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    idx += s
    middle_parameter_two = types.float64(0.0)
    if idx < len(relate_face):
        # if idx == 1314:
        # print(relate_face[idx])
        for i in range(3):
            # 对于每个点对应的face，求该face所包含的三个点和ref中的原始点的畸变关系
            # 然后对这个畸变关心进行量化，量化完之后，返回一个列表，同时对该列表的数据进行写入
            # 这里使用faces的目的是为了确定点相关平面所包含的三个点，并且得知他们的的vertex_index
            vertex_index2 = dis_faces[relate_face[idx]][i] - 1
            # 只能这样了，为什么这么写，因为当点相同，且畸变较小时
            # 两个曲率矩阵是一样的，这种情况下畸变为0
            if vertex_index2 == idx:
                continue
            # print(vertex_index2, '---gpu---')
            # print(eigen_values1[idx][0], eigen_values1[idx][1], eigen_values1[idx][2])
            # print(eigen_values2[vertex_index2][0], eigen_values2[vertex_index2][1], eigen_values2[vertex_index2][2])
            # print("vertex_index", dis_faces[relate_face[idx]][i])
            min_value_index1, max_value_index1 = getMinMaxValueIndex(
                eigen_values1[idx])
            min_value_index2, max_value_index2 = getMinMaxValueIndex(
                eigen_values2[vertex_index2])
            # print(eigen_values1[idx][min_value_index1])
            # print(eigen_values1[idx][max_value_index1])
            theta_min = getAngula(
                eigen_vectors1[idx][0][min_value_index1],
                eigen_vectors1[idx][1][min_value_index1],
                eigen_vectors1[idx][2][min_value_index1],
                eigen_vectors2[vertex_index2][0][min_value_index2],
                eigen_vectors2[vertex_index2][1][min_value_index2],
                eigen_vectors2[vertex_index2][2][min_value_index2],
            )
            
            theta_max = getAngula(
                eigen_vectors1[idx][0][max_value_index1],
                eigen_vectors1[idx][1][max_value_index1],
                eigen_vectors1[idx][2][max_value_index1],
                eigen_vectors2[vertex_index2][0][max_value_index2],
                eigen_vectors2[vertex_index2][1][max_value_index2],
                eigen_vectors2[vertex_index2][2][max_value_index2],
            )

            epsilon1 = mean(eigen_values1[idx])*0.05
            epsilon2 = mean(eigen_values2[vertex_index2])*0.05
    # 为什么两个不同的eigenvalues得到一样的值
            # print(idx, '---', eigen_values1[idx][min_value_index1])
            # print(vertex_index2, '---', \
            #     eigen_values2[vertex_index2][min_value_index2])

            delta_min = abs(eigen_values1[idx][min_value_index1] - \
                        eigen_values2[vertex_index2][min_value_index2]) / \
                        abs(eigen_values1[idx][min_value_index1] + \
                        eigen_values2[vertex_index2][min_value_index2] + epsilon1)

            delta_max = abs(eigen_values1[idx][max_value_index1] - \
                        eigen_values2[vertex_index2][max_value_index2]) / \
                        abs(eigen_values1[idx][max_value_index1] + \
                        eigen_values2[vertex_index2][max_value_index2] + epsilon2)

            # print(delta_min, delta_max)
            middle_parameter_two += theta_min/pi*2*delta_min + \
                                    theta_max/pi*2*delta_max
            # print(middle_parameter_two)

        res_list[idx - s] = middle_parameter_two/3

def writeParameterTwo(index, length, res_list, upper_dir):
    # path = 'E:\\zy_QA\\SVR-mesh-v2\\data\\parameter2.csv'
    path = upper_dir + '\\parameter2.csv'
    file = open(path, 'a')
    csv_writer = csv.writer(file)
    for para1 in range(len(res_list)):
        if index + para1 < length:
            csv_writer.writerow([res_list[para1]])
        else:
            break

def AccLTD(related_path, ref_vv_path, dis_vv_path, dis_mesh_path, upper_dir):
    '''
    测试LTD的函数，如果成功，就写一个start——v2的程序，来调度所有的function
    
    然后，结果就出来了
    '''


    # 设定所有的需要进行GPU的参数，并且对这些参数进行必要的初始化内容。
    relate_face = loadRelatedFace(related_path)
    dis_vertices, dis_faces = loadobj(dis_mesh_path)
    eigen_values1, eigen_vectors1 = loadValueAndVector(ref_vv_path)
    eigen_values2, eigen_vectors2 = loadValueAndVector(dis_vv_path)
    res_list = np.zeros(2000)

    # 数据传输到GPU中
    relate_facee = cuda.to_device(relate_face)
    dis_facess = cuda.to_device(dis_faces)
    eigen_values11 = cuda.to_device(eigen_values1)
    eigen_values22 = cuda.to_device(eigen_values2)
    eigen_vectors11 = cuda.to_device(eigen_vectors1)
    eigen_vectors22 = cuda.to_device(eigen_vectors2)
    res_listt = cuda.to_device(res_list)
    length = len(dis_vertices)
    l = int(length/2000)+1

    # nan_list = []
    # 得到的结果返回，并且写入
    for i in range(l):
        gpuLTD[20,100](i*2000, relate_facee, dis_facess, \
            eigen_values11, eigen_vectors11, \
            eigen_values22, eigen_vectors22, \
            res_listt)
        res_listt.copy_to_host(res_list)
        writeParameterTwo(i*2000, length, res_list, upper_dir)
    # print(nan_list)
    # print(relate_face[10])
    # print(dis_faces[relate_face[10]])

'''
为什么计算出来的LTP值会出现nan？
    1. 因为当畸变模型和参考模型之间的畸变很小时，会出现
    ref_vertex=10 relate to dis_vertex=10,11,12
    这种时候10对应10的话，在计算结果会出现NAN，具体是
    两个模型10处的畸变矩阵对应的曲率幅度相同，自然的deltamin分子
    就是0，所以会出现NAN
    2. 简单解决，跳过
'''

