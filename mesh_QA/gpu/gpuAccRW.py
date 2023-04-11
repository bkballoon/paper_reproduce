import sys 
sys.path.append("..") 

import math
import numpy as np
import csv
# import scipy

import numba
from math import acos
from numba import typed, types, jit
from numba import cuda

from utils import \
    loadobj, getAllNormVector, buildMapVerMesh, \
    prepareLinesLists, preparePointSets, prepareVtoFace, \
    loadCurvatureEigen, loadValueAndVector, writeValueAndVector

'''
1. 本页所有的操作只涉及Ref模型
    1. 求本质矩阵的GPU形式
    2. 参数涉及：point-lists；曲率矩阵lists；idx；
    3. 由于需要求一组数的平均数，所以需要给定一个列表，同时由于numba
    不支持列表，所以在给定参数时，直接给一个列表，这个列表用来存储这个数组
    4. 目前来看需要的GPU函数需要有：getAngular(),mean(),eig()
2. 一些细节的更改
    1. 由于numpy求特征值的速度很快，可以在求得特征矩阵之后，在cpu求特征向量和
    特征值，然后写进csv文件中。
    2. 然后再利用这些特征值和特征向量进行操作

'''
@cuda.jit(device=True)
def getAngula(a, b, c, d, e, f):
    return math.acos(dot(a,b,c,d,e,f)/norm(a,b,c)/norm(d,e,f))

@cuda.jit(device=True)
def norm(a, b, c):
    return math.sqrt(a*a + b*b + c*c)

@cuda.jit(device=True)
def dot(a, b, c, d, e, f):
    return a*d + b*e + c*f

@cuda.jit(device=True)
def mean(li):
    tmp = types.float64(0.0)
    length = len(li)
    for i in range(length):
        tmp += li[i]
    return tmp/length

# 
# 这个函数不行，在精度设置方面一旦出现误差就会导致特征值为0
@cuda.jit(device=True)
def eig(mat):    
    cols = mat.shape[1]

    Q = np.eye(cols)
    R = np.copy(mat)

    for col in range(cols-1):
        a = np.linalg.norm(R[col:, col])
        e = np.zeros((cols- col))
        e[0] = 1.0
        num = R[col:, col] -a*e
        den = np.linalg.norm(num)

        u = num / den
        H = np.eye((cols))
        H[col:, col:] = np.eye((cols- col))- 2*u.reshape(-1, 1).dot(u.reshape(1, -1))
        R = H.dot(R)

        Q = Q.dot(H)

    return Q, R
# GPU中的求特征值和特征向量的函数
# 貌似numba支持求特征值特征向量的函数

# 从其他函数的测试可以看出，numpy可以导入字符串，但是只能输出字符不能输出字符串
# 或者可以这么说，GPU中的print和c/c++中的printf相似，只能输出单个“东西”
@cuda.jit
def testEig(index):
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    print(idx)
    # print(value, vector)
    # a = ((1, 2, 3), (2,3,4), (3,4,5))
    # for i in range(len(a)):
    #     print(a[i][0])
    #     print(a[i][1])
    #     print(a[i][2])

# 自己写求矩阵特征值和特征向量的GPU函数
# 按照QR分解的houser方法来看，这种求解的方式虽然有着精度高，耗时短的有点
# 但是在工程中，由于存在一定的精度误差，导致在写实际代码的过程中，反而出现了
# “设置有误，特征值不准确”的现象。


# 特征值构成的列表
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

# 计算投影向量
@cuda.jit(device=True)
def getPorjectVector(vx, vy, vz, normal):
    value1 = dot(vx, vy, vz, normal[0], normal[1], normal[2])
    value2 = dot(normal[0], normal[1], normal[2], \
                normal[0], normal[1], normal[2])
    return (vx - normal[0]*value1/value2, \
            vy - normal[1]*value1/value2, \
            vz - normal[2]*value2/value2)

@cuda.jit
def gpuRW(s, v_to_face, point_sets, normals, eigen_values, eigen_vectors, \
    angle_list, res_list):
    '''
    1. 本函数进行RW参数的并行化
    2. eigen_values是一个2d的矩阵，矩阵的每一行是特征矩阵的特征值
    3. eigen_vectors是一个3d的矩阵，矩阵的每一个元素是一个特征向量构成的矩阵
    '''
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    idx += s
    mean_k = types.float64(0.0)
    if idx < len(v_to_face):
        count = 0
        for i in range(len(point_sets[idx])):
            # print(eigen_values[point_sets[idx][i]])
            v_index = int(point_sets[idx][i]) - 1
            if v_index > 0:
                # print(v_index, eigen_values[v_index][0])
                mean_k += mean(eigen_values[v_index])
                min_value_index, max_value_index = getMinMaxValueIndex(
                    eigen_values[v_index]
                )
                # print(eigen_vectors[v_index][0][min_value_index])
                # 计算投影向量
                prej_min = getPorjectVector(
                    eigen_vectors[v_index][0][min_value_index],
                    eigen_vectors[v_index][1][min_value_index],
                    eigen_vectors[v_index][2][min_value_index],
                    normals[v_index]
                )
                prej_max = getPorjectVector(
                    eigen_vectors[v_index][0][max_value_index],
                    eigen_vectors[v_index][1][max_value_index],
                    eigen_vectors[v_index][2][max_value_index],
                    normals[v_index]
                )
                angle_list[i] = getAngula(
                    prej_min[0], prej_min[1], prej_min[2],
                    prej_max[0], prej_max[1], prej_max[2]
                )
                count += 1

        # # # rw_k
        mean_value = mean_k/count
        # # rw_gamma
        mean_gamma = types.float64(0.0)
        mean_gamma = mean(angle_list)
        res = types.float64(0.0)
        for i in range(count):
            res += (angle_list[i] - mean_gamma) * (angle_list[i] - mean_gamma)
        rwgamma = math.sqrt(res)
        res_list[idx - s] = rwgamma*mean_value
        # print(rwgamma*mean_value)

def writeParameterOne(index, length, res_list, upper_dir):
    path = upper_dir + '\\parameter1.csv'
    file = open(path, 'a')
    csv_writer = csv.writer(file)
    for para1 in range(len(res_list)):
        if index + para1 < length:
            csv_writer.writerow([res_list[para1]])
        else:
            break

def AccRW(upper_dir, ref_mesh_path):
    writeValueAndVector(upper_dir)
    # e1, e2 = loadValueAndVector()
    # relate_face = loadRelatedFace()
    # print(relate_face[:3])

    # v_to_face, point_sets, normals, eigen_values, eigen_vectors, \
    # angle_list, res_list
    # path = 'E:\\zy_QA\\SVR-mesh-v2\\data\\parameter1.csv'
    
    ref_vertices, ref_faces = loadobj(ref_mesh_path)

    v_to_mesh = buildMapVerMesh(ref_vertices, ref_faces)
    v_to_face = prepareVtoFace(v_to_mesh)
    point_sets = preparePointSets(ref_vertices, ref_faces)
    normals = getAllNormVector(ref_vertices, ref_faces)
    normals = np.array(normals)
    path = upper_dir + '\\eigen_0_vv.csv'
    eigen_values, eigen_vectors = loadValueAndVector(path)

    angle_list = np.zeros(20)
    res_list = np.zeros(2000)

    v_to_facee = cuda.to_device(v_to_face)
    point_setss = cuda.to_device(point_sets)
    normalss = cuda.to_device(normals)
    eigen_valuess = cuda.to_device(eigen_values)
    eigen_vectorss = cuda.to_device(eigen_vectors)
    angle_listt = cuda.to_device(angle_list)
    res_listt = cuda.to_device(res_list)
    
    length = len(ref_vertices)
    l = int(length/2000)+1

    for i in range(l):
        gpuRW[20,100](i*2000, v_to_facee, point_setss, normalss, eigen_valuess, eigen_vectorss, \
            angle_listt, res_listt)
        res_list = np.zeros(2000)
        res_listt.copy_to_host(res_list)
        # 因为v_to_face 填充了一个（0，0，0）作为起始0
        writeParameterOne(i*2000, len(v_to_face)-1, res_list, upper_dir)

    
    # testEig[1,2](10)
    # print(eigen_values[5296])
    # print(eigen_values[33996])

    # print(eigen_vectors[1])

'''
用那种思路更好？
    1. 可在GPU中判断其在列表中的index，然后根据index来写入
    2. 或者可以仿照AccEigen的方法，在GPU中套定一个2000和线程长度相同的列表，然后更新这个列表
    这个方法的问题在于，列表最后的177个点的数据不好同步。    

'''

'''
1. 虽然不清楚bug怎么产生，但是从结果上来看，4200-823=40177可以得知，
parameter1的最后823个点是38000-40000那个迭代中的最后823个点，所以.zeros
这个函数并没有起作用，0没有覆盖上一个2000线程得出的结果。
2. 所以就另一种思路来看，在index大于vertices的长度之后直接不进行写入即可。
'''