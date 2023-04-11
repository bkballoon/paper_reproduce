# 这个代码用于求解网格中的每个点的平均曲率
# 我好像将模型整个缩小了100倍，因为这个bunny文件太大了

# 先用normal cycle理论的曲率计算公式来进行后期的处理，看看效果，然后在用他自己提的曲率计算方法

# 使用getEigen函数求解bunny模型每个点的曲率矩阵，然后得到平均曲率和曲率方向

import yang.myutil as myutil
from mayavi import mlab
import numpy as np
import csv

def load_curvature_csv(path):
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

def compute_mean_curvature(curvature_list):
    vertex_mean_curvature = []
    nan_len = 0
    for curvature in curvature_list:
        try:
            values, vectors = np.linalg.eig(curvature)
        except:
            nan_len += 1
            values = np.array([0])
        vertex_mean_curvature.append(np.mean(values))

    return vertex_mean_curvature

def visual_the_bunny_curvature():
    obj_data = myutil.load_obj("/home/simula/Dataset/textured_model/bunny/bunny.obj")
    vertices, faces = obj_data[0], obj_data[1]

    curvature_list = load_curvature_csv("/home/simula/Dataset/textured_model/bunny/saliency.csv")
    curvature_list = np.array(curvature_list)
    # compute the vertex mean curvature
    mean_curvature = compute_mean_curvature(curvature_list)
    mc_min, mc_max = min(mean_curvature), max(mean_curvature)
    mean_curvature = np.array(mean_curvature)
    mean_curvature /= mc_max - mc_min
    mean_curvature = np.array(mean_curvature, dtype=np.float)
    print(mean_curvature.shape)
    with open("/home/simula/Dataset/textured_model/bunny/saliency.txt", 'a') as file:
        for i in mean_curvature:
            print(str(i) + '\n')
            file.write(str(i) + '\n')
    # 这句话为了将numpy复杂的float类型转换为普通的float类型，因为复杂
    # 的float类型，vtk无法识别，进而也就无法显示
    mean_curvature = np.array(mean_curvature, np.float)
    mlab = myutil.mayavi_with_custom_point(vertices, faces, mean_curvature)
    mlab.show()


visual_the_bunny_curvature()

