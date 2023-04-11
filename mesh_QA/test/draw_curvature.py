from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

import csv

def histeq(array):
    hist, bins = np.histogram(array)
    cdf = hist.cumsum()
    cdf = 1000 * cdf / cdf[-1]
    array2 = np.interp(array, bins[:-1], cdf)
    return array2

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

def draw(ref_vertices, eigen_value_mean_color_index):
    rate_dis_v2 = histeq(eigen_value_mean_color_index)

    figure = plt.figure()
    axes = figure.add_subplot(111, projection='3d')
    axes.scatter3D(ref_vertices[:, 0], ref_vertices[:, 1], ref_vertices[:, 2], \
        c = rate_dis_v2)

    # axes = figure.add_subplot(122, projection='3d')
    # axes.scatter3D(ref_vertices[:, 0], ref_vertices[:, 1], ref_vertices[:, 2])
    plt.show()

def drawCurvaturePng(ref_curvature_path, ref_mesh_path):
    
    ref_vertices, ref_faces = loadobj(ref_mesh_path)
    eigen_value, eigen_vectors = loadValueAndVector(ref_curvature_path)
    eigen_value_mean = [max(i) for i in eigen_value]
    sorted_eigen_value_mean = sorted(eigen_value_mean)
    each_dice = (sorted_eigen_value_mean[-1] - sorted_eigen_value_mean[0])/100
    eigen_value_mean_color_index = \
        [int((eigen_value_mean[i]-sorted_eigen_value_mean[0])/each_dice) \
        for i in range(len(eigen_value_mean))]
    color_list = ["#FF6666", "#FF9966", "#FFCC66", "#FFFF66", "#CCFF66", "99FF66", "66FF66", "66FF99", "66FFCC", "66FFFF", "66FFFF", "66CCFF", "6699FF", "6666FF", "9966FF", "#CC66FF", "FF66FF", "FF66CC", "FF6699"]

    # eigen_value_mean_color = [color_list[i] for i in eigen_value_mean_color_index]
    ref_vertices = np.array(ref_vertices)

    draw(ref_vertices, eigen_value_mean_color_index)

def drawScattor(ref_curvature_path, ref_mesh_path):
    
    ref_vertices, ref_faces = loadobj(ref_mesh_path)
    eigen_value, eigen_vectors = loadValueAndVector(ref_curvature_path)
    eigen_value_mean = [np.mean(i) for i in eigen_value]
    eigen_value_max = [max(i) for i in eigen_value]
    sorted_eigen_value_mean = sorted(eigen_value_mean)

    # each_dice = (sorted_eigen_value_mean[-1] - sorted_eigen_value_mean[0])/180
    each_dice = (sorted_eigen_value_mean[-1] - sorted_eigen_value_mean[0])/1000
    rate_dis = [int((eigen_value_mean[i] - sorted_eigen_value_mean[0])/each_dice) for i in range(len(eigen_value_mean))]

    rate_dis_dict = {}
    for i in rate_dis:
        if i not in rate_dis_dict.keys():
            rate_dis_dict[i] = 1
        else:
            rate_dis_dict[i] += 1

    X = rate_dis_dict.keys()
    Y = rate_dis_dict.values()
    X = [i*each_dice for i in X]
    Y = [i for i in Y]
    index = Y.index(max(Y))
    # 拟合正态分布
    # Z = np.polyfit(X, Y, 3)
    # P = np.poly1d(Z)
    # Y_fit = P(X)

    figure = plt.figure()
    axes = figure.add_subplot(111)
    axes.scatter(X, Y)
    X_p = [X[index] for i in range(1000)]
    Y_p = np.linspace(0, Y[index], 1000)
    axes.plot(X_p, Y_p, c='red')
    plt.xlabel(X[index])
    plt.show()


if __name__ == "__main__":

    ref_curvature_path = 'E:\\zy_QA\\SVR-mesh-v3\\data\\noiserockerArm-Noise001\\eigen_1_vv.csv'
    ref_mesh_path = 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Noise001.obj'
    list1 = [
        'E:\\zy_QA\\SVR-mesh-v3\\data\\noiserockerArm-Noise001\\eigen_0_vv.csv',
        'E:\\zy_QA\\SVR-mesh-v3\\data\\noiserockerArm-Noise001\\eigen_1_vv.csv',
        'E:\\zy_QA\\SVR-mesh-v3\\data\\noiserockerArm-NoiseRough0005\\eigen_1_vv.csv'
    ]

    list2 = [
        'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm.obj',
        'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Noise001.obj',
        'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseRough0005.obj'
    ]
    for i in range(3):
        drawCurvaturePng(list1[i], list2[i])

