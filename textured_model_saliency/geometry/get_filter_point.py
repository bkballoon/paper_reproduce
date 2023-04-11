'''
功能：求得每个点radius范围内的直接相关点和球面和直线交点
解决：
1.对于模型中的任意一点，先求其范围内的直接相关点
2.然后做两次深度扩展，得到2环邻域内的所有点以及线段，然后判断线段和
球面的交点
3.得到1和2的交点并合成，保存

直观的思路就是统计相关的面中出现的线段
然后判断线段的两端是否为球的内外两侧，然后得到所有的window内的点
假设radius为一环邻域内的最近的点
'''
import random
import yang.myutil as myutil
import numpy as np
from matplotlib import pyplot as plt
from mayavi import mlab
# from get_mean_curvature import load_curvature_csv, compute_mean_curvature

def visual_mc_saliency(path):
    file = open(path)
    lines = file.readlines()
    feature_map = np.array([float(s) for s in lines])
    return feature_map

# 可视化点相关的点和面
def visual_vertex_to_face_vertex():
    obj_data = myutil.load_obj("arma.obj")
    vertices, faces = obj_data[0], obj_data[1]
    v_to_face = myutil.get_vertex_to_face(faces)
    vertex_index = 1
    relate_face = v_to_face[vertex_index]
    vertex = vertices[vertex_index]
    new_faces = np.array([faces[i] for i in relate_face])
    face_data = [random.random() for i in range(len(new_faces))]
    myutil.mayavi_with_custom_face(vertices, new_faces, face_data)

# 可视化二环邻域内的点
def visual_two_ring_points(vertices, faces, v_to_face, radius, mean_curvature):
    origin_index = 1
    point_set1 = myutil.get_one_ring_vertex(v_to_face, 0, faces)
    point_set2, two_ring_segments = myutil.get_two_ring_vertex(v_to_face, origin_index, faces)
    # 不同窗口下的顶点及顶点的平均曲率
    window_point_list_1r, window_point_mc_1r = get_window_point(origin_index, 1 * radius, two_ring_segments, vertices, mean_curvature)

    new_faces = []
    for i in point_set2:
        new_faces += v_to_face[i]
    new_faces = list(set(new_faces))
    end_faces = []
    for i in new_faces:
        end_faces.append(faces[i])

    faces_color = [random.random() for i in range(len(end_faces))]
    new_points = [vertices[i] for i in point_set2]
    for i in window_point_list_1r:
        new_points.append(i)
    new_points = np.array(new_points)
    one_ring_points = np.array([vertices[i] for i in point_set1])

    mlab = myutil.mayavi_with_custom_face(vertices, end_faces, faces_color)
    mlab.points3d(new_points[:, 0], new_points[:, 1], new_points[:, 2], colormap='cool')
    mlab.points3d(one_ring_points[:, 0], one_ring_points[:, 1], one_ring_points[:, 2])
    mlab.show()

# 面积
def face_s(i1, i2, i3, vertices):
    p1, p2, p3 = np.array(vertices[i1]), np.array(vertices[i2]), np.array(vertices[i3])
    end = np.cross(p1 - p2, p1 - p3)
    return np.sqrt(end[0] * end[0] + end[1] * end[1] + end[2] * end[2]) / 2

# 计算定义的半径
def compute_R(vertices, faces):
    all_s = 0
    for i in faces:
        s = face_s(i[0], i[1], i[2], vertices)
        all_s += s

    average_s = all_s / len(faces)
    R = np.sqrt(4 * average_s / np.sqrt(3)) * np.sqrt(3) / 3
    return R

# 计算最近相关点
def get_nearest_point(origin_index, v_to_face, vertices, faces):
    point_set = myutil.get_one_ring_vertex(v_to_face, origin_index, faces)
    min_dist = 99999
    origin_point = vertices[origin_index]
    for index in point_set:  # 遍历相关的点，找到最近的点，计算dist记radius
        if index != origin_index:
            point = vertices[index]
            t_dist = myutil.distance_between_two_point(point, origin_point)
            if t_dist < min_dist:
                min_dist = t_dist
    return min_dist, point_set

# 计算交点
def get_intersection(inter_point, outer_point, radius):
    vec_AC = np.array(inter_point) - np.array(outer_point)
    len_AC = np.sqrt(
        vec_AC[0] * vec_AC[0] + vec_AC[1] * vec_AC[1] + vec_AC[2] * vec_AC[2])
    n = radius / len_AC
    new_point = np.array(inter_point) + n * vec_AC
    return n, new_point

# 计算filter window
def get_window_point(origin_index, radius, segment_sets, vertices, mean_curvature):
    # 获取radius之后，计算radius范围内所有的交点
    window_point_list = []
    window_point_mc = []
    window_index_set = set()
    origin_point = vertices[origin_index]
    for segment in segment_sets:  # 对于每条线段
        i1, i2 = segment.split('+')
        i1, i2 = int(i1), int(i2)
        if i1 == i2:
            continue
        inter, outer = -1, -1
        if i1 == origin_index or i2 == origin_index:  # 如果一个点圆心，则另一个点加入到window中
            if i1 == origin_index:
                inter = i1
                outer = i2
            elif i2 == origin_index:
                inter = i2
                outer = i1
            t_dist = myutil.distance_between_two_point(vertices[inter], vertices[outer])
            if t_dist <= radius:
                window_index_set.add(outer)
                window_point_list.append(vertices[outer])
                window_point_mc.append(mean_curvature[outer])
            else:
                n, new_point = get_intersection(vertices[inter], vertices[outer], radius)
                new_point_mc = n * mean_curvature[outer] + (1 - n) * mean_curvature[inter]
                window_point_list.append(new_point)
                window_point_mc.append(new_point_mc)
            continue

        # 否则判断点是否是分布在球内外的
        d1 = myutil.distance_between_two_point(origin_point, vertices[i1])
        d2 = myutil.distance_between_two_point(origin_point, vertices[i2])
        if d1 < radius and d2 < radius:
            if i1 not in window_index_set:
                window_index_set.add(i1)
                window_point_list.append(vertices[i1])
                window_point_mc.append(mean_curvature[i1])
            if i2 not in window_index_set:
                window_index_set.add(i2)
                window_point_list.append(vertices[i2])
                window_point_mc.append(mean_curvature[i2])
        elif d1 < radius and d2 > radius:
            inter, outer = i1, i2
        elif d1 > radius and d2 < radius:
            inter, outer = i2, i1
        elif d1 >= radius and d2 >= radius:
            # 后面的操作都不用做了，因为线段和求无关
            var = None
            continue
        if inter < 0 or outer < 0:
            continue
        print("inter = {}, outer = {}".format(inter, outer))
        n, new_point = get_intersection(vertices[inter], vertices[outer], radius)
        print("n = {}, new_point = {}".format(n, new_point))
        print(type(outer), type(inter))
        new_point_mc = n * mean_curvature[outer] + (1 - n) * mean_curvature[inter]
        window_point_list.append(new_point)
        window_point_mc.append(new_point_mc)

    return window_point_list, window_point_mc
    # --------------------visual - function---------------------------------
    # v_to_face = myutil.get_vertex_to_face(faces)
    # relate_face = v_to_face[origin_index]
    #
    # new_faces = np.array([faces[i] for i in relate_face])
    # face_color = [random.random() for i in range(len(new_faces))]
    # mlab = myutil.mayavi_with_custom_face(vertices, new_faces, face_color)
    # new_point = np.array(window_point_list)
    # mlab.points3d(new_point[:, 0], new_point[:, 1], new_point[:, 2])
    #
    # mlab.show()
    # --------------------visual - function---------------------------------

# 计算高斯加权平均，思路就是遍历window内的每个点，计算其和圆心的距离和MC的加权平均
def compute_gaussian_average(origin_point, window_point_list, window_mc_list, radius):
    upper = 0
    downer = 0
    for index in range(len(window_point_list)):
        point = window_point_list[index]
        t_dist = myutil.distance_between_two_point(origin_point, point)
        item = - t_dist / (2 * radius * radius)
        upper += window_mc_list[index] * np.exp(item)
        downer += np.exp(item)
    gw_mc = upper / downer
    return gw_mc

# visual_vertex_to_face_vertex()
# get_nearest_point()
# get_window_point()
def main():
    obj_data = myutil.load_obj("/home/simula/Dataset/textured_model/buddha2/buddha2.obj")
    vertices, faces = obj_data[0], obj_data[1]
    feature_map = []
    v_to_face = myutil.get_vertex_to_face(faces)

    # 定义r
    radius = compute_R(vertices, faces)
    radius = np.power(2, 1.4) * radius / 2
    print("radius = ", radius)

    # 导入csv文件
    # curvature_list = load_curvature_csv("arma.csv")
    # curvature_list = np.array(curvature_list)
    # # compute the vertex mean curvature
    # mean_curvature = compute_mean_curvature(curvature_list)
    path = "/home/simula/Dataset/textured_model/buddha2/saliency.txt"
    mean_curvature = visual_mc_saliency(path)

    ratio = []
    diff_scale = []

    alpha = 1.0
    beta = 300
    for i in range(len(vertices)):
        # for i in range(2):
        # 显示循环的进展
        if i % 500 == 0:
            print(i)

        origin_index = i
        origin_point = vertices[origin_index]
        # min_dist, point_set = get_nearest_point(origin_index, v_to_face)
        two_ring_points, two_ring_segments = myutil.get_n_ring_vertex(v_to_face, origin_index, faces, 4)
        # 不同窗口下的顶点及顶点的平均曲率
        window_point_list_1r, window_point_mc_1r = get_window_point(origin_index, 1 * radius, two_ring_segments, vertices, mean_curvature)
        window_point_list_2r, window_point_mc_2r = get_window_point(origin_index, 2 * radius, two_ring_segments, vertices, mean_curvature)
        window_point_list_3r, window_point_mc_3r = get_window_point(origin_index, 3 * radius, two_ring_segments, vertices, mean_curvature)
        window_point_list_4r, window_point_mc_4r = get_window_point(origin_index, 4 * radius, two_ring_segments, vertices, mean_curvature)
        # 不同尺度的高斯均值
        gw_mc_1r = compute_gaussian_average(origin_point, window_point_list_1r, window_point_mc_1r, 1 * radius)
        gw_mc_2r = compute_gaussian_average(origin_point, window_point_list_2r, window_point_mc_2r, 2 * radius)
        gw_mc_3r = compute_gaussian_average(origin_point, window_point_list_3r, window_point_mc_3r, 3 * radius)
        gw_mc_4r = compute_gaussian_average(origin_point, window_point_list_4r, window_point_mc_4r, 4 * radius)
        gl = abs(gw_mc_2r - gw_mc_1r) + abs(gw_mc_3r - gw_mc_2r) + abs(gw_mc_4r - gw_mc_3r)

        gf_line = alpha * mean_curvature[i] + beta * gl
        # gf_line = beta * gl
        # ratio.append(mean_curvature[i] / gl)
        # diff_scale.append(gl)

        feature_map.append(gf_line)
        # print("mc = {}, gl = {}, gf = {}, ratio = {}, beta gl = {}".format(mean_curvature[i], gl, gf_line, mean_curvature[i] / gl, beta * gl))
    feature_map = np.array(feature_map, np.float)
    mlab = myutil.mayavi_with_custom_point(vertices, faces, feature_map)
    mlab.show()

# plt.subplot(131)
# plt.hist(mean_curvature)
# plt.subplot(132)
# plt.hist(diff_scale)
# plt.subplot(133)
# plt.hist(ratio)
# plt.show()


# 这份代码写得非常糟糕
