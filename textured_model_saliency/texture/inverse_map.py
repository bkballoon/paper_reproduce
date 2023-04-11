# 目的：将模型上的filter内的点统统映射到纹理空间上
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt
import yang.myutil as myutil
from yang.geometry.get_filter_point import get_window_point, compute_R
from mayavi import mlab

def visual_mc_saliency(path):
    file = open(path)
    lines = file.readlines()
    feature_map = np.array([float(s) for s in lines])
    return feature_map

def visual_n_ring_vertex(origin_index, n, k):
    obj_data = myutil.load_obj("/home/simula/Dataset/textured_model/buddha2/buddha2.obj")
    vertices, faces = obj_data[0], obj_data[1]
    v_to_face = myutil.get_vertex_to_face(faces)
    # compute radius
    radius = compute_R(vertices, faces)
    radius = np.power(2, 1.4) * radius / 2

    n_ring_points_index, n_ring_segments = myutil.get_n_ring_vertex(v_to_face, origin_index, faces, n)
    print(n_ring_points_index)
    # visual the vertices
    new_faces_index = list()
    for i in n_ring_points_index:
        new_faces_index += v_to_face[i]
    new_faces_index = list(set(new_faces_index))
    # build the local faces index
    new_faces = []
    for i in new_faces_index:
        new_faces.append(faces[i])
    # build the n ring neighbour vertices
    new_vertices = []
    new_vertices = np.array([vertices[i] for i in n_ring_points_index])
    # give the face different color
    new_faces_color = []
    for i in new_faces_index:
        new_faces_color.append(random.random())
    # build the line
    for i in n_ring_segments:
        p1, p2 = i.split('+')
        p1, p2 = vertices[int(p1)], vertices[int(p2)]
        mlab.plot3d([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]])
        # 求p1 和 p2到圆心的距离
        p1, p2 = i.split('+')
        p1, p2 = int(p1), int(p2)
        if p1 == p2: continue
        p1_center_dis = two_points_dist(vertices[p1], vertices[origin_index])
        p2_center_dis = two_points_dist(vertices[p2], vertices[origin_index])
        print(p1_center_dis, p2_center_dis, radius)

    mlab.points3d(new_vertices[:, 0], new_vertices[:, 1], new_vertices[:, 2], line_width=0.1)
    mayavi_sphere(k * radius, vertices[origin_index])
    # myutil.mayavi_with_custom_face(vertices, new_faces, new_faces_color)
    mlab.show()


def mayavi_sphere(radius, origin_point):
    # Create a sphere
    r = radius
    pi = np.pi
    cos = np.cos
    sin = np.sin
    phi, theta = np.mgrid[0:pi:101j, 0:2 * pi:101j]

    x = r * sin(phi) * cos(theta) + origin_point[0]
    y = r * sin(phi) * sin(theta) + origin_point[1]
    z = r * cos(phi)              + origin_point[2]

    mlab.mesh(x, y, z, transparent=True)
    # return mlab

# 计算两个点的距离
def two_points_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = p1 - p2
    return myutil.norm(p3)

# 获取交点坐标
def get_intersection(inter_point, outer_point, radius):
    vec_AC = np.array(outer_point) - np.array(inter_point)
    len_AC = np.sqrt(
        vec_AC[0] * vec_AC[0] + vec_AC[1] * vec_AC[1] + vec_AC[2] * vec_AC[2])
    n = radius / len_AC
    new_point = np.array(inter_point) + n * vec_AC
    return n, new_point

# parameter : 圆心位置，顶点，uv lists, uv_index_lists, 顶点对应的面，顶点对应的uvs，球体半径，n环邻域
def n_scale_vertex_uvs(origin_index, vertices, faces, uvs, uv_index, v_to_face, v_to_uvs, radius, n_ring):
    '''
    获取n环邻域的点在uv上的映射
    '''
    # 用于存储每个点的uv值的列表
    res_uvs_sec, res_uvs_inter = [], []
    res_points_sec, res_points_inter = [], []
    # 遍历所有的边，然后得到所有的邻域点（相交或者不想交），然后统统转化为uv坐标，结果就是uv坐标的集合
    n_ring_points_index, n_ring_segments = myutil.get_n_ring_vertex(v_to_face, origin_index, faces, n_ring)
    # 遍历边
    window_index_set = set()
    intersection_dict = dict()
    for segment in n_ring_segments:
        p1, p2 = segment.split('+')
        p1, p2 = int(p1), int(p2)
        # 求p1 和 p2到圆心的距离
        if p1 == p2: continue
        p1_center_dis = two_points_dist(vertices[p1], vertices[origin_index])
        p2_center_dis = two_points_dist(vertices[p2], vertices[origin_index])
        # 判断点和球面的关系
        if p1_center_dis < radius and p2_center_dis < radius:
            if p1 not in window_index_set:
                window_index_set.add(p1)
            if p2 not in window_index_set:
                window_index_set.add(p2)

        elif p1_center_dis < radius < p2_center_dis:
            # print("-> 1")
            n, new_point = get_intersection(vertices[p1], vertices[p2], radius)
            # 由于顶点和uv坐标不直接对应，所以不可以用顶点对应uv来进行uv的插值，需要借助面片来进行
            p1_uv_index, p2_uv_index = -1, -1
            p1_faces = v_to_face[p1]
            p2_faces = v_to_face[p2]
            new_face_index_list = list(set(p2_faces).intersection(set(p1_faces)))  # 两个顶点相连的线段对应的面片
            # face_len.add(len(new_face_index_list))
            new_face_index = new_face_index_list[0]
            new_face = faces[new_face_index]
            for i in range(3):
                if new_face[i] == p1:
                    p1_index = i
                    p1_uv_index = uv_index[new_face_index][p1_index]
                if new_face[i] == p2:
                    p2_index = i
                    p2_uv_index = uv_index[new_face_index][p2_index]
            # p1_uv_index, p2_uv_index = list(v_to_uvs[p1])[0], list(v_to_uvs[p2])[0]
            p1_uv, p2_uv = uvs[p1_uv_index], uvs[p2_uv_index]
            # print(p1_uv, p2_uv)
            p1_uv, p2_uv = np.array(p1_uv, dtype=np.float), np.array(p2_uv, dtype=np.float)
            new_uv = p1_uv + n * (p2_uv - p1_uv)
            res_uvs_sec.append(new_uv)
            res_points_sec.append(new_point)
            # print(new_uv)

        elif p2_center_dis < radius < p1_center_dis:
            # print("-> 2")
            n, new_point = get_intersection(vertices[p2], vertices[p1], radius)
            # 如果以面为单位进行顶点uv计算的话，下面的注释代码需要进行反注释
            p1_uv_index, p2_uv_index = -1, -1
            p1_faces = v_to_face[p1]
            p2_faces = v_to_face[p2]
            new_face_index_list = list(set(p2_faces).intersection(set(p1_faces)))
            # face_len.add(len(new_face_index_list))
            new_face_index = new_face_index_list[0]
            new_face = faces[new_face_index]
            for i in range(3):
                if new_face[i] == p1:
                    p1_index = i
                    p1_uv_index = uv_index[new_face_index][p1_index]
                if new_face[i] == p2:
                    p2_index = i
                    p2_uv_index = uv_index[new_face_index][p2_index]
            # p1_uv_index, p2_uv_index = list(v_to_uvs[p1])[0], list(v_to_uvs[p2])[0]
            p1_uv, p2_uv = uvs[p1_uv_index], uvs[p2_uv_index]
            # print(p1_uv, p2_uv)
            p1_uv, p2_uv = np.array(p1_uv, dtype=np.float), np.array(p2_uv, dtype=np.float)
            new_uv = p2_uv + n * (p1_uv - p2_uv)
            res_uvs_sec.append(new_uv)
            res_points_sec.append(new_point)
            # print(new_uv)

    if origin_index in window_index_set:
        window_index_set.remove(origin_index)
    for i in window_index_set:
        temp_face = list(v_to_uvs[i])
        res_uvs_inter.append(np.array(uvs[temp_face[0]], dtype=np.float))
        res_points_inter.append(vertices[i])
    # 在这里，将得到的faceindex赋到uv坐标上，然后得到两个点的uv值，然后插值出来结果得到uv值
    # 然后将uv值结合起来

    return res_uvs_sec, res_uvs_inter


# result uv
def main(n_ring, k):
    obj_data = myutil.load_obj("/home/simula/Dataset/textured_model/buddha2/buddha2.obj")
    vertices, faces, uvs, uv_index = obj_data[0], obj_data[1], obj_data[2], obj_data[3]
    v_to_face = myutil.get_vertex_to_face(faces)
    v_to_uvs = myutil.get_v_to_uv(vertices, faces, uv_index)
    # compute radius
    radius = compute_R(vertices, faces)
    radius = np.power(2, 1.4) * radius / 2
    print("<<====================================>>")

    res_uv_lists = []
    for i in range(len(vertices)):
    # for i in range(10):
        if i % 1000 == 0:
            print(i)
        res_uvs_sec, res_uvs_inter = n_scale_vertex_uvs(
            i, vertices, faces, uvs, uv_index, v_to_face, v_to_uvs, k * radius, n_ring)
        # res_uv_lists.append(res_uvs_sec + res_uvs_inter)
        res_uv_lists.append(res_uvs_sec)

    img2 = np.zeros(1024 * 1024 * 3, dtype=np.uint8).reshape([1024, 1024, 3])

    count = 0
    for res_uvs in res_uv_lists:
        color = [int(random.random()*255) for i in range(3)]
        r_list = []
        origin_uv = list(v_to_uvs[count])[0]
        # print("res = {}".format(res_uvs))
        for uv in res_uvs:
            # texel_coord = uv_coord * [width, height], width
            new_uv = [round(uv[0] * 1024), round(uv[1] * 1024)]
            cv2.circle(img2, (new_uv[0], new_uv[1]), 1, color, -1)
            r = np.array(origin_uv) - np.array(new_uv)
            r_list.append(myutil.norm(r))
        # print(r_list)
        count += 1
    return img2

# 逆时针旋转90
def inverse_transpose(img):
    img = cv2.flip(img, 1)
    img = cv2.transpose(img)
    return img

# 可视化纹理点到2d图像的映射
def visual_texcoord_to_image():
    obj_data = myutil.load_obj("/home/simula/Dataset/textured_model/buddha2/buddha2.obj")
    vertices, faces, uvs, uv_index = obj_data[0], obj_data[1], obj_data[2], obj_data[3]

    path = "/home/simula/Dataset/textured_model/buddha2/buddha2-atlas.jpg"
    img1 = cv2.imread(path)
    img2 = np.zeros(1024 * 1024 * 3, dtype=np.uint8).reshape([1024, 1024, 3])
    for uv in uv_index:
        i1, i2, i3 = uv[0], uv[1], uv[2]
        t1, t2, t3 = uvs[i1], uvs[i2], uvs[i2]
        t1, t2, t3 = np.array(t1, dtype=np.float), np.array(t2, dtype=np.float), np.array(t3, dtype=np.float)
        new_uv = [t1[0] * 1024, t1[1] * 1024]
        cv2.circle(img2, (round(new_uv[0]), round(new_uv[1])), 1, (0, 0, 213), -1)
        new_uv = [t2[0] * 1024, t2[1] * 1024]
        cv2.circle(img2, (round(new_uv[0]), round(new_uv[1])), 1, (0, 0, 213), -1)
        new_uv = [t3[0] * 1024, t3[1] * 1024]
        cv2.circle(img2, (round(new_uv[0]), round(new_uv[1])), 1, (0, 0, 213), -1)
    return img2

# 可视化邻域的点在纹理上的显示
# n环，k倍radius
def visual_n_ring_points_in_texture():
    img_path = "/home/simula/Dataset/textured_model/buddha2/buddha2-atlas.jpg"
    img1 = cv2.imread(img_path)
    gaussian_img = cv2.GaussianBlur(img1, (3, 3), 3)

    img2 = visual_texcoord_to_image()
    # 这是将邻域点映射回texture的图像
    # n环邻域，k倍半径
    n_ring, k = 2, 1
    img3 = main(n_ring, k)
    # img2 = inverse_transpose(img2)
    # img3 = inverse_transpose(img3)

    plt.subplot(121)
    plt.imshow(img2)
    plt.subplot(122)
    plt.imshow(img3)
    plt.show()

# 获取intensity-saliency
def get_intensity(x, y, img):
    pixel = img[x][y]
    return np.mean(pixel)

# 获取c-saliency
def get_c(x, y, img):
    pixel = img[x][y]
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    rg = (r - g) / max(pixel)
    by = (b - min([r, g])) / max(pixel)
    return rg + by

# 对于n环邻域内的uv值
def get_ge(uvs, img):
    gr_sum = 0
    for uv in uvs:
        x, y = uv[0], uv[1]
        pixel = img[x][y]
        gr = 0.3*pixel[0] + 0.59*pixel[1] + 0.11*pixel[2]
        gr_sum += gr
    return gr_sum / len(uvs)

# 开始计算纹理显著度
def start_to_compute(img_path, path):
    img1 = cv2.imread(img_path)
    gaussian_img = cv2.GaussianBlur(img1, (3, 3), 3)

    # 处理模型有关的数据
    obj_data = myutil.load_obj(path)
    vertices, faces, uvs, uv_index = obj_data[0], obj_data[1], obj_data[2], obj_data[3]  # uv_index是和面相关的uv的index

    v_to_face = myutil.get_vertex_to_face(faces)
    v_to_uvs = myutil.get_v_to_uv(vertices, faces, uv_index)
    print(len(uvs))
    # print(v_to_uvs[:10])
    # count = 10
    # print(list(v_to_uvs[count])[0])
    # exit()
    # compute radius
    radius = compute_R(vertices, faces)
    radius = np.power(2, 1.4) * radius / 2
    print("<<====================================>>")

    # 获取所有的相交点
    k = 2
    n_ring = 3
    res_uv_lists = []
    for i in range(len(vertices)):
    # for i in range(1):
        if i % 1000 == 0:
            print(i)
        # sec是相交点，inter是圈内的点，这两者都已经映射到texture-img上了
        res_uvs_sec, res_uvs_inter = n_scale_vertex_uvs(
            i, vertices, faces, uvs, uv_index, v_to_face, v_to_uvs, k * radius, n_ring)

        # res_uv_lists.append(res_uvs_sec + res_uvs_inter)
        res_uv_lists.append(res_uvs_sec)  # sec uv的范围是0-1

    # 可视化：顶点v和相关点vn在uv上的形式究竟为如何？

    # 所有的相交点就是origin-index的相关点
    # 遍历所有的相关点，然后判断圆心的uv点是哪个？

    print("v_to_vs_len", len(v_to_uvs))
    print("res uvs len", len(res_uv_lists))
    count = 0
    origin_index_list = []
    radius_list = []
    for res_uvs in res_uv_lists:
        # 判断所有的纹理uv点，判断哪个中心点的分布最为接近? --》我也不知道我写的是什么东西
        if len(list(v_to_uvs[count])) == 2:
            origin_index1 = list(v_to_uvs[count])[0]  # 得到的是uv值index，顶点i对应的uv值index1和index2
            origin_index2 = list(v_to_uvs[count])[1]
            r_list1 = []
            r_list2 = []
            for uv in res_uvs:
                # new_uv = [round(uv[0] * 1024), round(uv[1] * 1024)]
                new_uv = uv
                center1 = np.array(uvs[origin_index1], dtype=np.float)
                center2 = np.array(uvs[origin_index2], dtype=np.float)
                r_list1.append(myutil.norm(center1 - new_uv))
                r_list2.append(myutil.norm(center2 - new_uv))
            m1 = np.mean(r_list1)
            m2 = np.mean(r_list2)
            if m1 < m2:
                origin_index = origin_index1
                r_list = r_list1
            else:
                origin_index = origin_index2
                r_list = r_list2
            origin_index_list.append(origin_index)
            radius_list.append(r_list)
        else:
            origin_index = list(v_to_uvs[count])[0]
            r_list = []
            for uv in res_uvs:
                new_uv = [round(uv[0] * 1024), round(uv[1] * 1024)]
                center = np.array(uvs[origin_index], dtype=np.float)
                r_list.append(myutil.norm(center - new_uv))
            origin_index_list.append(origin_index)
            radius_list.append(r_list)
        count += 1
    # 现在得到所有的中心点uv值和相关点的uv值，分别是orgin-index-list和res-uv-list
    # 以及对应的r-list

    # 接下来就是计算每个顶点及其相关点在平面上的I和C显著度，然后叠加得到最后的结果
    intensity_list = []
    c_list = []
    # 思路是：计算每个点的I*EXP求和然后再除以EXP的求和
    for i in range(len(vertices)):
        # 获取圆心的uv值
        origin_index = i  # 获取index
        origin_uv_index = list(v_to_uvs[origin_index])[0]  # 获取uv的index值
        origin_uv = np.array(uvs[origin_uv_index], dtype=np.float)  # 获取uv值
        origin_x, origin_y = [round(origin_uv[0] * 1024), round(origin_uv[1] * 1024)]
        origin_pixel = gaussian_img[origin_x][origin_y]
        res_uv = res_uv_lists[i]
        intensity = 0
        c = 0
        down = 0
        # 这里进行公式的计算：
        # print('--> ', end=" ")
        for uv in res_uv:  # res_uv的每个值都是相交点的uv坐标
            x, y = [round(uv[0] * 1024), round(uv[1] * 1024)]  #
            if x >= 1024: x = 1023
            if y >= 1024: y = 1023
            pixel = gaussian_img[x][y]
            pixel_diff = np.array(origin_pixel, np.int) - np.array(pixel, np.int)
            exp_value = np.exp(-0.5 * myutil.norm2(pixel_diff) / myutil.norm(1024*origin_uv - 1024*uv) / myutil.norm(1024*origin_uv - 1024*uv))
            # print("pixel diff", myutil.norm2(pixel_diff))
            # print("dis", myutil.norm(1024*origin_uv - 1024*uv))
            intensity += np.mean(pixel) * exp_value
            c += get_c(x, y, gaussian_img) * exp_value
            # print(exp_value, end=' ')
            down += exp_value
        # print()
        if down == 0:
            down = 1
            print(i, " index = 0 ", down)
        # print(intensity / down)
        intensity_list.append(intensity / down)
        c_list.append(c / down)

    # 然后就获取了每个点的intensity和c值
    # 将这两个值进行合成，然后投影到模型上就得到最后的结果了

    # 合成，合成，合成
    i_min, i_max = min(intensity_list), max(intensity_list)
    c_min, c_max = min(c_list), max(c_list)
    intensity_list = np.array(intensity_list)
    c_list = np.array(c_list)
    intensity_list /= i_max - i_min
    c_list /= c_max - c_min

    final_list = 0.5 * intensity_list + 0.5 * c_list

    return final_list

# face_len = set()
# visual_n_ring_points_in_texture()
# print(face_len)

# 5环邻域的点和球面的交点，原点index是0
# visual_n_ring_vertex(origin_index=0, n=2, k=1)


# final_list = start_to_compute()


# 代码有问题，导致结果太小了。
