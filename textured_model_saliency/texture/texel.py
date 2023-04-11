'''
将纹理图像上的点映射到3D上，得到模型上的每个点的pixel，然后对texel模型进行处理得到结果
'''
import cv2
import yang.myutil as myutil
import numpy as np
from PIL import Image
from mayavi import mlab
from matplotlib import pyplot as plt

# 归一化
def normalized(arr):
    arr_max = np.max(arr)
    arr_min = np.min(arr)
    return arr/(arr_max - arr_min)

# 获取intensity-saliency
def get_intensity(x, y, img):
    pixel = img[y][x]
    return np.mean(pixel)

# 获取c-saliency
def get_c(x, y, img):
    pixel = img[y][x]
    r, g, b = int(pixel[0]), int(pixel[1]), int(pixel[2])
    rg = (r - g) / max(pixel)
    by = (b - min([r, g])) / max(pixel)
    return rg + by

# 获取相交点的参数
def get_intersection(inter_point, outer_point, radius):
    vec_AC = np.array(outer_point) - np.array(inter_point)
    len_AC = np.sqrt(
        vec_AC[0] * vec_AC[0] + vec_AC[1] * vec_AC[1] + vec_AC[2] * vec_AC[2])
    n = radius / len_AC
    new_point = np.array(inter_point) + n * vec_AC
    return n, new_point

# 计算两个点的距离
def two_points_dist(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
    p3 = p1 - p2
    return myutil.norm(p3)

def n_scale_vertex_uvs_v2(origin_index, vertices, faces, uvs, edge_to_pixels, v_to_face, v_to_uvs, radius, n_ring):
    n_ring_points_index, n_ring_segments = myutil.get_n_ring_vertex(v_to_face, origin_index, faces, n_ring)
    # 存储数据的容器
    res_uvs_sec, res_points_sec = [], []

    res_points_sec, res_points_inter = [], []
    # 遍历n环内的素有的点
    for segment in n_ring_segments:
        p1, p2 = segment.split("+")
        p1, p2 = int(p1), int(p2)

        # 求边的两端和圆心之间的距离
        p1_center_dis = two_points_dist(vertices[p1], vertices[origin_index])
        p2_center_dis = two_points_dist(vertices[p2], vertices[origin_index])

        if p1_center_dis < radius < p2_center_dis:
            n, new_point = get_intersection(vertices[p1], vertices[p2], radius)
            ps = edge_to_pixels[segment][0]
            ps1, ps2 = ps.split("+")
            p1_uv, p2_uv = uvs[int(ps1)], uvs[int(ps2)]
            p1_uv, p2_uv = np.array(p1_uv, dtype=np.float), np.array(p2_uv, dtype=np.float)
            new_uv = p1_uv + n * (p2_uv - p1_uv)
            if new_uv.all() > 0:
                res_uvs_sec.append(new_uv)
                res_points_sec.append(new_point)
            else:
                new_uv = p2_uv + n * (p1_uv - p2_uv)
                res_uvs_sec.append(new_uv)
                res_points_sec.append(new_point)

        elif p2_center_dis < radius < p1_center_dis:
            n, new_point = get_intersection(vertices[p1], vertices[p2], radius)
            ps = edge_to_pixels[segment][0]
            ps1, ps2 = ps.split("+")
            try:
                p1_uv, p2_uv = uvs[int(ps1)], uvs[int(ps2)]
            except:
                print(len(uvs))
                exit()
            p1_uv, p2_uv = np.array(p1_uv, dtype=np.float), np.array(p2_uv, dtype=np.float)
            new_uv = p2_uv + n * (p1_uv - p2_uv)
            if new_uv.all() > 0:
                res_uvs_sec.append(new_uv)
                res_points_sec.append(new_point)
            else:
                new_uv = p1_uv + n * (p2_uv - p1_uv)
                res_uvs_sec.append(new_uv)
                res_points_sec.append(new_point)

    return res_uvs_sec, res_points_sec


# 用于获取n环邻域内的点的window内的点和交点
def n_scale_vertex_uvs(origin_index, vertices, faces, uvs, uv_index, v_to_face, v_to_uvs, radius, n_ring):
    '''
    获取n环邻域的点在uv上的映射
    parameter : 圆心位置，顶点，uv lists, uv_index_lists, 顶点对应的面，顶点对应的uvs，球体半径，n环邻域
    '''
    # 用于存储每个点的uv值的列表
    res_uvs_sec, res_uvs_inter = [], []
    res_points_sec, res_points_inter = [], []
    # 遍历所有的边，然后得到所有的邻域点（相交或者不想交），然后统统转化为uv坐标，结果就是uv坐标的集合
    n_ring_points_index, n_ring_segments = myutil.get_n_ring_vertex(v_to_face, origin_index, faces, n_ring)

    window_index_set = set()
    intersection_dict = dict()

    # 遍历所有的边
    for segment in n_ring_segments:
        # p1, p2是线段的两端
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
            # 插值得到uv
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

    return res_uvs_sec, res_points_sec

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

# 逆时针旋转180度，然后翻转，得到和坐标分布类似的图像
def rotation(img):
    rows, cols = img.shape[:2]
    M2 = cv2.getRotationMatrix2D((cols / 2, rows / 2), 180, 1)
    img = cv2.warpAffine(img, M2, (rows, cols))
    img = cv2.flip(img, 1)
    return img

# 对图像做高斯模糊
def gaussian_img(img):
    # img = rotation(img)
    gaussian_img = cv2.GaussianBlur(img, (3, 3), 3)
    return gaussian_img

# 获取空间中的点的像素
def vertex_pixel(k, n_ring, radius, vertices, faces, uvs, uv_index, edge_to_pixels):
    '''
    1. 获取点的window内的点，然后逆获取window内的点的pixel
    2. 获取window和边的交点，记录下边的两端，将两端端点逆到texture上，根据点在边的parameter进行距离插值
    '''
    v_to_face = myutil.get_vertex_to_face(faces)
    v_to_uvs = myutil.get_v_to_uv(vertices, faces, uv_index)

    res_uv_lists = []
    res_points_lists = []
    for i in range(len(vertices)):
        if i % 1000 == 0:
            print(i)
        # res_uvs_sec, res_points_sec = n_scale_vertex_uvs(
        #     i, vertices, faces, uvs, uv_index, v_to_face, v_to_uvs, k * radius, n_ring)
        res_uvs_sec, res_points_sec = n_scale_vertex_uvs_v2(
            i, vertices, faces, uvs, edge_to_pixels, v_to_face, v_to_uvs, k * radius, n_ring)

        section_xy = np.array(res_uvs_sec)
        res_uv_lists.append(section_xy)
        res_points_lists.append(res_points_sec)
        # if i == 3000: break

    return res_uv_lists, res_points_lists

# 获取每个texel的I和C
def texel_I_and_C(res_uv_lists, img):
    vertex_section_pc = []
    vertex_section_pi = []
    h, w, _ = img.shape
    for xy_lists in res_uv_lists:
        tc, ti = [], []
        for xy in xy_lists:
            x, y = xy[0], xy[1]
            p_c = get_c(x, y, img)
            p_i = get_intensity(x, y, img)
            tc.append(p_c)
            ti.append(p_i)
        vertex_section_pc.append(tc)
        vertex_section_pi.append(ti)
    return vertex_section_pc, vertex_section_pi

# 然后根据公式计算color Saliency
def compute_IW_CW(section_c, section_i, vertices, res_points_lists, radius):
    ver_c, ver_i = [], []
    for v_i in range(len(section_i)):
        vertex = vertices[v_i]
        section_vertices = res_points_lists[v_i]
        section_vertex_c = section_c[v_i]
        section_vertex_i = section_i[v_i]
        up_i, up_c, down = 0, 0, 0
        for index in range(len(section_vertices)):
            up_i += section_vertex_i[index]*np.exp(
                -myutil.norm(vertex-section_vertices[index])/2/radius/radius
            )
            up_c += section_vertex_c[index]*np.exp(
                -myutil.norm(vertex-section_vertices[index])/2/radius/radius
            )
            down += np.exp(
                -myutil.norm(vertex-section_vertices[index])/2/radius/radius
            )
        try:
            ver_c.append(up_c/down)
            ver_i.append(up_i/down)
        except:
            print(v_i, " down ", down)
            ver_i.append(-1)
            ver_c.append(-1)

    return ver_c, ver_i

def compute_color_saliency(img_path, obj_path):
    # 处理图像
    img = cv2.imread(img_path)
    img = gaussian_img(img)

    # 处理模型有关的数据
    obj_data = myutil.load_obj(obj_path)
    vertices, faces, uvs_, uv_index = obj_data[0], obj_data[1], obj_data[2], obj_data[3]  # uv_index是和面相关的uv的index
    edge_to_pixels = myutil.edge_to_pixel(faces, uv_index)

    radius = compute_R(vertices, faces)
    radius = np.power(2, 1.4) * radius / 2

    # 获取不同尺度下每个点的I和C
    v_c_list = []
    v_i_list = []
    for k in range(1, 5):
        colors= []
        print("---><---", k)
        n_ring = k+1
        # res_uv_lists保存每个点的n环交点的uv坐标
        res_uv_lists, res_points_lists = vertex_pixel(
            k, n_ring, radius, vertices, faces, uvs_, uv_index, edge_to_pixels)

        res_uv_lists_new = []
        for uvs in res_uv_lists:
            uv_new = []
            if 512 in img.shape:
                for uv in uvs:
                    x = int(uv[0]*1024)
                    if x >= 1024:
                        x = 1023
                    y = 511 - int(uv[1]*512)
                    if y >= 511:
                        y = 511
                    uv_new.append([x, y])
            else:
                for uv in uvs:
                    x = int(uv[0] * 1024)
                    if x >= 1024:
                        x = 1023
                    y = 1024 - int(uv[1] * 1024)
                    if y >= 1024:
                        y = 1023
                    uv_new.append([x, y])
            res_uv_lists_new.append(uv_new)

        colors = []
        for uv in res_uv_lists_new:
            x, y = uv[0], uv[1]
            colors.append(img[y][x])
        myutil.mayavi_point_cloud(res_points_lists, colors)
        mlab.show()
        exit()

        # for uvs in res_uv_lists_new:
        #     for uv in uvs:
        #         cv2.circle(img, (uv[0], uv[1]), 3, (200, 100, 0))
        #
        # cv2.circle(img, (100, 200), 10, (200, 100, 0))
        # plt.imshow(img)
        # plt.show()
        # exit()

        section_c, section_i = texel_I_and_C(res_uv_lists_new, img)  # 获取每个点相交点的I和C
        v_c, v_i = compute_IW_CW(section_c, section_i, vertices, res_points_lists, radius)

        v_c = np.array(v_c)
        v_i = np.array(v_i)

        for i in range(len(v_c)):
            if v_i[i] == -1:
                v_c[i] = np.mean(v_c)
                v_i[i] = np.mean(v_i)
        v_c_list.append(v_c)
        v_i_list.append(v_i)

    # 然后对每个点I和C，求其差别，然后归一化到0-1之间，再线性拟合
    cm1 = abs(v_c_list[1] - v_c_list[0])
    cm2 = abs(v_c_list[2] - v_c_list[1])
    cm3 = abs(v_c_list[3] - v_c_list[2])

    im1 = abs(v_i_list[1] - v_i_list[0])
    im2 = abs(v_i_list[2] - v_i_list[1])
    im3 = abs(v_i_list[3] - v_i_list[2])

    cm1 = normalized(cm1)
    cm2 = normalized(cm2)
    cm3 = normalized(cm3)
    im1 = normalized(im1)
    im2 = normalized(im2)
    im3 = normalized(im3)

    v_s = 0.5*(cm1+cm2+cm3) + 0.5*(im1+im2+im3)

    # myutil.mayavi_with_custom_point(vertices, faces, v_s)
    # mlab.show()
    return v_s
