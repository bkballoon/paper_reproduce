import matplotlib.pyplot as plt
import numpy as np
from mayavi import mlab
import yang.myutil as myutil

def write_obj_only_vf(vertices, faces):
    write_file = open("bunny_change.obj", "a")
    for i in vertices:
        v = 'v' + ' ' + str(i[0]) + ' ' + str(i[1]) + ' ' + str(i[2]) + '\n'
        write_file.write(v)
    for i in faces:
        f = 'f' + ' ' + str(i[0]+1) + ' ' + str(i[1]+1) + ' ' + str(i[2]+1) + '\n'
        write_file.write(f)
    write_file.close()

def visual_mc_saliency(path):
    file = open(path)
    lines = file.readlines()
    feature_map = np.array([np.float(s) for s in lines])
    return feature_map

def compute_triangle_area(vertices, faces, face_index):
    three_point = faces[face_index]
    p1, p2, p3 = vertices[three_point[0]], vertices[three_point[1]], vertices[three_point[2]]
    return myutil.norm(np.cross(p1 - p2, p1 - p3))

def compute_mean_curvature():
    obj_data = myutil.load_obj("/home/simula/Dataset/textured_model/buddha2/buddha2.obj")
    vertices, faces = obj_data[0], obj_data[1]
    feature_map = []
    v_to_face = myutil.get_vertex_to_face(faces)
    # feature_map = visual_mc_saliency()
    # mlab = myutil.mayavi_with_custom_point(vertices, faces, feature_map)
    # mlab.show()
    # exit()

    mc_curvature = []
    for vertex_index in range(len(vertices)):
        one_ring_set = myutil.get_one_ring_vertex(v_to_face, vertex_index, faces)
        one_ring_set.remove(vertex_index)
        sum = 0
        area = 0
        for one_ring_index in one_ring_set:
            cot = 0
            vertex_faces = v_to_face[vertex_index]
            onering_faces = v_to_face[one_ring_index]
            a = set(vertex_faces)
            b = set(onering_faces)
            c = a.intersection(b)  # a = { face_index1, face_index2 }
            for faces_index in c:  # 对两条边合成的面片index进行测量
                for i in faces[faces_index]:  # 对每个面片计算其余切值
                    if i != vertex_index and i != one_ring_index:
                        A_line1 = vertices[one_ring_index] - vertices[i]
                        A_line2 = vertices[vertex_index] - vertices[i]
                        l1 = myutil.norm(A_line1)
                        l2 = myutil.norm(A_line2)
                        c = A_line2.dot(A_line1)
                        cos_ = c / l1 / l2

                        angle = np.arccos(cos_)
                        cot_ = 1 / np.tan(angle)
                        cot += cot_
            weight = cot
            # 公式中的f_i - f_j无法理解，按照别人的代码来看，好像边向量的模
            sum += weight * np.linalg.norm(vertices[vertex_index] - vertices[one_ring_index])

        for i in v_to_face[vertex_index]:
            area += compute_triangle_area(i)

        mc_curvature.append(sum / area)

    print("mc = ", mc_curvature[:10])
    mlab = myutil.mayavi_with_custom_point(vertices, faces, mc_curvature)
    mlab.show()


# path = "/home/simula/Dataset/textured_model/buddha2/saliency.txt"
# obj_data = myutil.load_obj("/home/simula/Dataset/textured_model/buddha2/buddha2.obj")
# vertices, faces = obj_data[0], obj_data[1]
# feature_map = visual_mc_saliency(path)
# mlab = myutil.mayavi_with_custom_point(vertices, faces, feature_map)
# mlab.show()
