from PIL import ImageOps
import numpy as np
import sys
sys.path.append("..")
import myutil
# import yang.myutil as myutil
from geometry.test_vertex_saliency import visual_mc_saliency
from matplotlib import pyplot as plt
# from skimage import exposure
# from texture.inverse_map import start_to_compute
# from texture.texel import compute_color_saliency
from mayavi import mlab


def write_vs(path, vertex_saliency):
    file = open(path, 'a')
    for i in vertex_saliency:
        file.write(str(i))
        file.write('\n')

def normalized_array(li):
    li_min, li_max = min(li), max(li)
    li = np.array(li)
    return li / (li_max - li_min)

def read_vs(path):
    vs = open(path, "r")
    liens = vs.readlines()
    v_s = []
    for i in liens:
        v_s.append(np.float(i))
    return v_s

dataname = "bunny"
# obj_path = "/home/simula/Dataset/textured_model/buddha2/buddha2.obj"
# obj_path = "/home/simula/Dataset/textured_model/bunny/bunny.obj"
# obj_path = "/home/simula/Dataset/textured_model/feline/feline.obj"
obj_path = "E:\\mycodes\\Dataset\\textured_model\\" + dataname +"\\" + dataname + ".obj"

obj_data = myutil.load_obj(obj_path)
vertices, faces = obj_data[0], obj_data[1]

# path = "/home/simula/Dataset/textured_model/buddha2/saliency.txt"
# path = "/home/simula/Dataset/textured_model/bunny/saliency.txt"
path = "/home/simula/Dataset/textured_model/" + dataname +"/" + "saliency.txt"
path_aug = "/home/simula/Pic/paper_" + dataname + "/aug/cat_aug.txt"

# img_path = "/home/simula/Dataset/textured_model/buddha2/buddha2-atlas.jpg"
# img_path = "/home/simula/Dataset/textured_model/bunny/bunny-atlas.jpg"
img_path = "/home/simula/Dataset/textured_model/" + dataname +"/" + dataname + "-atlas.jpg"

# texture_saliency_list = start_to_compute(img_path, obj_path)
# texture_saliency_list = compute_color_saliency(img_path, obj_path)
# geometry_saliency_list = visual_mc_saliency(path)

# 对得到的顶点显著度列表进行一次直方图均衡化，将几何显著度进行全域扩散
# geometry_saliency_list = exposure.equalize_hist(geometry_saliency_list)
# texture_saliency_list = exposure.equalize_hist(texture_saliency_list)

# texture_saliency_list = normalized_array(texture_saliency_list)
# geometry_saliency_list = normalized_array(geometry_saliency_list)

# yang_saliency_list = geometry_saliency_list
# yang_saliency_list = texture_saliency_list
yang_saliency_list = read_vs("E:\\mycodes\\Dataset\\pic\\paper_bunny\\aug\\bunny_yang.txt")
# yang_saliency_list = 0.8 * texture_saliency_list + 0.2 * geometry_saliency_list
# yang_saliency_list = normalized_array(yang_saliency_list)
# write_vs(dataname+"_yang.txt", yang_saliency_list)


fig = mlab.figure(size=(512, 512 + 48), bgcolor=(1, 1, 1))
mlab = myutil.mayavi_with_custom_point(vertices, faces, yang_saliency_list)
mlab.show()

'''
CC -> 0.5575164384540074
SIM-> 0.8195359904480696
KLD-> 0.5790606296554135
Yang
CC -> 0.5422093080054537
SIM-> 0.8159725636251751
KLD-> 0.5825618959057862
'''