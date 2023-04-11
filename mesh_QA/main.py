import sys
import csv
import os
import time

from gpu.gpuRelateFace import AccRelateFace
from gpu.gpuAccEigen   import AccEigen
from gpu.gpuAccLTD     import AccLTD
from gpu.gpuAccRW      import AccRW

from utils import readAllObj
# 一个加速的可选方法，将所有的数据全部load进来，然后直接在函数之间传递
# 自然的就减少了运行时间

def main():
    # ref_file_path = 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm.obj'
    # dis_file_path = 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Noise001.obj'
    
    data_uppper_path = 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\'
    path_lists = ['E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Noise0005.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Noise00075.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Noise001.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseRough0005.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseRough00075.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseRough001.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseRoughHalf0005.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseRoughHalf00075.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseRoughHalf001.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseSmooth0005.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseSmooth00075.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-NoiseSmooth001.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Taubin10.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Taubin15.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-Taubin20.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-TaubinRough10.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-TaubinRough15.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-TaubinRough20.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-TaubinRoughHalf10.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-TaubinRoughHalf15.obj', 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm-TaubinRoughHalf20.obj']
    # print(path_lists)
    for i in path_lists[:1]:
        print("enter the array = {}".format(i))
        ref_file_path = 'E:\\zy_QA\\LIRIS_EPFL_GenPurpose\\RockerArm_models\\rockerArm.obj'
        dis_file_path = i
        name = i.split('\\')[-1][:-4]

        upper_dir = "E:\\zy_QA\\SVR-mesh-v3\\data\\noise" + str(name)
        os.mkdir(upper_dir)

        AccRelateFace(ref_file_path, dis_file_path, upper_dir)

        # 0 means ref mesh while 1 means dis mesh
        AccEigen(ref_file_path, upper_dir, 0)
        AccEigen(dis_file_path, upper_dir, 1)
        AccRW(upper_dir, ref_file_path)
        ref_vv_path = upper_dir + '\\eigen_0_vv.csv'
        dis_vv_path = upper_dir + '\\eigen_1_vv.csv'
        related_path = upper_dir + '\\related_face.csv'
        dis_mesh_path = dis_file_path
        AccLTD(related_path, ref_vv_path, dis_vv_path, dis_mesh_path, upper_dir)

if __name__ == "__main__":
    
    # meshpath = "/home/simula/Pro/textured_mesh_saliency/yang/geometry/arma.obj"

    meshpath = "/home/simula/Dataset/textured_model/bunny/bunny.obj"
    AccEigen(meshpath, "none", 0)

    # main()

