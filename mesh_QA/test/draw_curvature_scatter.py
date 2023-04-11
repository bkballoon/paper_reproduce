from matplotlib import pyplot as plt
import numpy as np
import csv

import seaborn as sns


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