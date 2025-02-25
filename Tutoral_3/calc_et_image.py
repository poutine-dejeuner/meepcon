import os
import numpy as np
import matplotlib.pyplot as plt

from calculFOM import compute_FOM_parallele 

file = 'generated_tree_to_high_fom.npy'
path = '~/scratch/nanophoto/'
path = os.path.join(path, file)
path = os.path.expanduser(path)

images = np.load(path)

fom = compute_FOM_parallele(images)

