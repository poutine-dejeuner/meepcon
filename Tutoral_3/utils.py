import os
import numpy as np


def double_with_mirror(image):
    channels = '~/scratch/nanophoto/lowfom/nodata/fields/channels.npy'
    channels = np.load(os.path.expanduser(channels))
    mirrored_image = np.fliplr(image)  # Crée l'image miroir
    doubled_image = np.concatenate((mirrored_image[:, :-1], image), axis=1)
    return doubled_image

def normalise(image):
    image = (image - image.min()) / (image.max() - image.min())
    return image

def npstats(arr):
    print('mean', np.mean(arr))
    print('std', np.std(arr))
    print('min', np.min(arr))
    print('max', np.max(arr))