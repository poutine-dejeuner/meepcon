import os
import numpy as np
from scipy import fftpack
from icecream import ic
from scipy.signal import convolve2d


def smooth_image_fft(image, sigma, resolution=30):
    """
    Takes the convlution of an image with a gaussian with variance sigma.
    The convolution is performed in the Fourier domain.
    """
    if image.ndim == 2:
        image = np.expand_dims(image, 2)
    t = np.linspace(0, 10, resolution)
    bump = np.exp(-t**2/sigma)
    bump /= np.trapz(bump)
    kernel = bump[:, np.newaxis] * bump[np.newaxis, :]
    kernel_ft = fftpack.fft2(kernel, shape=image.shape[:2], axes=(0, 1))
    ic(kernel_ft.shape)

    img_ft = fftpack.fft2(image, axes=(0, 1))
    ic(img_ft.shape)
    img2_ft = kernel_ft[:, :, np.newaxis] * img_ft
    img2 = fftpack.ifft2(img2_ft, axes=(0, 1)).real

    img2 = np.clip(img2, 0, 1)
    img2 = img2.squeeze()

    return img2


def smooth_image(image, sigma=20.):                       
    """                                                      
    Smooths the image from trees by convolution with a gaussi an in
    image: (N,1,d1,d2) torch.float tensor
    """
    def gkern(l, sigma):                                     
        """                                                  
        creates gaussian kernel with side length l and varian
ce sigma                                                     
        """                                                  
        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)     
        gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma
))                                                           
        kernel = np.outer(gauss, gauss)                      
        return kernel / np.sum(kernel)                       
    l = int(sigma*5)
    kernel = gkern(l, sigma)
    out = convolve2d(image, kernel, mode='same', boundary='symm')
                                                                  
    return out


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
