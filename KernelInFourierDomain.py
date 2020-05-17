import numpy as np
from skimage.io import imread, imsave
from scipy.signal import convolve2d as conv2

def kernpad(K,size):
    ksize = np.asarray(K.shape)
    padsize = np.asarray(size)-ksize
    t = -(ksize-1)/2
    tfin = tuple(t.astype(int))
    Ko1 = np.lib.pad(K,((0,padsize[0]),(0,padsize[1])),'constant')
    Ko = np.roll(Ko1,tfin,axis=(0,1))  
    return Ko



########################## Support code 

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img = np.float32(imread(fn('inputs/input_image.png')))/255.

# Create Gaussian Kernel
x = np.float32(range(-21,22))
x,y = np.meshgrid(x,x)
G = np.exp(-(x*x+y*y)/2/9.)
G = G / np.sum(G[:])


# Traditional convolve
v1 = conv2(img,G,'same','wrap')

# Convolution in Fourier domain
G = kernpad(G,img.shape)
v2f = np.fft.fft2(G)*np.fft.fft2(img)
v2 = np.real(np.fft.ifft2(v2f))

# Stack them together and save
out = np.concatenate([img,v1,v2],axis=1)
out = np.minimum(1.,np.maximum(0.,out))

imsave(fn('outputs/output_image.png'),out)


                 
