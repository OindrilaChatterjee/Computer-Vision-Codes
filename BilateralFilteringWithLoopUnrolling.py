import numpy as np
from skimage.io import imread, imsave


def bfilt(X,K,sgm_s,sgm_i):
    ## X :input image
    ## K :Kernel matrix
    ## sgm_s : spatial stndard deviation
    ## sgm_i : image standard deviation
    
    Y = np.zeros_like(X)
    [H,W,D] = X.shape
    Bsum = np.empty([H,W])
    for ky in range(-K,K+1):
        for kx in range(-K,K+1):
            Xshifted = np.roll(X,(ky,kx),axis=(0,1))
            if ky < 0:
                oly1 = 0
                oly2 = H+ky
            else:
                oly1 = ky
                oly2 = H
            if kx < 0:
                olx1 = 0
                olx2 = W+kx
            else:
                olx1 = kx
                olx2 = W
            G1 = np.exp(-(ky**2 + kx**2)/(2*(sgm_s**2)))
            imgdiff =  Xshifted-X
            imgdiffwindow = imgdiff[oly1:oly2,olx1:olx2,:]
            intensity_dist = np.square(imgdiffwindow[:,:,0])+np.square(imgdiffwindow[:,:,1])+np.square(imgdiffwindow[:,:,2])
            G2 = np.exp(-intensity_dist/(2*(sgm_i**2)))
            B = G1*G2
            Y[oly1:oly2,olx1:olx2,0] += B*Xshifted[oly1:oly2,olx1:olx2,0] 
            Y[oly1:oly2,olx1:olx2,1] += B*Xshifted[oly1:oly2,olx1:olx2,1] 
            Y[oly1:oly2,olx1:olx2,2] += B*Xshifted[oly1:oly2,olx1:olx2,2] 
            Bsum[oly1:oly2,olx1:olx2] += B
    Y[:,:,0] = Y[:,:,0] /Bsum
    Y[:,:,1] = Y[:,:,1] /Bsum
    Y[:,:,2] = Y[:,:,2] /Bsum
    
    return Y

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



########################### Support code 

def clip(im):
    return np.maximum(0.,np.minimum(1.,im))

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

img1 = np.float32(imread(fn('inputs/input_img1.png')))/255.
img2 = np.float32(imread(fn('inputs/input_img2.png')))/255.

K=9

print("Creating outputs/output_img1_a.png")
im1A = bfilt(img1,K,2,0.5)
imsave(fn('outputs/output_img1_a.png'),clip(im1A))


print("Creating outputs/output_img1_b.png")
im1B = bfilt(img1,K,4,0.25)
imsave(fn('outputs/output_img1_b.png'),clip(im1B))

print("Creating outputs/output_img1_c.png")
im1C = bfilt(img1,K,16,0.125)
imsave(fn('outputs/output_img1_c.png'),clip(im1C))

# Repeated application
print("Creating outputs/output_img1_rep.png")
im1D = bfilt(img1,K,2,0.125)
for i in range(8):
    im1D = bfilt(im1D,K,2,0.125)
imsave(fn('outputs/output_img1_rep.png'),clip(im1D))

# Try this on image with more noise    
print("Creating outputs/output_img2.png")
im2D = bfilt(img2,2,8,0.125)
for i in range(16):
    im2D = bfilt(im2D,K,2,0.125)
imsave(fn('outputs/output_img2_rep.png'),clip(im2D))
