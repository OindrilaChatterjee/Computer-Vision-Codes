import numpy as np


#########################################
### Hamming distance computation
### Calls the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the element-wise hamming distance.
hd8bit = np.zeros((256,))
for i in range(256):
    v = i
    for k in range(8):
        hd8bit[i] = hd8bit[i] + v%2
        v=v//2


def hamdist(x,y):
    dist = np.zeros(x.shape)
    g = x^y
    for i in range(4):
        dist = dist + hd8bit[g%256]
        g = g // 256
    return dist
#########################################


# Given left and right grayscale images and max disparity D_max, builds a HxWx(D_max+1) array
# corresponding to the cost volume. For disparity d where x-d < 0, fills a cost
# value of 24 (the maximum possible hamming distance).
def census(img):
    
    [H,W] = img.shape
    img_pad = np.pad(img,(2,),'constant',constant_values=(255,))
    c = np.zeros([H,W],dtype=np.uint32)
    ind = 0
    for i in range(5):
        for j in range (5):
            if (abs(i)+abs(j)!=0):
                neighbormat = img_pad[i:i+H,j:j+W]
                csum = np.array(1*(img>neighbormat),dtype=np.uint32)
                c = c+pow(2,23-ind)*csum
                ind+=1
    
    return c

## Build cost volume using hamming distance based on census transform
def buildcv(left,right,dmax):
    d = np.zeros(left.shape)
    cv = 24 * np.ones([left.shape[0],left.shape[1],dmax+1], dtype=np.float32)
    cenleft = census(left)
    cenright = census(right)
    [H,W] = left.shape
    for i in range(0,dmax+1):
        hd = hamdist(cenleft[:,i:],cenright[:,0:W-i])
        cv[:,i:,i] = hd
    return cv



## CV is the cost-volume to be filtered.
## X is the left color image that will serve as guidance.
## K is the support of the filter (2K+1)x(2K+1)
## sgm_s is std of spatial gaussian
## sgm_i is std of intensity gaussian

## Implement bilateral filtering
def bfilt(cv,X,K,sgm_s,sgm_i):
    
    Y = np.zeros_like(cv)
    [H,W,D] = X.shape
    Bsum = np.empty([H,W])
    for ky in range(-K,K+1):
        for kx in range(-K,K+1):
            Xshifted = np.roll(X,(ky,kx),axis=(0,1))
            cvshifted = np.roll(cv,(ky,kx),axis=(0,1))
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
            Y[oly1:oly2,olx1:olx2,:] += cvshifted[oly1:oly2,olx1:olx2,:]*B[:,:,np.newaxis]
            Bsum[oly1:oly2,olx1:olx2] += B
    Y = np.divide(Y,Bsum[:,:,np.newaxis])


    return Y
    
########################## Support code 

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)

                   
cv0 = buildcv(left_g,right_g,50)

cv1 = bfilt(cv0,left,5,2,0.5)
    

d0 = np.argmin(cv0,axis=2)
d1 = np.argmin(cv1,axis=2)
#
## Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d0.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d0.shape[0],d0.shape[1],3])
imsave(fn('outputs/output_nofilter.jpg'),dimg)

dimg = cm.jet(np.minimum(1,np.float32(d1.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d1.shape[0],d1.shape[1],3])
imsave(fn('outputs/output_bilateral.jpg'),dimg)
