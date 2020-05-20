import numpy as np
from skimage.io import imread, imsave

def im2wv(img,nLev):
    
    ## img : input image
    ## nLev : number of levels

    if nLev == 0:
        return [img]
    
    hA = (img[0::2,:] + img[1::2,:])/2.
    hB = (-img[0::2,:] + img[1::2,:])/2.
    L = hA[:,0::2]+hA[:,1::2]
    h1 = hB[:,0::2]+hB[:,1::2]
    h2 = -hA[:,0::2]+hA[:,1::2]
    h3 = -hB[:,0::2]+hB[:,1::2]


    return [[h1,h2,h3]] + im2wv(L,nLev-1)

 
def wv2im(pyr):

    while len(pyr) > 1:
        L0 = pyr[-1]

        Hs = pyr[-2]
        H1 = Hs[0]
        H2 = Hs[1]
        H3 = Hs[2]
        
        
        sz = L0.shape
        L = np.zeros([sz[0]*2,sz[1]*2],dtype=np.float32)

        L[::2,::2] = (L0-H1-H2+H3)/2.
        L[1::2,::2] = (L0+H1-H2-H3)/2.
        L[::2,1::2] = (L0-H1+H2-H3)/2.
        L[1::2,1::2] = (L0+H1+H2+H3)/2.
        
        pyr = pyr[:-2] + [L]

    return pyr[0]

# Return corresponding coefficients x (same shape/size)
# that minimizes (x - y)^2 + lmbda * abs(x)
def denoise_coeff(y,lmbda):
    x = np.zeros_like(y)
    x[np.where(np.abs(y)>0.5*lmbda)] = y[np.where(np.abs(y)>0.5*lmbda)]-0.5*lmbda*np.sign(y[np.where(np.abs(y)>0.5*lmbda)])
    return x



########################## Support code 

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Clip intensities b/w 0 and 1
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############# Main Program

lmain = 0.88

img = np.float32(imread(fn('inputs/input_image.png')))/255.

pyr = im2wv(img,4)
for i in range(len(pyr)-1):
    for j in range(2):
        pyr[i][j] = denoise_coeff(pyr[i][j],lmain/(2**i))
    pyr[i][2] = denoise_coeff(pyr[i][2],np.sqrt(2)*lmain/(2**i))
    
im = wv2im(pyr)        
imsave(fn('outputs/output_image.png'),clip(im))
