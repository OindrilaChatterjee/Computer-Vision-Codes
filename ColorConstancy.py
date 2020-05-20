import numpy as np
from skimage.io import imread, imsave

### METHOD 1

## Takes in color image, and returns 'white balanced' color image. For each
## channel, find the average intensity across all pixels.
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.

def balance2a(img):
    imgflat = np.reshape(img,(-1,3)) 
    alphavec = np.mean(imgflat,axis=0)
    alphamat = np.diag(np.reciprocal(alphavec))
    alphamat = alphamat*3/np.trace(alphamat)
    Yflat = np.matmul(imgflat,alphamat.T)
    Y = np.reshape(Yflat,img.shape) 
    return Y


### METHOD 2

## Take color image, and return 'white balanced' color image. In each channel, find
## top 10% of the brightest intensities, take their average.
## Now multiply each channel by multipliers that are inversely
## proportional to these averages, but add upto 3.
def balance2b(img):
    imgflat = np.reshape(img,(-1,3)) 
    intensitysort = np.sort(imgflat,axis=0)[::-1,:]
    k = np.around(0.1*imgflat.shape[0])
    intmat = imgflat[:np.int32(k)+1,:]
    alphavec = np.mean(intmat,axis=0)
    alphamat = np.diag(np.reciprocal(alphavec))
    alphamat = alphamat*3/np.trace(alphamat)
    Yflat = np.matmul(imgflat,alphamat.T)
    Y = np.reshape(Yflat,img.shape) 
    return Y



########################## Support code

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')


# Clip intensities b/w 0 and 1
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))



############## Main Program
im1 = np.float32(imread(fn('inputs/CC/input_img1.jpg')))/255.
im2 = np.float32(imread(fn('inputs/CC/einput_img2.jpg')))/255.
im3 = np.float32(imread(fn('inputs/CC/input_img3.jpg')))/255.


im1a = balance2a(im1)
im2a = balance2a(im2)
im3a = balance2a(im3)

imsave(fn('outputs/output_img1_M1.png'),clip(im1a))
imsave(fn('outputs/output_img2_M1.png'),clip(im2a))
imsave(fn('outputs/output_img3_M1.png'),clip(im3a))

im1b = balance2b(im1)
im2b = balance2b(im2)
im3b = balance2b(im3)

imsave(fn('outputs/output_img1_M2.png'),clip(im1b))
imsave(fn('outputs/output_img2_M2.png'),clip(im2b))
imsave(fn('outputs/output_img3_M2.png'),clip(im3b))
