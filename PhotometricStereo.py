import numpy as np
from skimage.io import imread, imsave
from scipy import linalg as linear

# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns nrm:
#    nrm: HxWx3 Unit normal vector at each location.

def pstereo_n(imgs, L, mask): ## finds normal vectors at each location
    intmat = np.asarray([np.sum(np.reshape(img,(-1,3)),axis=1) for img in imgs])*mask.flatten()
    b = np.matmul(L.T,intmat)
    Q = np.matmul(L.T,L)
    nmat = linear.solve(Q,b)
    normvec = np.zeros((1,nmat.shape[1]))
    normvec = np.broadcast_to(linear.norm(nmat,axis=0),(nmat.shape))
    nmat_norm = np.divide(nmat,normvec)
    nmat_norm[np.where(1*np.isnan(nmat_norm)==1)]=0
    nrm = np.reshape(nmat_norm.T,imgs[i].shape)   
    return nrm


# Inputs:
#    imgs: A list of N color images, each of which is HxWx3
#    nrm:  HxWx3 Unit normal vector at each location (from pstereo_n)
#    L:    An Nx3 matrix where each row corresponds to light vector
#          for corresponding image.
#    mask: A 0-1 mask of size HxW, showing where observed data is 'valid'.
#
# Returns alb:
#    alb: HxWx3 RGB Color Albedo values

def pstereo_alb(imgs, nrm, L, mask):  ## finds albedo values at each location
    intmat = np.asarray([np.reshape((img*mask[:,:,np.newaxis]),(-1,3)) for img in imgs])
    Lnmat = np.matmul(L,np.reshape(nrm,(-1,3)).T)
    Lnmatsqr = np.sum(np.square(Lnmat),axis = 0)
    ILnmat = np.sum(intmat*Lnmat[:,:,np.newaxis],axis=0).T
    albvec = np.divide(ILnmat,np.broadcast_to(Lnmatsqr,(ILnmat.shape)))
    albvec[np.where(1*np.isnan(albvec)==1)]=0
    alb = np.reshape(albvec.T,imgs[i].shape)
    return alb
    
########################## Support code

from os.path import normpath as fn # Fixes window/linux path conventions
import warnings
warnings.filterwarnings('ignore')

### Light directions matrix
L = np.float32( \
                [[  4.82962877e-01,   2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,   2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,   2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,   2.58819044e-01,   8.36516261e-01],
                 [ -5.00000060e-01,   0.00000000e+00,   8.66025388e-01],
                 [ -2.58819044e-01,   0.00000000e+00,   9.65925813e-01],
                 [ -4.37113883e-08,   0.00000000e+00,   1.00000000e+00],
                 [  2.58819073e-01,   0.00000000e+00,   9.65925813e-01],
                 [  4.99999970e-01,   0.00000000e+00,   8.66025448e-01],
                 [  4.82962877e-01,  -2.58819044e-01,   8.36516321e-01],
                 [  2.50000030e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.22219593e-08,  -2.58819044e-01,   9.65925813e-01],
                 [ -2.50000000e-01,  -2.58819044e-01,   9.33012664e-01],
                 [ -4.82962966e-01,  -2.58819044e-01,   8.36516261e-01]])


# Clip intensities b/w 0 and 1
def clip(im):
    return np.maximum(0.,np.minimum(1.,im))


############# Main Program


# Load image data
imgs = []
for i in range(L.shape[0]):
    imgs = imgs + [np.float32(imread(fn('inputs/phstereo/input_image.png' % i)))/255.]

mask = np.float32(imread(fn('inputs/phstereo/mask.png')) > 0)

nrm = pstereo_n(imgs,L,mask)

nimg = nrm/2.0+0.5
nimg = clip(nimg * mask[:,:,np.newaxis])
imsave(fn('outputs/output_normals.png'),nimg)


alb = pstereo_alb(imgs,nrm,L,mask)

alb = alb / np.max(alb[:])
alb = clip(alb * mask[:,:,np.newaxis])

imsave(fn('outputs/output_albedos.png'),alb)
