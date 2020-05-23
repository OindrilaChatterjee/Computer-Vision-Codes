import numpy as np


#########################################
### Hamming distance computation
### Calls the function hamdist with two
### uint32 bit arrays of the same size. It will
### return another array of the same size with
### the elmenet-wise hamming distance.
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

## Builds cost volume
    
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

# Does SGM. First computes the augmented / smoothed cost volumes along 4
# directions (LR, RL, UD, DU), and then computes the disparity map as
# the argmin of the sum of these cost volumes. 
def SGM(cv,P1,P2):
    
    [H,W,D] = cv.shape
    cbarlr = np.copy(cv)
    cbarrl = np.copy(cv) 
    cbarud = np.copy(cv)
    cbardu = np.copy(cv)
    
    # left to right    
    for x in range(1,W):
        compmat = np.zeros([H,D,4])
        compmat[:,:,0] = P2*np.ones([H,D])
        cbarprevcol = cbarlr[:,x-1,:]
        mincbprevcol = np.min(cbarprevcol,axis=1)
        compmat[:,:,1] = np.roll(cbarprevcol,-1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([H,D])
        compmat[:,:,3] = np.roll(cbarprevcol,1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([H,D])
        compmat[:,:,2] = cbarprevcol-mincbprevcol[:,np.newaxis]
        mincomp = np.min(compmat,axis=2)
        cbarlr[:,x,:] = mincomp+mincbprevcol[:,np.newaxis]+ cv[:,x,:]
        
  # right to left      
    for x in range(W-2,-1):
        compmat = np.zeros([H,D,4])
        compmat[:,:,0] = P2*np.ones([H,D])
        cbarprevcol = cbarrl[:,x+1,:]
        mincbprevcol = np.min(cbarprevcol,axis=1)
        compmat[:,:,1] = np.roll(cbarprevcol,-1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([H,D])
        compmat[:,:,3] = np.roll(cbarprevcol,1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([H,D])
        compmat[:,:,2] = cbarprevcol-mincbprevcol[:,np.newaxis]
        mincomp = np.min(compmat,axis=2)
        cbarrl[:,x,:] = mincomp+mincbprevcol[:,np.newaxis]+ cv[:,x,:]
    
  # up to down      
    for x in range(1,H):
        compmat = np.zeros([W,D,4])
        compmat[:,:,0] = P2*np.ones([W,D])
        cbarprevcol = cbarud[x-1,:,:]
        mincbprevcol = np.min(cbarprevcol,axis=1)
        compmat[:,:,1] = np.roll(cbarprevcol,-1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([W,D])
        compmat[:,:,3] = np.roll(cbarprevcol,1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([W,D])
        compmat[:,:,2] = cbarprevcol-mincbprevcol[:,np.newaxis]
        mincomp = np.min(compmat,axis=2)
        cbarud[x,:,:] = mincomp+mincbprevcol[:,np.newaxis]+ cv[x,:,:]    
        
    
      # down to up    
    for x in range(H-2,-1):
        compmat = np.zeros([W,D,4])
        compmat[:,:,0] = P2*np.ones([W,D])
        cbarprevcol = cbardu[x+1,:,:]
        mincbprevcol = np.min(cbarprevcol,axis=1)
        compmat[:,:,1] = np.roll(cbarprevcol,-1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([W,D])
        compmat[:,:,3] = np.roll(cbarprevcol,1,axis=1)-mincbprevcol[:,np.newaxis]+P1*np.ones([W,D])
        compmat[:,:,2] = cbarprevcol-mincbprevcol[:,np.newaxis]
        mincomp = np.min(compmat,axis=2)
        cbardu[x,:,:] = mincomp+mincbprevcol[:,np.newaxis]+ cv[x,:,:]    

    cbar = cbarlr+cbarrl+cbarud+cbardu
#    dmap = np.argmin(cbar,axis=2)


    return np.argmin(cbar,axis=2)

    
    
########################## Support code below

from skimage.io import imread, imsave
from os.path import normpath as fn # Fixes window/linux path conventions
import matplotlib.cm as cm
import warnings
warnings.filterwarnings('ignore')


left = np.float32(imread(fn('inputs/left.jpg')))/255.
right = np.float32(imread(fn('inputs/right.jpg')))/255.

left_g = np.mean(left,axis=2)
right_g = np.mean(right,axis=2)
                   
cv = buildcv(left_g,right_g,50)
d = SGM(cv,0.5,16)

# Map to color and save
dimg = cm.jet(np.minimum(1,np.float32(d.flatten())/50.))[:,0:3]
dimg = dimg.reshape([d.shape[0],d.shape[1],3])
imsave(fn('outputs/output.jpg'),dimg)
